"""
Shuttle tracking with TrackNetV3 (third_party/tracknetv3, MIT licence).

TrackNet finds the shuttle in a heatmap from 8 consecutive frames plus the median background,
then InpaintNet fills short gaps in the trajectory. This mirrors the "nonoverlap" mode of the
upstream predict.py, run in-process. On hand-labelled clean broadcast frames it is far more
accurate than our YOLO + motion tracker (see eval_shuttle_tracking.py).

Weights (not in git, 142 MB): models/tracknetv3/TrackNet_best.pt and InpaintNet_best.pt from
TrackNetV3_ckpts.zip linked in the upstream README.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from badminton.court import PROJECT_ROOT
from third_party.tracknetv3.dataset import Shuttlecock_Trajectory_Dataset
from third_party.tracknetv3.utils.general import COOR_TH, HEIGHT, WIDTH, get_model

WEIGHTS_DIR = PROJECT_ROOT / "models" / "tracknetv3"
TRACKNET_WEIGHTS = WEIGHTS_DIR / "TrackNet_best.pt"
INPAINTNET_WEIGHTS = WEIGHTS_DIR / "InpaintNet_best.pt"
CHUNK_FRAMES = 300   # frames held in memory at once (~1.5 GB peak at 720p); each chunk gets its own median background


def available() -> bool:
    return TRACKNET_WEIGHTS.is_file() and INPAINTNET_WEIGHTS.is_file()


def _heatmap_point(heatmap: np.ndarray) -> Tuple[int, int]:
    """Centre of the largest blob in a thresholded heatmap, (0, 0) if empty (upstream predict_location)."""
    if heatmap.max() == 0:
        return 0, 0
    contours, _ = cv2.findContours(heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = max((cv2.boundingRect(c) for c in contours), key=lambda r: r[2] * r[3])
    return int(x + w / 2), int(y + h / 2)


def _inpaint_mask(ys, vis, th_h: float):
    """Frames inside a gap that should be filled: the shuttle vanished and came back on court
    (upstream generate_inpaint_mask)."""
    ys, vis = np.asarray(ys), np.asarray(vis)
    mask = np.zeros(len(vis))
    i = j = 0
    while j < len(vis):
        while i < len(vis) - 1 and vis[i] == 1:
            i += 1
        j = i
        while j < len(vis) - 1 and vis[j] == 0:
            j += 1
        if j == i:
            break
        if i == 0 and ys[j] > th_h:
            mask[:j] = 1
        elif i > 1 and ys[i - 1] > th_h and ys[j] > th_h:
            mask[i:j] = 1
        i = j
    return mask.tolist()


class TrackNetShuttle:
    def __init__(self, device: Optional[str] = None, batch_size: int = 8):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        ck = torch.load(TRACKNET_WEIGHTS, map_location=self.device, weights_only=True)
        self.seq_len, self.bg_mode = ck["param_dict"]["seq_len"], ck["param_dict"]["bg_mode"]
        self.tracknet = get_model("TrackNet", self.seq_len, self.bg_mode).to(self.device).eval()
        self.tracknet.load_state_dict(ck["model"])
        ck = torch.load(INPAINTNET_WEIGHTS, map_location=self.device, weights_only=True)
        self.inpaint_len = ck["param_dict"]["seq_len"]
        self.inpaintnet = get_model("InpaintNet").to(self.device).eval()
        self.inpaintnet.load_state_dict(ck["model"])

    def _collect(self, indices, points, out):
        """Keeps the first prediction of each frame index (sequences are padded at the end)."""
        for seq_i, seq_p in zip(indices, points):
            prev = -1
            for (_, f), p in zip(seq_i.tolist(), seq_p):
                if f == prev:
                    break
                out.setdefault(int(f), p)
                prev = f

    @torch.no_grad()
    def _track_frames(self, frames) -> Dict[int, Optional[Tuple[int, int]]]:
        h, w = frames[0].shape[:2]
        sx, sy = w / WIDTH, h / HEIGHT
        rgb = np.array(frames)[:, :, :, ::-1]

        # 1) TrackNet heatmaps -> pixel points
        data = Shuttlecock_Trajectory_Dataset(seq_len=self.seq_len, sliding_step=self.seq_len, data_mode="heatmap",
                                              bg_mode=self.bg_mode, frame_arr=rgb, padding=True)
        raw = {}
        for idx, x in DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=0):
            heat = (self.tracknet(x.float().to(self.device)) > 0.5).cpu().numpy()
            pts = [[_heatmap_point((hm * 255).astype(np.uint8)) for hm in seq] for seq in heat]
            pts = [[(int(px * sx), int(py * sy)) for px, py in seq] for seq in pts]
            self._collect(idx, pts, raw)
        order = sorted(raw)
        pred = {"Frame": order, "X": [raw[f][0] for f in order], "Y": [raw[f][1] for f in order],
                "Visibility": [0 if raw[f] == (0, 0) else 1 for f in order]}
        pred["Inpaint_Mask"] = _inpaint_mask(pred["Y"], pred["Visibility"], th_h=h * 0.05)
        pred["Img_scaler"], pred["Img_shape"] = (sx, sy), (w, h)

        # 2) InpaintNet fills the masked gaps
        data = Shuttlecock_Trajectory_Dataset(seq_len=self.inpaint_len, sliding_step=self.inpaint_len,
                                              data_mode="coordinate", pred_dict=pred, padding=True)
        final = {}
        for idx, coor, mask in DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=0):
            coor, mask = coor.float(), mask.float()
            fixed = self.inpaintnet(coor.to(self.device), mask.to(self.device)).cpu()
            fixed = fixed * mask + coor * (1 - mask)
            fixed[(fixed[:, :, 0] < COOR_TH) & (fixed[:, :, 1] < COOR_TH)] = 0.0
            pts = [[(int(cx * WIDTH * sx), int(cy * HEIGHT * sy)) for cx, cy in seq.tolist()] for seq in fixed]
            self._collect(idx, pts, final)
        return {f: (None if p == (0, 0) else p) for f, p in final.items()}

    def track_video(self, video_path, max_frames: Optional[int] = None, progress=None):
        """{frame index: (x, y) pixel point or None} for the whole video."""
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        if max_frames:
            total = min(total, max_frames)
        points, start = {}, 0
        while start < total:
            frames = []
            while len(frames) < CHUNK_FRAMES and start + len(frames) < total:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            if len(frames) < self.seq_len:
                points.update({start + i: None for i in range(len(frames))})
                break
            points.update({start + f: p for f, p in self._track_frames(frames).items()})
            start += len(frames)
            if progress:
                progress(start / total)
        cap.release()
        return points
