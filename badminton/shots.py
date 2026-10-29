"""
Shot-type classifier: given the hitting player's pose around a hit (YOLOv8-pose keypoints, 17 joints
x/y/conf in pixels, one row per frame), predict which of 18 shot types was played.

Trained by train_shot_classifier.py on the group member's keypoint dataset (dataset/shot_keypoints.csv,
fixed-camera recordings). Poses are centred on the hips and scaled by body height, so pixel size and
where the player stands in the frame don't matter.
"""

import json
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from badminton.court import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "runs" / "shot_model"
MODEL_PATH = MODEL_DIR / "shot_classifier.pt"

# Label ids 0-17 in the order of the training CSV
SHOT_TYPES = [
    "Short Serve", "Cross Court Flight", "Lift", "Tap Smash", "Block", "Drop Shot", "Push Shot",
    "Transitional Slice", "Cut", "Rush Shot", "Defensive Clear", "Defensive Drive", "Clear", "Long Serve",
    "Smash", "Flat Shot", "Rear Court Flat Drive", "Short Flat Shot",
]

# Coarse groups for when the 18 fine classes are too hard to tell apart
SHOT_GROUPS = ["serve", "clear/lift", "smash", "drop/net", "drive/push", "defensive"]
GROUP_OF = {
    "Short Serve": "serve", "Long Serve": "serve",
    "Lift": "clear/lift", "Clear": "clear/lift",
    "Smash": "smash", "Tap Smash": "smash", "Rush Shot": "smash",   # rush shot = net kill
    "Drop Shot": "drop/net", "Cut": "drop/net", "Transitional Slice": "drop/net",
    "Cross Court Flight": "drive/push", "Push Shot": "drive/push", "Flat Shot": "drive/push",
    "Rear Court Flat Drive": "drive/push", "Short Flat Shot": "drive/push",
    "Block": "defensive", "Defensive Clear": "defensive", "Defensive Drive": "defensive",
}
GROUP_IDS = np.array([SHOT_GROUPS.index(GROUP_OF[name]) for name in SHOT_TYPES])

SEQ_LEN = 30
MIN_CONF = 0.3
# COCO keypoints: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips,
# 13/14 knees, 15/16 ankles (odd = left, even = right)
FLIP_ORDER = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
N_FEATURES = 17 * 2 + 17 + 17 * 2 + 2   # positions, visibility, velocities, hip movement


def normalise_clip(kpts: np.ndarray) -> np.ndarray:
    """
    (T, 17, 3) pixel keypoints -> (T, 17, 3) with x/y centred on the mid-hip and divided by the
    body height, joints below MIN_CONF zeroed (x, y and conf). The hip position is kept separately
    in normalise_root so the player's own movement (lunge, jump) isn't lost.
    """
    kpts = np.asarray(kpts, dtype=np.float32).reshape(-1, 17, 3)
    xy, conf = kpts[..., :2], kpts[..., 2]
    seen = conf >= MIN_CONF
    root = _root_track(xy, seen)
    scale = _body_height(xy, seen)
    out = np.zeros_like(kpts)
    out[..., :2] = (xy - root[:, None, :]) / scale
    out[..., 2] = conf
    out[~seen] = 0.0
    return out


def normalise_root(kpts: np.ndarray) -> np.ndarray:
    """(T, 2) mid-hip position relative to the clip's middle frame, in body heights."""
    kpts = np.asarray(kpts, dtype=np.float32).reshape(-1, 17, 3)
    seen = kpts[..., 2] >= MIN_CONF
    root = _root_track(kpts[..., :2], seen)
    return (root - root[len(root) // 2]) / _body_height(kpts[..., :2], seen)


def _root_track(xy, seen) -> np.ndarray:
    """Mid-hip per frame (mean of the confident joints if neither hip is seen), gaps filled in."""
    root = np.full((len(xy), 2), np.nan, dtype=np.float32)
    for t in range(len(xy)):
        hips = seen[t, [11, 12]]
        if hips.any():
            root[t] = xy[t, [11, 12]][hips].mean(axis=0)
        elif seen[t].any():
            root[t] = xy[t][seen[t]].mean(axis=0)
    ok = ~np.isnan(root[:, 0])
    if not ok.any():
        return np.zeros((len(xy), 2), dtype=np.float32)
    idx = np.arange(len(xy))
    for d in range(2):
        root[:, d] = np.interp(idx, idx[ok], root[ok, d])
    return root


def _body_height(xy, seen) -> float:
    """Median over the clip of the top-to-bottom extent of the confident joints (one scale per clip)."""
    heights = [np.ptp(xy[t, seen[t], 1]) for t in range(len(xy)) if seen[t].sum() >= 4]
    h = float(np.median(heights)) if heights else 0.0
    return h if h > 1.0 else 1.0


def fit_length(seq: np.ndarray, length: int = SEQ_LEN, offset: int = 0) -> np.ndarray:
    """Centre-crop (shifted by offset frames) or edge-pad (repeat first/last frame) to length frames."""
    n = len(seq)
    if n >= length:
        start = int(np.clip((n - length) // 2 + offset, 0, n - length))
        return seq[start:start + length]
    before = (length - n) // 2
    return np.concatenate([np.repeat(seq[:1], before, axis=0), seq,
                           np.repeat(seq[-1:], length - n - before, axis=0)])


def flip_clip(clip: np.ndarray, root: np.ndarray):
    """Mirror left-right: negate x and swap left/right joints (a left-hander's version of the shot)."""
    clip = clip[:, FLIP_ORDER].copy()
    clip[..., 0] *= -1
    root = root.copy()
    root[:, 0] *= -1
    return clip, root


def clip_features(clip: np.ndarray, root: np.ndarray) -> np.ndarray:
    """(T, 17, 3) normalised clip + (T, 2) hip movement -> (T, N_FEATURES) model input."""
    pos = clip[..., :2].reshape(len(clip), -1)
    vis = (clip[..., 2] > 0).astype(np.float32)
    vel = np.diff(pos, axis=0, prepend=pos[:1])
    vel *= np.repeat(vis, 2, axis=1) * np.repeat(np.vstack([vis[:1], vis[:-1]]), 2, axis=1)  # only seen->seen
    return np.concatenate([pos, vis, vel, root], axis=1).astype(np.float32)


def prepare(kpts: np.ndarray, offset: int = 0, flip: bool = False) -> np.ndarray:
    """Raw (T, 17, 3) pixel keypoints -> (SEQ_LEN, N_FEATURES) model input."""
    clip, root = fit_length(normalise_clip(kpts), offset=offset), fit_length(normalise_root(kpts), offset=offset)
    if flip:
        clip, root = flip_clip(clip, root)
    return clip_features(clip, root)


def build_model(arch: str, n_classes: int = len(SHOT_TYPES), in_dim: int = N_FEATURES):
    import torch
    import torch.nn as nn

    class ShotLSTM(nn.Module):
        """Same shape as the group member's model (2-layer LSTM 128, fc 64), on normalised input."""

        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, 128, num_layers=2, batch_first=True, dropout=0.3)
            self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_classes))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1])

    class ShotGRU(nn.Module):
        """Bidirectional GRU, mean + max pooled over time."""

        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(in_dim, 96, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4 * 96, n_classes))

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(torch.cat([out.mean(1), out.amax(1)], dim=1))

    class ShotCNN(nn.Module):
        """Small temporal CNN: three conv blocks over the 30 frames, then global pooling."""

        def __init__(self):
            super().__init__()
            layers, ch = [], in_dim
            for out_ch, dilation in ((128, 1), (128, 2), (128, 4)):
                layers += [nn.Conv1d(ch, out_ch, 3, padding=dilation, dilation=dilation),
                           nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(0.2)]
                ch = out_ch
            self.conv = nn.Sequential(*layers)
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * ch, n_classes))

        def forward(self, x):
            h = self.conv(x.transpose(1, 2))
            return self.head(torch.cat([h.mean(2), h.amax(2)], dim=1))

    return {"lstm": ShotLSTM, "gru": ShotGRU, "cnn": ShotCNN}[arch]()


def group_probs(probs: np.ndarray) -> np.ndarray:
    """(..., 18) class probabilities -> (..., 6) group probabilities."""
    out = np.zeros(probs.shape[:-1] + (len(SHOT_GROUPS),), dtype=np.float64)
    for cls, group in enumerate(GROUP_IDS):
        out[..., group] += probs[..., cls]
    return out


class ShotClassifier:
    """Loads the model saved by train_shot_classifier.py; ready is False when it hasn't been trained."""

    def __init__(self, path: Path = MODEL_PATH, device: str = None):
        self.model = None
        if not path.is_file():
            return
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        bundle = torch.load(path, map_location=self.device, weights_only=False)
        self.arch = bundle["arch"]
        self.model = build_model(self.arch).to(self.device).eval()
        self.model.load_state_dict(bundle["state_dict"])
        self.info = {k: v for k, v in bundle.items() if k != "state_dict"}

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict_batch(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        """List of (T, 17, 3) pixel keypoint sequences -> (N, 18) probabilities (averaged with the mirror)."""
        x = np.stack([prepare(k, flip=f) for k in clips for f in (False, True)])
        with self.torch.no_grad():
            logits = self.model(self.torch.from_numpy(x).to(self.device))
            probs = self.torch.softmax(logits, dim=1).cpu().numpy()
        return probs.reshape(len(clips), 2, -1).mean(axis=1)

    def predict(self, seq: np.ndarray) -> Tuple[int, np.ndarray]:
        """(T, 17, 3) keypoints of the hitting player around the hit -> (class id, 18 probabilities)."""
        if self.model is None:
            raise RuntimeError(f"No shot model at {MODEL_PATH}; run train_shot_classifier.py first.")
        probs = self.predict_batch([seq])[0]
        return int(np.argmax(probs)), probs

    @staticmethod
    def name(class_id: int) -> str:
        return SHOT_TYPES[class_id]

    @staticmethod
    def group(class_id: int) -> str:
        return GROUP_OF[SHOT_TYPES[class_id]]


def load_metrics() -> dict:
    path = MODEL_DIR / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
