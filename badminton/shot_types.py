"""
Shot types for detected hits: YOLOv8-pose on the hitting player around each hit, then the
shot classifier (badminton/shots.py) on the 31-frame keypoint sequence.

The pose model runs on a crop around the tracked player box (grown for the racket arm), so the
other player, umpire and crowd can't be picked up; keypoints are mapped back to frame pixels.

The classifier's 18 types don't carry over from its fixed-camera training data to broadcast
footage, so when finetune_shot_classifier.py has saved a broadcast model (runs/shot_model/
broadcast_groups.pt), events get the 6 shot groups from it and no fine type.
"""

import pickle
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from badminton.court import PROJECT_ROOT
from badminton.shots import GROUP_IDS, MODEL_DIR, SHOT_GROUPS, ShotClassifier, prepare

POSE_MODEL = PROJECT_ROOT / "yolov8n-pose.pt"
BROADCAST_MODEL = MODEL_DIR / "broadcast_groups.pt"
HALF_WINDOW = 15     # frames either side of the hit (the classifier sees 30)
CROP_GROW = 0.5      # grow the player box by this share of its height, for arms and racket
SERVE = SHOT_GROUPS.index("serve")


def _player_box(frame_data, pid: int):
    for p, box, _ in frame_data["players"]:
        if p == pid:
            return box
    return None


def _crop(frame, box):
    x1, y1, x2, y2 = box
    g = CROP_GROW * (y2 - y1)
    h, w = frame.shape[:2]
    cx1, cy1 = int(max(x1 - g, 0)), int(max(y1 - g, 0))
    cx2, cy2 = int(min(x2 + g, w)), int(min(y2 + g / 2, h))
    return frame[cy1:cy2, cx1:cx2], (cx1, cy1)


def hitter_poses(video_path, frames: List[dict], events: List[dict], pose_model_path: Path = POSE_MODEL,
                 offset: int = 0):
    """
    (sequences, has_pose): a (31, 17, 3) pixel keypoint sequence of the hitting player per event
    (zeros where no pose was found), and whether enough frames had one to classify.
    offset: frames between the hit and the centre of the pose window.
    """
    from ultralytics import YOLO
    pose = YOLO(str(pose_model_path))

    # Which frames each shot needs, and whose box to crop there
    wanted = {}
    for k, ev in enumerate(events):
        pid = 1 if ev["side"] == "far" else 2
        for f in range(ev["frame"] + offset - HALF_WINDOW, ev["frame"] + offset + HALF_WINDOW + 1):
            if 0 <= f < len(frames):
                box = _player_box(frames[f], pid)
                if box is not None:
                    wanted.setdefault(f, []).append((k, box))

    sequences = [np.zeros((2 * HALF_WINDOW + 1, 17, 3), dtype=np.float32) for _ in events]
    cap = cv2.VideoCapture(str(video_path))
    index, last = 0, max(wanted, default=-1)
    while index <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for k, box in wanted.get(index, []):
            crop, (ox, oy) = _crop(frame, box)
            if crop.size == 0:
                continue
            res = pose.predict(crop, conf=0.25, verbose=False)[0]
            if res.keypoints is None or len(res.keypoints) == 0:
                continue
            kp = res.keypoints.data[int(res.boxes.conf.argmax())].cpu().numpy().copy()
            kp[:, 0] += ox
            kp[:, 1] += oy
            sequences[k][index - (events[k]["frame"] + offset - HALF_WINDOW)] = kp
        index += 1
    cap.release()

    has_pose = [int((s[:, :, 2] > 0.3).any(axis=1).sum()) >= 8 for s in sequences]
    return sequences, has_pose


def model_inputs(seqs, flip=False):
    import torch
    return torch.from_numpy(np.stack([prepare(s, flip=flip) for s in seqs]))


def embed(classifier: ShotClassifier, x) -> np.ndarray:
    """Pooled conv features of the pretrained CNN (the layer before its classification head)."""
    import torch
    with torch.no_grad():
        h = classifier.model.conv(x.to(classifier.device).transpose(1, 2))
        return torch.cat([h.mean(2), h.amax(2)], dim=1).cpu().numpy()


def type_probs_to_groups(probs: np.ndarray) -> np.ndarray:
    return np.stack([np.bincount(GROUP_IDS, weights=p, minlength=len(SHOT_GROUPS)) for p in probs])


def embed_lr_probs(classifier: ShotClassifier, lr, seqs) -> np.ndarray:
    """Group probabilities from a logistic regression on the embeddings, averaged with the mirror."""
    probs = np.zeros((len(seqs), len(SHOT_GROUPS)))
    for flip in (False, True):
        probs[:, lr.classes_] += lr.predict_proba(embed(classifier, model_inputs(seqs, flip))) / 2
    return probs


class BroadcastGroups:
    """The broadcast shot-group model saved by finetune_shot_classifier.py (ready is False without one)."""

    def __init__(self, classifier: ShotClassifier, path: Path = BROADCAST_MODEL):
        self.classifier, self.kind = classifier, None
        if not path.is_file() or not classifier.ready:
            return
        import torch
        bundle = torch.load(path, map_location=classifier.device, weights_only=False)
        self.kind = bundle["kind"]
        if self.kind == "embed_lr":
            self.lr = pickle.loads(bundle["lr"])
        else:
            self.finetuned = ShotClassifier(device=classifier.device)
            self.finetuned.model.load_state_dict(bundle["state_dict"])

    @property
    def ready(self) -> bool:
        return self.kind is not None

    def predict(self, seqs) -> np.ndarray:
        if self.kind == "embed_lr":
            return embed_lr_probs(self.classifier, self.lr, seqs)
        return type_probs_to_groups(self.finetuned.predict_batch(seqs))


def classify_shots(video_path, frames: List[dict], events: List[dict], classifier: Optional[ShotClassifier] = None,
                   pose_model_path: Path = POSE_MODEL, offset: int = 0, use_broadcast: bool = True) -> List[dict]:
    """
    Adds shot_group and shot_group_conf to each event (None when no pose was found), from the
    broadcast model. Without one nothing is guessed, since the original classifier scores below
    chance on broadcast footage; use_broadcast=False uses it anyway (for evaluation) and also adds
    its shot_type and shot_type_conf.
    """
    classifier = classifier or ShotClassifier()
    if not classifier.ready or not events or not Path(pose_model_path).is_file():
        return events
    groups = BroadcastGroups(classifier) if use_broadcast else None
    if groups is not None and not groups.ready:
        for ev in events:
            ev.update(shot_type=None, shot_type_conf=None, shot_group=None, shot_group_conf=None)
        return events
    sequences, has_pose = hitter_poses(video_path, frames, events, pose_model_path, offset)
    seqs = [s for s, ok in zip(sequences, has_pose) if ok]
    if groups is not None and groups.ready:
        probs, fine = (groups.predict(seqs) if seqs else []), False
    else:
        probs, fine = (classifier.predict_batch(seqs) if seqs else []), True
    it = iter(probs)
    for ev, ok in zip(events, has_pose):
        ev.update(shot_type=None, shot_type_conf=None, shot_group=None, shot_group_conf=None)
        if not ok:
            continue
        p = next(it)
        if fine:
            cls = int(np.argmax(p))
            g = type_probs_to_groups(p[None])[0]
            ev.update(shot_type=classifier.name(cls), shot_type_conf=float(p[cls]))
        else:
            g = p
        g = np.asarray(g, dtype=float)
        if ev.get("shot_no", 1) > 1 and g.sum() > g[SERVE]:  # only a rally's first shot can be a serve
            g = g.copy()
            g[SERVE] = 0.0
            g /= g.sum()
        ev.update(shot_group=SHOT_GROUPS[int(np.argmax(g))], shot_group_conf=float(np.max(g)))
    return events
