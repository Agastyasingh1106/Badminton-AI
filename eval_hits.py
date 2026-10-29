"""
Scores hit detection against the hand labels in hit_labels.json (label_hits.py).

A detected hit matches a labelled one within --tolerance frames (one-to-one, closest first);
reports precision, recall, the median frame offset and whether the hitter's side is right.
Shot groups are scored separately, with leave-one-rally-out, by finetune_shot_classifier.py.

Tracks are cached in runs/shot_eval/cache, so re-scoring after changing badminton/events.py is fast.

  python eval_hits.py
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from analyze_match import track_video
from badminton.court import CourtMapper, load_corners
from badminton.events import detect_hits
from track_rallies import cached_tracknet

PROJECT_ROOT = Path(__file__).resolve().parent
HIT_LABELS = PROJECT_ROOT / "hit_labels.json"
VIDEO_DIR = PROJECT_ROOT / "dataset" / "rallies"
CACHE_DIR = PROJECT_ROOT / "runs" / "shot_eval" / "cache"
OUT_FILE = PROJECT_ROOT / "runs" / "shot_eval" / "hits.json"


def labelled_rallies():
    return {v: e for v, e in json.loads(HIT_LABELS.read_text(encoding="utf-8")).items()
            if e.get("hits_done") and e.get("hits")}


def tracked(video: str, mapper: CourtMapper):
    """(frames, info) from track_video with TrackNetV3 points, cached per video."""
    path = CACHE_DIR / f"{video}.pkl"
    if path.is_file():
        return pickle.loads(path.read_bytes())
    frames, info = track_video(VIDEO_DIR / f"{video}.mp4", mapper, PROJECT_ROOT / "yolov8n.pt",
                               shuttle_points=cached_tracknet(video))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps((frames, info)))
    return frames, info


def match(detected, labelled, tolerance):
    """Pairs (detected index, labelled index), closest frames first, each used once."""
    pairs = sorted((abs(d["frame"] - l["frame"]), i, j) for i, d in enumerate(detected)
                   for j, l in enumerate(labelled) if abs(d["frame"] - l["frame"]) <= tolerance)
    used_d, used_l, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_d and j not in used_l:
            used_d.add(i); used_l.add(j); out.append((i, j))
    return out


def score(tolerance: int, **hit_params):
    """Hit detection scores over all labelled rallies; hit_params go to detect_hits."""
    corners = load_corners()
    n_det = n_lab = n_match = n_side = 0
    offsets, per_video = [], {}
    for video, entry in sorted(labelled_rallies().items()):
        mapper = CourtMapper(corners[video])
        frames, info = tracked(video, mapper)
        points = {i: f["shuttle"][0] for i, f in enumerate(frames) if f["shuttle"] is not None}
        boxes = lambda i: [b for _, b, _ in frames[i]["players"]] if 0 <= i < len(frames) else []
        detected, _ = detect_hits(points, mapper, fps=info["fps"], players_at=boxes, **hit_params)
        hits = sorted(entry["hits"], key=lambda h: h["frame"])
        pairs = match(detected, hits, tolerance)
        n_det, n_lab, n_match = n_det + len(detected), n_lab + len(hits), n_match + len(pairs)
        n_side += sum(detected[i]["side"] == hits[j]["side"] for i, j in pairs)
        offsets += [detected[i]["frame"] - hits[j]["frame"] for i, j in pairs]
        per_video[video] = {"labelled": len(hits), "detected": len(detected), "matched": len(pairs)}
    p = n_match / n_det if n_det else 0.0
    r = n_match / n_lab if n_lab else 0.0
    return {"tolerance_frames": tolerance, "videos": len(per_video), "labelled": n_lab, "detected": n_det,
            "matched": n_match, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "side_correct": n_side / n_match if n_match else None,
            "median_offset_frames": float(np.median(offsets)) if offsets else None, "per_video": per_video}


def main():
    parser = argparse.ArgumentParser(description="Score hit detection against hand labels.")
    parser.add_argument("--tolerance", type=int, default=4, help="Frames between a detected and a labelled hit.")
    args = parser.parse_args()
    res = score(args.tolerance)
    for video, row in res["per_video"].items():
        print(f"  {video}: labelled {row['labelled']}, detected {row['detected']}, matched {row['matched']}")
    print(f"\nHits within {args.tolerance} frames: precision {res['precision']:.0%}  recall {res['recall']:.0%}  "
          f"F1 {res['f1']:.2f}  ({res['matched']} matched / {res['detected']} detected / {res['labelled']} labelled)"
          f"\nhitter side right on {res['side_correct']:.0%} of matches, median offset {res['median_offset_frames']} frames")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
