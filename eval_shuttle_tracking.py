"""
Scores shuttle tracking against hand labels on clean broadcast footage.

Labels: shuttle_labels_00467.json (label_shuttle.py) and the "shuttle" entries of
hit_labels.json (label_hits.py, rallies in dataset/rallies). Each label is a pixel point,
or null when the shuttle is not visible.

Trackers compared:
  ours      - our YOLO + motion tracker (analyze_match.track_video, shuttle_backend="yolo")
  tracknet  - TrackNetV3 output in runs/tracknet/<video_id>_ball.csv, if present (the pipeline's
              default shuttle backend, badminton/tracknet.py, gives the same points within 1 px)

Per tracker: share of visible-shuttle frames tracked within --tolerance px, frames where a
shuttle was reported although none is visible, and the median error. The training images
of the dataset's *_predict videos have TrackNet's dot drawn on the shuttle, so they are not
used here.

  python eval_shuttle_tracking.py                       # all labelled videos
  python eval_shuttle_tracking.py --model runs/detect/shuttlecock_v3/weights/best.pt
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_match import track_video
from badminton.court import CourtMapper, load_corners
from detect_shuttle import DEFAULT_MODEL

PROJECT_ROOT = Path(__file__).resolve().parent
HIT_LABELS = PROJECT_ROOT / "hit_labels.json"
TRACKNET_DIR = PROJECT_ROOT / "runs" / "tracknet"
OUT_FILE = PROJECT_ROOT / "runs" / "shuttle_eval" / "benchmark.json"
VIDEO_DIRS = (PROJECT_ROOT / "sample_dataset", PROJECT_ROOT / "dataset" / "rallies")


def load_labels():
    """{video_id: {frame: (x, y) or None}} from every label file."""
    videos = {}
    for path in sorted(PROJECT_ROOT.glob("shuttle_labels_*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        videos[path.stem.split("_")[-1]] = {int(f): (tuple(p) if p else None) for f, p in raw.items()}
    if HIT_LABELS.is_file():
        for video, entry in json.loads(HIT_LABELS.read_text(encoding="utf-8")).items():
            shuttle = entry.get("shuttle") or {}
            if shuttle:
                videos.setdefault(video, {}).update(
                    {int(f): (tuple(p) if p else None) for f, p in shuttle.items()})
    return videos


def find_video(video: str):
    for folder in VIDEO_DIRS:
        if (folder / f"{video}.mp4").is_file():
            return folder / f"{video}.mp4"
    return None


def ours(video_path: Path, video: str, model: Path):
    frames, _ = track_video(video_path, CourtMapper(load_corners()[video]), PROJECT_ROOT / "yolov8n.pt",
                            shuttle_model_path=model, shuttle_backend="yolo")
    return {i: (f["shuttle"][0] if f["shuttle"] is not None else None) for i, f in enumerate(frames)}


def tracknet(video: str):
    path = TRACKNET_DIR / f"{video}_ball.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return {int(r["Frame"]): ((float(r["X"]), float(r["Y"])) if r["Visibility"] == "1" else None)
                for r in csv.DictReader(f)}


def score(pred: dict, labels: dict, tolerance: float):
    """Counts for one video: (visible correct, visible total, false shuttles, invisible total, errors)."""
    vis_ok = vis_n = false_pos = invis_n = 0
    errors = []
    for frame, truth in labels.items():
        p = pred.get(frame)
        if truth is None:
            invis_n += 1
            false_pos += p is not None
            continue
        vis_n += 1
        if p is not None:
            err = float(np.hypot(p[0] - truth[0], p[1] - truth[1]))
            errors.append(err)
            vis_ok += err <= tolerance
    return vis_ok, vis_n, false_pos, invis_n, errors


def summarise(counts):
    vis_ok, vis_n, false_pos, invis_n = (sum(c[k] for c in counts) for k in range(4))
    errors = [e for c in counts for e in c[4]]
    return {"visible_correct": vis_ok, "visible_total": vis_n,
            "accuracy": vis_ok / vis_n if vis_n else None,
            "false_shuttles": false_pos, "not_visible_total": invis_n,
            "median_error_px": float(np.median(errors)) if errors else None}


def main():
    parser = argparse.ArgumentParser(description="Score shuttle tracking against hand labels.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Shuttle YOLO weights for 'ours'.")
    parser.add_argument("--tolerance", type=float, default=10.0, help="Pixels from the label that count as correct.")
    parser.add_argument("--videos", nargs="*", help="Only these video ids.")
    args = parser.parse_args()

    labels = load_labels()
    if args.videos:
        labels = {v: l for v, l in labels.items() if v in args.videos}
    corners = load_corners()
    per_video, counts = {}, {"ours": [], "tracknet": []}
    for video, lab in sorted(labels.items()):
        path = find_video(video)
        if path is None or video not in corners:
            print(f"  {video}: skipped (no video or court corners)")
            continue
        row = {}
        for name, pred in (("ours", ours(path, video, args.model)), ("tracknet", tracknet(video))):
            if pred is None:
                continue
            c = score(pred, lab, args.tolerance)
            counts[name].append(c)
            row[name] = summarise([c])
        per_video[video] = row
        print(f"  {video}: " + "   ".join(f"{n} {r['visible_correct']}/{r['visible_total']} "
                                          f"(false {r['false_shuttles']}/{r['not_visible_total']})"
                                          for n, r in row.items()))

    totals = {name: summarise(c) for name, c in counts.items() if c}
    print(f"\nAll labelled frames (tolerance {args.tolerance:.0f} px):")
    for name, t in totals.items():
        med = f"{t['median_error_px']:.1f}px" if t["median_error_px"] is not None else "-"
        print(f"  {name:<9} visible tracked {t['visible_correct']}/{t['visible_total']} ({t['accuracy']:.0%})   "
              f"false shuttles {t['false_shuttles']}/{t['not_visible_total']}   median error {med}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({"tolerance_px": args.tolerance, "model": str(args.model),
                                    "totals": totals, "per_video": per_video}, indent=2), encoding="utf-8")
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
