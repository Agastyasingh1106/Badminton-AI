"""
Scores shuttle tracking against the hand-clicked frames from label_shuttle.py.

A frame counts as correct when the tracked point is within --tolerance pixels of the click.
Frames marked "not visible" are scored as correct when the tracker reports nothing.
Reports all labelled frames and the held-out (val) frames the retrained model never saw.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_match import track_video
from badminton.court import CourtMapper, load_corners, video_id_from_path
from detect_shuttle import DEFAULT_MODEL, ORIGINAL_MODEL

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = PROJECT_ROOT / "sample_dataset" / "00467.mp4"


def score(frames, labels, tolerance, frame_ids):
    hits, errors = 0, []
    for index in frame_ids:
        truth, tracked = labels[str(index)], frames[index]["shuttle"]
        if truth is None:
            hits += tracked is None
            continue
        if tracked is not None:
            err = float(np.hypot(tracked[0][0] - truth[0], tracked[0][1] - truth[1]))
            errors.append(err)
            hits += err <= tolerance
    return hits, len(frame_ids), errors


def main():
    parser = argparse.ArgumentParser(description="Score shuttle tracking against hand labels.")
    parser.add_argument("video", nargs="?", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--tolerance", type=float, default=15.0, help="Pixels from the click that count as correct.")
    args = parser.parse_args()

    video_id = video_id_from_path(args.video)
    labels = json.loads((PROJECT_ROOT / f"shuttle_labels_{video_id}.json").read_text())
    ordered = sorted(int(i) for i in labels)
    val_ids = [i for n, i in enumerate(ordered) if n % 4 == 3]   # same split as label_shuttle.py
    mapper = CourtMapper(load_corners()[video_id])
    visible = sum(labels[str(i)] is not None for i in ordered)
    print(f"{len(ordered)} labelled frames ({visible} with the shuttle visible), {len(val_ids)} held out\n")

    setups = [("original model, no player filter", ORIGINAL_MODEL, False),
              ("original model + player filter", ORIGINAL_MODEL, True)]
    if DEFAULT_MODEL != ORIGINAL_MODEL:
        setups.append(("retrained model + player filter", DEFAULT_MODEL, True))

    results = {}
    for name, model, use_filter in setups:
        frames, _ = track_video(args.video, mapper, PROJECT_ROOT / "yolov8n.pt",
                                shuttle_model_path=model, player_filter=use_filter)
        all_hits, all_n, errors = score(frames, labels, args.tolerance, ordered)
        val_hits, val_n, _ = score(frames, labels, args.tolerance, val_ids)
        median = f"{np.median(errors):.0f}px" if errors else "-"
        print(f"{name:<34} correct {all_hits}/{all_n} ({all_hits / all_n:.0%})   "
              f"held-out {val_hits}/{val_n} ({val_hits / max(val_n, 1):.0%})   median error {median}")
        results[name] = {"correct": all_hits, "total": all_n, "held_out_correct": val_hits, "held_out_total": val_n}

    out = PROJECT_ROOT / "runs" / "analysis" / video_id / "tracking_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tolerance_px": args.tolerance, "results": results}, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
