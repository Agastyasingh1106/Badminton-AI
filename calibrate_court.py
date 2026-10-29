"""
Click the 4 outer court corners for each video, saved to court_corners.json.

Click order: far-left, far-right, near-right, near-left (outer doubles lines).
Keys:  s = reuse previous video's corners   u = undo last click
       n = skip this video                  q = save and quit
Videos that already have corners are skipped unless --redo is given.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from badminton.court import PROJECT_ROOT, load_corners, save_corners, video_id_from_path


DATASET_DIR = PROJECT_ROOT / "training" / "shuttlecock_yolo_dataset" / "images"
SAMPLE_VIDEO = PROJECT_ROOT / "sample_dataset" / "00467.mp4"
CORNER_NAMES = ["far-left", "far-right", "near-right", "near-left"]
WINDOW = "Court calibration"


def first_frames():
    """One reference frame per video: training/val images plus the sample clip."""
    frames = {}
    for split in ("train", "val"):
        for img_path in sorted((DATASET_DIR / split).glob("*.jpg")):
            frames.setdefault(video_id_from_path(img_path), img_path)
    frames[video_id_from_path(SAMPLE_VIDEO)] = SAMPLE_VIDEO
    return dict(sorted(frames.items()))


def read_frame(path: Path):
    if path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
    return cv2.imread(str(path))


def draw(frame, points, video_id, index, total):
    view = frame.copy()
    for i, p in enumerate(points):
        cv2.circle(view, p, 6, (0, 0, 255), -1)
        cv2.putText(view, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if len(points) > 1:
        cv2.polylines(view, [np.array(points, dtype=np.int32)], len(points) == 4, (0, 255, 255), 2)
    hint = f"click {CORNER_NAMES[len(points)]}" if len(points) < 4 else "ENTER = save"
    text = f"[{index}/{total}] video {video_id}: {hint}   (s=reuse prev  u=undo  n=skip  q=quit)"
    cv2.rectangle(view, (0, 0), (view.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(view, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return view


def calibrate(sources, redo=False):
    corners = load_corners()
    todo = {vid: path for vid, path in sources.items() if redo or vid not in corners}
    print(f"{len(corners)} videos already calibrated, {len(todo)} to do.")
    previous = next(reversed(corners.values()), None) if corners else None

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    points = []
    cv2.setMouseCallback(WINDOW, lambda e, x, y, *_: points.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN and len(points) < 4 else None)

    for index, (video_id, path) in enumerate(todo.items(), start=1):
        frame = read_frame(path)
        if frame is None:
            print(f"  could not read {path}, skipping")
            continue
        points.clear()
        while True:
            cv2.imshow(WINDOW, draw(frame, points, video_id, index, len(todo)))
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                save_corners(corners)
                cv2.destroyAllWindows()
                print(f"Saved {len(corners)} videos to court_corners.json")
                return
            if key == ord("u") and points:
                points.pop()
            if key == ord("n"):
                break
            if key == ord("s") and previous is not None:
                points[:] = [tuple(map(int, p)) for p in previous]
            if key in (13, 10) and len(points) == 4:
                corners[video_id] = [list(map(float, p)) for p in points]
                previous = corners[video_id]
                save_corners(corners)
                break

    cv2.destroyAllWindows()
    print(f"Saved {len(corners)} videos to court_corners.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Click court corners for each video.")
    parser.add_argument("videos", nargs="*", type=Path, help="Specific videos/images (default: all training videos + sample clip).")
    parser.add_argument("--redo", action="store_true", help="Re-click videos that already have corners.")
    args = parser.parse_args()

    sources = {video_id_from_path(p): p for p in args.videos} if args.videos else first_frames()
    calibrate(sources, redo=args.redo)
