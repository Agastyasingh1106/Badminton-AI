"""
Finds the 4 court corners automatically for rally videos that have none in court_corners.json.

For each video: median of frames spread over the rally (removes the players) -> mask of the white
court lines -> the standard court line layout is projected with a candidate set of corners and
the corners are adjusted until the projected lines sit on the white lines. Candidates are the
corners already in court_corners.json (the same broadcast cameras come back across rallies).

A fit is accepted only when most of the court lines land on white line pixels. Videos that fail
are listed so they can be clicked by hand with calibrate_court.py.

  python auto_calibrate_court.py dataset/rallies            # fit and save new videos
  python auto_calibrate_court.py --check                    # leave-one-out test on clicked videos
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from badminton.court import (COURT_LENGTH, COURT_WIDTH, NET_Y, PROJECT_ROOT, CourtMapper, load_corners,
                             save_corners, video_id_from_path)

DATASET_IMAGES = PROJECT_ROOT / "training" / "shuttlecock_yolo_dataset" / "images"
FIT_LOG = PROJECT_ROOT / "runs" / "court_fit" / "fits.json"
PREVIEW_DIR = PROJECT_ROOT / "runs" / "court_fit" / "previews"

SINGLES_INSET = 0.46       # singles sideline, from the doubles sideline
LONG_SERVICE_INSET = 0.76  # doubles long service line, from the baseline
SHORT_SERVICE = 1.98       # short service line, from the net
CAP_PX = 12.0              # distance cap so occluded lines don't dominate the score
INLIER_PX = 3.0
MIN_INLIERS = 0.6          # share of line points within INLIER_PX of a white pixel


def court_line_points(step: float = 0.05) -> np.ndarray:
    """Points (metres) along every painted court line."""
    xs = [0.0, SINGLES_INSET, COURT_WIDTH - SINGLES_INSET, COURT_WIDTH]
    ys = [0.0, LONG_SERVICE_INSET, NET_Y - SHORT_SERVICE, NET_Y + SHORT_SERVICE,
          COURT_LENGTH - LONG_SERVICE_INSET, COURT_LENGTH]
    pts = []
    for x in xs:
        pts += [(x, y) for y in np.arange(0.0, COURT_LENGTH + 1e-6, step)]
    for y in ys:
        pts += [(x, y) for x in np.arange(0.0, COURT_WIDTH + 1e-6, step)]
    centre = COURT_WIDTH / 2
    pts += [(centre, y) for y in np.arange(0.0, NET_Y - SHORT_SERVICE + 1e-6, step)]
    pts += [(centre, y) for y in np.arange(NET_Y + SHORT_SERVICE, COURT_LENGTH + 1e-6, step)]
    return np.array(pts, dtype=np.float32)


LINE_POINTS = court_line_points()


def background_frame(source, samples: int = 25):
    """Median frame of a video (or of a list of image paths): moving players fade out."""
    frames = []
    if isinstance(source, (list, tuple)):
        for path in source[:: max(len(source) // samples, 1)][:samples]:
            img = cv2.imread(str(path))
            if img is not None:
                frames.append(img)
    else:
        cap = cv2.VideoCapture(str(source))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        for idx in np.linspace(0, total - 1, samples).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, img = cap.read()
            if ok:
                frames.append(img)
        cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8) if frames else None


def line_distance_map(frame) -> np.ndarray:
    """Pixel distance to the nearest white-line pixel."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    sat = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
    mask = ((tophat > 25) & (sat < 90)).astype(np.uint8)
    return cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)


def _project(corners: np.ndarray) -> np.ndarray:
    court = np.array([[0, 0], [COURT_WIDTH, 0], [COURT_WIDTH, COURT_LENGTH], [0, COURT_LENGTH]], np.float32)
    h = cv2.getPerspectiveTransform(court, corners.reshape(4, 2).astype(np.float32))
    return cv2.perspectiveTransform(LINE_POINTS[None], h)[0]


def line_distances(corners: np.ndarray, dist: np.ndarray) -> np.ndarray:
    pts = _project(corners)
    h, w = dist.shape
    x, y = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    d = np.full(len(pts), CAP_PX, dtype=np.float32)
    d[inside] = np.minimum(dist[y[inside], x[inside]], CAP_PX)
    return d


def fit_corners(dist: np.ndarray, candidates):
    """Best refined corners over all candidate starts: (corners, inlier share)."""
    def cost(c):
        return float(line_distances(c, dist).mean())

    # Rank the starts cheaply, refine the most promising few
    starts = sorted((np.asarray(c, np.float64).ravel() for c in candidates), key=cost)[:5]
    best, best_cost = None, None
    for start in starts:
        res = minimize(cost, start, method="Powell", options={"xtol": 0.3, "ftol": 1e-3, "maxfev": 4000})
        if best_cost is None or res.fun < best_cost:
            best, best_cost = res.x, res.fun
    inliers = float((line_distances(best, dist) <= INLIER_PX).mean())
    return best.reshape(4, 2), inliers


def save_preview(frame, corners, video_id, inliers):
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    view = frame.copy()
    for x, y in _project(np.asarray(corners)):
        cv2.circle(view, (int(round(x)), int(round(y))), 1, (0, 0, 255), -1)
    CourtMapper(corners).draw_zone_grid(view, color=(0, 255, 255), labels=False)
    cv2.putText(view, f"{video_id}  lines on white: {inliers:.0%}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255) if inliers < MIN_INLIERS else (0, 255, 0), 2)
    cv2.imwrite(str(PREVIEW_DIR / f"{video_id}.jpg"), view)


def clicked_video_frames():
    """{video_id: [image paths]} for the videos in the shuttle dataset."""
    frames = {}
    for path in sorted(DATASET_IMAGES.glob("*/*.jpg")):
        frames.setdefault(video_id_from_path(path), []).append(path)
    return frames


def check(corners: dict):
    """Leave-one-out: fit each clicked video from the other videos' corners, compare to the clicks."""
    frames = clicked_video_frames()
    errors = []
    for video, clicked in sorted(corners.items()):
        if video not in frames:
            continue
        bg = background_frame(frames[video])
        fit, inliers = fit_corners(line_distance_map(bg), [c for v, c in corners.items() if v != video])
        err = float(np.linalg.norm(fit - np.asarray(clicked), axis=1).max())
        accepted = inliers >= MIN_INLIERS
        errors.append((video, err, inliers, accepted))
        print(f"  {video}: worst corner off by {err:5.1f} px   lines on white {inliers:.0%}"
              f"   {'accepted' if accepted else 'rejected'}")
    acc = [e for e in errors if e[3]]
    print(f"\nAccepted {len(acc)}/{len(errors)}; worst-corner error on accepted: "
          f"median {np.median([e[1] for e in acc]):.1f} px, max {max(e[1] for e in acc):.1f} px")
    bad = [e for e in acc if e[1] > 15]
    print(f"Accepted but off by more than 15 px: {[(v, round(e, 1)) for v, e, _, _ in bad]}")


def calibrate_folder(folder: Path, redo: bool = False):
    corners = load_corners()
    candidates = list(corners.values())
    log = json.loads(FIT_LOG.read_text(encoding="utf-8")) if FIT_LOG.is_file() else {}
    videos = sorted(folder.glob("*.mp4"))
    failed = []
    for i, path in enumerate(videos, 1):
        video = video_id_from_path(path)
        if video in corners and not redo and log.get(video, {}).get("auto", True) is False:
            continue
        if video in corners and not redo and video not in log:
            continue  # clicked by hand
        if video in log and not redo:
            if not log[video]["accepted"]:
                failed.append(video)
            continue
        bg = background_frame(path)
        if bg is None:
            failed.append(video)
            continue
        fit, inliers = fit_corners(line_distance_map(bg), candidates)
        accepted = inliers >= MIN_INLIERS
        log[video] = {"accepted": accepted, "inliers": round(inliers, 3),
                      "corners": [[round(float(x), 1), round(float(y), 1)] for x, y in fit]}
        if accepted:
            corners[video] = log[video]["corners"]
        else:
            failed.append(video)
        save_preview(bg, fit, video, inliers)
        print(f"[{i}/{len(videos)}] {video}: lines on white {inliers:.0%}  {'ok' if accepted else 'REJECTED'}")
        if i % 20 == 0:
            save_corners(corners)
            FIT_LOG.parent.mkdir(parents=True, exist_ok=True)
            FIT_LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")

    save_corners(corners)
    FIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    FIT_LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"\n{len(videos) - len(failed)}/{len(videos)} videos calibrated. Previews in {PREVIEW_DIR}")
    if failed:
        print(f"Click these by hand with calibrate_court.py: {' '.join(failed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit court corners automatically.")
    parser.add_argument("folder", nargs="?", type=Path, help="Folder of rally videos to calibrate.")
    parser.add_argument("--check", action="store_true", help="Leave-one-out test on the clicked videos.")
    parser.add_argument("--redo", action="store_true", help="Refit videos that already have a fit.")
    args = parser.parse_args()
    if args.check:
        check(load_corners())
    elif args.folder:
        calibrate_folder(args.folder, args.redo)
    else:
        parser.error("give a folder of videos or --check")
