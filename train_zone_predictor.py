"""
Trains the next-zone predictor from the labelled shuttle positions in
training/shuttlecock_yolo_dataset (one label every 4th frame, 51 rally videos), plus the
rallies tracked by our own pipeline with track_rallies.py (runs/rally_tracks, no labels needed).

Steps: trajectory per video -> hits with court zones (needs court_corners.json)
       -> features -> Random Forest. Scored with grouped cross-validation against two simple
       baselines, with and without the tracked rallies, then refit on all videos for the dashboard.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from badminton.court import CourtMapper, load_corners
from badminton.events import attach_targets, detect_hits
from badminton.predictor import (FEATURE_NAMES, LOCAL_ZONES, MODEL_DIR, MODEL_PATH, MarkovBaseline,
                                 dataset_from_rallies)

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "training" / "shuttlecock_yolo_dataset"
LABEL_NAME = re.compile(r"(?P<video>\d+)_predict_f(?P<frame>\d+)$")


def load_trajectories(split: str):
    """{video_id: {frame: (x_px, y_px)}} from the YOLO label files of one split."""
    tracks = defaultdict(dict)
    sizes = {}
    for label in sorted((DATASET_DIR / "labels" / split).glob("*.txt")):
        match = LABEL_NAME.match(label.stem)
        if not match:
            continue
        video, frame = match["video"], int(match["frame"])
        if video not in sizes:
            img = cv2.imread(str(DATASET_DIR / "images" / split / f"{label.stem}.jpg"))
            sizes[video] = img.shape[:2] if img is not None else (720, 1280)
        h, w = sizes[video]
        tracks[video]  # keep videos that never show the shuttle
        line = label.read_text(encoding="utf-8").strip().splitlines()
        if line:
            _, cx, cy, *_ = map(float, line[0].split())
            tracks[video][frame] = (cx * w, cy * h)
    # Hand-labelled clips (label_shuttle.py) are sparse and off the 4-frame grid: not rally tracks
    return {video: pts for video, pts in tracks.items() if all(f % 4 == 0 for f in pts)}


def split_trajectories():
    """Train/val trajectories with no video in both (a shared video is merged into train)."""
    train, val = load_trajectories("train"), load_trajectories("val")
    for video in set(train) & set(val):
        train[video].update(val.pop(video))
        print(f"Video {video} had frames in both splits; using it for training only.")
    return train, val


class PlayerFinder:
    """Detects on-court players in the training images (only for frames where a hit is checked)."""

    def __init__(self, model_path=PROJECT_ROOT / "yolov8n.pt"):
        from ultralytics import YOLO
        self.model = YOLO(str(model_path))

    def for_video(self, video: str, mapper: CourtMapper):
        polygon = mapper.polygon_px()

        def players_at(frame: int):
            matches = list((DATASET_DIR / "images").glob(f"*/{video}_predict_f{frame:06d}.jpg"))
            if not matches:
                return []
            result = self.model.predict(str(matches[0]), classes=[0], conf=0.25, verbose=False)[0]
            boxes = []
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                if cv2.pointPolygonTest(polygon, ((x1 + x2) / 2, y2), True) >= -30:
                    boxes.append((x1, y1, x2, y2))
            return boxes

        return players_at


def rallies_for(trajectories: dict, corners: dict, finder: "PlayerFinder"):
    rallies, skipped = {}, []
    for video, points in trajectories.items():
        if video not in corners:
            skipped.append(video)
            continue
        mapper = CourtMapper(corners[video])
        events, samples = detect_hits(points, mapper, players_at=finder.for_video(video, mapper))
        rallies[video] = attach_targets(events, samples, mapper)
    return rallies, skipped


def tracked_rallies(corners: dict, exclude=()):
    """Rallies from runs/rally_tracks: our tracker's shuttle path and player boxes, no labels."""
    from track_rallies import HELD_OUT, TRACKS_DIR

    rallies = {}
    for path in sorted(TRACKS_DIR.glob("*.json")):
        video = path.stem
        if video in HELD_OUT or video in exclude or video not in corners:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        points = {int(f): tuple(p) for f, p in data["shuttle"].items()}
        boxes = {int(f): b for f, b in data["players"].items()}
        mapper = CourtMapper(corners[video])
        events, samples = detect_hits(points, mapper, fps=data["info"]["fps"],
                                      players_at=lambda f: boxes.get(f, []))
        rallies[video] = attach_targets(events, samples, mapper)
    return rallies


def top_k_accuracy(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    top = np.argsort(-probs, axis=1)[:, :k] + 1
    return float(np.mean([t in row for t, row in zip(y, top)])) if len(y) else 0.0


def make_model():
    # Fairly shallow forest: with ~3,500 shots, depth 8 beat depth 5 and 12, unlimited depth and
    # gradient boosting in grouped cross-validation; deeper trees start memorising rallies
    return RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)


def as_local_probs(model, X) -> np.ndarray:
    probs = np.zeros((len(X), len(LOCAL_ZONES)))
    for cls, col in zip(model.classes_, model.predict_proba(X).T):
        probs[:, int(cls) - 1] = col
    return probs


def evaluate(labelled: dict, tracked: dict, folds: int = 5):
    """
    Grouped k-fold over all rallies: every rally is scored by models that never saw it.
    Each fold trains the forest twice, on the labelled rallies only and on labelled + tracked,
    so both are scored on exactly the same unseen shots (split by where the test rally came from).
    """
    from sklearn.model_selection import GroupKFold

    X, y, zones, groups, source = [], [], [], [], []
    for name, rallies in (("labelled", labelled), ("tracked", tracked)):
        for video, events in rallies.items():
            Xv, yv, zv = dataset_from_rallies([events])
            X.extend(Xv); y.extend(yv); zones.extend(zv)
            groups.extend([video] * len(yv)); source.extend([name] * len(yv))
    X, y, groups, source = np.array(X), np.array(y), np.array(groups), np.array(source)
    print(f"Shots with a known next zone: {len(y)} from {len(set(groups))} rallies "
          f"({(source == 'labelled').sum()} labelled, {(source == 'tracked').sum()} tracked)")
    if len(y) < 30:
        raise SystemExit("Not enough shots to train. Calibrate more videos with calibrate_court.py.")

    methods = ("random_forest", "random_forest_labelled_only", "markov_baseline", "majority_baseline")
    test_sets = ("all", "labelled", "tracked")
    scores = {m: {t: {1: [], 2: []} for t in test_sets} for m in methods}
    importances = []
    for train_idx, test_idx in GroupKFold(folds).split(X, y, groups):
        model = make_model().fit(X[train_idx], y[train_idx])
        lab_idx = train_idx[source[train_idx] == "labelled"]
        model_lab = make_model().fit(X[lab_idx], y[lab_idx])
        importances.append(model.feature_importances_)
        markov = MarkovBaseline().fit([zones[i] for i in train_idx], y[train_idx])
        majority = Counter(y[train_idx].tolist()).most_common(1)[0][0]
        mj = np.zeros((len(test_idx), len(LOCAL_ZONES)))
        mj[:, majority - 1] = 1.0
        mj[:, (majority % 6)] = 0.5  # arbitrary second guess so top-2 is defined
        all_probs = {"random_forest": as_local_probs(model, X[test_idx]),
                     "random_forest_labelled_only": as_local_probs(model_lab, X[test_idx]),
                     "markov_baseline": np.array([markov.predict_proba(zones[i]) for i in test_idx]),
                     "majority_baseline": mj}
        for test_set in test_sets:
            keep = np.ones(len(test_idx), bool) if test_set == "all" else source[test_idx] == test_set
            if not keep.any():
                continue
            for name, probs in all_probs.items():
                for k in (1, 2):
                    scores[name][test_set][k].append(
                        (top_k_accuracy(probs[keep], y[test_idx][keep], k), int(keep.sum())))

    def pooled(by_k):
        if not by_k[1]:
            return None
        return {key: float(sum(a * n for a, n in by_k[k]) / sum(n for _, n in by_k[k]))
                for key, k in (("accuracy", 1), ("top2_accuracy", 2))}

    metrics = {name: pooled(scores[name]["all"]) for name in methods}
    metrics["random_chance"] = {"accuracy": 1 / 6, "top2_accuracy": 2 / 6}
    metrics["by_test_rallies"] = {t: {name: pooled(scores[name][t]) for name in methods} for t in test_sets}
    metrics["evaluation"] = f"{folds}-fold cross-validation grouped by rally video"
    metrics["n_shots"], metrics["n_videos"] = int(len(y)), int(len(set(groups)))
    metrics["n_videos_labelled"], metrics["n_videos_tracked"] = len(labelled), len(tracked)
    metrics["feature_importance"] = dict(sorted(
        zip(FEATURE_NAMES, map(float, np.mean(importances, axis=0))), key=lambda kv: -kv[1]))
    return metrics


def preview_hits(video: str, rally, corners, out_dir: Path):
    """Saves one image per hit with the zone grid, to eyeball the hit detection."""
    import glob
    mapper = CourtMapper(corners[video])
    out_dir.mkdir(parents=True, exist_ok=True)
    for ev in rally:
        matches = glob.glob(str(DATASET_DIR / "images" / "*" / f"{video}_predict_f{ev['frame']:06d}.jpg"))
        if not matches:
            continue
        img = cv2.imread(matches[0])
        mapper.draw_zone_grid(img)
        cv2.circle(img, tuple(map(int, ev["px"])), 12, (0, 0, 255), 2)
        foot = tuple(map(int, mapper.to_image(ev["court"])))
        cv2.circle(img, foot, 6, (0, 255, 0), -1)
        cv2.putText(img, f"shot {ev['shot_no']} {ev['side']} {ev['zone']} -> {ev.get('next_zone')}  "
                         f"{ev['speed_kmh']:.0f} km/h", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imwrite(str(out_dir / f"{video}_shot{ev['shot_no']:02d}.jpg"), img)
    print(f"Saved hit previews for {video} to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train the next-zone predictor.")
    parser.add_argument("--preview", nargs="*", default=[], help="Video ids to save hit preview images for.")
    parser.add_argument("--labelled-only", action="store_true",
                        help="Ignore the rallies tracked with track_rallies.py.")
    args = parser.parse_args()

    corners = load_corners()
    train_tracks, val_tracks = split_trajectories()
    finder = PlayerFinder()
    train_rallies, skip_tr = rallies_for(train_tracks, corners, finder)
    val_rallies, skip_va = rallies_for(val_tracks, corners, finder)
    for name, rallies in (("train", train_rallies), ("val", val_rallies)):
        evs = [e for r in rallies.values() for e in r]
        print(f"{name}: hit zone taken from the hitting player for "
              f"{sum(e['from_player'] for e in evs)}/{len(evs)} shots")
    if skip_tr or skip_va:
        print(f"Skipped {len(skip_tr) + len(skip_va)} videos without court corners: {sorted(skip_tr + skip_va)}")

    for name, rallies in (("train", train_rallies), ("val", val_rallies)):
        counts = [len(r) for r in rallies.values()]
        print(f"{name}: {len(rallies)} videos, {sum(counts)} shots detected "
              f"(avg {np.mean(counts) if counts else 0:.1f} per video)")

    labelled = {**train_rallies, **val_rallies}
    tracked = {} if args.labelled_only else tracked_rallies(corners, exclude=labelled)
    if tracked:
        evs = [e for r in tracked.values() for e in r]
        print(f"tracked: {len(tracked)} videos, {len(evs)} shots detected "
              f"(avg {len(evs) / len(tracked):.1f} per video)")
    metrics = evaluate(labelled, tracked)
    print(f"\nResults on unseen rallies ({metrics['evaluation']}):")
    for test_set, by_method in metrics["by_test_rallies"].items():
        if by_method["random_forest"] is None:
            continue
        print(f"  tested on {test_set} rallies:")
        for name, m in (*by_method.items(), ("random_chance", metrics["random_chance"])):
            print(f"    {name:<28} accuracy {m['accuracy']:.1%}   top-2 {m['top2_accuracy']:.1%}")
    print("  Top features:", ", ".join(list(metrics["feature_importance"])[:5]))
    all_rallies = {**labelled, **tracked}

    # Final model for the dashboard: refit on every calibrated video
    X, y, zones = dataset_from_rallies(list(all_rallies.values()))
    model = make_model().fit(X, y)
    markov = MarkovBaseline().fit(zones, y)

    import joblib
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "markov": markov, "features": FEATURE_NAMES}, MODEL_PATH)
    transitions = Counter((z, f"{'N' if z[0] == 'F' else 'F'}{t}") for z, t in zip(zones, y))
    metrics["training_transitions"] = {f"{a}->{b}": c for (a, b), c in transitions.most_common()}
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics to {MODEL_DIR / 'metrics.json'}")

    for video in args.preview:
        rally = all_rallies.get(video)
        if rally:
            preview_hits(video, rally, corners, MODEL_DIR / "hit_previews")


if __name__ == "__main__":
    main()
