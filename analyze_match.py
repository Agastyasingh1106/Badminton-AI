"""
Full match analysis: players + shuttle tracking, court zones, shot events and next-zone prediction.

Pass 1 tracks everything and stores positions; shots are then detected on the whole trajectory;
pass 2 re-reads the video and draws the annotated output.

Outputs in runs/analysis/<video_id>/: annotated.mp4 (H.264), events.csv, summary.json
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from badminton.court import CourtMapper, load_corners, video_id_from_path, zone_of
from badminton.events import EARLY_SAMPLES, STRIDE, attach_targets, detect_hits
from badminton.predictor import ZonePredictor
from detect_shuttle import DEFAULT_MODEL as SHUTTLE_MODEL
from detect_shuttle import (SHUTTLE_AIRSPACE_POLYGON, ShuttleTrajectoryTracker, draw_shuttle_overlay,
                            drop_candidates_on_players, extract_temporal_motion_candidates,
                            select_best_shuttle_candidate)
from yolo_inference import TwoPlayerTracker, draw_detection, point_is_near_polygon

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = PROJECT_ROOT / "sample_dataset" / "00467.mp4"
OUTPUT_ROOT = PROJECT_ROOT / "runs" / "analysis"
# Measured with eval_shuttle_tracking.py on hand-labelled frames: 960 px finds the tiny shuttle
# more often than 640, and motion blobs (often shoes/shirts) must not outrank weak model detections
SHUTTLE_IMGSZ = 960
MOTION_CONF = 0.03
STRONG_CONF = 0.3    # model detections this confident override the tracker's expectation
START_CONF = 0.05    # minimum confidence to start a new track
PLAYER_COLOR = (0, 0, 255)
PREDICT_COLOR = (0, 140, 255)
ACTUAL_COLOR = (0, 220, 0)


def _detect_shuttle(model, tracker, f_prev, f_curr, f_next, airspace_mask, airspace_polygon, player_boxes):
    """Same fusion as detect_shuttle.process_video: YOLO + 3-frame motion, then the tracker,
    with candidates on the players' bodies removed first."""
    result = model.predict(f_curr, conf=0.001, imgsz=SHUTTLE_IMGSZ, verbose=False)[0]
    detections = [(*box.xyxy[0].tolist(), float(box.conf.item())) for box in result.boxes]
    if f_prev is not None and f_next is not None:
        motion = extract_temporal_motion_candidates(f_prev, f_curr, f_next, airspace_mask)
        detections.extend((*m[:4], MOTION_CONF) for m in motion)
    detections = drop_candidates_on_players(detections, player_boxes)

    # A confident model detection wins even far from where the tracker expected the shuttle,
    # so a track stuck on the wrong object is dropped at once
    strong = [d for d in detections if d[4] >= STRONG_CONF]
    best = select_best_shuttle_candidate(strong or detections, tracker, airspace_polygon, frame=f_curr)

    # With no track running, a weak candidate (usually a motion blob) is not enough to start one
    if best is not None and tracker.pos is None and best[1] < START_CONF:
        best = None
    if best is not None:
        box, conf = best
        point = tracker.update((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)), conf)
        return point, conf, "detected"
    point = tracker.update(None)
    return (point, 0.2, "predicted") if point is not None else (None, 0.0, None)


def track_video(video_path: Path, mapper: CourtMapper, person_model_path: Path, max_frames=None,
                progress: Optional[Callable] = None, shuttle_model_path: Path = SHUTTLE_MODEL,
                player_filter: bool = True):
    """Pass 1: per-frame players, rackets and shuttle positions."""
    person_model = YOLO(str(person_model_path))
    shuttle_model = YOLO(str(shuttle_model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    if max_frames:
        total = min(total, max_frames)

    court_polygon = mapper.polygon_px()
    airspace_polygon = np.round(SHUTTLE_AIRSPACE_POLYGON * [width, height]).astype(np.int32)
    airspace_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(airspace_mask, [airspace_polygon], 255)

    players = TwoPlayerTracker()
    shuttle = ShuttleTrajectoryTracker(fps=fps)
    frames = []

    ok, f_curr = cap.read()
    f_prev, index = None, 0
    while ok and (max_frames is None or index < max_frames):
        ok_next, f_next = cap.read()
        f_next = f_next if ok_next else None

        result = person_model.track(f_curr, persist=True, tracker="bytetrack.yaml", classes=[0, 38],
                                    conf=0.20, verbose=False)[0]
        candidates, rackets = [], []
        for det in result.boxes:
            x1, y1, x2, y2 = det.xyxy[0].tolist()
            conf, cls = float(det.conf.item()), int(det.cls.item())
            if cls == 0:
                foot = (int((x1 + x2) / 2), int(y2))
                if point_is_near_polygon(foot, court_polygon, tolerance=-30.0):
                    candidates.append({"box": (x1, y1, x2, y2), "conf": conf, "foot": foot})
            elif cls == 38:
                rackets.append(((x1, y1, x2, y2), conf))

        tracked = players.track(candidates)
        point, s_conf, kind = _detect_shuttle(shuttle_model, shuttle, f_prev, f_curr, f_next,
                                              airspace_mask, airspace_polygon,
                                              [box for _, box, _ in tracked] if player_filter else [])
        frames.append({
            "players": [(pid, tuple(map(float, box)), conf) for pid, box, conf in tracked],
            "rackets": [(tuple(map(float, box)), conf) for box, conf in rackets],
            "shuttle": (tuple(map(int, point)), s_conf, kind) if point is not None else None,
        })

        index += 1
        if progress and index % 5 == 0:
            progress(min(index / total, 1.0) * 0.7, f"Tracking frame {index}/{total}")
        f_prev, f_curr, ok = f_curr, f_next, f_next is not None

    cap.release()
    return frames, {"width": width, "height": height, "fps": fps}


def build_events(frames, mapper: CourtMapper, fps: float, predictor: ZonePredictor):
    points = {i: f["shuttle"][0] for i, f in enumerate(frames) if f["shuttle"] is not None}
    players_at = lambda i: [box for _, box, _ in frames[i]["players"]] if 0 <= i < len(frames) else []
    events, samples = detect_hits(points, mapper, fps=fps, players_at=players_at)
    events = attach_targets(events, samples, mapper)
    for k, ev in enumerate(events):
        ev["player"] = 1 if ev["side"] == "far" else 2
        ev["prediction"] = predictor.predict_proba(events, k)
        ev["predicted_zone"] = max(ev["prediction"], key=ev["prediction"].get)
        ev["prediction_ready_frame"] = ev["frame"] + EARLY_SAMPLES * STRIDE
        ev["correct"] = (ev["predicted_zone"] == ev["next_zone"]) if ev.get("next_zone") else None
    return events


def player_movement(frames, mapper: CourtMapper, fps: float):
    """Court positions and distance covered for Player 1 and Player 2."""
    positions = {1: [], 2: []}
    for i, f in enumerate(frames):
        for pid, box, _ in f["players"]:
            foot = ((box[0] + box[2]) / 2, box[3])
            x, y = mapper.to_court(foot)
            positions[pid].append((i, round(x, 3), round(y, 3)))
    stats = {}
    for pid, pts in positions.items():
        xy = np.array([(x, y) for _, x, y in pts]) if pts else np.zeros((0, 2))
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1) if len(xy) > 1 else np.array([])
        steps = steps[steps < 1.0]  # ignore ID jumps
        stats[pid] = {
            "distance_m": float(steps.sum()),
            "frames_seen": len(pts),
            "avg_speed_mps": float(steps.sum() / max(len(pts) / fps, 1e-6)),
            "zones": {z: sum(1 for _, x, y in pts if zone_of((x, y)) == z) for z in
                      sorted({zone_of((x, y)) for _, x, y in pts})},
        }
    return positions, stats


def _panel(frame, lines, origin=(None, 20), width=330):
    x0 = frame.shape[1] - width - 20 if origin[0] is None else origin[0]
    y0 = origin[1]
    h = 22 * len(lines) + 16
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + h), (20, 20, 20), cv2.FILLED)
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + h), (0, 140, 255), 1)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (x0 + 10, y0 + 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def render_video(video_path, frames, events, mapper: CourtMapper, info, out_path: Path,
                 progress: Optional[Callable] = None):
    """Pass 2: draw zones, players, shuttle trail, shot info and live next-zone prediction."""
    cap = cv2.VideoCapture(str(video_path))
    raw_path = out_path.with_name(out_path.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"],
                             (info["width"], info["height"]))
    shuttle = ShuttleTrajectoryTracker(fps=info["fps"])

    for i, data in enumerate(frames):
        ok, frame = cap.read()
        if not ok:
            break
        mapper.draw_zone_grid(frame, color=(230, 230, 230))

        current = None
        for ev in events:
            if ev["frame"] <= i:
                current = ev
        if current is not None:
            if i >= current["prediction_ready_frame"]:
                mapper.fill_zone(frame, current["predicted_zone"], PREDICT_COLOR, 0.35)
            # When a new shot starts, flash the zone it was played from (= where the last shot went)
            if current["shot_no"] > 1 and i < current["frame"] + 15:
                mapper.fill_zone(frame, current["zone"], ACTUAL_COLOR, 0.45)

        for pid, box, conf in data["players"]:
            draw_detection(frame, box, f"Player {pid} {conf:.2f}", color=PLAYER_COLOR, thickness=3)
        for box, conf in data["rackets"]:
            draw_detection(frame, box, f"racket {conf:.2f}", color=(0, 165, 255), thickness=2)

        if data["shuttle"] is not None:
            point, conf, kind = data["shuttle"]
            shuttle.update(point, conf)
            box = (point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8)
            draw_shuttle_overlay(frame, box, conf, shuttle)
        else:
            shuttle.update(None)
            draw_shuttle_overlay(frame, None, 0.0, shuttle)

        shots_so_far = sum(1 for e in events if e["frame"] <= i)
        lines = [(f"RALLY ANALYSIS   shots: {shots_so_far}", (255, 255, 255))]
        if current is not None:
            lines.append((f"Shot {current['shot_no']}: Player {current['player']} from {current['zone']}",
                          (255, 255, 255)))
            lines.append((f"Shot speed: {current['speed_kmh']:.0f} km/h", (0, 255, 255)))
            if i >= current["prediction_ready_frame"]:
                p = current["prediction"][current["predicted_zone"]]
                lines.append((f"Predicted next zone: {current['predicted_zone']} ({p:.0%})", PREDICT_COLOR))
            else:
                lines.append(("Predicting next zone...", (180, 180, 180)))
        _panel(frame, lines)
        writer.write(frame)
        if progress and i % 10 == 0:
            progress(0.7 + 0.25 * i / max(len(frames), 1), f"Rendering frame {i}/{len(frames)}")

    cap.release()
    writer.release()
    to_h264(raw_path, out_path)


def to_h264(src: Path, dst: Path):
    """Re-encode to H.264 so browsers (and Streamlit) can play the video."""
    try:
        import imageio_ffmpeg
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-movflags", "+faststart", str(dst)]
        subprocess.run(cmd, check=True)
        src.unlink(missing_ok=True)
    except Exception as exc:  # keep the mp4v file if ffmpeg is unavailable
        print(f"H.264 conversion failed ({exc}); keeping {src.name}")
        src.replace(dst)


def write_outputs(out_dir: Path, video_path, events, movement, stats, info, corners):
    with (out_dir / "events.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["shot_no", "frame", "time_s", "player", "side", "zone", "speed_kmh",
                         "next_zone", "predicted_zone", "predicted_prob", "correct"])
        for ev in events:
            writer.writerow([ev["shot_no"], ev["frame"], round(ev["frame"] / info["fps"], 2), ev["player"],
                             ev["side"], ev["zone"], round(ev["speed_kmh"], 1), ev.get("next_zone") or "",
                             ev["predicted_zone"], round(ev["prediction"][ev["predicted_zone"]], 3),
                             "" if ev["correct"] is None else ev["correct"]])

    judged = [e for e in events if e["correct"] is not None]
    summary = {
        "video": str(video_path), "video_id": video_id_from_path(video_path), "info": info,
        "corners": corners, "n_shots": len(events),
        "rally_duration_s": round(len(movement[1] or movement[2] or []) / info["fps"], 2),
        "avg_shot_speed_kmh": float(np.mean([e["speed_kmh"] for e in events])) if events else 0.0,
        "max_shot_speed_kmh": float(max([e["speed_kmh"] for e in events], default=0.0)),
        "prediction_accuracy": (sum(e["correct"] for e in judged) / len(judged)) if judged else None,
        "players": stats,
        "events": [{k: v for k, v in e.items() if k != "early"} for e in events],
        "player_positions": {str(pid): pts[::3] for pid, pts in movement.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def analyze_video(video_path, corners=None, person_model="yolov8n.pt", max_frames=None,
                  progress: Optional[Callable] = None, out_dir: Optional[Path] = None):
    video_path = Path(video_path)
    video_id = video_id_from_path(video_path)
    if corners is None:
        corners = load_corners().get(video_id)
    if corners is None:
        raise ValueError(f"No court corners for {video_id}. Run calibrate_court.py {video_path} first.")
    mapper = CourtMapper(corners)

    person_model_path = Path(person_model)
    if not person_model_path.is_file() and (PROJECT_ROOT / person_model).is_file():
        person_model_path = PROJECT_ROOT / person_model
    out_dir = out_dir or OUTPUT_ROOT / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, info = track_video(video_path, mapper, person_model_path, max_frames, progress)
    predictor = ZonePredictor()
    events = build_events(frames, mapper, info["fps"], predictor)
    movement, stats = player_movement(frames, mapper, info["fps"])
    info["n_frames"] = len(frames)
    info["predictor"] = "random_forest" if predictor.ready else "uniform (no trained model)"

    render_video(video_path, frames, events, mapper, info, out_dir / "annotated.mp4", progress)
    summary = write_outputs(out_dir, video_path, events, movement, stats, info,
                            [list(map(float, c)) for c in corners])
    if progress:
        progress(1.0, "Done")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a badminton rally video.")
    parser.add_argument("video", nargs="?", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--person-model", default="yolov8n.pt")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    result = analyze_video(args.video, person_model=args.person_model, max_frames=args.max_frames,
                           progress=lambda p, msg: print(f"[{p:5.0%}] {msg}"))
    print(f"\nShots: {result['n_shots']}   avg speed {result['avg_shot_speed_kmh']:.0f} km/h   "
          f"prediction accuracy: {result['prediction_accuracy']}")
    for ev in result["events"]:
        print(f"  shot {ev['shot_no']:>2} frame {ev['frame']:>4}  P{ev['player']} {ev['zone']} -> "
              f"{ev.get('next_zone')}  predicted {ev['predicted_zone']}  {ev['speed_kmh']:.0f} km/h")
    print(f"Outputs in {OUTPUT_ROOT / result['video_id']}")
