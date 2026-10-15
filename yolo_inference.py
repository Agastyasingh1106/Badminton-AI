import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from detect_shuttle import (
        ShuttleTrajectoryTracker,
        select_best_shuttle_candidate,
        draw_shuttle_overlay,
        extract_temporal_motion_candidates,
        SHUTTLE_AIRSPACE_POLYGON,
    )
except ImportError:
    ShuttleTrajectoryTracker = None
    select_best_shuttle_candidate = None
    draw_shuttle_overlay = None
    extract_temporal_motion_candidates = None
    SHUTTLE_AIRSPACE_POLYGON = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = PROJECT_ROOT / "sample_dataset" / "00467.mp4"
PERSON_MODEL_PATH = PROJECT_ROOT / "yolov8x.pt"
SHUTTLE_MODEL_PATH = (
    PROJECT_ROOT / "runs" / "detect" / "shuttlecock" / "weights" / "best.pt"
)
OUTPUT_DIR = PROJECT_ROOT / "runs" / "detect" / "predict"

# Court corners for the fixed broadcast view, in normalized image coordinates.
COURT_POLYGON = np.array(
    [[0.36, 0.37], [0.64, 0.37], [0.80, 1.00], [0.20, 1.00]],
    dtype=np.float32,
)


def point_is_near_polygon(point, polygon, tolerance=-25.0):
    """Returns True if point is inside or within tolerance margin of the court polygon."""
    return cv2.pointPolygonTest(polygon, point, True) >= tolerance


def draw_detection(frame, box, label, color, text_color=(255, 255, 255), thickness=3):
    """Draws a high-visibility bounding box with a solid colored header banner and white text."""
    x1, y1, x2, y2 = map(int, box)
    # Draw outer bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Calculate text badge size
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2
    )
    badge_y1 = max(y1 - text_h - 10, 0)
    badge_y2 = y1
    badge_x2 = min(x1 + text_w + 10, frame.shape[1] - 1)

    # Filled banner for maximum visibility
    cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, cv2.FILLED)
    cv2.putText(
        frame,
        label,
        (x1 + 5, badge_y2 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        text_color,
        2,
        cv2.LINE_AA,
    )


class TwoPlayerTracker:
    """Robust tracker for badminton singles that consistently identifies Player 1 and Player 2.
    
    Assigns:
      - Player 1: Far Court player (top half of court, smaller Y coordinate)
      - Player 2: Near Court player (bottom half of court, larger Y coordinate)
    Ensures both players maintain stable IDs without dropping or switching.
    """

    def __init__(self):
        self.player1_pos = None  # Far/Top player foot coordinate
        self.player2_pos = None  # Near/Bottom player foot coordinate

    def track(self, candidate_detections):
        """
        candidate_detections: list of dicts with keys: 'box', 'conf', 'foot'
        Returns: list of tuples (player_id, box, conf)
        """
        if not candidate_detections:
            return []

        # Sort candidate players by foot Y coordinate (top of screen to bottom)
        sorted_candidates = sorted(candidate_detections, key=lambda d: d["foot"][1])

        assigned = []
        if len(sorted_candidates) >= 2:
            # Top-most player inside court is Player 1 (Far court)
            # Bottom-most player inside court is Player 2 (Near court)
            p1 = sorted_candidates[0]
            p2 = sorted_candidates[-1]
            self.player1_pos = p1["foot"]
            self.player2_pos = p2["foot"]
            assigned.append((1, p1["box"], p1["conf"]))
            assigned.append((2, p2["box"], p2["conf"]))
        elif len(sorted_candidates) == 1:
            cand = sorted_candidates[0]
            foot = cand["foot"]
            # Reassociate with closest player slot from previous frame
            if self.player1_pos is not None and self.player2_pos is not None:
                d1 = np.hypot(foot[0] - self.player1_pos[0], foot[1] - self.player1_pos[1])
                d2 = np.hypot(foot[0] - self.player2_pos[0], foot[1] - self.player2_pos[1])
                pid = 1 if d1 <= d2 else 2
            else:
                # Default: Far court (y < 400) is Player 1, Near court is Player 2
                pid = 1 if foot[1] < 400 else 2

            if pid == 1:
                self.player1_pos = foot
            else:
                self.player2_pos = foot
            assigned.append((pid, cand["box"], cand["conf"]))

        return assigned


def predict_video(video_path, output_path, max_frames=None, person_model_path=PERSON_MODEL_PATH):
    person_model = YOLO(str(person_model_path))
    shuttle_model = (
        YOLO(str(SHUTTLE_MODEL_PATH)) if SHUTTLE_MODEL_PATH.is_file() else None
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    court_polygon = np.round(COURT_POLYGON * [width, height]).astype(np.int32)
    air_poly = SHUTTLE_AIRSPACE_POLYGON if SHUTTLE_AIRSPACE_POLYGON is not None else COURT_POLYGON
    airspace_polygon = np.round(air_poly * [width, height]).astype(np.int32)
    airspace_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(airspace_mask, [airspace_polygon], 255)

    player_tracker = TwoPlayerTracker()
    shuttle_tracker = ShuttleTrajectoryTracker(fps=fps) if ShuttleTrajectoryTracker is not None else None
    f_prev = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not create output video: {output_path}")

    try:
        for frame_index, result in enumerate(
            person_model.track(
                source=str(video_path),
                stream=True,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0, 38],
                conf=0.20,
                verbose=False,
            )
        ):
            if max_frames is not None and frame_index >= max_frames:
                break
            frame = result.orig_img.copy()

            person_candidates = []
            rackets = []

            for detection in result.boxes:
                class_id = int(detection.cls.item())
                if class_id not in (0, 38):
                    continue

                x1, y1, x2, y2 = detection.xyxy[0].tolist()
                conf = float(detection.conf.item())

                if class_id == 0:
                    foot_point = (int((x1 + x2) / 2), int(y2))
                    # Allow slight margin outside court polygon for lunges/jump-smashes
                    if point_is_near_polygon(foot_point, court_polygon, tolerance=-30.0):
                        person_candidates.append({
                            "box": (x1, y1, x2, y2),
                            "conf": conf,
                            "foot": foot_point,
                        })
                elif class_id == 38:
                    rackets.append(((x1, y1, x2, y2), conf))

            # Consistently track Player 1 and Player 2
            tracked_players = player_tracker.track(person_candidates)

            # Draw players in bold RED with high-visibility banner
            # Color in OpenCV BGR: Red = (0, 0, 255)
            RED_COLOR = (0, 0, 255)
            for player_id, box, conf in tracked_players:
                label = f"Player {player_id} {conf:.2f}"
                draw_detection(frame, box, label, color=RED_COLOR, thickness=3)

            # Draw badminton rackets in orange/amber
            for box, conf in rackets:
                draw_detection(
                    frame,
                    box,
                    f"racket {conf:.2f}",
                    color=(0, 165, 255),
                    thickness=2,
                )

            # Draw shuttlecock if model is loaded
            if shuttle_model is not None:
                shuttle_result = shuttle_model.predict(
                    frame, conf=0.001, imgsz=640, verbose=False
                )[0]
                if shuttle_tracker is not None and select_best_shuttle_candidate is not None:
                    s_detections = []
                    for detection in shuttle_result.boxes:
                        coords = detection.xyxy[0].tolist()
                        s_conf = float(detection.conf.item())
                        s_detections.append((*coords, s_conf))

                    # Fuse temporal motion cues if previous frame is available
                    if f_prev is not None and extract_temporal_motion_candidates is not None:
                        motion_cands = extract_temporal_motion_candidates(f_prev, frame, frame, airspace_mask)
                        s_detections.extend(motion_cands)

                    best_candidate = select_best_shuttle_candidate(
                        s_detections, shuttle_tracker, airspace_polygon, frame=frame
                    )
                    if best_candidate is not None:
                        best_box, best_conf = best_candidate
                        cx = int((best_box[0] + best_box[2]) / 2)
                        cy = int((best_box[1] + best_box[3]) / 2)
                        shuttle_tracker.update((cx, cy), best_conf)
                        draw_shuttle_overlay(frame, best_box, best_conf, shuttle_tracker)
                    else:
                        extrapolated = shuttle_tracker.update(None)
                        if extrapolated is not None:
                            ex_box = (
                                extrapolated[0] - 8,
                                extrapolated[1] - 8,
                                extrapolated[0] + 8,
                                extrapolated[1] + 8,
                            )
                            draw_shuttle_overlay(frame, ex_box, 0.20, shuttle_tracker)
                        else:
                            shuttle_tracker.draw_trajectory(frame)
                else:
                    for detection in shuttle_result.boxes:
                        s_box = detection.xyxy[0].tolist()
                        s_conf = float(detection.conf.item())
                        draw_detection(
                            frame,
                            s_box,
                            f"shuttlecock {s_conf:.2f}",
                            color=(0, 255, 255),
                            thickness=2,
                        )

            writer.write(frame)
            f_prev = frame.copy()
    finally:
        capture.release()
        writer.release()

    if shuttle_model is None:
        print(
            "Note: Shuttlecock model not found at runs/detect/shuttlecock/weights/best.pt. "
            "Run 'python train_shuttlecock.py' to train it."
        )
    print(f"Saved filtered video to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect badminton players and gear.")
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=DEFAULT_VIDEO,
        help="Input video (defaults to sample_dataset/00467.mp4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "00467_filtered.mp4",
        help="Path for the annotated output video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optionally process only the first N frames.",
    )
    parser.add_argument(
        "--person-model",
        type=Path,
        default=PERSON_MODEL_PATH,
        help="YOLO model for person tracking (defaults to yolov8x.pt).",
    )
    arguments = parser.parse_args()
    predict_video(arguments.video, arguments.output, arguments.max_frames, person_model_path=arguments.person_model)
