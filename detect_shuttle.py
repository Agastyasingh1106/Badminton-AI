"""
High-accuracy badminton shuttlecock detector and trajectory tracker.
Features physics-based motion smoothing, velocity gating, court polygon filtering,
and TV-broadcast style flight trajectory visualization.
"""

import argparse
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = PROJECT_ROOT / "sample_dataset" / "00467.mp4"
DEFAULT_MODEL = PROJECT_ROOT / "runs" / "detect" / "shuttlecock" / "weights" / "best.pt"
FALLBACK_MODEL = PROJECT_ROOT / "yolov8n.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "detect" / "shuttle_predict"

# 3D Flight Airspace polygon for broadcast view (includes high clears and lifts)
SHUTTLE_AIRSPACE_POLYGON = np.array(
    [[0.25, 0.05], [0.75, 0.05], [0.88, 1.00], [0.12, 1.00]],
    dtype=np.float32,
)
COURT_POLYGON = SHUTTLE_AIRSPACE_POLYGON


def extract_temporal_motion_candidates(f_prev, f_curr, f_next, airspace_mask):
    """Detects small fast-moving white objects using 3-frame temporal differencing."""
    if f_prev is None or f_curr is None or f_next is None:
        return []
    g_prev = cv2.cvtColor(f_prev, cv2.COLOR_BGR2GRAY)
    g_curr = cv2.cvtColor(f_curr, cv2.COLOR_BGR2GRAY)
    g_next = cv2.cvtColor(f_next, cv2.COLOR_BGR2GRAY)

    d = cv2.bitwise_and(cv2.absdiff(g_curr, g_prev), cv2.absdiff(g_next, g_curr))
    if airspace_mask is not None:
        d = cv2.bitwise_and(d, airspace_mask)
    _, thresh = cv2.threshold(d, 8, 255, cv2.THRESH_BINARY)

    hsv = cv2.cvtColor(f_curr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 175), (180, 60, 255))
    shuttle_motion = cv2.bitwise_and(thresh, white_mask)

    contours, _ = cv2.findContours(shuttle_motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_cands = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if 2 <= bw <= 35 and 2 <= bh <= 35:
            motion_cands.append((float(x), float(y), float(x + bw), float(y + bh), 0.30))
    return motion_cands


class ShuttleTrajectoryTracker:
    """Robust physics-based tracker for badminton shuttlecock trajectory."""

    def __init__(
        self,
        max_trail: int = 24,
        max_jump_px: float = 240.0,
        fps: float = 30.0,
        pixels_per_meter: float = 45.0,
    ):
        self.max_trail = max_trail
        self.max_jump_px = max_jump_px
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.history: Deque[Tuple[int, int]] = deque(maxlen=max_trail)
        self.confs: Deque[float] = deque(maxlen=max_trail)
        self.pos: Optional[Tuple[float, float]] = None
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.missed_frames = 0
        self.max_missed_tolerance = 10
        self.static_counts = {}

    def predict_next(self) -> Optional[Tuple[float, float]]:
        """Predicts position in current frame using velocity and subtle downward gravity."""
        if self.pos is None:
            return None
        pred_x = self.pos[0] + self.vel[0]
        pred_y = self.pos[1] + self.vel[1] + 1.2
        return (pred_x, pred_y)

    def update(
        self, detection: Optional[Tuple[int, int]], conf: float = 0.0
    ) -> Optional[Tuple[int, int]]:
        """Updates tracker with verified candidate detection or extrapolates during motion blur."""
        if detection is not None:
            cx, cy = float(detection[0]), float(detection[1])
            if self.pos is not None:
                new_vx = cx - self.pos[0]
                new_vy = cy - self.pos[1]
                # Smooth velocity with exponential moving average
                self.vel = 0.65 * np.array([new_vx, new_vy], dtype=np.float32) + 0.35 * self.vel
            self.pos = (cx, cy)
            self.history.append((int(cx), int(cy)))
            self.confs.append(conf)
            self.missed_frames = 0
            return (int(cx), int(cy))

        # Extrapolate through temporary motion blur or occlusion
        if self.pos is not None and self.missed_frames < self.max_missed_tolerance:
            pred = self.predict_next()
            if pred is not None:
                self.pos = pred
                self.history.append((int(pred[0]), int(pred[1])))
                last_conf = self.confs[-1] if self.confs else 0.05
                self.confs.append(last_conf * 0.85)
                self.vel *= 0.96  # aerodynamic air drag
                self.missed_frames += 1
                return (int(pred[0]), int(pred[1]))

        # Track is fully lost: clear history so tracker can immediately re-acquire at next hit
        self.pos = None
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.missed_frames += 1
        return None

    def get_current_speed_kmh(self) -> float:
        """Estimates shuttle speed in km/h based on recent frame displacement."""
        if len(self.history) < 2:
            return 0.0
        p1, p2 = self.history[-2], self.history[-1]
        pixel_dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        meters_per_sec = (pixel_dist / self.pixels_per_meter) * self.fps
        return min(meters_per_sec * 3.6, 380.0)

    def draw_trajectory(self, frame: np.ndarray):
        """Draws glowing broadcast-style trajectory trail with fading gradient."""
        pts = list(self.history)
        n = len(pts)
        if n < 2:
            return

        for i in range(1, n):
            alpha = float(i) / float(n)
            thickness = max(1, int(4 * alpha))
            # Vibrant cyan/yellow gradient flight arc
            color = (
                int(20 + 220 * (1 - alpha)),
                int(245 * alpha),
                255,
            )
            cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)
            cv2.circle(frame, pts[i], max(1, int(3 * alpha)), color, -1, cv2.LINE_AA)


def is_near_court(point: Tuple[int, int], court_polygon: np.ndarray, tolerance: float = -30.0) -> bool:
    """Checks if point is inside or within reasonable margin of the court."""
    return cv2.pointPolygonTest(court_polygon, point, True) >= tolerance


def select_best_shuttle_candidate(
    detections: List[Tuple[float, float, float, float, float]],
    tracker: ShuttleTrajectoryTracker,
    court_polygon: np.ndarray,
    frame: Optional[np.ndarray] = None,
) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
    """Selects the most probable shuttlecock detection combining confidence, motion, and visual cues."""
    if not detections:
        return None

    pred = tracker.predict_next()
    valid_candidates = []

    for x1, y1, x2, y2, conf in detections:
        bw = x2 - x1
        bh = y2 - y1
        if not (3 <= bw <= 55 and 3 <= bh <= 55):
            continue
        aspect_ratio = bw / max(bh, 1.0)
        if not (0.25 <= aspect_ratio <= 4.0):
            continue

        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if not is_near_court((int(cx), int(cy)), court_polygon):
            continue

        # Suppress static clutter (e.g. net posts, umpire chair clips, court logos)
        grid_key = (int(cx // 8), int(cy // 8))
        count = tracker.static_counts.get(grid_key, 0) + 1
        tracker.static_counts[grid_key] = count
        if count > 4:
            continue

        # Visual brightness check: shuttlecock is high-brightness white
        if frame is not None:
            iy1, iy2 = max(0, int(y1)), min(frame.shape[0], int(y2))
            ix1, ix2 = max(0, int(x1)), min(frame.shape[1], int(x2))
            patch = frame[iy1:iy2, ix1:ix2]
            if patch.size > 0:
                mean_val = float(patch.mean())
                if mean_val < 85.0:
                    continue

        score = conf * 100.0
        if pred is not None:
            dist = np.hypot(cx - pred[0], cy - pred[1])
            if dist < tracker.max_jump_px:
                # Proximity boost when matching the flight trajectory
                score += max(0.0, (tracker.max_jump_px - dist) / 15.0)
            else:
                score *= 0.25

        valid_candidates.append(((int(x1), int(y1), int(x2), int(y2)), score, conf))

    if not valid_candidates:
        return None

    valid_candidates.sort(key=lambda item: item[1], reverse=True)
    best_box, _, best_conf = valid_candidates[0]
    return best_box, best_conf


def draw_shuttle_overlay(
    frame: np.ndarray,
    box: Optional[Tuple[int, int, int, int]],
    conf: float,
    tracker: ShuttleTrajectoryTracker,
):
    """Draws targeting crosshairs, speed overlay, and status HUD."""
    # Draw trajectory line
    tracker.draw_trajectory(frame)

    if box is not None:
        x1, y1, x2, y2 = box
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        # Distinct shuttle indicator: Outer glowing ring + inner crosshair
        cv2.circle(frame, (cx, cy), 12, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)

        label = f"Shuttle {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y = max(y1 - 6, th + 4)
        cv2.rectangle(
            frame,
            (x1 - 2, tag_y - th - 3),
            (x1 + tw + 4, tag_y + 3),
            (0, 255, 255),
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label,
            (x1, tag_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # Top-left HUD Card
    speed_kmh = tracker.get_current_speed_kmh()
    status_text = "TRACKING" if len(tracker.history) > 0 and tracker.missed_frames == 0 else "LOST / SEARCHING"
    status_color = (0, 255, 0) if "TRACKING" in status_text else (0, 165, 255)

    cv2.rectangle(frame, (20, 20), (280, 85), (20, 20, 20), cv2.FILLED)
    cv2.rectangle(frame, (20, 20), (280, 85), (0, 255, 255), 1)

    cv2.putText(
        frame,
        f"SHUTTLECOCK DETECTOR",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"STATUS: {status_text}",
        (30, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        status_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"EST. SPEED: {speed_kmh:.1f} km/h",
        (30, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


def process_video(
    video_path: Path,
    output_path: Path,
    model_path: Path,
    conf_threshold: float = 0.001,
    imgsz: int = 640,
    max_frames: Optional[int] = None,
):
    """Processes video file, detecting and tracking shuttlecock with trajectory overlay."""
    if not model_path.is_file():
        if FALLBACK_MODEL.is_file():
            print(f"Warning: Model {model_path} not found. Falling back to {FALLBACK_MODEL}.")
            model_path = FALLBACK_MODEL
        else:
            raise FileNotFoundError(f"Model weights not found at {model_path}")

    print(f"Loading shuttlecock model: {model_path}")
    model = YOLO(str(model_path))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open input video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    court_polygon = np.round(COURT_POLYGON * [width, height]).astype(np.int32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not initialize output video writer: {output_path}")

    tracker = ShuttleTrajectoryTracker(max_trail=24, fps=fps)
    airspace_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(airspace_mask, [court_polygon], 255)

    frame_index = 0
    detected_frames = 0

    print(f"Processing video {video_path.name} ({width}x{height} @ {fps:.1f} FPS)...")
    try:
        ret, f_curr = capture.read()
        if not ret:
            return

        f_prev = None
        while True:
            ret, f_next = capture.read()
            if not ret:
                f_next = None

            if max_frames is not None and frame_index >= max_frames:
                break

            # 1. Deep Learning candidate proposals
            results = model.predict(f_curr, conf=conf_threshold, imgsz=imgsz, verbose=False)[0]
            detections = []
            for box in results.boxes:
                coords = box.xyxy[0].tolist()
                conf = float(box.conf.item())
                detections.append((*coords, conf))

            # 2. Temporal motion candidate proposals
            if f_prev is not None and f_next is not None:
                motion_detections = extract_temporal_motion_candidates(f_prev, f_curr, f_next, airspace_mask)
                detections.extend(motion_detections)

            # 3. Associate and score best candidate
            best_candidate = select_best_shuttle_candidate(detections, tracker, court_polygon, frame=f_curr)

            if best_candidate is not None:
                best_box, best_conf = best_candidate
                cx = int((best_box[0] + best_box[2]) / 2)
                cy = int((best_box[1] + best_box[3]) / 2)
                tracker.update((cx, cy), best_conf)
                draw_shuttle_overlay(f_curr, best_box, best_conf, tracker)
                detected_frames += 1
            else:
                extrapolated = tracker.update(None)
                if extrapolated is not None:
                    ex_box = (extrapolated[0] - 8, extrapolated[1] - 8, extrapolated[0] + 8, extrapolated[1] + 8)
                    draw_shuttle_overlay(f_curr, ex_box, 0.20, tracker)
                    detected_frames += 1
                else:
                    draw_shuttle_overlay(f_curr, None, 0.0, tracker)

            writer.write(f_curr)
            frame_index += 1
            if frame_index % 30 == 0:
                print(f"Processed frame {frame_index}... (Shuttle tracked in {detected_frames}/{frame_index} frames)")

            if f_next is None:
                break
            f_prev = f_curr
            f_curr = f_next

    finally:
        capture.release()
        writer.release()

    print(f"\nFinished processing {frame_index} frames.")
    print(f"Shuttlecock tracked in {detected_frames} frames.")
    print(f"Output saved to: {output_path}")


def process_image(
    image_path: Path,
    output_path: Path,
    model_path: Path,
    conf_threshold: float = 0.005,
    imgsz: int = 640,
):
    """Processes a single image and overlays detection."""
    model = YOLO(str(model_path))
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    court_polygon = np.round(COURT_POLYGON * [w, h]).astype(np.int32)
    tracker = ShuttleTrajectoryTracker()

    results = model.predict(img, conf=conf_threshold, imgsz=imgsz, verbose=False)[0]
    detections = []
    for box in results.boxes:
        coords = box.xyxy[0].tolist()
        conf = float(box.conf.item())
        detections.append((*coords, conf))

    best_candidate = select_best_shuttle_candidate(detections, tracker, court_polygon)
    if best_candidate is not None:
        best_box, best_conf = best_candidate
        cx = int((best_box[0] + best_box[2]) / 2)
        cy = int((best_box[1] + best_box[3]) / 2)
        tracker.update((cx, cy), best_conf)
        draw_shuttle_overlay(img, best_box, best_conf, tracker)
    else:
        draw_shuttle_overlay(img, None, 0.0, tracker)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"Saved annotated image to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accurate Shuttlecock Detector and Trajectory Tracker.")
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_VIDEO, help="Input video or image.")
    parser.add_argument("--output", type=Path, default=None, help="Output destination path.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to trained shuttlecock weights.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold (default: 0.001).")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution (default: 640).")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames to process.")
    args = parser.parse_args()

    input_source = args.source
    if input_source.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        out = args.output or DEFAULT_OUTPUT_DIR / f"{input_source.stem}_shuttle_detected{input_source.suffix}"
        process_image(input_source, out, args.model, args.conf, args.imgsz)
    else:
        out = args.output or DEFAULT_OUTPUT_DIR / f"{input_source.stem}_shuttle_tracked.mp4"
        process_video(input_source, out, args.model, args.conf, args.imgsz, args.max_frames)
