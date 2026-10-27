"""
Court geometry: maps image pixels to court metres and splits each half into 6 zones.

Court frame (metres), as seen from the broadcast camera:
  x: 0 (left sideline) -> 6.1 (right sideline)
  y: 0 (far baseline)  -> 13.4 (near baseline), net at y = 6.7

Zones per side (F = far half, N = near half), numbered as seen on screen:
  1 2 3  -> front row (next to the net)
  4 5 6  -> back row (next to the baseline)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORNERS_FILE = PROJECT_ROOT / "court_corners.json"

COURT_WIDTH = 6.1
COURT_LENGTH = 13.4
NET_Y = COURT_LENGTH / 2
HALF_DEPTH = NET_Y / 2

# Corner order used everywhere: far-left, far-right, near-right, near-left
COURT_CORNERS_M = np.array(
    [[0.0, 0.0], [COURT_WIDTH, 0.0], [COURT_WIDTH, COURT_LENGTH], [0.0, COURT_LENGTH]],
    dtype=np.float32,
)
SIDE_ZONES = {"far": [f"F{i}" for i in range(1, 7)], "near": [f"N{i}" for i in range(1, 7)]}
ALL_ZONES = SIDE_ZONES["far"] + SIDE_ZONES["near"]

# Typical racket contact height, used to drop an airborne hit point onto the floor
HIT_HEIGHT_M = 2.0


def video_id_from_path(path) -> str:
    """'00467.mp4' / '00352_predict_f000016.jpg' -> '00467' / '00352'."""
    return Path(path).stem.split("_")[0]


def load_corners(path: Path = CORNERS_FILE) -> Dict[str, List[List[float]]]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_corners(corners: Dict[str, List[List[float]]], path: Path = CORNERS_FILE):
    path.write_text(json.dumps(corners, indent=2, sort_keys=True), encoding="utf-8")


def zone_of(point_m: Tuple[float, float]) -> str:
    """Zone id for a court point in metres. Points outside the lines snap to the nearest zone."""
    x = float(np.clip(point_m[0], 0.0, COURT_WIDTH - 1e-6))
    y = float(np.clip(point_m[1], 0.0, COURT_LENGTH - 1e-6))
    col = int(x // (COURT_WIDTH / 3))
    if y < NET_Y:
        row = 0 if y >= HALF_DEPTH else 1          # far half: front row is next to the net
        return f"F{row * 3 + col + 1}"
    row = 0 if y < NET_Y + HALF_DEPTH else 1       # near half
    return f"N{row * 3 + col + 1}"


def side_of(point_m: Tuple[float, float]) -> str:
    return "far" if point_m[1] < NET_Y else "near"


def zone_center_m(zone: str) -> Tuple[float, float]:
    idx = int(zone[1:]) - 1
    row, col = divmod(idx, 3)
    x = (col + 0.5) * COURT_WIDTH / 3
    if zone[0] == "F":
        y = NET_Y - (row + 0.5) * HALF_DEPTH
    else:
        y = NET_Y + (row + 0.5) * HALF_DEPTH
    return x, y


def zone_rect_m(zone: str) -> Tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of a zone in court metres."""
    cx, cy = zone_center_m(zone)
    hw, hh = COURT_WIDTH / 6, HALF_DEPTH / 2
    return cx - hw, cy - hh, cx + hw, cy + hh


class CourtMapper:
    """Converts between image pixels and court metres using the 4 court corners."""

    def __init__(self, corners_px):
        self.corners_px = np.asarray(corners_px, dtype=np.float32).reshape(4, 2)
        self.img_to_court = cv2.getPerspectiveTransform(self.corners_px, COURT_CORNERS_M)
        self.court_to_img = cv2.getPerspectiveTransform(COURT_CORNERS_M, self.corners_px)

    def to_court(self, point_px) -> Tuple[float, float]:
        p = np.array([[point_px]], dtype=np.float32)
        x, y = cv2.perspectiveTransform(p, self.img_to_court)[0, 0]
        return float(x), float(y)

    def to_image(self, point_m) -> Tuple[float, float]:
        p = np.array([[point_m]], dtype=np.float32)
        x, y = cv2.perspectiveTransform(p, self.court_to_img)[0, 0]
        return float(x), float(y)

    def px_per_metre(self, point_px) -> float:
        """Local image scale at a pixel location, measured across the court width."""
        cx, cy = self.to_court(point_px)
        left = self.to_image((cx - 0.5, cy))
        right = self.to_image((cx + 0.5, cy))
        return max(float(np.hypot(right[0] - left[0], right[1] - left[1])), 1e-3)

    def ground_point(self, point_px, height_m: float = HIT_HEIGHT_M) -> Tuple[float, float]:
        """Court position under an airborne point, assuming it is height_m above the floor."""
        x, y = float(point_px[0]), float(point_px[1])
        foot_y = y
        for _ in range(3):
            foot_y = y + height_m * self.px_per_metre((x, foot_y))
        return self.to_court((x, foot_y))

    def polygon_px(self) -> np.ndarray:
        return np.round(self.corners_px).astype(np.int32)

    def zone_polygon_px(self, zone: str) -> np.ndarray:
        x0, y0, x1, y1 = zone_rect_m(zone)
        pts = [self.to_image(p) for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        return np.round(np.array(pts)).astype(np.int32)

    def draw_zone_grid(self, frame, color=(255, 255, 255), labels: bool = True):
        """Draws the 12 zones on the frame."""
        for zone in ALL_ZONES:
            poly = self.zone_polygon_px(zone)
            cv2.polylines(frame, [poly], True, color, 1, cv2.LINE_AA)
            if labels:
                cx, cy = self.to_image(zone_center_m(zone))
                cv2.putText(frame, zone, (int(cx) - 10, int(cy) + 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, color, 1, cv2.LINE_AA)
        x0, y0 = self.to_image((0.0, NET_Y))
        x1, y1 = self.to_image((COURT_WIDTH, NET_Y))
        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 220, 255), 2, cv2.LINE_AA)

    def fill_zone(self, frame, zone: str, color, alpha: float = 0.35):
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.zone_polygon_px(zone)], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


def mapper_for_video(video_id: str, corners: Optional[dict] = None) -> Optional[CourtMapper]:
    corners = corners if corners is not None else load_corners()
    return CourtMapper(corners[video_id]) if video_id in corners else None
