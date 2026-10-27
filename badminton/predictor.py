"""
Next-zone predictor: given the shot just played, predict which of the opponent's 6 zones
the shuttle is going to (where the opponent will play their next shot from).
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from badminton.court import PROJECT_ROOT
from badminton.events import EARLY_SAMPLES

MODEL_DIR = PROJECT_ROOT / "runs" / "zone_model"
MODEL_PATH = MODEL_DIR / "zone_predictor.joblib"
LOCAL_ZONES = [1, 2, 3, 4, 5, 6]

FEATURE_NAMES = (
    ["hitter_near", "zone_row", "zone_col", "court_x", "depth_from_net", "prev_row", "prev_col", "shot_no",
     "speed_kmh", "aim_x"]
    + [f"move{j}_{axis}" for j in range(1, EARLY_SAMPLES + 1) for axis in ("dx", "forward")]
)


def _row_col(zone: Optional[str]):
    if not zone:
        return -1, -1
    return divmod(int(zone[1:]) - 1, 3)


def opponent_zone(hitter_zone: str, local: int) -> str:
    return ("N" if hitter_zone[0] == "F" else "F") + str(local)


def build_features(events: List[dict], k: int) -> List[float]:
    """Feature vector for shot k, using only shot k, the shot before it and its first moves."""
    ev = events[k]
    row, col = _row_col(ev["zone"])
    prev_row, prev_col = _row_col(events[k - 1]["zone"]) if k > 0 else (-1, -1)
    near = ev["side"] == "near"
    early = ev.get("early") or [(0.0, 0.0)] * EARLY_SAMPLES
    # 'forward' = movement towards the opponent (up the image for the near player, down for the far one),
    # so shots from both sides look alike to the model
    moves = [v for dx, dy in early for v in (dx, -dy if near else dy)]
    aim_x = ev["court"][0] + 3.0 * early[-1][0]   # rough sideways landing guess from the flight direction
    return [
        1.0 if near else 0.0, row, col, ev["court"][0], abs(ev["court"][1] - 6.7),
        prev_row, prev_col, min(ev.get("shot_no", k + 1), 20), ev.get("speed_kmh", 0.0), aim_x,
    ] + moves


def dataset_from_rallies(rallies: List[List[dict]]):
    X, y, zones = [], [], []
    for events in rallies:
        for k, ev in enumerate(events):
            if ev.get("next_zone"):
                X.append(build_features(events, k))
                y.append(int(ev["next_zone"][1:]))
                zones.append(ev["zone"])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), zones


class MarkovBaseline:
    """Predicts the most common next zone seen after the current zone in training."""

    def fit(self, zones: List[str], y):
        self.table: Dict[str, Counter] = defaultdict(Counter)
        self.overall = Counter()
        for zone, target in zip(zones, y):
            self.table[zone][int(target)] += 1
            self.overall[int(target)] += 1
        return self

    def predict_proba(self, zone: str) -> np.ndarray:
        counts = self.table.get(zone) or self.overall
        probs = np.array([counts.get(z, 0) + 0.5 for z in LOCAL_ZONES], dtype=np.float64)
        return probs / probs.sum()


class ZonePredictor:
    """Random Forest wrapper; falls back to the Markov table when the model file is missing."""

    def __init__(self, path: Path = MODEL_PATH):
        self.model = None
        self.markov = None
        if path.is_file():
            import joblib
            bundle = joblib.load(path)
            self.model, self.markov = bundle["model"], bundle["markov"]

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict_proba(self, events: List[dict], k: int) -> Dict[str, float]:
        """{opponent zone id: probability} for shot k."""
        hitter_zone = events[k]["zone"]
        if self.model is not None:
            raw = self.model.predict_proba([build_features(events, k)])[0]
            probs = np.zeros(len(LOCAL_ZONES))
            for cls, p in zip(self.model.classes_, raw):
                probs[int(cls) - 1] = p
        elif self.markov is not None:
            probs = self.markov.predict_proba(hitter_zone)
        else:
            probs = np.full(len(LOCAL_ZONES), 1 / len(LOCAL_ZONES))
        return {opponent_zone(hitter_zone, z): float(p) for z, p in zip(LOCAL_ZONES, probs)}
