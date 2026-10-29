"""
Turns a shuttle trajectory into shot events (hits) with court zones.

Trajectories are resampled to every STRIDE-th frame so live tracking matches the
training labels (which exist for every 4th frame only).
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from badminton.court import CourtMapper, zone_of, side_of

STRIDE = 4
MAX_GAP_SAMPLES = 3          # interpolate across up to 3 missing samples
MIN_HIT_GAP_SAMPLES = 3      # two hits are at least 12 frames (0.4 s) apart
TURN_ANGLE_DEG = 55.0        # direction change that counts as a hit
SPEED_JUMP = 1.8             # speed ratio that counts as a hit
MIN_STEP_PX = 10.0           # ignore jitter of a near-still shuttle (eval_hits.py: F1 0.64 -> 0.67 vs 6 px)
EARLY_SAMPLES = 3            # samples after a hit used for prediction features


def resample(points: Dict[int, Tuple[float, float]], stride: int = STRIDE) -> List[Tuple[int, Optional[np.ndarray]]]:
    """{frame: (x, y)} -> [(frame, xy or None)] every `stride` frames, filling short gaps."""
    if not points:
        return []
    frames = sorted(points)
    start = frames[0] - frames[0] % stride
    samples = [(f, np.array(points[f], dtype=np.float32) if f in points else None)
               for f in range(start, frames[-1] + 1, stride)]
    # Linear interpolation over short gaps
    known = [i for i, (_, p) in enumerate(samples) if p is not None]
    for a, b in zip(known, known[1:]):
        if 1 < b - a <= MAX_GAP_SAMPLES + 1:
            for i in range(a + 1, b):
                t = (i - a) / (b - a)
                samples[i] = (samples[i][0], samples[a][1] * (1 - t) + samples[b][1] * t)
    return samples


def split_segments(samples) -> List[List[Tuple[int, np.ndarray]]]:
    """Continuous runs of known positions."""
    segments, current = [], []
    for frame, p in samples:
        if p is None:
            if len(current) >= 3:
                segments.append(current)
            current = []
        else:
            current.append((frame, p))
    if len(current) >= 3:
        segments.append(current)
    return segments


def _angle(v1, v2) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


def hitter_foot(shuttle_px, boxes) -> Optional[Tuple[float, float]]:
    """Foot point of the player whose box is closest to the shuttle (box grown upwards for the racket)."""
    best, best_d = None, None
    for x1, y1, x2, y2 in boxes:
        reach = 0.6 * (y2 - y1)
        dx = max(x1 - reach / 2 - shuttle_px[0], 0, shuttle_px[0] - x2 - reach / 2)
        dy = max(y1 - reach - shuttle_px[1], 0, shuttle_px[1] - y2)
        d = float(np.hypot(dx, dy))
        if best_d is None or d < best_d:
            best, best_d = ((x1 + x2) / 2, y2), d
    return best


def detect_hits(points: Dict[int, Tuple[float, float]], mapper: CourtMapper, fps: float = 30.0,
                stride: int = STRIDE, players_at: Optional[Callable[[int], list]] = None,
                turn_angle: float = TURN_ANGLE_DEG, speed_jump: float = SPEED_JUMP,
                min_gap_samples: int = MIN_HIT_GAP_SAMPLES, min_step_px: float = MIN_STEP_PX
                ) -> Tuple[List[dict], List[Tuple[int, np.ndarray]]]:
    """
    Finds hits in a shuttle trajectory.
    Returns (events, samples). Each event: frame, px, court position, side, zone, speed and the
    shuttle's first EARLY_SAMPLES moves after the hit (court metres, used by the predictor).
    players_at(frame) -> list of player boxes; when given, the hit zone is the hitting player's
    foot position (reliable) instead of the shuttle dropped to the floor (rough).
    The thresholds default to the module constants (tuned with eval_hits.py).
    """
    samples = resample(points, stride)
    candidates = []
    for seg in split_segments(samples):
        pts = np.array([p for _, p in seg])
        vel = np.diff(pts, axis=0)
        for i in range(1, len(vel)):
            v_in, v_out = vel[i - 1], vel[i]
            s_in, s_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
            if s_out < min_step_px:
                continue
            turn = _angle(v_in, v_out)
            jump = s_out / max(s_in, 1.0)
            if turn >= turn_angle or jump >= speed_jump:
                score = turn / turn_angle + jump / speed_jump
                candidates.append((seg[i][0], seg[i][1], score))
        # The first point of a segment is a hit when the shuttle starts moving there (e.g. a serve)
        if len(vel) and np.linalg.norm(vel[0]) >= min_step_px:
            candidates.append((seg[0][0], seg[0][1], 0.5))

    candidates.sort(key=lambda c: c[0])

    # Keep the strongest candidate within each short window
    hits = []
    for frame, p, score in candidates:
        if hits and frame - hits[-1][0] < min_gap_samples * stride:
            if score > hits[-1][2]:
                hits[-1] = (frame, p, score)
            continue
        hits.append((frame, p, score))

    events = []
    for frame, p, _ in hits:
        foot = hitter_foot(p, players_at(frame)) if players_at is not None else None
        court = mapper.to_court(foot) if foot is not None else mapper.ground_point(p)
        events.append({"frame": int(frame), "px": (float(p[0]), float(p[1])), "court": court,
                       "side": side_of(court), "zone": zone_of(court), "from_player": foot is not None})

    # Shots go back and forth: of two hits in a row on the same side, keep the later one
    alternated = []
    for ev in events:
        if alternated and alternated[-1]["side"] == ev["side"]:
            alternated[-1] = ev
        else:
            alternated.append(ev)
    events = alternated

    sample_index = {f: p for f, p in samples if p is not None}
    for k, ev in enumerate(events):
        ev["early"] = early_moves(ev, sample_index, mapper, stride)
        ev["speed_kmh"] = shot_speed_kmh(ev, sample_index, mapper, fps, stride)
        ev["shot_no"] = k + 1
    return events, samples


def early_moves(event, sample_index, mapper, stride=STRIDE) -> List[Tuple[float, float]]:
    """Image displacement (in metres at the hit point's scale) for the first samples after a hit."""
    scale = mapper.px_per_metre(event["px"])
    origin = np.array(event["px"])
    moves, last = [], (0.0, 0.0)
    for j in range(1, EARLY_SAMPLES + 1):
        p = sample_index.get(event["frame"] + j * stride)
        if p is not None:
            last = tuple(((np.array(p) - origin) / scale).tolist())
        moves.append(last)
    return moves


def shot_speed_kmh(event, sample_index, mapper, fps=30.0, stride=STRIDE) -> float:
    """
    Speed right after the hit: the mean of the first two moves on the court plane (points dropped to
    the floor at hit height), or the smaller one when they differ over 3x (a tracking jump).
    Measuring across the court width alone undercounts shots that travel down the court, which move
    mostly up or down the image. A near-side smash still reads low: moving away from the camera and
    down at once barely moves the shuttle on screen.
    """
    pts = [sample_index.get(event["frame"] + j * stride) for j in range(3)]
    steps = [float(np.hypot(*np.subtract(mapper.ground_point(b), mapper.ground_point(a))))
             for a, b in zip(pts, pts[1:]) if a is not None and b is not None]
    if not steps:
        return 0.0
    metres = min(steps) if max(steps) > 3 * max(min(steps), 1e-3) else float(np.mean(steps))
    return float(min(metres * fps / stride * 3.6, 420.0))


def landing_zone(samples, mapper: CourtMapper) -> Optional[str]:
    """Zone of the last tracked point of the rally (the shuttle is on/near the floor)."""
    known = [p for _, p in samples if p is not None]
    if not known:
        return None
    return zone_of(mapper.ground_point(known[-1], height_m=0.3))


def attach_targets(events: List[dict], samples, mapper: CourtMapper) -> List[dict]:
    """Each shot's target = zone of the next hit (or the landing zone for the last shot)."""
    final = landing_zone(samples, mapper)
    for k, ev in enumerate(events):
        if k + 1 < len(events):
            ev["next_zone"] = events[k + 1]["zone"]
        else:
            ev["next_zone"] = final
        # A shot should go to the other side; drop targets that don't
        if ev["next_zone"] is not None and ev["next_zone"][0] == ev["zone"][0]:
            ev["next_zone"] = None
    return events
