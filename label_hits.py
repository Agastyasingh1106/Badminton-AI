"""
Hand-label hits, shot types and shuttle positions in broadcast rally videos, one pass per rally.

1. Hits pass: step to the frame where the racket meets the shuttle, press f (far player) or
   n (near player), then pick the shot type with 0-9 / a-h (u = unsure, ESC = cancel).
   Keys:  d / RIGHT = next frame    a / LEFT = previous frame    c / z = +10 / -10 frames
          SPACE = play / pause      x = delete the hit nearest this frame    u = undo
          ENTER = hits done, go to the shuttle pass    q = save and quit
2. Shuttle pass: ~40 evenly spaced frames plus every hit frame. The ring is a guess from TrackNetV3
   (runs/tracknet/<id>_ball.csv) when that exists, else from our tracker (runs/rally_tracks/<id>.json).
   Keys:  ENTER = accept the ring (or the clicked point)    click = correct it: click near the
          shuttle -> 4x zoom -> click it exactly    n = shuttle not visible    r = redo the click
          f = hold to flick to the next video frame    b = back one frame    h = back to the hits pass
          q = save and quit
Labels go to hit_labels.json (saved on every change); a partly labelled video resumes where it stopped.
With no video argument it works through a fixed list of broadcast rallies.
"""

import argparse
import copy
import csv
import json
from pathlib import Path

import cv2

from badminton.court import video_id_from_path

PROJECT_ROOT = Path(__file__).resolve().parent
RALLIES_DIR = PROJECT_ROOT / "dataset" / "rallies"
TRACKS_DIR = PROJECT_ROOT / "runs" / "rally_tracks"
TRACKNET_DIR = PROJECT_ROOT / "runs" / "tracknet"
LABELS_FILE = PROJECT_ROOT / "hit_labels.json"
WINDOW = "Hit labelling"
PANEL_W = 330
SHUTTLE_SAMPLES = 40
ZOOM = 4

SHOT_TYPES = [
    "Short Serve", "Cross Court Flight", "Lift", "Tap Smash", "Block", "Drop Shot",
    "Push Shot", "Transitional Slice", "Cut", "Rush Shot", "Defensive Clear", "Defensive Drive",
    "Clear", "Long Serve", "Smash", "Flat Shot", "Rear Court Flat Drive", "Short Flat Shot",
]
TYPE_KEYS = "0123456789abcdefgh"   # key for shot type id i is TYPE_KEYS[i]

# Every 21st broadcast rally of 00401-00800 (starting at the 11th), skipping 00467. Broadcast =
# mean V of HSV >= 70 over the near-court floor frame[450:650, 300:980] of frame 5; the darker
# videos are practice-hall footage. Same as every 21st id (from the 11th) of
# runs/tracknet/broadcast_ids.txt, so TrackNet proposals exist for all of them. All also have a
# tracker file in runs/rally_tracks (00467 is the only rally without one). 15 rallies, ~3.5 minutes.
DEFAULT_RALLIES = [
    "00412", "00438", "00475", "00498", "00529", "00551", "00580", "00608",
    "00632", "00658", "00684", "00711", "00736", "00761", "00783",
]

KEY_LEFT, KEY_RIGHT = 2424832, 2555904   # cv2.waitKeyEx codes on Windows
KEY_LEFT_GTK, KEY_RIGHT_GTK = 65361, 65363
FONT, WHITE, YELLOW, RED = cv2.FONT_HERSHEY_SIMPLEX, (255, 255, 255), (0, 255, 255), (0, 0, 255)


def read_frames(video: Path):
    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def load_labels(path: Path):
    labels = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    for entry in labels.values():
        # Before TrackNet proposals, "accepted" always meant our tracker's guess
        sources = entry.get("shuttle_source", {})
        for frame, src in sources.items():
            if src == "accepted":
                sources[frame] = "accepted:ours"
    return labels


def save_labels(labels, path: Path):
    path.write_text(json.dumps(labels, indent=1), encoding="utf-8")


def load_proposals(video_id):
    """Shuttle guesses ({frame: (x, y)}, source name); frames where nothing was found are absent.
    TrackNetV3's output is preferred, our own tracker is the fallback."""
    path = TRACKNET_DIR / f"{video_id}_ball.csv"
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as f:
            rows = csv.DictReader(f)
            return {int(r["Frame"]): (float(r["X"]), float(r["Y"])) for r in rows
                    if int(float(r["Visibility"])) != 0}, "tracknet"
    path = TRACKS_DIR / f"{video_id}.json"
    if not path.is_file():
        return {}, "none"
    shuttle = json.loads(path.read_text(encoding="utf-8")).get("shuttle", {})
    return {int(f): tuple(p) for f, p in shuttle.items() if p is not None}, "ours"


def shuttle_frames(n_frames, hits):
    step = max(1, n_frames // SHUTTLE_SAMPLES)
    sampled = range(step // 2, n_frames, step)
    return sorted(set(sampled) | {hit["frame"] for hit in hits})


def type_name(shot_type):
    return "unsure" if shot_type is None else f"{shot_type} {SHOT_TYPES[shot_type]}"


def with_panel(frame, bar):
    h, w = frame.shape[:2]
    view = cv2.copyMakeBorder(frame, 0, 0, 0, PANEL_W, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.rectangle(view, (0, 0), (w, 30), (0, 0, 0), -1)
    text(view, bar, 8, 20)
    return view


def text(view, s, x, y, color=WHITE, scale=0.5):
    cv2.putText(view, s, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def draw_hits(frame, index, total, hits, playing, pending_side, video_id):
    h, w = frame.shape[:2]
    status = "PLAY" if playing else "pause"
    view = with_panel(frame, f"{video_id} HITS  frame {index}/{total - 1}  [{status}]   "
                             "f/n=hit far/near  a/d=-1/+1  z/c=-10/+10  SPACE=play  x=delete  u=undo  ENTER=done  q=quit")

    this_hit = next((hit for hit in hits if hit["frame"] == index), None)
    if this_hit is not None:
        cv2.rectangle(view, (0, 0), (w - 1, h - 1), RED, 6)
        text(view, f"HIT {this_hit['side']}: {type_name(this_hit['shot_type'])}", w + 10, h - 20, RED, 0.55)

    # Side panel: the hits so far, the one nearest the current frame highlighted
    x0 = w + 10
    text(view, f"Hits ({len(hits)})", x0, 55, YELLOW, 0.6)
    nearest = min(hits, key=lambda hit: abs(hit["frame"] - index)) if hits else None
    rows = sorted(hits, key=lambda hit: hit["frame"])
    if len(rows) > 30:   # keep the rows around the current frame
        k = rows.index(nearest)
        rows = rows[max(0, min(k - 15, len(rows) - 30)):][:30]
    for i, hit in enumerate(rows):
        color = YELLOW if hit is nearest else WHITE
        text(view, f"{hit['frame']:5d}  {hit['side']:4s}  {type_name(hit['shot_type'])}", x0, 85 + 21 * i, color, 0.45)

    if pending_side is not None:
        # Shot-type menu over the frame
        top = 70
        cv2.rectangle(view, (20, top), (430, top + 30 + 22 * (len(SHOT_TYPES) + 2)), (0, 0, 0), -1)
        text(view, f"{pending_side.upper()} hit at frame {index}: shot type?", 30, top + 22, YELLOW, 0.55)
        for i, name in enumerate(SHOT_TYPES):
            text(view, f"{TYPE_KEYS[i]}  {name}", 30, top + 48 + 22 * i)
        text(view, "u  unsure      ESC  cancel", 30, top + 48 + 22 * len(SHOT_TYPES), YELLOW)
    return view


def hits_pass(frames, entry, save, video_id):
    """Returns "shuttle" when the hits are done, "quit" when the user quit."""
    hits = entry["hits"]
    index = min(max((hit["frame"] for hit in hits), default=-1) + 1, len(frames) - 1)
    playing, pending_side = False, None
    history = []   # snapshots of the hit list for undo

    def change(new_hits):
        history.append(copy.deepcopy(hits))
        hits[:] = sorted(new_hits, key=lambda hit: hit["frame"])
        save()

    while True:
        cv2.imshow(WINDOW, draw_hits(frames[index], index, len(frames), hits, playing, pending_side, video_id))
        key = cv2.waitKeyEx(33 if playing else 0)
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return "quit"   # window closed with the mouse
        char = chr(key) if 0 <= key < 256 else ""

        if pending_side is not None:
            if key == 27:
                pending_side = None
            elif char == "u" or (char and char in TYPE_KEYS):
                shot_type = None if char == "u" else TYPE_KEYS.index(char)
                others = [hit for hit in hits if hit["frame"] != index]
                change(others + [{"frame": index, "side": pending_side, "shot_type": shot_type}])
                pending_side = None
            continue

        if key == -1:   # timeout while playing
            if index + 1 < len(frames):
                index += 1
            else:
                playing = False
        elif char in ("f", "n"):
            playing, pending_side = False, "far" if char == "f" else "near"
        elif char == "d" or key in (KEY_RIGHT, KEY_RIGHT_GTK):
            playing, index = False, min(index + 1, len(frames) - 1)
        elif char == "a" or key in (KEY_LEFT, KEY_LEFT_GTK):
            playing, index = False, max(index - 1, 0)
        elif char == "c":
            playing, index = False, min(index + 10, len(frames) - 1)
        elif char == "z":
            playing, index = False, max(index - 10, 0)
        elif char == " ":
            playing = not playing
        elif char == "x" and hits:
            nearest = min(hits, key=lambda hit: abs(hit["frame"] - index))
            change([hit for hit in hits if hit is not nearest])
        elif char == "u" and history:
            hits[:] = history.pop()
            save()
        elif key in (10, 13):
            entry["hits_done"] = True
            save()
            print(f"{video_id}: {len(hits)} hits.")
            return "shuttle"
        elif char == "q":
            save()
            return "quit"


def shuttle_pass(frames, entry, save, video_id):
    """Returns "done" when every target frame has a shuttle label, "hits" to go back, "quit"."""
    shuttle = entry.setdefault("shuttle", {})
    source = entry.setdefault("shuttle_source", {})
    proposals, proposal_source = load_proposals(video_id)
    targets = shuttle_frames(len(frames), entry["hits"])
    hit_frames = {hit["frame"] for hit in entry["hits"]}
    h, w = frames[0].shape[:2]
    state = {"zoom_origin": None, "point": None}

    def on_mouse(event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN or x >= w:
            return
        if state["zoom_origin"] is None:
            # First click: zoom into a crop centred on it (stored as the crop's top-left corner)
            zw, zh = w // ZOOM, h // ZOOM
            state["zoom_origin"] = (int(min(max(x - zw // 2, 0), w - zw)), int(min(max(y - zh // 2, 0), h - zh)))
        else:
            ox, oy = state["zoom_origin"]
            state["point"] = (ox + x / ZOOM, oy + y / ZOOM)
            state["zoom_origin"] = None

    cv2.setMouseCallback(WINDOW, on_mouse)
    k = 0
    while k < len(targets):
        index = targets[k]
        if str(index) in shuttle:
            k += 1
            continue
        frame = frames[index]
        proposal = proposals.get(index)
        state.update(zoom_origin=None, point=None)
        while True:
            if state["zoom_origin"] is not None:
                ox, oy = state["zoom_origin"]
                zw, zh = w // ZOOM, h // ZOOM
                view = cv2.resize(frame[oy:oy + zh, ox:ox + zw], (w, h), interpolation=cv2.INTER_CUBIC)
                hint = "ZOOM: click the shuttle exactly"
                if proposal is not None:
                    p = (int((proposal[0] - ox) * ZOOM), int((proposal[1] - oy) * ZOOM))
                    cv2.circle(view, p, 12 * ZOOM, (255, 255, 0), 2)
            else:
                view = frame.copy()
                if proposal is not None:
                    cv2.circle(view, tuple(map(int, proposal)), 14, (255, 255, 0), 2)
                if state["point"] is not None:
                    p = tuple(map(int, state["point"]))
                    cv2.circle(view, p, 12, YELLOW, 2)
                    cv2.circle(view, p, 2, RED, -1)
                    hint = "ENTER = save click   r = redo"
                elif proposal is not None:
                    hint = "ENTER = ring is right   click = correct it   n = not visible"
                else:
                    hint = "tracker found nothing: click the shuttle or n = not visible"
            done_count = sum(1 for t in targets if str(t) in shuttle)
            view = with_panel(view, f"{video_id} SHUTTLE [{done_count}/{len(targets)}] frame {index}: {hint}")
            x0 = w + 10
            text(view, "Shuttle pass", x0, 55, YELLOW, 0.6)
            for i, line in enumerate([f"ring = {proposal_source} guess", "ENTER  accept", "click  correct (zoom)",
                                      "n  not visible", "r  redo click", "f  flick next frame",
                                      "b  back one", "h  back to hits", "q  save and quit"]):
                text(view, line, x0, 85 + 21 * i, WHITE, 0.45)
            if index in hit_frames:
                text(view, "(hit frame)", x0, h - 20, RED, 0.55)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKeyEx(30)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return "quit"
            char = chr(key) if 0 <= key < 256 else ""
            if char == "f" and index + 1 < len(frames) and state["zoom_origin"] is None:
                cv2.imshow(WINDOW, with_panel(frames[index + 1], f"{video_id} frame {index + 1} (flick)"))
                cv2.waitKey(250)
            elif char == "r":
                state.update(zoom_origin=None, point=None)
            elif char == "n":
                shuttle[str(index)], source[str(index)] = None, "none"
                break
            elif key in (10, 13) and state["zoom_origin"] is None:
                if state["point"] is not None:
                    shuttle[str(index)] = [round(v, 1) for v in state["point"]]
                    source[str(index)] = "clicked"
                    break
                if proposal is not None:
                    shuttle[str(index)], source[str(index)] = list(proposal), f"accepted:{proposal_source}"
                    break
            elif char == "b":
                previous = [t for t in targets[:k] if str(t) in shuttle]
                if previous:
                    shuttle.pop(str(previous[-1]))
                    source.pop(str(previous[-1]), None)
                    save()
                    k = targets.index(previous[-1]) - 1
                    break
            elif char == "h":
                entry["hits_done"] = False
                save()
                return "hits"
            elif char == "q":
                save()
                return "quit"
        save()
        k += 1
    return "done"


def label_video(video: Path, labels, labels_file: Path):
    """Label one video (hits, then shuttle). Returns False if the user quit."""
    video_id = video_id_from_path(video)
    frames = read_frames(video)
    if not frames:
        print(f"Could not read {video}")
        return True
    entry = labels.setdefault(video_id, {"hits": [], "done": False})
    print(f"{video_id}: {len(frames)} frames, {len(entry['hits'])} hits and "
          f"{len(entry.get('shuttle', {}))} shuttle points already labelled.")

    def save():
        save_labels(labels, labels_file)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280 + PANEL_W, 720)
    while True:
        if not entry.get("hits_done") and hits_pass(frames, entry, save, video_id) == "quit":
            return False
        result = shuttle_pass(frames, entry, save, video_id)
        if result == "quit":
            return False
        if result == "done":
            entry["done"] = True
            save()
            print(f"{video_id}: done, {len(entry['shuttle'])} shuttle points.")
            return True


def resolve_video(arg: str) -> Path:
    path = Path(arg)
    return path if path.suffix else RALLIES_DIR / f"{arg}.mp4"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand-label hits, shot types and shuttle positions in rally videos.")
    parser.add_argument("video", nargs="?", help="Video path or rally id (e.g. 00412). Default: the fixed broadcast list.")
    parser.add_argument("--labels", type=Path, default=LABELS_FILE, help="Output JSON (default hit_labels.json).")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    if args.video:
        label_video(resolve_video(args.video), labels, args.labels)
    else:
        for video_id in DEFAULT_RALLIES:
            entry = labels.get(video_id, {})
            state = "done" if entry.get("done") else \
                f"{len(entry.get('hits', []))} hits, {len(entry.get('shuttle', {}))} shuttle points"
            print(f"  {video_id}  {state}")
        todo = [v for v in DEFAULT_RALLIES if not labels.get(v, {}).get("done")]
        print(f"{len(DEFAULT_RALLIES) - len(todo)}/{len(DEFAULT_RALLIES)} rallies done.")
        for video_id in todo:
            if not label_video(RALLIES_DIR / f"{video_id}.mp4", labels, args.labels):
                break
    cv2.destroyAllWindows()
    print(f"Labels saved to {args.labels}")
