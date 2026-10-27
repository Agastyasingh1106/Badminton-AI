"""
Click the shuttle in frames of a video to make extra YOLO training labels.

For each frame: click near the shuttle -> a 4x zoom opens -> click the shuttle exactly.
Keys:  ENTER = save point    n = shuttle not visible (saved as an empty label)
       r = redo this frame   f = hold to flick to the next video frame (spot the moving shuttle)
       b = back one frame    q = save and quit
Every 4th labelled frame goes to the val split so the retrained model can be checked on frames it never saw.
"""

import argparse
import json
from pathlib import Path

import cv2

from badminton.court import video_id_from_path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "training" / "shuttlecock_yolo_dataset"
DEFAULT_VIDEO = PROJECT_ROOT / "sample_dataset" / "00467.mp4"
BOX_PX = 20          # same box size as the existing labels
ZOOM = 4
WINDOW = "Shuttle labelling"


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


def write_label(video_id, index, frame, point, split):
    name = f"{video_id}_predict_f{index:06d}"
    img_path = DATASET_DIR / "images" / split / f"{name}.jpg"
    lbl_path = DATASET_DIR / "labels" / split / f"{name}.txt"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_path), frame)
    h, w = frame.shape[:2]
    text = "" if point is None else f"0 {point[0] / w:.6f} {point[1] / h:.6f} {BOX_PX / w:.6f} {BOX_PX / h:.6f}\n"
    lbl_path.write_text(text, encoding="utf-8")


def label(video: Path, count: int):
    video_id = video_id_from_path(video)
    frames = read_frames(video)
    step = max(1, len(frames) // count)
    targets = list(range(step // 2, len(frames), step))[:count]
    progress_file = PROJECT_ROOT / f"shuttle_labels_{video_id}.json"
    done = json.loads(progress_file.read_text()) if progress_file.is_file() else {}
    print(f"{len(frames)} frames, labelling {len(targets)} of them ({len(done)} already done).")

    state = {"zoom_origin": None, "point": None}

    def on_mouse(event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state["zoom_origin"] is None:
            # First click: zoom into a crop centred on it (stored as the crop's top-left corner)
            w, h = state["size"]
            zw, zh = w // ZOOM, h // ZOOM
            state["zoom_origin"] = (int(min(max(x - zw // 2, 0), w - zw)), int(min(max(y - zh // 2, 0), h - zh)))
        else:
            ox, oy = state["zoom_origin"]
            state["point"] = (ox + x / ZOOM, oy + y / ZOOM)
            state["zoom_origin"] = None

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    cv2.setMouseCallback(WINDOW, on_mouse)

    k = 0
    while k < len(targets):
        index = targets[k]
        if str(index) in done:
            k += 1
            continue
        frame = frames[index]
        h, w = frame.shape[:2]
        state.update(zoom_origin=None, point=None, size=(w, h))
        while True:
            if state["zoom_origin"] is not None:
                ox, oy = state["zoom_origin"]
                zw, zh = w // ZOOM, h // ZOOM
                view = cv2.resize(frame[oy:oy + zh, ox:ox + zw], (w, h), interpolation=cv2.INTER_CUBIC)
                hint = "ZOOM: click the shuttle exactly"
            else:
                view = frame.copy()
                if state["point"] is not None:
                    p = tuple(map(int, state["point"]))
                    cv2.circle(view, p, 12, (0, 255, 255), 2)
                    cv2.circle(view, p, 2, (0, 0, 255), -1)
                    hint = "ENTER = save   r = redo"
                else:
                    hint = "click near the shuttle (n = not visible, f = flick to next frame)"
            bar = f"[{k + 1}/{len(targets)}] frame {index}: {hint}   b=back q=quit"
            cv2.rectangle(view, (0, 0), (w, 34), (0, 0, 0), -1)
            cv2.putText(view, bar, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("f") and index + 1 < len(frames) and state["zoom_origin"] is None:
                cv2.imshow(WINDOW, frames[index + 1])
                cv2.waitKey(250)
            elif key == ord("r"):
                state.update(zoom_origin=None, point=None)
            elif key == ord("n"):
                done[str(index)] = None
                break
            elif key in (10, 13) and state["point"] is not None:
                done[str(index)] = list(state["point"])
                break
            elif key == ord("b"):
                k = max(k - 1, 0)
                done.pop(str(targets[k]), None)
                k -= 1
                break
            elif key == ord("q"):
                k = len(targets)
                break
        progress_file.write_text(json.dumps(done, indent=1), encoding="utf-8")
        k += 1

    cv2.destroyAllWindows()

    # Write YOLO images + labels; every 4th labelled frame is held out for validation
    for n, index in enumerate(sorted(int(i) for i in done)):
        split = "val" if n % 4 == 3 else "train"
        point = done[str(index)]
        write_label(video_id, index, frames[index], point, split)
    visible = sum(1 for p in done.values() if p is not None)
    print(f"Saved {len(done)} labels ({visible} with a visible shuttle) for video {video_id}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Click the shuttle to create training labels.")
    parser.add_argument("video", nargs="?", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--count", type=int, default=40, help="Number of frames to label (spread across the video).")
    args = parser.parse_args()
    label(args.video, args.count)
