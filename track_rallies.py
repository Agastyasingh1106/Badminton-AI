"""
Runs pass 1 of analyze_match (players + shuttle tracking) on a folder of rally videos and saves
the tracks, so train_zone_predictor.py can learn from rallies that have no shuttle labels.

Only videos with court corners (court_corners.json) are tracked; finished videos are skipped,
so the script can be stopped and restarted. Run several shards in parallel to use more cores:

  python track_rallies.py dataset/rallies --shard 0/2
  python track_rallies.py dataset/rallies --shard 1/2

Output: runs/rally_tracks/<video_id>.json  {"info": ..., "shuttle": {frame: [x, y]},
                                           "players": {frame: [[x1, y1, x2, y2], ...]}}
"""

import argparse
import json
import time
from pathlib import Path

from badminton.court import PROJECT_ROOT, CourtMapper, load_corners, video_id_from_path

TRACKS_DIR = PROJECT_ROOT / "runs" / "rally_tracks"
# The benchmark clip shown in the dashboard: never used for training
HELD_OUT = {"00467"}


def save_tracks(path: Path, frames, info):
    data = {
        "info": info,
        "shuttle": {str(i): list(f["shuttle"][0]) for i, f in enumerate(frames) if f["shuttle"] is not None},
        "players": {str(i): [[round(v, 1) for v in box] for _, box, _ in f["players"]]
                    for i, f in enumerate(frames) if f["players"]},
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Track players and shuttle in rally videos.")
    parser.add_argument("folder", type=Path, help="Folder of rally .mp4 files.")
    parser.add_argument("--shard", default="0/1", help="i/n: process every n-th video starting at i.")
    parser.add_argument("--person-model", type=Path, default=PROJECT_ROOT / "yolov8n.pt")
    args = parser.parse_args()

    from analyze_match import track_video

    shard, n_shards = map(int, args.shard.split("/"))
    corners = load_corners()
    TRACKS_DIR.mkdir(parents=True, exist_ok=True)
    videos = [p for p in sorted(args.folder.glob("*.mp4"))
              if video_id_from_path(p) in corners and video_id_from_path(p) not in HELD_OUT]
    videos = videos[shard::n_shards]
    todo = [p for p in videos if not (TRACKS_DIR / f"{video_id_from_path(p)}.json").is_file()]
    print(f"Shard {args.shard}: {len(videos)} calibrated videos, {len(todo)} still to track")

    for i, path in enumerate(todo, 1):
        video = video_id_from_path(path)
        start = time.time()
        frames, info = track_video(path, CourtMapper(corners[video]), args.person_model)
        info["n_frames"] = len(frames)
        save_tracks(TRACKS_DIR / f"{video}.json", frames, info)
        seen = sum(f["shuttle"] is not None for f in frames)
        print(f"[{i}/{len(todo)}] {video}: {len(frames)} frames, shuttle in {seen}, "
              f"{len(frames) / (time.time() - start):.1f} fps", flush=True)


if __name__ == "__main__":
    main()
