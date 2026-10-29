# Badminton AI 🏸

Advanced Computer Vision system for automated badminton match analysis, player tracking with persistent identities, and high-accuracy aerial shuttlecock trajectory detection.

![Badminton AI Overview](sample_dataset/00467img.png)

---

## 🌟 Key Features

1. **High-Visibility Player Tracking (Player 1 & 2)**
   - Bold red bounding boxes (`#FF0000`) with solid colored header badges and white text for high contrast against green court surfaces.
   - **`TwoPlayerTracker`**: Spatial court-partition algorithm that stably assigns **Player 1** to the far court (top half) and **Player 2** to the near court (bottom half). Prevents ID flipping or player loss during fast smashes and lunges.

2. **Hybrid Shuttlecock Detector & Flight Tracker (`detect_shuttle.py`)**
   - **Dual-Stream Detection**: Fuses YOLOv8 deep learning proposals with 3-frame symmetric temporal motion differencing ($|F_t - F_{t-1}| \land |F_{t+1} - F_t|$) and white reflectance masking.
   - **3D Flight Airspace Polygon**: Extends tracking vertically above the court floor to capture high clears, lifts, and steep smashes.
   - **Physics-Guided Trajectory Predictor**: Employs velocity smoothing, aerodynamic drag, and downward gravity modeling to bridge occlusions across the net.
   - **Telemetry HUD**: Displays real-time estimated flight velocity in **km/h**, tracking status, and broadcast-style fading trajectory trails.

3. **Dedicated Training Pipeline (`train_shuttlecock.py`)**
   - Curated dataset loader with balanced positive (visible shuttlecock) and negative frames.
   - Tuned small-object hyperparameters (`box=8.5`, `mosaic=0.5`, `scale=0.3`, `close_mosaic=3`).
   - Note: the dataset's `*_predict` videos (rallies 351–400) have TrackNet's red dot drawn on the shuttle, so the original model's 99.4% mAP50 on them mostly measures dot detection. Tracking on clean footage is measured with `eval_shuttle_tracking.py` against hand labels instead.

---

## 📁 Repository Structure

```text
Badminton AI/
├── output_videos/               # Processed demonstration videos
│   ├── 00467_players_and_shuttle.mp4   # Full match video with players & shuttle
│   └── 00467_shuttle_tracked.mp4       # Dedicated shuttle flight trail & speed HUD
├── sample_dataset/              # Sample broadcast video & preview
│   ├── 00467.mp4                # Match rally sample clip
│   └── 00467img.png             # Frame snapshot
├── training/                    # Dataset & notebook training utilities
│   ├── shuttlecock_yolo_dataset/# YOLO format images, labels & split configs
│   └── shuttlecock_detector_training.ipynb
├── runs/                        # Trained model weights & inference outputs
│   └── detect/shuttlecock/weights/best.pt
├── detect_shuttle.py            # Dedicated shuttlecock detector & speed tracker
├── yolo_inference.py            # Full player & shuttlecock match inference
├── train_shuttlecock.py         # Shuttlecock YOLO fine-tuning script
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/Badminton-AI.git
cd "Badminton AI"
pip install -r requirements.txt
```
On Windows `pip` installs the CPU-only PyTorch. With an NVIDIA GPU, swap in the CUDA build (about 2.6× faster tracking on an RTX 4050):
```bash
pip install --force-reinstall --no-deps torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
```

### 2. Run Full Match Detection (Players + Shuttlecock)
```bash
python yolo_inference.py sample_dataset/00467.mp4 --person-model yolov8n.pt --output output_videos/match_annotated.mp4
```

### 3. Run Dedicated Shuttlecock Trajectory & Speed Tracker
```bash
python detect_shuttle.py sample_dataset/00467.mp4 --output output_videos/shuttle_trajectory.mp4
```

### 4. Train / Fine-Tune Shuttlecock Model
```bash
python train_shuttlecock.py --epochs 15 --batch 16 --subset 400
```

---

## 🗺️ Court Zones, Next-Zone Prediction & Dashboard

- **Court zones** (`badminton/court.py`): the 4 court corners give a homography from pixels to court metres. Each half is split into 6 zones (F1–F6 far side, N1–N6 near side; 1–3 next to the net, 4–6 near the baseline). Airborne hit points are dropped onto the floor assuming a ~2 m contact height.
- **Shot events** (`badminton/events.py`): hits are found where the shuttle's path turns sharply or speeds up, on a trajectory sampled every 4th frame (the same rate as the training labels). Each shot gets its player, zone, speed and the zone it was sent to.
- **Next-zone predictor** (`badminton/predictor.py`, `train_zone_predictor.py`): a Random Forest uses the hit zone, previous zone, shot speed and the shuttle's first 0.4 s of flight to predict which of the opponent's 6 zones the shuttle goes to. It is trained on the 45 broadcast labelled rallies plus 320 unlabelled broadcast rallies tracked by our own pipeline with TrackNetV3 (3,413 shots from 362 rallies with a known next zone; practice-hall footage is left out) and scored on unseen rallies against simple baselines: **54.4% top-1 / 81.7% top-2**, vs 44.9% / 72.6% with the earlier YOLO shuttle tracks, 45.0% / 74.9% when trained on the labelled rallies alone and 39.6% for always guessing the most common zone.
- **Unlabelled rallies** (`auto_calibrate_court.py`, `track_rallies.py`): court corners are fitted automatically by matching the standard court line layout to the white lines (median error 4.7 px against hand clicks), then players and shuttle are tracked with the match pipeline. Only broadcast rallies are tracked (`--ids runs/tracknet/broadcast_ids.txt`); the dark practice-hall footage is not used.
- **Shuttle tracking** (`badminton/tracknet.py`): [TrackNetV3](https://github.com/qaz812345/TrackNetV3) (MIT, vendored in `third_party/tracknetv3/`) finds the shuttle from 8-frame sequences and fills short gaps. On hand-labelled clean broadcast frames it is far more accurate than our YOLO + motion tracker (`eval_shuttle_tracking.py`), (within 10 px on 87% of 737 visible frames from 16 rallies, median error 2.8 px, vs 36% and 61 px), so the pipeline uses it whenever its weights are installed and falls back to YOLO otherwise.
- **Hand labels** (`label_hits.py`): one pass per rally marks each hit (frame, hitter side, shot type) and confirms or corrects the shuttle position on ~40 frames, saved to `hit_labels.json`. 15 rallies are labelled (200 hits). `eval_hits.py` scores hit detection against them: F1 0.67 within 4 frames (precision 68%, recall 66%), hitter side right on 92%.
- **Shot types** (`badminton/shots.py`, `badminton/shot_types.py`): YOLOv8-pose on the hitting player around each hit, then a pose classifier. Trained on the group member's fixed-camera dataset (`train_shot_classifier.py`, 60% top-1 on unseen sessions there), it does not carry over to broadcast footage, so `finetune_shot_classifier.py` adapts it to our labelled hits as 6 shot groups (serve, clear/lift, smash, drop/net, drive/push, defensive) and saves a model only if it beats always guessing the most common group on unseen rallies. Fine-tuning the whole network does: 46% on unseen rallies (leave-one-rally-out over 195 hits) vs 33% for the most common group and 19% for the original model. Without a saved model, no shot group is shown.
- **Match pipeline** (`analyze_match.py`): players + rackets + shuttle tracking, zones, shots, shot groups and live predictions → annotated H.264 video, `events.csv` and `summary.json` in `runs/analysis/<video_id>/`.
- **Dashboard** (`app.py`): Streamlit app with the annotated video, shot timeline, zone heatmaps, prediction viewer, model accuracy and player movement.

```bash
# 1. Click the 4 court corners for each video (saved to court_corners.json)
python calibrate_court.py

#    or fit them automatically for a folder of rally videos (check runs/court_fit/previews)
python auto_calibrate_court.py dataset/rallies

# 2. Track the unlabelled rallies (GPU: ~25 frames/s per process; run shards in parallel),
#    then train the next-zone predictor (a few minutes)
python track_rallies.py dataset/rallies --ids runs/tracknet/broadcast_ids.txt --shard 0/2
python track_rallies.py dataset/rallies --ids runs/tracknet/broadcast_ids.txt --shard 1/2
python train_zone_predictor.py

# TrackNetV3 weights (142 MB, not in git): download TrackNetV3_ckpts.zip from the link in
# https://github.com/qaz812345/TrackNetV3 and put TrackNet_best.pt and InpaintNet_best.pt in models/tracknetv3/

# Hand labels, then score shuttle tracking, hit detection and adapt the shot classifier
python label_hits.py
python eval_shuttle_tracking.py
python eval_hits.py
python train_shot_classifier.py          # on dataset/shot_keypoints.csv (group member's pose data)
python finetune_shot_classifier.py       # broadcast shot groups from hit_labels.json

# 3. Analyse a rally from the command line
python analyze_match.py sample_dataset/00467.mp4

# 4. Or open the dashboard
streamlit run app.py
```

---

## 📊 Performance & Results

- **Player Tracking**: 100% persistent Player 1 & 2 identity retention across rallies.
- **Shuttlecock Tracking**: TrackNetV3 is within 10 px of the hand-labelled shuttle on 87% of visible frames (median error 2.8 px) across 16 broadcast rallies, vs 36% for the earlier YOLO + motion tracker (`runs/shuttle_eval/benchmark.json`).
- **Speed Estimation**: shot speed from the shuttle's first two 4-frame moves on the court plane. From a single camera it is an estimate: clears and lifts read ~130 km/h, drives ~30 km/h, and near-side smashes read low because moving away from the camera and down at once barely moves the shuttle on screen.

---

## 📜 License
MIT License
