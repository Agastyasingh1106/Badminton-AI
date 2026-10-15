# Badminton AI Dataset

This directory contains the dataset configurations, sample match footage, and instructions for training the shuttlecock detection models.

## Structure

```text
dataset/
├── sample/                  # Benchmark match video clip & frame preview
│   ├── 00467.mp4            # Benchmark test match video (1280x720 @ 30 FPS, 274 frames)
│   └── 00467img.png         # Reference court perspective screenshot
├── data.yaml                # Standard YOLO dataset definition for full training
├── data_curated.yaml        # Curated high-signal training split config
└── README.md
```

## Annotation Format (YOLO)
Each annotated frame corresponds to a `.txt` label file in YOLO normalized coordinates:
`<class_id> <center_x> <center_y> <width> <height>`

- `class_id`: `0` (shuttlecock)
- Coordinates are normalized between `0.0` and `1.0`.

## Training
To train the shuttlecock detector using these dataset configurations:
```bash
python train_shuttlecock.py --epochs 15 --batch 16 --subset 400
```
