"""
Shuttlecock YOLO training script for badminton match videos.
Optimized for small-object detection with transfer learning from YOLOv8n.
"""

import argparse
import random
from pathlib import Path

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "training" / "shuttlecock_yolo_dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "detect" / "shuttlecock"


def prepare_curated_split(
    dataset_dir: Path,
    subset_size: int = 450,
    negative_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    """Prepares a balanced, high-signal subset of annotated images for fast and accurate training."""
    train_img_dir = dataset_dir / "images" / "train"
    train_lbl_dir = dataset_dir / "labels" / "train"
    val_img_dir = dataset_dir / "images" / "val"
    val_lbl_dir = dataset_dir / "labels" / "val"

    if not train_img_dir.is_dir() or not train_lbl_dir.is_dir():
        raise FileNotFoundError(f"Training dataset not found in {dataset_dir}")

    positives, negatives = [], []
    for lbl_path in sorted(train_lbl_dir.glob("*.txt")):
        img_path = train_img_dir / f"{lbl_path.stem}.jpg"
        if not img_path.is_file():
            continue
        content = lbl_path.read_text(encoding="utf-8").strip()
        if content:
            positives.append(img_path.as_posix())
        else:
            negatives.append(img_path.as_posix())

    val_positives = []
    for lbl_path in sorted(val_lbl_dir.glob("*.txt")):
        img_path = val_img_dir / f"{lbl_path.stem}.jpg"
        if not img_path.is_file():
            continue
        if lbl_path.read_text(encoding="utf-8").strip():
            val_positives.append(img_path.as_posix())

    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    rng.shuffle(val_positives)

    selected_pos = positives[:subset_size] if subset_size > 0 else positives
    num_neg = int(len(selected_pos) * negative_ratio)
    selected_neg = negatives[:num_neg]

    train_list = selected_pos + selected_neg
    rng.shuffle(train_list)

    val_list = val_positives[: min(150, len(val_positives))] if subset_size > 0 else val_positives

    train_txt = dataset_dir / "train_curated.txt"
    val_txt = dataset_dir / "val_curated.txt"

    train_txt.write_text("\n".join(train_list), encoding="utf-8")
    val_txt.write_text("\n".join(val_list), encoding="utf-8")

    yaml_config = dataset_dir / "data_curated.yaml"
    config_dict = {
        "path": dataset_dir.as_posix(),
        "train": train_txt.as_posix(),
        "val": val_txt.as_posix(),
        "names": {0: "shuttlecock"},
    }
    yaml_config.write_text(yaml.safe_dump(config_dict, sort_keys=False), encoding="utf-8")

    print(f"Prepared curated dataset:")
    print(f"  - Positive train samples: {len(selected_pos)}")
    print(f"  - Negative train samples: {len(selected_neg)}")
    print(f"  - Validation samples:     {len(val_list)}")
    print(f"  - Config file:            {yaml_config}")
    return yaml_config


def train_shuttlecock(
    epochs: int = 15,
    batch_size: int = 16,
    imgsz: int = 640,
    subset_size: int = 400,
    base_model: str = "yolov8n.pt",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
):
    """Fine-tunes YOLO for shuttlecock detection with small-object hyperparameter tuning."""
    base_model_path = PROJECT_ROOT / base_model
    if not base_model_path.is_file():
        # Fallback to model name if not in root
        base_model_path = Path(base_model)

    print(f"\n==========================================")
    print(f"Starting Shuttlecock Detector Training")
    print(f"Base model:  {base_model_path}")
    print(f"Target dir:  {output_dir}")
    print(f"Epochs:      {epochs}")
    print(f"Batch size:  {batch_size}")
    print(f"Image size:  {imgsz}")
    print(f"Subset size: {subset_size if subset_size > 0 else 'Full dataset'}")
    print(f"==========================================\n")

    if subset_size > 0:
        data_yaml = prepare_curated_split(DATASET_DIR, subset_size=subset_size)
    else:
        data_yaml = DATASET_DIR / "data.yaml"

    model = YOLO(str(base_model_path))

    # Hyperparameters tuned specifically for tiny, fast-moving shuttlecock:
    # 1. Higher box weight (box=8.5) forces accurate bounding box regression on small patches.
    # 2. Moderate mosaic (mosaic=0.5) to avoid excessive downscaling of tiny objects.
    # 3. scale=0.3 preserves minimum visible pixel size of the shuttlecock.
    # 4. close_mosaic=3 disables mosaic near end of training for realistic single-image refinement.
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device="cpu",
        workers=0,
        box=8.5,
        cls=1.2,
        mosaic=0.5,
        mixup=0.0,
        scale=0.3,
        degrees=0.0,
        close_mosaic=3,
        patience=8,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        verbose=True,
    )

    best_weights = output_dir / "weights" / "best.pt"
    if best_weights.is_file():
        print(f"\nTraining completed successfully!")
        print(f"Best model weights saved to: {best_weights}")
    else:
        print(f"\nTraining finished. Check outputs at: {output_dir}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 for shuttlecock detection.")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs (default: 12).")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution for training (default: 640).")
    parser.add_argument(
        "--subset",
        type=int,
        default=400,
        help="Number of positive images to train on for fast CPU convergence (default: 400, use 0 for full).",
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights (default: yolov8n.pt).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save runs and weights.",
    )
    args = parser.parse_args()

    train_shuttlecock(
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        subset_size=args.subset,
        base_model=args.model,
        output_dir=args.output_dir,
    )
