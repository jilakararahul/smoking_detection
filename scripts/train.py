"""
YOLOv11 Training Script — Smoking Detection
============================================
Classes:
    0: Cigarette
    1: smoking  (smoke + vape + smoking behaviour)
    2: cigarette_like_object  (pens, pencils, straws)

Run on GPU (Colab / local with CUDA):
    python train.py

Run quick test on CPU (sanity check, 1 epoch):
    python train.py --test
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch

DATA_YAML = str(Path(__file__).parent / "data" / "merged_dataset" / "data.yaml")
RUNS_DIR  = str(Path(__file__).parent / "runs")

def main(test_mode: bool = False):
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {'GPU (CUDA)' if device == '0' else 'CPU — consider using Colab for full training'}")
    print(f"Data: {DATA_YAML}\n")

    # Use nano for test/CPU, small for real GPU training
    model_weights = "yolo11n.pt" if (test_mode or device == "cpu") else "yolo11s.pt"
    model = YOLO(model_weights)

    train_args = dict(
        data      = DATA_YAML,
        epochs    = 2 if test_mode else 100,
        imgsz     = 640,
        batch     = 4 if device == "cpu" else 16,
        device    = device,
        patience  = 20,           # early stopping if no improvement for 20 epochs
        optimizer = "AdamW",
        lr0       = 0.001,
        lrf       = 0.01,
        momentum  = 0.937,
        weight_decay = 0.0005,
        augment   = True,
        hsv_h     = 0.015,        # hue augmentation
        hsv_s     = 0.7,          # saturation augmentation
        hsv_v     = 0.4,          # value/brightness augmentation
        flipud    = 0.0,          # no vertical flip (real-world cameras)
        fliplr    = 0.5,
        mosaic    = 1.0,
        mixup     = 0.1,
        project   = RUNS_DIR,
        name      = "smoking_v2_test" if test_mode else "smoking_v2",
        exist_ok  = True,
        verbose   = True,
    )

    print("Starting training...")
    results = model.train(**train_args)

    if not test_mode:
        print("\n=== Training complete. Running validation... ===")
        metrics = model.val()
        print(f"\n--- Results ---")
        print(f"mAP50      : {metrics.box.map50:.4f}")
        print(f"mAP50-95   : {metrics.box.map:.4f}")
        print(f"Precision  : {metrics.box.p.mean():.4f}")
        print(f"Recall     : {metrics.box.r.mean():.4f}")

        print(f"\nBest model saved to: {RUNS_DIR}/smoking_v2/weights/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Run 2-epoch sanity check (CPU-friendly)")
    args = parser.parse_args()
    main(test_mode=args.test)
