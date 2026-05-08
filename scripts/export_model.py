"""
Model Export & Quantisation Script
====================================
Exports the trained YOLOv11s model to multiple formats and benchmarks
each format for size and inference speed.

Exported formats
----------------
  best.pt          — original PyTorch FP32  (baseline)
  best_fp32.onnx   — ONNX FP32              (cross-platform, edge-friendly)
  best_int8.onnx   — ONNX INT8 quantised    (smaller, faster, ~slight accuracy drop)
  best.torchscript — TorchScript            (PyTorch mobile / C++ deployment)

Usage
-----
    python scripts/export_model.py
    python scripts/export_model.py --model models/best.pt --imgsz 640
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "best.pt"
EXPORT_DIR    = PROJECT_ROOT / "models"
IMGSZ         = 640
WARMUP_RUNS   = 5
BENCH_RUNS    = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def benchmark_pt(model_path: Path, imgsz: int, runs: int) -> float:
    """Return mean inference time (ms) for a .pt model."""
    model = YOLO(str(model_path))
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(WARMUP_RUNS):
        model(dummy, verbose=False)
    t0 = time.perf_counter()
    for _ in range(runs):
        model(dummy, verbose=False)
    return (time.perf_counter() - t0) / runs * 1000


def benchmark_onnx(model_path: Path, imgsz: int, runs: int) -> float:
    """Return mean inference time (ms) for an ONNX model via onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        return float("nan")

    sess = ort.InferenceSession(str(model_path))
    inp_name = sess.get_inputs()[0].name
    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)

    for _ in range(WARMUP_RUNS):
        sess.run(None, {inp_name: dummy})
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {inp_name: dummy})
    return (time.perf_counter() - t0) / runs * 1000


def print_table(rows: list[dict]) -> None:
    headers = ["Format", "File", "Size (MB)", "Inf. time (ms)", "vs FP32"]
    col_w   = [18, 26, 12, 16, 10]

    header = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        line = "  ".join(str(row.get(h, "")).ljust(w) for h, w in zip(headers, col_w))
        print(line)
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(model_path: Path, imgsz: int) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\nSource model : {model_path}")
    print(f"Export dir   : {EXPORT_DIR}")
    print(f"Image size   : {imgsz}x{imgsz}")
    print(f"Benchmark    : {BENCH_RUNS} runs (after {WARMUP_RUNS} warm-up)\n")

    model = YOLO(str(model_path))
    results = []

    # ------------------------------------------------------------------
    # 1. Baseline — PyTorch FP32
    # ------------------------------------------------------------------
    print("[1/3] Benchmarking baseline PyTorch FP32 ...")
    pt_time = benchmark_pt(model_path, imgsz, BENCH_RUNS)
    pt_size = file_size_mb(model_path)
    fp32_time = pt_time
    results.append({
        "Format":         "PyTorch FP32",
        "File":           model_path.name,
        "Size (MB)":      f"{pt_size:.1f}",
        "Inf. time (ms)": f"{pt_time:.1f}",
        "vs FP32":        "1.00x  (baseline)",
    })

    # ------------------------------------------------------------------
    # 2. ONNX FP32
    # ------------------------------------------------------------------
    print("[2/3] Exporting ONNX FP32 ...")
    onnx_fp32_path = EXPORT_DIR / "best_fp32.onnx"
    model.export(format="onnx", imgsz=imgsz, simplify=True,
                 dynamic=False, half=False,
                 project=str(EXPORT_DIR), name="best_fp32")
    # Ultralytics saves to <model_stem>.onnx next to the .pt by default
    # Locate it wherever ultralytics placed it
    candidate = model_path.with_suffix(".onnx")
    if candidate.exists() and not onnx_fp32_path.exists():
        candidate.rename(onnx_fp32_path)

    if onnx_fp32_path.exists():
        onnx_fp32_size = file_size_mb(onnx_fp32_path)
        onnx_fp32_time = benchmark_onnx(onnx_fp32_path, imgsz, BENCH_RUNS)
        ratio = f"{fp32_time / onnx_fp32_time:.2f}x" if not np.isnan(onnx_fp32_time) else "n/a"
        results.append({
            "Format":         "ONNX FP32",
            "File":           onnx_fp32_path.name,
            "Size (MB)":      f"{onnx_fp32_size:.1f}",
            "Inf. time (ms)": f"{onnx_fp32_time:.1f}" if not np.isnan(onnx_fp32_time) else "n/a",
            "vs FP32":        ratio,
        })
    else:
        print("  ONNX FP32 export path not found — skipping benchmark.")

    # ------------------------------------------------------------------
    # 3. ONNX INT8 (post-training quantisation)
    # ------------------------------------------------------------------
    print("[3/3] Exporting ONNX INT8 (quantised) ...")
    onnx_int8_path = EXPORT_DIR / "best_int8.onnx"
    try:
        model.export(format="onnx", imgsz=imgsz, simplify=True,
                     dynamic=False, int8=True,
                     project=str(EXPORT_DIR), name="best_int8")
        candidate_int8 = model_path.parent / (model_path.stem + "_int8.onnx")
        if candidate_int8.exists() and not onnx_int8_path.exists():
            candidate_int8.rename(onnx_int8_path)

        if onnx_int8_path.exists():
            int8_size = file_size_mb(onnx_int8_path)
            int8_time = benchmark_onnx(onnx_int8_path, imgsz, BENCH_RUNS)
            ratio_int8 = f"{fp32_time / int8_time:.2f}x" if not np.isnan(int8_time) else "n/a"
            results.append({
                "Format":         "ONNX INT8",
                "File":           onnx_int8_path.name,
                "Size (MB)":      f"{int8_size:.1f}",
                "Inf. time (ms)": f"{int8_time:.1f}" if not np.isnan(int8_time) else "n/a",
                "vs FP32":        ratio_int8,
            })
        else:
            print("  INT8 ONNX path not found after export — skipping benchmark.")
    except Exception as e:
        print(f"  INT8 export failed: {e}")
        print("  (INT8 quantisation may require a calibration dataset on some backends)")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print_table(results)
    print("\nExported files in models/:")
    for f in sorted(EXPORT_DIR.glob("best*")):
        print(f"  {f.name:<30} {file_size_mb(f):.1f} MB")

    print("\nNotes:")
    print("  - INT8 quantisation reduces model size ~4x and speeds up CPU inference.")
    print("  - Accuracy impact is typically <1% mAP on well-trained models.")
    print("  - ONNX models run on ONNX Runtime (CPU/GPU), suitable for Raspberry Pi / Jetson.")
    print("  - Use onnxruntime-gpu for GPU acceleration on Jetson Nano.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and benchmark YOLOv11 model formats")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to best.pt")
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="Inference image size")
    args = parser.parse_args()
    main(Path(args.model), args.imgsz)
