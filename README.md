# Smoking Detection System

Real-time cigarette detection using YOLOv11m, built as part of a Computer Vision dissertation. The system detects cigarettes and cigarette-like objects (pens, straws) in images and live camera feeds, with optional SMS alerts via Twilio.

---

## Features

- **Image Upload** — Upload a JPG/PNG, get annotated bounding boxes and a detection table
- **Live Camera** — Smooth 30 fps MJPEG stream with real-time YOLO detection overlay
- **2-Class Detection** — `cigarette` and `cigarette_like_object` (reduces false positives from pens/straws)
- **SMS Alerts** — Fires once after sustained detection (>=50% of frames over 15 seconds)
- **Model Export** — ONNX FP32, ONNX INT8, TorchScript via `scripts/export_model.py`

---

## Project Structure

```
smoking_detection/
├── app.py                          # Streamlit web application
├── requirements.txt
├── generate_diagrams.py            # Architecture diagram generator
│
├── models/
│   └── best.pt                     # Trained weights (download separately)
│
├── utils/
│   ├── detector.py                 # YOLO inference wrapper
│   ├── camera_stream.py            # 3-thread MJPEG camera stream
│   └── alerts.py                   # Twilio SMS + sliding-window tracker
│
├── scripts/
│   └── export_model.py             # Export to ONNX / TorchScript
│
├── notebooks/
│   ├── 01_yolov11s_roboflow_baseline.ipynb
│   ├── 02_yolov12s_roboflow_comparison.ipynb
│   ├── 03_yolov11s_merged_2class_final.ipynb
│   ├── 04_yolov11m_attempt.ipynb
│   ├── 05_yolov11s_vs_yolov12s_comparison.ipynb   # YOLOv11m vs YOLOv12m
│   └── Yolo12m.ipynb
│
├── assets/
│   ├── diagrams/                   # Architecture diagrams (PNG)
│   ├── training_results/           # CSV logs, curves, confusion matrices
│   └── sample_images/
│
└── .streamlit/
    ├── config.toml                 # Dark theme
    └── secrets.toml.example        # Twilio credentials template
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/jilakararahul/smoking_detection.git
cd smoking_detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download model weights

The trained weights are not stored in this repository. Download `best.pt` from Google Drive and place it at:

```
models/best.pt
```

### 5. Configure SMS alerts (optional)

Copy the example secrets file and fill in your Twilio credentials:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`:

```toml
[twilio]
account_sid = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
auth_token  = "your_auth_token_here"
from_number = "+14155238886"   # your Twilio number
```

> `secrets.toml` is gitignored — never commit real credentials.

---

## Running the App

```bash
streamlit run app.py
```

Or without activating the venv:

```bash
.venv\Scripts\streamlit run app.py   # Windows
.venv/bin/streamlit run app.py       # macOS / Linux
```

Open `http://localhost:8501` in your browser.

---

## Usage

### Image Upload
1. Click the **Image Upload** tab
2. Upload a JPG or PNG file
3. Annotated image and detection table appear automatically
4. Download the annotated result using the button below

### Live Camera
1. Click the **Live Camera** tab
2. Select the camera index (0 = default webcam)
3. Click **Start live feed**
4. Detection runs in a background thread — the video stays smooth at ~30 fps
5. Click **Stop** to end the stream

### SMS Alerts
1. Enable **SMS Alerts** in the sidebar
2. Enter the recipient phone number in international format (e.g. `+447700900000`)
3. Alert fires once a cigarette has been detected in >=50% of frames over 15 seconds
4. 30-second cooldown between consecutive alerts

### Confidence Threshold
Adjust the **Confidence threshold** slider in the sidebar. Changes apply on the next frame without restarting the stream.

---

## Model Export

Export the trained model to deployment formats:

```bash
python scripts/export_model.py
# or with a custom path:
python scripts/export_model.py --model models/best.pt --imgsz 640
```

Outputs to `models/`:

| Format | Use case |
|--------|----------|
| `best.pt` | PyTorch — standard inference |
| `best_fp32.onnx` | ONNX — cross-platform, edge deployment |
| `best_int8.onnx` | ONNX INT8 — quantised, ~4x smaller, faster on CPU |

---

## Model Details

| | YOLOv11m (selected) | YOLOv12m (comparison) |
|---|---|---|
| Parameters | 20.1 M | 20.1 M |
| GFLOPs | 68.0 | 67.1 |
| mAP@50 | **74.81%** | 74.5% |
| Precision | **79.63%** | 75.4% |
| Recall | 68.69% | **69.3%** |
| Training time | **3.49 h** | 6.21 h |
| Dataset | Merged 2-class | Merged 2-class |

**Dataset:** Merged from 4 sources — Roboflow cigarette dataset, custom smoking images, pen dataset, and straw dataset. Label-remapped to 2 classes: `cigarette` and `cigarette_like_object`.

---

## Architecture Diagrams

High-resolution diagrams are in `assets/diagrams/`:

| File | Description |
|------|-------------|
| `01_full_system_pipeline.png` | End-to-end pipeline from data collection to deployment |
| `02_webapp_architecture.png` | Streamlit app internals, 3-thread MJPEG, SMS system |
| `03_dataset_pipeline.png` | Dataset construction and label remapping |
| `04_model_architecture.png` | YOLOv11m backbone / neck / head |

Regenerate with:

```bash
python generate_diagrams.py
```

---

## Requirements

- Python 3.10+
- Webcam required for live camera mode
- Twilio account required for SMS alerts
- CUDA GPU recommended for training; CPU is sufficient for inference
