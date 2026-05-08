"""
Live Webcam Demo — Smoking Detection
=====================================
Uses the trained YOLOv11 model to run real-time detection on webcam feed.

Usage:
    python live_demo.py                          # default webcam, best.pt
    python live_demo.py --model path/to/best.pt  # custom model path
    python live_demo.py --conf 0.5               # higher confidence threshold
    python live_demo.py --cam 1                  # use second camera

Controls:
    Q  — quit
    S  — save screenshot
    +  — increase confidence threshold
    -  — decrease confidence threshold
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# Class colours (BGR)
COLOURS = {
    0: (0,   140, 255),   # Cigarette    — orange
    1: (0,   0,   220),   # smoking      — red
    2: (50,  200, 50),    # cigarette_like_object — green
}

LABELS = {
    0: "Cigarette",
    1: "Smoking",
    2: "Cig-like Object",
}


def draw_box(frame, box, cls_id, conf):
    x1, y1, x2, y2 = map(int, box)
    colour = COLOURS.get(cls_id, (200, 200, 200))
    label  = f"{LABELS.get(cls_id, cls_id)} {conf:.0%}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    # Label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(frame, fps, conf_thresh, detections):
    h, w = frame.shape[:2]

    # FPS + threshold
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Conf: {conf_thresh:.2f}  (+/-  to adjust)",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Detection counts per class
    y = 85
    for cls_id, name in LABELS.items():
        count = sum(1 for d in detections if d == cls_id)
        colour = COLOURS[cls_id]
        cv2.putText(frame, f"{name}: {count}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
        y += 22

    # Controls hint
    cv2.putText(frame, "Q: quit  S: screenshot", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


def main(model_path: str, cam_index: int, conf: float):
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    print("Model loaded. Opening camera...")

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {cam_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    conf_thresh = conf
    screenshot_dir = Path("screenshots")
    screenshot_dir.mkdir(exist_ok=True)

    prev_time = time.time()
    fps = 0.0

    print("Running — press Q to quit, S to save screenshot, +/- to adjust confidence")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Run inference
        results = model(frame, conf=conf_thresh, verbose=False)[0]

        # Draw detections
        detected_classes = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                draw_box(frame, box.xyxy[0].tolist(), cls_id, conf_val)
                detected_classes.append(cls_id)

        # FPS calculation
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time + 1e-9))
        prev_time = now

        draw_hud(frame, fps, conf_thresh, detected_classes)

        cv2.imshow("Smoking Detection — Live", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            path = screenshot_dir / f"shot_{int(time.time())}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"Screenshot saved: {path}")
        elif key == ord("+") or key == ord("="):
            conf_thresh = min(0.95, round(conf_thresh + 0.05, 2))
            print(f"Confidence threshold: {conf_thresh}")
        elif key == ord("-"):
            conf_thresh = max(0.05, round(conf_thresh - 0.05, 2))
            print(f"Confidence threshold: {conf_thresh}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Default model path — update this to wherever Colab saved your best.pt
    DEFAULT_MODEL = str(
        Path(__file__).parent / "runs" / "smoking_v2" / "weights" / "best.pt"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Path to trained best.pt")
    parser.add_argument("--cam",   type=int, default=0,
                        help="Camera index (0 = default webcam)")
    parser.add_argument("--conf",  type=float, default=0.4,
                        help="Confidence threshold (default 0.4)")
    args = parser.parse_args()

    main(args.model, args.cam, args.conf)
