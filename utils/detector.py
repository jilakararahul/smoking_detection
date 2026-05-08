"""
Smoking Detection — YOLO inference wrapper.

Classes
-------
0  cigarette             — an actual cigarette being held/smoked
1  cigarette_like_object — pen, straw, pencil, etc. visually similar to a cigarette

Model: YOLOv11s trained on merged 2-class dataset (smoking_v3_2class, 150 epochs)
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from ultralytics import YOLO

# ------------------------------------------------------------------
# Class metadata
# ------------------------------------------------------------------
CLASS_NAMES: dict[int, str] = {
    0: "Cigarette",
    1: "Cigarette-like Object",
}

# RGB colours used for bounding boxes and UI badges
CLASS_COLORS_RGB: dict[int, tuple[int, int, int]] = {
    0: (255, 120, 20),   # orange  — cigarette
    1: (50,  140, 230),  # blue    — cigarette-like object
}

# BGR equivalents for OpenCV drawing
CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    k: (v[2], v[1], v[0]) for k, v in CLASS_COLORS_RGB.items()
}

HEX_COLORS: dict[int, str] = {
    0: "#FF7814",
    1: "#3250E6",
}


class Detection(NamedTuple):
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class SmokingDetector:
    """Thin wrapper around a YOLO model for smoking / cigarette-like detection."""

    def __init__(self, model_path: str | Path, conf: float = 0.40):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{self.model_path}'.\n"
                "Download smoking_v3_best.pt from Google Drive and place it at models/best.pt"
            )
        self.model = YOLO(str(self.model_path))
        self.conf = conf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image_bgr: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        """
        Run detection on a BGR NumPy array (as returned by OpenCV).

        Returns
        -------
        annotated_bgr : np.ndarray
            A copy of the input image with bounding boxes drawn.
        detections : list[Detection]
            Metadata for each detected object.
        """
        results = self.model(image_bgr, conf=self.conf, verbose=False)[0]
        annotated = image_bgr.copy()
        detections: list[Detection] = []

        if results.boxes is not None:
            for box in results.boxes:
                cls_id   = int(box.cls[0])
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                bgr   = CLASS_COLORS_BGR.get(cls_id, (180, 180, 180))
                label = f"{CLASS_NAMES.get(cls_id, str(cls_id))}  {conf_val:.0%}"

                # Bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)

                # Label background
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                )
                cv2.rectangle(
                    annotated,
                    (x1, y1 - th - 10),
                    (x1 + tw + 6, y1),
                    bgr, -1,
                )
                cv2.putText(
                    annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )

                detections.append(Detection(
                    class_id=cls_id,
                    class_name=CLASS_NAMES.get(cls_id, str(cls_id)),
                    confidence=conf_val,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))

        return annotated, detections
