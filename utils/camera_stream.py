"""
Background MJPEG camera stream with YOLO detection.

Three daemon threads run in parallel:
  Thread 1 — Capture   : reads frames from webcam at full camera FPS, draws
                          last-known bboxes, encodes to JPEG
  Thread 2 — Inference : runs YOLO on the latest raw frame, updates detections
  Thread 3 — Server    : serves JPEG buffer as MJPEG over HTTP

Decoupling capture from inference keeps the video smooth at ~30 fps even
when YOLO inference is slow (5–15 fps on CPU).
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2

from utils.alerts import SlidingWindowTracker
from utils.detector import CLASS_COLORS_BGR, CLASS_NAMES, Detection


def _draw_detections(frame, detections: list[Detection]) -> None:
    """Draw bounding boxes on a frame in-place."""
    for d in detections:
        bgr   = CLASS_COLORS_BGR.get(d.class_id, (180, 180, 180))
        label = f"{d.class_name}  {d.confidence:.0%}"

        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), bgr, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(
            frame,
            (d.x1, d.y1 - th - 10),
            (d.x1 + tw + 6, d.y1),
            bgr, -1,
        )
        cv2.putText(
            frame, label, (d.x1 + 3, d.y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )


class CameraStream:
    """Manages the capture, inference, and MJPEG server threads."""

    def __init__(
        self,
        port: int = 8765,
        window_seconds: float = 15.0,
        ratio_threshold: float = 0.50,
    ):
        self.port = port

        self._out_lock  = threading.Lock()
        self._frame_bytes: bytes | None = None

        self._raw_lock  = threading.Lock()
        self._raw_frame = None

        self._det_lock  = threading.Lock()
        self._last_dets: list[Detection] = []

        self._running      = False
        self._detector     = None
        self._on_cigarette = None

        self._tracker = SlidingWindowTracker(
            window_seconds=window_seconds,
            ratio_threshold=ratio_threshold,
        )

        self._server: HTTPServer | None = None

    @property
    def running(self) -> bool:
        return self._running

    def start(self, cam_index: int, detector, on_cigarette=None) -> bool:
        """Start all threads. Returns False if the camera cannot be opened."""
        if self._running:
            return True

        probe = cv2.VideoCapture(int(cam_index))
        if not probe.isOpened():
            probe.release()
            return False
        probe.release()

        self._detector     = detector
        self._on_cigarette = on_cigarette
        self._running      = True
        self._tracker.reset()

        threading.Thread(
            target=self._capture_loop, args=(int(cam_index),), daemon=True
        ).start()
        threading.Thread(target=self._inference_loop, daemon=True).start()
        self._start_server()
        return True

    def stop(self) -> None:
        """Stop all threads and shut down the HTTP server."""
        self._running = False
        if self._server:
            self._server.shutdown()
            self._server = None
        self._frame_bytes = None
        self._raw_frame   = None
        self._last_dets   = []
        self._tracker.reset()

    def update_conf(self, conf: float) -> None:
        if self._detector is not None:
            self._detector.conf = conf

    # ------------------------------------------------------------------
    # Thread 1 — Capture
    # ------------------------------------------------------------------

    def _capture_loop(self, cam_index: int) -> None:
        cap = cv2.VideoCapture(cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            with self._raw_lock:
                self._raw_frame = frame.copy()

            with self._det_lock:
                dets = list(self._last_dets)

            annotated = frame.copy()
            _draw_detections(annotated, dets)

            n_cig  = sum(1 for d in dets if d.class_id == 0)
            n_like = sum(1 for d in dets if d.class_id == 1)
            ratio  = self._tracker.detection_ratio

            lines = [
                f"Cigarettes : {n_cig}",
                f"Cig-like   : {n_like}",
                f"Conf thresh: {self._detector.conf:.2f}",
                f"Det ratio  : {ratio:.0%} / 15 s window" if n_cig > 0 else "Det ratio  : --",
            ]
            y = 32
            for line in lines:
                cv2.putText(annotated, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 1, cv2.LINE_AA)
                y += 30

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._out_lock:
                self._frame_bytes = buf.tobytes()

        cap.release()
        self._running = False

    # ------------------------------------------------------------------
    # Thread 2 — Inference
    # ------------------------------------------------------------------

    def _inference_loop(self) -> None:
        while self._running:
            with self._raw_lock:
                frame = self._raw_frame

            if frame is None:
                time.sleep(0.01)
                continue

            _, dets = self._detector.predict(frame)

            with self._det_lock:
                self._last_dets = dets

            n_cig = sum(1 for d in dets if d.class_id == 0)
            if self._on_cigarette is not None:
                if self._tracker.update(n_cig > 0):
                    try:
                        self._on_cigarette(n_cig)
                    except Exception:
                        pass
            else:
                self._tracker.update(n_cig > 0)

    # ------------------------------------------------------------------
    # Thread 3 — MJPEG server
    # ------------------------------------------------------------------

    def _start_server(self) -> None:
        stream = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache, no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                try:
                    while stream._running:
                        with stream._out_lock:
                            data = stream._frame_bytes
                        if data:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                            self.wfile.write(data)
                            self.wfile.write(b"\r\n")
                        time.sleep(0.033)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        try:
            self._server = HTTPServer(("localhost", self.port), _Handler)
        except OSError:
            self.port += 1
            self._server = HTTPServer(("localhost", self.port), _Handler)

        threading.Thread(target=self._server.serve_forever, daemon=True).start()
