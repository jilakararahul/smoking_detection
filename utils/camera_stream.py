"""
Background MJPEG camera stream with embedded YOLO detection.

Architecture
------------
  CameraStream.start()
    ├─ Thread A : OpenCV capture + YOLO detection → JPEG bytes in shared buffer
    └─ Thread B : HTTPServer serves those bytes as MJPEG to the browser

The browser loads the stream as a plain <img> tag — no WebRTC, no HTTPS needed.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2


class CameraStream:
    """Manages a background capture thread and a local MJPEG HTTP server."""

    def __init__(self, port: int = 8765):
        self.port = port
        self._lock = threading.Lock()
        self._frame_bytes: bytes | None = None
        self._running = False
        self._cam_thread: threading.Thread | None = None
        self._server: HTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._detector = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    def start(self, cam_index: int, detector, on_cigarette=None) -> bool:
        """
        Start the camera capture and MJPEG server.
        Returns False if the camera could not be opened.
        """
        if self._running:
            return True

        # Verify camera opens before committing
        probe = cv2.VideoCapture(int(cam_index))
        if not probe.isOpened():
            probe.release()
            return False
        probe.release()

        self._detector    = detector
        self._on_cigarette = on_cigarette  # callable(n_cigs) or None
        self._running = True

        self._cam_thread = threading.Thread(
            target=self._capture_loop,
            args=(int(cam_index),),
            daemon=True,
        )
        self._cam_thread.start()
        self._start_server()
        return True

    def stop(self) -> None:
        """Stop the capture thread and shut down the HTTP server."""
        self._running = False
        if self._server:
            self._server.shutdown()
            self._server = None
        self._frame_bytes = None

    def update_conf(self, conf: float) -> None:
        """Update detection confidence threshold from the Streamlit slider."""
        if self._detector is not None:
            self._detector.conf = conf

    # ------------------------------------------------------------------
    # Internal — capture thread
    # ------------------------------------------------------------------

    def _capture_loop(self, cam_index: int) -> None:
        cap = cv2.VideoCapture(cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep latency minimal

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, dets = self._detector.predict(frame)

            # Overlay counters on the frame
            n_cig  = sum(1 for d in dets if d.class_id == 0)
            n_like = sum(1 for d in dets if d.class_id == 1)

            # Fire alert callback (runs in background thread — keep it fast)
            if n_cig > 0 and self._on_cigarette is not None:
                try:
                    self._on_cigarette(n_cig)
                except Exception:
                    pass

            lines = [
                f"Cigarettes : {n_cig}",
                f"Cig-like   : {n_like}",
                f"Conf thresh: {self._detector.conf:.2f}",
            ]
            y = 32
            for line in lines:
                cv2.putText(
                    annotated, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                    (0, 0, 0), 3, cv2.LINE_AA,
                )
                cv2.putText(
                    annotated, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )
                y += 30

            _, buf = cv2.imencode(
                ".jpg", annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
            with self._lock:
                self._frame_bytes = buf.tobytes()

        cap.release()
        self._running = False

    # ------------------------------------------------------------------
    # Internal — MJPEG HTTP server
    # ------------------------------------------------------------------

    def _start_server(self) -> None:
        stream = self   # captured in handler closure

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.send_header("Cache-Control", "no-cache, no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                try:
                    while stream._running:
                        with stream._lock:
                            data = stream._frame_bytes
                        if data:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                            self.wfile.write(data)
                            self.wfile.write(b"\r\n")
                        time.sleep(0.033)   # ~30 fps ceiling
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass  # silence access logs

        try:
            self._server = HTTPServer(("localhost", self.port), _Handler)
        except OSError:
            # Port already in use — try the next one
            self.port += 1
            self._server = HTTPServer(("localhost", self.port), _Handler)

        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
        )
        self._server_thread.start()
