"""
Smoking Detection — Streamlit Web Application

Two modes:
  Image Upload : Upload a JPG/PNG image and get annotated detection results.
  Live Camera  : Real-time webcam feed with YOLO detection overlay.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from utils.alerts import SMSAlerter
from utils.camera_stream import CameraStream
from utils.detector import (
    CLASS_COLORS_RGB,
    CLASS_NAMES,
    HEX_COLORS,
    SmokingDetector,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smoking Detection System",
    page_icon=":camera:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Top header strip */
        .app-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 2rem 2.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            border: 1px solid #0f3460;
        }
        .app-header h1 {
            color: #e0e0e0;
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        .app-header p {
            color: #9ba8b5;
            margin: 0.4rem 0 0 0;
            font-size: 0.95rem;
        }

        /* Detection badges */
        .badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 600;
            margin: 0.2rem 0.2rem 0.2rem 0;
        }

        /* Result cards */
        .result-card {
            background: #1e1e2e;
            border: 1px solid #2d2d42;
            border-radius: 10px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
        }
        .result-card-title {
            font-size: 0.78rem;
            color: #9ba8b5;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.5rem;
        }
        .result-card-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #e0e0e0;
        }

        /* Alert box */
        .alert-box {
            background: #2a1a1a;
            border-left: 4px solid #e05252;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            margin-top: 0.75rem;
            color: #f0a0a0;
            font-size: 0.9rem;
        }
        .safe-box {
            background: #1a2a1a;
            border-left: 4px solid #52c052;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            margin-top: 0.75rem;
            color: #a0e0a0;
            font-size: 0.9rem;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: #0e0e1a;
        }
        section[data-testid="stSidebar"] .stSlider label {
            color: #c0c0d0;
        }

        /* Image captions */
        .img-caption {
            text-align: center;
            font-size: 0.8rem;
            color: #9ba8b5;
            margin-top: 0.4rem;
        }

        /* Tab styling */
        button[data-baseweb="tab"] {
            font-size: 0.95rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "models/best.pt"
CONF_DEFAULT  = 0.40


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Model Configuration")

    model_path_str = st.text_input(
        "Model weights path",
        value=str(DEFAULT_MODEL),
        help="Path to the trained best.pt file",
    )
    model_path = Path(model_path_str)

    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.95,
        value=CONF_DEFAULT,
        step=0.05,
        help="Detections below this score are ignored",
    )

    st.markdown("---")
    st.markdown("### Model Status")

    if model_path.exists():
        st.success(f"Model loaded  \n`{model_path.name}`")
        model_ok = True
    else:
        st.error("Model not found")
        st.markdown(
            f"""
            **Setup required:**
            1. Download `smoking_v3_best.pt` from Google Drive
            2. Copy it to `models/best.pt`
            3. Or paste the full path above
            """
        )
        model_ok = False

    st.markdown("---")
    st.markdown("### Class Legend")
    for cls_id, name in CLASS_NAMES.items():
        r, g, b = CLASS_COLORS_RGB[cls_id]
        st.markdown(
            f'<span class="badge" style="background:rgba({r},{g},{b},0.2);'
            f'color:rgb({r},{g},{b});border:1px solid rgb({r},{g},{b});">'
            f"{name}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### SMS Alerts")
    sms_enabled = st.toggle("Enable SMS alerts", value=False)
    sms_number  = ""
    if sms_enabled:
        sms_number = st.text_input(
            "Alert phone number",
            placeholder="+447700900000",
            help="International format. Requires Twilio credentials in .streamlit/secrets.toml",
        )
        st.caption("Alert sent when a cigarette is detected (30 s cooldown).")

    st.markdown("---")
    st.caption("YOLOv11s · 2-class · 150 epochs  \nDataset: merged cigarette + pen/straw")


# ---------------------------------------------------------------------------
# Helper: SMS alerter (one per session)
# ---------------------------------------------------------------------------
def maybe_send_sms(detections: list, source: str) -> None:
    """Fire an SMS alert if cigarettes were found and SMS is configured."""
    if not sms_enabled or not sms_number:
        return
    cig_count = sum(1 for d in detections if d.class_id == 0)
    if cig_count == 0:
        return
    key = "sms_alerter"
    if key not in st.session_state or st.session_state[key].to_number != sms_number:
        st.session_state[key] = SMSAlerter(to_number=sms_number)
    alerter: SMSAlerter = st.session_state[key]
    try:
        sent = alerter.send(
            f"Smoking Detection Alert: {cig_count} cigarette(s) detected in {source}."
        )
        if sent:
            st.toast("SMS alert sent.", icon="📱")
    except RuntimeError as e:
        st.warning(f"SMS not sent: {e}")


# ---------------------------------------------------------------------------
# Helper: load detector (cached so we don't reload on every interaction)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model weights...")
def load_detector(path: str, conf: float) -> SmokingDetector:
    return SmokingDetector(model_path=path, conf=conf)


def get_detector() -> SmokingDetector | None:
    if not model_ok:
        return None
    try:
        det = load_detector(str(model_path), conf_threshold)
        # Update conf dynamically without reloading the model
        det.conf = conf_threshold
        return det
    except FileNotFoundError as e:
        st.error(str(e))
        return None


# ---------------------------------------------------------------------------
# Helper: render detection summary
# ---------------------------------------------------------------------------
def render_summary(detections: list) -> None:
    cig_dets  = [d for d in detections if d.class_id == 0]
    like_dets = [d for d in detections if d.class_id == 1]
    total     = len(detections)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="result-card">'
            f'<div class="result-card-title">Total detections</div>'
            f'<div class="result-card-value">{total}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="result-card">'
            f'<div class="result-card-title">Cigarettes</div>'
            f'<div class="result-card-value" style="color:{HEX_COLORS[0]};">'
            f"{len(cig_dets)}</div></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="result-card">'
            f'<div class="result-card-title">Cig-like Objects</div>'
            f'<div class="result-card-value" style="color:{HEX_COLORS[1]};">'
            f"{len(like_dets)}</div></div>",
            unsafe_allow_html=True,
        )

    if cig_dets:
        st.markdown(
            '<div class="alert-box">Smoking detected in frame.</div>',
            unsafe_allow_html=True,
        )
    elif total == 0:
        st.markdown(
            '<div class="safe-box">No smoking detected.</div>',
            unsafe_allow_html=True,
        )

    if detections:
        st.markdown("**Detection details**")
        rows = [
            {
                "Class":      d.class_name,
                "Confidence": f"{d.confidence:.1%}",
                "Bounding Box": f"({d.x1}, {d.y1}) → ({d.x2}, {d.y2})",
            }
            for d in detections
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>Smoking Detection System</h1>
        <p>YOLOv11s &nbsp;·&nbsp; 2-class detection &nbsp;·&nbsp;
           Cigarette &amp; Cigarette-like Objects</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_image, tab_camera = st.tabs(["Image Upload", "Live Camera"])


# ===== TAB 1: IMAGE UPLOAD =================================================
with tab_image:
    st.markdown("Upload an image to run detection. Supported formats: JPG, JPEG, PNG.")

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        detector = get_detector()
        if detector is None:
            st.stop()

        # Decode image
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Could not decode the uploaded image.")
            st.stop()

        # Run inference
        with st.spinner("Running detection..."):
            annotated_bgr, detections = detector.predict(img_bgr)

        # Display side by side
        col_orig, col_det = st.columns(2)

        with col_orig:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)
            st.markdown('<div class="img-caption">Original</div>', unsafe_allow_html=True)

        with col_det:
            ann_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(ann_rgb, use_container_width=True)
            st.markdown(
                f'<div class="img-caption">Detected — {len(detections)} object(s)</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        render_summary(detections)
        maybe_send_sms(detections, source=f"uploaded image ({uploaded.name})")

        # Download annotated image
        ann_pil = Image.fromarray(ann_rgb)
        buf = io.BytesIO()
        ann_pil.save(buf, format="JPEG", quality=92)
        st.download_button(
            "Download annotated image",
            data=buf.getvalue(),
            file_name=f"detected_{uploaded.name}",
            mime="image/jpeg",
        )
    else:
        st.info("Upload an image to begin detection.")


# ===== TAB 2: LIVE CAMERA ==================================================
with tab_camera:
    # ------------------------------------------------------------------
    # Architecture: Python/OpenCV captures frames in a background thread,
    # runs detection, and serves annotated frames as MJPEG over a local
    # HTTP server.  The browser loads the stream as a plain <img> — no
    # WebRTC, no HTTPS, no browser camera permission required.
    # ------------------------------------------------------------------
    detector_cam = get_detector()

    if detector_cam is None:
        st.warning("Load a valid model first (see sidebar).")
    else:
        if "cam_stream" not in st.session_state:
            st.session_state.cam_stream = CameraStream(port=8765)

        stream: CameraStream = st.session_state.cam_stream

        # Keep conf threshold in sync with the sidebar slider
        stream.update_conf(conf_threshold)

        cam_col, btn_start, btn_stop = st.columns([1, 1, 1])
        with cam_col:
            cam_index = st.number_input(
                "Camera index", min_value=0, max_value=10, value=0, step=1,
                help="0 = default webcam, 1 = second camera, etc.",
            )
        with btn_start:
            if st.button(
                "Start live feed",
                disabled=stream.running,
                use_container_width=True,
            ):
                # Build SMS callback for the live stream background thread
                _sms_cb = None
                if sms_enabled and sms_number:
                    _alerter = SMSAlerter(to_number=sms_number)
                    def _sms_cb(n, _a=_alerter, _num=sms_number):
                        try:
                            _a.send(f"Smoking Detection Alert: {n} cigarette(s) detected in live camera feed.")
                        except Exception:
                            pass
                ok = stream.start(int(cam_index), detector_cam, on_cigarette=_sms_cb)
                if not ok:
                    st.error(f"Could not open camera index {int(cam_index)}.")
                st.rerun()
        with btn_stop:
            if st.button(
                "Stop",
                disabled=not stream.running,
                use_container_width=True,
            ):
                stream.stop()
                st.rerun()

        if stream.running:
            # Embed the MJPEG stream — renders as smooth native video in browser
            components.html(
                f"""
                <style>
                  body {{ margin: 0; background: #0e0e1a; }}
                  img  {{ width: 100%; border-radius: 8px; display: block; }}
                </style>
                <img src="http://localhost:{stream.port}/"
                     alt="Live detection feed"
                     onerror="setTimeout(()=>this.src=this.src+'?'+Date.now(), 500)">
                """,
                height=500,
                scrolling=False,
            )
            st.caption(
                "Smooth MJPEG stream — detection runs in a background thread. "
                "Adjust confidence in the sidebar; it applies on the next frame."
            )
        else:
            st.info("Click **Start live feed** to begin real-time detection.")
