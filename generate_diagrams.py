"""
Generate system architecture diagrams for dissertation report.
Run:  py generate_diagrams.py
Outputs to:  assets/diagrams/
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
os.makedirs("assets/diagrams", exist_ok=True)

# ── Shared colours ────────────────────────────────────────────────────────────
BG      = "#0e0e1a"
PANEL   = "#1a1a2e"
BORDER  = "#3a3a5c"
ORANGE  = "#FF7814"
CYAN    = "#4CC9F0"
GREEN   = "#52c052"
RED     = "#e05252"
PURPLE  = "#9d4edd"
BLUE    = "#3a86ff"
YELLOW  = "#ffbe0b"
TEXT    = "#e0e0e0"
SUBTEXT = "#9ba8b5"


def box(ax, x, y, w, h, label, sublabel="", color=PANEL, border=BORDER,
        fontsize=9, subfontsize=7.5, text_color=TEXT, bold=False):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.02",
                          facecolor=color, edgecolor=border, linewidth=1.2,
                          zorder=3)
    ax.add_patch(rect)
    fw = "bold" if bold else "normal"
    ax.text(x, y + (0.08 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight=fw, zorder=4)
    if sublabel:
        ax.text(x, y - 0.13, sublabel,
                ha="center", va="center", fontsize=subfontsize,
                color=SUBTEXT, zorder=4)


def arrow(ax, x1, y1, x2, y2, color=SUBTEXT, lw=1.5, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=2)


def label_arrow(ax, x1, y1, x2, y2, text, color=SUBTEXT):
    arrow(ax, x1, y1, x2, y2, color=color)
    mx, my = (x1+x2)/2, (y1+y2)/2
    ax.text(mx + 0.04, my, text, fontsize=6.5, color=SUBTEXT,
            va="center", zorder=5)


def title(ax, text, sub=""):
    ax.text(0.5, 0.97, text, transform=ax.transAxes,
            ha="center", va="top", fontsize=13, fontweight="bold",
            color=TEXT)
    if sub:
        ax.text(0.5, 0.93, sub, transform=ax.transAxes,
                ha="center", va="top", fontsize=8, color=SUBTEXT)


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 1 — Full System Pipeline (end-to-end)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

title(ax, "Smoking Detection System — Full Pipeline Architecture",
      "From raw data to real-time detection and SMS alerts")

# ── Phase labels (left column) ────────────────────────────────────────────────
phases = [
    (0.7, 7.5, "Phase 1–2", ORANGE),
    (0.7, 5.8, "Phase 3–4", CYAN),
    (0.7, 3.8, "Phase 5",   GREEN),
    (0.7, 1.8, "Deploy",    PURPLE),
]
for px, py, lbl, col in phases:
    ax.text(px, py, lbl, ha="center", va="center", fontsize=8,
            color=col, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec=col, lw=1.2))

# ── Row 1: Data Collection ─────────────────────────────────────────────────────
ax.text(8, 8.5, "DATA COLLECTION & PREPARATION", ha="center",
        fontsize=9, color=SUBTEXT, fontweight="bold")

box(ax, 2.8, 7.7, 2.2, 0.6, "Roboflow Dataset",       "5-class, 800 images",      color="#1e1e2e", border=ORANGE)
box(ax, 5.3, 7.7, 2.2, 0.6, "Pen Dataset (v1+v2)",    "815 images collected",     color="#1e1e2e", border=ORANGE)
box(ax, 7.8, 7.7, 2.2, 0.6, "Straw Dataset",          "615 images collected",     color="#1e1e2e", border=ORANGE)
box(ax, 10.3,7.7, 2.2, 0.6, "Smoking Dataset",        "5,786 cigarette images",   color="#1e1e2e", border=ORANGE)

# merge arrow
ax.annotate("", xy=(8, 6.9), xytext=(2.8, 7.4),
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2,
                            connectionstyle="arc3,rad=0.3"), zorder=2)
ax.annotate("", xy=(8, 6.9), xytext=(5.3, 7.4),
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2,
                            connectionstyle="arc3,rad=0.1"), zorder=2)
ax.annotate("", xy=(8, 6.9), xytext=(7.8, 7.4),
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2,
                            connectionstyle="arc3,rad=-0.1"), zorder=2)
ax.annotate("", xy=(8, 6.9), xytext=(10.3, 7.4),
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2,
                            connectionstyle="arc3,rad=-0.3"), zorder=2)

box(ax, 8, 6.55, 4.5, 0.6, "Merged 2-Class Dataset",
    "3,478 train / 920 valid · cigarette (0) · cigarette_like_object (1)",
    color=PANEL, border=ORANGE, bold=True, fontsize=10)

ax.text(12.8, 6.55, "Label Remap:", fontsize=7.5, color=SUBTEXT, va="center")
ax.text(12.8, 6.3,  "cls 0,1 → cigarette", fontsize=7, color=ORANGE, va="center")
ax.text(12.8, 6.1,  "cls 2   → cig_like",   fontsize=7, color=CYAN,   va="center")

# ── Row 2: Model Training ──────────────────────────────────────────────────────
arrow(ax, 8, 6.25, 8, 5.65, color=CYAN, lw=2)
ax.text(8, 5.85, "MODEL TRAINING (NVIDIA L4 · Google Colab Pro)",
        ha="center", fontsize=9, color=SUBTEXT, fontweight="bold")

box(ax, 3.5, 5.2, 3.0, 0.75, "YOLOv11s  (Phase 1)",
    "148 ep · 1.78h · mAP@50=66.2%\nP=82.5%  R=61.2%",
    color="#1e1e2e", border=CYAN, fontsize=8.5)
box(ax, 7.2, 5.2, 3.0, 0.75, "YOLOv11m  (Phase 4 ✓ Best)",
    "88 ep · 3.49h · mAP@50=74.81%\nP=79.6%  R=68.7%",
    color=PANEL, border=GREEN, bold=True, fontsize=8.5, text_color=GREEN)
box(ax, 11.0,5.2, 3.0, 0.75, "YOLOv12m  (Phase 5)",
    "105 ep · 6.21h · mAP@50=74.5%\nP=75.4%  R=69.3%",
    color="#1e1e2e", border=CYAN, fontsize=8.5)

arrow(ax, 8, 6.25, 3.5,  5.6, color=CYAN, lw=1.2)
arrow(ax, 8, 6.25, 7.2,  5.6, color=GREEN, lw=2)
arrow(ax, 8, 6.25, 11.0, 5.6, color=CYAN, lw=1.2)

ax.text(7.2, 4.78, "Selected: higher accuracy + faster training", ha="center",
        fontsize=7, color=GREEN)

# ── Row 3: Evaluation ─────────────────────────────────────────────────────────
arrow(ax, 7.2, 4.82, 7.2, 4.3, color=GREEN, lw=2)
ax.text(8, 4.1, "MODEL EVALUATION & EXPORT",
        ha="center", fontsize=9, color=SUBTEXT, fontweight="bold")

box(ax, 3.5, 3.65, 2.8, 0.65, "Model Comparison",    "Notebook 05\nYOLOv11m vs YOLOv12m", color="#1e1e2e", border=GREEN, fontsize=8)
box(ax, 7.2, 3.65, 2.8, 0.65, "Quantization Export", "ONNX FP32 / INT8\nTorchScript",       color="#1e1e2e", border=GREEN, fontsize=8)
box(ax, 11.0,3.65, 2.8, 0.65, "Validation Metrics",  "mAP@50 · P · R · F1\nConfusion Matrix", color="#1e1e2e", border=GREEN, fontsize=8)

arrow(ax, 7.2, 4.3, 3.5,  3.98, color=GREEN, lw=1.2)
arrow(ax, 7.2, 4.3, 7.2,  3.98, color=GREEN, lw=1.2)
arrow(ax, 7.2, 4.3, 11.0, 3.98, color=GREEN, lw=1.2)

# ── Row 4: Deployment ─────────────────────────────────────────────────────────
arrow(ax, 7.2, 3.32, 7.2, 2.75, color=PURPLE, lw=2)
ax.text(8, 2.6, "STREAMLIT WEB APPLICATION  (app.py)",
        ha="center", fontsize=9, color=SUBTEXT, fontweight="bold")

box(ax, 4.0, 2.1, 3.2, 0.75, "Image Upload Tab",
    "Upload JPG/PNG\nYOLO detect → annotated image\n+ download button",
    color="#1e1e2e", border=PURPLE, fontsize=7.5)
box(ax, 8.0, 2.1, 3.2, 0.75, "Live Camera Tab",
    "3-thread MJPEG stream\nCapture · Inference · Server\nSmooth 30fps feed",
    color="#1e1e2e", border=PURPLE, fontsize=7.5)
box(ax, 12.0,2.1, 2.6, 0.75, "SMS Alerts",
    "SlidingWindowTracker\n15s window · 50% ratio\nTwilio API",
    color="#1e1e2e", border=RED, fontsize=7.5)

arrow(ax, 7.2, 2.75, 4.0,  2.48, color=PURPLE, lw=1.2)
arrow(ax, 7.2, 2.75, 8.0,  2.48, color=PURPLE, lw=1.2)
arrow(ax, 7.2, 2.75, 12.0, 2.48, color=RED, lw=1.2)

# ── Row 5: Output ─────────────────────────────────────────────────────────────
arrow(ax, 4.0,  1.72, 4.0,  1.25, color=PURPLE, lw=1.2)
arrow(ax, 8.0,  1.72, 8.0,  1.25, color=PURPLE, lw=1.2)
arrow(ax, 12.0, 1.72, 12.0, 1.25, color=RED,    lw=1.2)

box(ax, 4.0,  0.95, 3.0, 0.5, "Annotated Image + Detection Table",
    color="#1e1e2e", border=PURPLE, fontsize=7.5)
box(ax, 8.0,  0.95, 3.0, 0.5, "Real-Time Annotated Video Feed",
    color="#1e1e2e", border=PURPLE, fontsize=7.5)
box(ax, 12.0, 0.95, 2.4, 0.5, "SMS to Supervisor Phone",
    color="#1e1e2e", border=RED, fontsize=7.5)

plt.tight_layout()
plt.savefig("assets/diagrams/01_full_system_pipeline.png", dpi=180,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved: 01_full_system_pipeline.png")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 2 — Web Application Architecture (3-thread detail)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis("off")

title(ax, "Web Application Architecture",
      "Streamlit + 3-Thread MJPEG Camera Stream + SMS Alert System")

# ── Streamlit sidebar ─────────────────────────────────────────────────────────
sidebar = FancyBboxPatch((0.2, 1.0), 2.4, 6.2,
                         boxstyle="round,pad=0.05",
                         facecolor="#12122a", edgecolor=BORDER, lw=1.2, zorder=2)
ax.add_patch(sidebar)
ax.text(1.4, 7.4, "SIDEBAR", ha="center", fontsize=8, color=SUBTEXT, fontweight="bold")
for sy, lbl, col in [
    (6.7, "Model path", TEXT),
    (6.2, "Confidence\nthreshold", TEXT),
    (5.5, "Class legend\n(colour badges)", TEXT),
    (4.7, "SMS toggle +\nphone number", TEXT),
    (4.0, "Model status\n(loaded / error)", GREEN),
]:
    ax.text(1.4, sy, lbl, ha="center", va="center", fontsize=7.5,
            color=col,
            bbox=dict(boxstyle="round,pad=0.25", fc=PANEL, ec=BORDER, lw=0.8))

# ── Main area: tabs ───────────────────────────────────────────────────────────
# Tab 1 — Image Upload
tab1 = FancyBboxPatch((2.9, 3.8), 4.8, 3.8,
                      boxstyle="round,pad=0.05",
                      facecolor="#1a1a2e", edgecolor=ORANGE, lw=1.5, zorder=2)
ax.add_patch(tab1)
ax.text(5.3, 7.4, "Tab 1 — Image Upload", ha="center", fontsize=9,
        color=ORANGE, fontweight="bold")

steps1 = [
    (5.3, 7.0,  "User uploads JPG/PNG"),
    (5.3, 6.55, "cv2.imdecode → BGR array"),
    (5.3, 6.1,  "SmokingDetector.predict()"),
    (5.3, 5.65, "Draw bounding boxes (cv2)"),
    (5.3, 5.2,  "Show original + annotated"),
    (5.3, 4.75, "Detection table (st.dataframe)"),
    (5.3, 4.3,  "Download annotated image"),
]
for sx, sy, sl in steps1:
    ax.text(sx, sy, sl, ha="center", va="center", fontsize=7.5,
            color=TEXT,
            bbox=dict(boxstyle="round,pad=0.2", fc=PANEL, ec=BORDER, lw=0.7))
    if sy > 4.3:
        arrow(ax, sx, sy-0.17, sx, sy-0.2, color=ORANGE, lw=1)

# Tab 2 — Live Camera
tab2 = FancyBboxPatch((8.0, 3.8), 5.8, 3.8,
                      boxstyle="round,pad=0.05",
                      facecolor="#1a1a2e", edgecolor=PURPLE, lw=1.5, zorder=2)
ax.add_patch(tab2)
ax.text(10.9, 7.4, "Tab 2 — Live Camera (3-Thread Architecture)",
        ha="center", fontsize=9, color=PURPLE, fontweight="bold")

# Thread boxes
for tx, ty, tname, tcol, tdesc in [
    (9.2,  6.2, "Thread 1\nCapture",   ORANGE,
     "cv2.VideoCapture\n30 fps\nDraw cached bboxes\nEncode JPEG"),
    (10.9, 6.2, "Thread 2\nInference", CYAN,
     "YOLO.predict()\n5–15 fps\nUpdate detection\ncache"),
    (12.6, 6.2, "Thread 3\nMJPEG Srv", GREEN,
     "HTTPServer\nlocalhost:8765\nserve_forever()\n~30 fps"),
]:
    tb = FancyBboxPatch((tx-0.75, ty-0.9), 1.5, 1.8,
                        boxstyle="round,pad=0.05",
                        facecolor=PANEL, edgecolor=tcol, lw=1.2, zorder=3)
    ax.add_patch(tb)
    ax.text(tx, ty+0.65, tname, ha="center", va="center",
            fontsize=7.5, color=tcol, fontweight="bold", zorder=4)
    ax.text(tx, ty-0.1, tdesc, ha="center", va="center",
            fontsize=6.5, color=TEXT, zorder=4)

# Shared buffers between threads
ax.text(10.05, 6.2, "raw\nframe", ha="center", va="center",
        fontsize=6.5, color=SUBTEXT,
        bbox=dict(boxstyle="round,pad=0.2", fc="#0e0e1a", ec=BORDER, lw=0.7))
ax.text(11.75, 6.2, "JPEG\nbuffer", ha="center", va="center",
        fontsize=6.5, color=SUBTEXT,
        bbox=dict(boxstyle="round,pad=0.2", fc="#0e0e1a", ec=BORDER, lw=0.7))

arrow(ax, 9.95, 6.2, 10.15, 6.2, color=SUBTEXT, lw=1)
arrow(ax, 11.65, 6.2, 11.85, 6.2, color=SUBTEXT, lw=1)

# Browser
box(ax, 10.9, 4.4, 3.5, 0.55,
    "Browser  <img src='http://localhost:8765'>",
    "MJPEG stream renders as smooth native video (no WebRTC)",
    color="#0e0e1a", border=PURPLE, fontsize=7.5)
arrow(ax, 12.6, 5.3, 12.6, 4.68, color=GREEN, lw=1.5)

# ── SMS Alert System ──────────────────────────────────────────────────────────
sms_bg = FancyBboxPatch((0.2, 0.15), 13.6, 3.4,
                        boxstyle="round,pad=0.05",
                        facecolor="#1a1a1a", edgecolor=RED, lw=1.5, zorder=2)
ax.add_patch(sms_bg)
ax.text(7.0, 3.35, "SMS ALERT SYSTEM  (utils/alerts.py + utils/camera_stream.py)",
        ha="center", fontsize=9, color=RED, fontweight="bold")

for bx, by, bl, bsub, bcol in [
    (1.5,  2.2, "Detection\nEvent",        "n_cig > 0\nper frame",          CYAN),
    (3.8,  2.2, "SlidingWindow\nTracker",  "15s rolling window\n≥50% ratio", YELLOW),
    (6.4,  2.2, "Threshold\nCrossed?",     "Returns True\nexactly once",    ORANGE),
    (8.9,  2.2, "SMSAlerter\n.send()",     "30s cooldown\nbetween alerts",  RED),
    (11.5, 2.2, "Twilio API",             "REST POST\nSMS delivered",       GREEN),
]:
    box(ax, bx, by, 1.9, 0.95, bl, bsub, color=PANEL, border=bcol,
        fontsize=8, subfontsize=7)

arrow(ax, 2.45, 2.2, 2.85, 2.2, color=CYAN,   lw=1.5)
arrow(ax, 4.75, 2.2, 5.45, 2.2, color=YELLOW, lw=1.5)
arrow(ax, 7.35, 2.2, 7.95, 2.2, color=ORANGE, lw=1.5)
arrow(ax, 9.85, 2.2, 10.55,2.2, color=RED,    lw=1.5)

# Reset path
ax.annotate("", xy=(3.8, 1.72), xytext=(6.4, 1.72),
            arrowprops=dict(arrowstyle="->", color=SUBTEXT, lw=1,
                            connectionstyle="arc3,rad=0"), zorder=2)
ax.text(5.1, 1.55, "No → keep accumulating frames", ha="center",
        fontsize=7, color=SUBTEXT)
ax.text(6.4, 1.85, "No", fontsize=7, color=SUBTEXT, ha="center")

ax.text(7.0, 0.55,
        "If cigarette disappears for >85% of frames in window → tracker resets → new event can trigger",
        ha="center", fontsize=7.5, color=SUBTEXT)

plt.tight_layout()
plt.savefig("assets/diagrams/02_webapp_architecture.png", dpi=180,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved: 02_webapp_architecture.png")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 3 — Dataset Construction Pipeline
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")

title(ax, "Dataset Construction Pipeline",
      "From raw Roboflow downloads to merged 2-class training-ready dataset")

# ── Source datasets ────────────────────────────────────────────────────────────
ax.text(7, 6.35, "SOURCE DATASETS", ha="center", fontsize=9,
        color=SUBTEXT, fontweight="bold")

sources = [
    (1.8,  5.6, "Roboflow\nCigarette Dataset", "800 images\n5 classes\nExternal source", ORANGE),
    (5.0,  5.6, "Smoking\nDataset",             "5,786 images\nCigarette only\nCollected", ORANGE),
    (8.5,  5.6, "Pen Dataset\n(v1 + v2)",       "148 + 667 = 815 imgs\nBallpoint / gel pens\nCollected", CYAN),
    (11.8, 5.6, "Straw\nDataset",               "615 images\nDrinking straws\nCollected", CYAN),
]
for sx, sy, sl, ssub, scol in sources:
    box(ax, sx, sy, 2.4, 1.1, sl, ssub, color="#1e1e2e", border=scol,
        fontsize=8.5, subfontsize=7.5)

# ── Remapping ─────────────────────────────────────────────────────────────────
ax.text(7, 4.5, "LABEL REMAPPING", ha="center", fontsize=9,
        color=SUBTEXT, fontweight="bold")

arrow(ax, 1.8,  5.05, 3.5,  4.15, color=ORANGE, lw=1.2)
arrow(ax, 5.0,  5.05, 4.5,  4.15, color=ORANGE, lw=1.2)
arrow(ax, 8.5,  5.05, 9.5,  4.15, color=CYAN,   lw=1.2)
arrow(ax, 11.8, 5.05, 10.5, 4.15, color=CYAN,   lw=1.2)

box(ax, 4.0, 3.7, 3.2, 0.75, "Cigarette Class  (0)",
    "Original cls 0, 1 remapped → 0\n5,786 + 800 = 6,586 annotated instances",
    color=PANEL, border=ORANGE, bold=True, fontsize=8.5)

box(ax, 10.0, 3.7, 3.2, 0.75, "Cigarette-like Object  (1)",
    "Original cls 2 remapped → 1\nPens + straws = 1,430 instances",
    color=PANEL, border=CYAN, bold=True, fontsize=8.5)

# ── Merge ─────────────────────────────────────────────────────────────────────
arrow(ax, 4.0,  3.32, 6.5, 2.7, color=ORANGE, lw=1.5)
arrow(ax, 10.0, 3.32, 7.5, 2.7, color=CYAN,   lw=1.5)

ax.text(7, 2.5, "MERGE + TRAIN / VALID / TEST SPLIT", ha="center",
        fontsize=9, color=SUBTEXT, fontweight="bold")

box(ax, 7, 2.05, 9.0, 0.75,
    "Merged 2-Class Dataset  (smoking_v3_2class)",
    "Train: 3,478 images · Valid: 920 images · Test: reserved\n"
    "Annotations: 5,132 cigarette + 2,296 cig-like = 7,428 total",
    color=PANEL, border=GREEN, bold=True, fontsize=10)

# ── Augmentation ──────────────────────────────────────────────────────────────
arrow(ax, 7, 1.67, 7, 1.2, color=GREEN, lw=2)
ax.text(7, 1.0, "AUGMENTATION PIPELINE (applied at training time by Ultralytics)",
        ha="center", fontsize=9, color=SUBTEXT, fontweight="bold")

augs = [
    (1.5,  0.5, "Mosaic\n(p=1.0)"),
    (3.5,  0.5, "MixUp\n(p=0.15)"),
    (5.5,  0.5, "Horizontal\nFlip"),
    (7.5,  0.5, "HSV Jitter\nH/S/V"),
    (9.5,  0.5, "Random\nErase"),
    (11.5, 0.5, "Rand\nAugment"),
]
for ax2, ay, al in augs:
    box(ax, ax2, ay, 1.6, 0.55, al, color="#1e1e2e", border=GREEN,
        fontsize=7.5)

plt.tight_layout()
plt.savefig("assets/diagrams/03_dataset_pipeline.png", dpi=180,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved: 03_dataset_pipeline.png")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 4 — Model Architecture (YOLOv11m)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis("off")

title(ax, "YOLOv11m Model Architecture",
      "20.1M parameters · 68.0 GFLOPs · Input: 640x640 · 2-class output")

# Input
box(ax, 1.0, 3.0, 1.4, 1.0, "Input", "640×640×3\nBGR image",
    color="#1e1e2e", border=BLUE, fontsize=8)
arrow(ax, 1.7, 3.0, 2.2, 3.0, color=BLUE, lw=1.5)

# Backbone
bb = FancyBboxPatch((2.3, 1.5), 4.5, 3.0, boxstyle="round,pad=0.05",
                    facecolor="#131326", edgecolor=ORANGE, lw=1.5, zorder=2)
ax.add_patch(bb)
ax.text(4.55, 4.3, "BACKBONE  (feature extraction)", ha="center",
        fontsize=8.5, color=ORANGE, fontweight="bold")

for by, bl, bsub in [
    (3.8, "Conv + BN + SiLU", "Stem: 3→32, stride 2"),
    (3.2, "C3k2 Block ×2",    "32→64, stride 2"),
    (2.6, "C3k2 Block ×2",    "64→128, stride 2  [P3]"),
    (2.0, "C3k2 Block ×2",    "128→256, stride 2 [P4]"),
    (1.65,"C2PSA + SPPF",     "256→512, stride 2 [P5]"),
]:
    box(ax, 4.55, by, 4.0, 0.42, bl, bsub, color=PANEL, border=ORANGE,
        fontsize=7.5, subfontsize=6.5)
    if by > 1.65:
        arrow(ax, 4.55, by-0.21, 4.55, by-0.3, color=ORANGE, lw=1)

arrow(ax, 6.8, 3.0, 7.3, 3.0, color=ORANGE, lw=1.5)

# Neck
neck = FancyBboxPatch((7.4, 1.5), 3.8, 3.0, boxstyle="round,pad=0.05",
                      facecolor="#131326", edgecolor=CYAN, lw=1.5, zorder=2)
ax.add_patch(neck)
ax.text(9.3, 4.3, "NECK  (feature fusion)", ha="center",
        fontsize=8.5, color=CYAN, fontweight="bold")

for ny, nl, ns in [
    (3.8, "FPN Upsample", "P5 → P4 scale"),
    (3.2, "C3k2 Fusion",  "P4 + P3 concat"),
    (2.6, "PAN Downsample","P3 → P4"),
    (2.0, "C3k2 Fusion",  "P4 + P5 concat"),
    (1.65,"Multi-scale",  "P3, P4, P5 outputs"),
]:
    box(ax, 9.3, ny, 3.4, 0.42, nl, ns, color=PANEL, border=CYAN,
        fontsize=7.5, subfontsize=6.5)
    if ny > 1.65:
        arrow(ax, 9.3, ny-0.21, 9.3, ny-0.3, color=CYAN, lw=1)

arrow(ax, 11.2, 3.0, 11.7, 3.0, color=CYAN, lw=1.5)

# Head
head = FancyBboxPatch((11.8, 1.5), 2.8, 3.0, boxstyle="round,pad=0.05",
                      facecolor="#131326", edgecolor=GREEN, lw=1.5, zorder=2)
ax.add_patch(head)
ax.text(13.2, 4.3, "HEAD  (detection)", ha="center",
        fontsize=8.5, color=GREEN, fontweight="bold")

for hy, hl, hs in [
    (3.8, "Detect Layer", "Decoupled head"),
    (3.2, "Box Regressor","xyxy + DFL"),
    (2.6, "Cls Classifier","2-class softmax"),
    (2.0, "NMS Filter",   "IoU=0.7, conf=0.40"),
    (1.65,"Output Boxes", "B × [x,y,w,h,cls]"),
]:
    box(ax, 13.2, hy, 2.4, 0.42, hl, hs, color=PANEL, border=GREEN,
        fontsize=7.5, subfontsize=6.5)
    if hy > 1.65:
        arrow(ax, 13.2, hy-0.21, 13.2, hy-0.3, color=GREEN, lw=1)

arrow(ax, 14.6, 3.0, 15.1, 3.0, color=GREEN, lw=1.5)
box(ax, 15.5, 3.0, 0.8, 1.0, "Output", "cigarette\ncig-like",
    color="#1e1e2e", border=GREEN, fontsize=7.5)

# Training config strip
strip = FancyBboxPatch((0.2, 0.1), 15.6, 0.8, boxstyle="round,pad=0.05",
                       facecolor="#131326", edgecolor=BORDER, lw=1, zorder=2)
ax.add_patch(strip)
cfg_items = [
    ("Loss", "BoxLoss + ClsLoss (w=1.5) + DFLoss"),
    ("Optimizer", "AdamW  lr=1e-3 → 1e-5 (cosine)"),
    ("Warmup", "5 epochs"),
    ("Early stop", "patience=10"),
    ("AMP", "Mixed precision FP16"),
]
for i, (ck, cv) in enumerate(cfg_items):
    x = 1.7 + i * 3.1
    ax.text(x, 0.62, ck, ha="center", fontsize=7, color=SUBTEXT, fontweight="bold")
    ax.text(x, 0.38, cv, ha="center", fontsize=6.5, color=TEXT)

plt.tight_layout()
plt.savefig("assets/diagrams/04_model_architecture.png", dpi=180,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved: 04_model_architecture.png")

print("\nAll diagrams saved to assets/diagrams/")
print("  01_full_system_pipeline.png")
print("  02_webapp_architecture.png")
print("  03_dataset_pipeline.png")
print("  04_model_architecture.png")
