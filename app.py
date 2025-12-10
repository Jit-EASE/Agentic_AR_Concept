import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from streamlit_autorefresh import st_autorefresh
import av
import cv2
import numpy as np

from backend.detector import Detector
from backend.agentic_reasoner import AgenticReasoner

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="AFI-AR – Agentic Farm Intelligence AR System",
    layout="wide"
)

# ----------------------------------------------------
# CSS Theme
# ----------------------------------------------------
with open("ui/spectre_theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    "<h1 class='spectre-title'>Agentic Farm Intelligence AR System - Ireland</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p class='spectre-subtitle'>
    System Architecture, Design and Engineering - Shubhojit Bagchi © 2025
    
    Real-time augmented intelligence system for field scanning, sustainability insights,
    supply-chain mapping, econometric reasoning, and LPIS awareness.  
    Concept Designed for Irish Agri-Food Research
    </p>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
st.sidebar.header("System Controls")
run_reasoner = st.sidebar.checkbox("Enable Agentic AI", value=False)

max_fps = st.sidebar.slider("Processing FPS", 1, 15, 5)
confidence_threshold = st.sidebar.slider("Detection Confidence", 0.2, 0.9, 0.5, 0.05)

camera_choice = st.sidebar.selectbox("Camera Source", ["rear (recommended)", "front"])
facing_mode = "environment" if camera_choice == "rear (recommended)" else "user"

st_autorefresh(interval=1500, key="intel_refresh")

# ----------------------------------------------------
# INIT MODELS
# ----------------------------------------------------
detector = Detector(confidence_threshold=confidence_threshold)
reasoner = AgenticReasoner(enabled=run_reasoner)

# ----------------------------------------------------
# VIDEO PROCESSOR
# ----------------------------------------------------
class ARVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_label = None
        self.last_box = None
        self.last_agents = None

        # Professional HUD font settings
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale = 0.55
        self.font_thickness = 1
        self.text_color = (255, 255, 255)      # white
        self.box_bg = (18, 20, 27)             # subtle dark grey
        self.box_border = (180, 180, 180)      # light grey border
        self.box_thickness = 1

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        process_interval = max(1, int(15 / max_fps))

        if self.frame_count % process_interval == 0:
            label, box = detector.detect_single(img)

            if label:
                self.last_label = label
                self.last_box = box
                self.last_agents = reasoner.explain_structured(label) if run_reasoner else None
            else:
                self.last_label = None
                self.last_box = None
                self.last_agents = None

                # ----------------------------------------------------
        # DRAW PROFESSIONAL, COMPACT MOBILE-FRIENDLY HUD
        # ----------------------------------------------------
        if self.last_label and self.last_box:
            x1, y1, x2, y2 = self.last_box

            # bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), self.box_border, 2)

            # ---------- TEXT WRAPPING FUNCTION ----------
            def wrap(text, max_len=38):
                words = text.split()
                lines = []
                line = ""
                for w in words:
                    if len(line) + len(w) + 1 <= max_len:
                        line += (" " + w) if line else w
                    else:
                        lines.append(line)
                        line = w
                if line:
                    lines.append(line)
                return lines

            # ---------- BUILD LINES ----------
            final_lines = []
            final_lines.append(self.last_label.upper())

            if self.last_agents:
                sust = self.last_agents.get("SUSTAINABILITY") or ""
                econ = self.last_agents.get("ECONOMETRICS") or ""
                haz  = self.last_agents.get("HAZARD") or ""
                sc   = self.last_agents.get("SUPPLY_CHAIN") or ""
                lpis = self.last_agents.get("LPIS_GEO") or ""

                for ln in wrap("SUST ▸ " + sust): final_lines.append(ln)
                for ln in wrap("ECON ▸ " + econ): final_lines.append(ln)
                for ln in wrap("HAZ ▸ "  + haz):  final_lines.append(ln)
                for ln in wrap("SCM ▸ "  + sc):   final_lines.append(ln)
                for ln in wrap("LPIS ▸ " + lpis): final_lines.append(ln)

            # ---------- FONT + COMPACT METRICS ----------
            font = self.font
            scale = 0.45        # Smaller for AR mobile
            thick = 1
            line_h = 20         # fixed-height per text line
            max_width = 320     # FORCE HUD WIDTH (IMPORTANT)

            # Measure actual max width
            w_list = [
                cv2.getTextSize(t, font, scale, thick)[0][0]
                for t in final_lines
            ]
            text_w = min(max(w_list), max_width)
            total_h = line_h * len(final_lines) + 10

            # HUD position (above box)
            hud_x = x1
            hud_y = max(0, y1 - total_h - 10)

            # ---------- DRAW BACKGROUND ----------
            cv2.rectangle(
                img,
                (hud_x, hud_y),
                (hud_x + text_w + 22, hud_y + total_h),
                self.box_bg,
                -1
            )

            cv2.rectangle(
                img,
                (hud_x, hud_y),
                (hud_x + text_w + 22, hud_y + total_h),
                self.box_border,
                1
            )

            # ---------- DRAW ALL TEXT LINES ----------
            y_text = hud_y + 18
            for line in final_lines:
                cv2.putText(
                    img,
                    line,
                    (hud_x + 10, y_text),
                    font,
                    scale,
                    self.text_color,
                    thick,
                    cv2.LINE_AA,
                )
                y_text += line_h

# ----------------------------------------------------
# PAGE LAYOUT
# ----------------------------------------------------
col_video, col_panel = st.columns([2.3, 1])

with col_video:
    st.markdown(
        "<div class='spectre-panel'><b>Instructions:</b> Rear camera is used by default on iPhone. "
        "Point at any object to receive multi-agent insights.</div>",
        unsafe_allow_html=True,
    )

    webrtc_ctx = webrtc_streamer(
        key="afi-ar-system-v2",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=ARVideoProcessor,
        media_stream_constraints={
            "video": {"facingMode": {"exact": facing_mode}},
            "audio": False
        },
        async_processing=True,
    )

# ----------------------------------------------------
# INTELLIGENCE PANEL
# ----------------------------------------------------
with col_panel:
    st.markdown("Intelligence Panel", unsafe_allow_html=True)
    panel = st.empty()

    intel = None

    if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.video_processor:
        vp = webrtc_ctx.video_processor
        if vp.last_label and vp.last_agents:
            intel = {
                "Object": vp.last_label,
                "Sust": vp.last_agents.get("SUSTAINABILITY"),
                "SC": vp.last_agents.get("SUPPLY_CHAIN"),
                "Econ": vp.last_agents.get("ECONOMETRICS"),
                "Haz": vp.last_agents.get("HAZARD"),
                "LPIS": vp.last_agents.get("LPIS_GEO"),
            }

    if intel:
        panel.markdown(
            f"""
            <div class='spectre-panel'>
              <b>Detected:</b> {intel["Object"]}<br><br>
              <b>Sustainability</b><br>{intel["Sust"]}<br><br>
              <b>Supply Chain</b><br>{intel["SC"]}<br><br>
              <b>Econometrics</b><br>{intel["Econ"]}<br><br>
              <b>Hazard</b><br>{intel["Haz"]}<br><br>
              <b>LPIS / GEO</b><br>{intel["LPIS"]}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        panel.markdown(
            "<div class='spectre-panel'>No object detected yet. Hold the camera steady.</div>",
            unsafe_allow_html=True,
        )
