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
    Designed for professional agricultural and MIS research environments.
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
        # DRAW PROFESSIONAL HUD
        # ----------------------------------------------------
        if self.last_label and self.last_box:
            x1, y1, x2, y2 = self.last_box

            # bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), self.box_border, 2)

            # HUD TEXT LINES
            lines = [self.last_label]

            if self.last_agents:
                clip = lambda s: s[:60] + "…" if len(s) > 60 else s
                sust = self.last_agents.get("SUSTAINABILITY") or ""
                econ = self.last_agents.get("ECONOMETRICS") or ""
                haz  = self.last_agents.get("HAZARD") or ""
                sc   = self.last_agents.get("SUPPLY_CHAIN") or ""
                lpis = self.last_agents.get("LPIS_GEO") or ""

                if sust: lines.append("SUST: " + clip(sust))
                if econ: lines.append("ECON: " + clip(econ))
                if haz:  lines.append("HAZ:  " + clip(haz))
                if sc:   lines.append("SC:   " + clip(sc))
                if lpis: lines.append("LPIS: " + clip(lpis))

            # TEXT MEASUREMENT
            sizes = [cv2.getTextSize(l, self.font, self.font_scale, self.font_thickness)[0] for l in lines]
            max_w = max(w for (w, h) in sizes)
            line_h = max(h for (w, h) in sizes) + 6
            total_h = line_h * len(lines) + 8

            box_top = max(0, y1 - total_h - 10)
            box_left = x1

            # BACKGROUND BOX
            cv2.rectangle(
                img,
                (box_left, box_top),
                (box_left + max_w + 20, box_top + total_h),
                self.box_bg,
                -1
            )

            # BORDER
            cv2.rectangle(
                img,
                (box_left, box_top),
                (box_left + max_w + 20, box_top + total_h),
                self.box_border,
                self.box_thickness
            )

            # DRAW TEXT
            y_text = box_top + 20
            for line in lines:
                cv2.putText(
                    img,
                    line,
                    (box_left + 10, y_text),
                    self.font,
                    self.font_scale,
                    self.text_color,
                    self.font_thickness,
                    cv2.LINE_AA
                )
                y_text += line_h

        return av.VideoFrame.from_ndarray(img, format="bgr24")


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
