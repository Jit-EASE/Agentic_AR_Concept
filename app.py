import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from streamlit_autorefresh import st_autorefresh
import av
import cv2
import numpy as np
from backend.detector import Detector
from backend.agentic_reasoner import AgenticReasoner

st.set_page_config(
    page_title="Spectre Agentic AR – Irish Object Lens (v2)",
    layout="wide"
)

# --- Spectre UI Styling ---
with open("ui/spectre_theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    "<h1 class='spectre-title'>Agentic Farm Intelligence - Augmented Reality System - Irish Agri Food Research</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class='spectre-subtitle'>
    ***System Architecture, Design and Engineering - Shubhojit Bagchi © 2025***
    
    Live multi-agent AR: point your camera at an object and see sustainability, supply chain,
    econometric, hazard, and LPIS insights overlaid and in the side intelligence panel.
    Rear camera is used by default on iPhone.
    </p>
    """,
    unsafe_allow_html=True
)

# Sidebar controls
st.sidebar.header("Agentic AR Controls")
run_reasoner = st.sidebar.checkbox("Enable Agentic Reasoner", value=False)
max_fps = st.sidebar.slider("Processing FPS", 1, 15, 5)
confidence_threshold = st.sidebar.slider("Detection confidence", 0.2, 0.9, 0.5, 0.05)

# Camera toggle UI
camera_choice = st.sidebar.selectbox(
    "Camera Source",
    ["rear (recommended)", "front"]
)

# Translate UI choice to WebRTC facingMode
facing_mode = "environment" if camera_choice == "rear (recommended)" else "user"

# Auto refresh for panel updates
st_autorefresh(interval=1500, key="intel_refresh")

# Init detector + reasoning engine
detector = Detector(confidence_threshold=confidence_threshold)
reasoner = AgenticReasoner(enabled=run_reasoner)


class ARVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_label = None
        self.last_box = None
        self.last_agents = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        process_interval = max(1, int(15 / max_fps))

        if self.frame_count % process_interval == 0:
            label, box = detector.detect_single(img)
            if label is not None:
                self.last_label = label
                self.last_box = box

                if run_reasoner:
                    self.last_agents = reasoner.explain_structured(label)
                else:
                    self.last_agents = None
            else:
                self.last_label = None
                self.last_box = None
                self.last_agents = None

        # Draw multi-line HUD
        if self.last_label and self.last_box is not None:
            x1, y1, x2, y2 = self.last_box

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # HUD lines
            lines = [self.last_label]

            if self.last_agents:
                def clip(s, n=60):
                    return (s[:n] + "…") if len(s) > n else s

                sust = self.last_agents.get("SUSTAINABILITY") or ""
                econ = self.last_agents.get("ECONOMETRICS") or ""
                hazard = self.last_agents.get("HAZARD") or ""
                sc = self.last_agents.get("SUPPLY_CHAIN") or ""
                lpis = self.last_agents.get("LPIS_GEO") or ""

                if sust: lines.append("Sust: " + clip(sust))
                if econ: lines.append("Econ: " + clip(econ))
                if hazard: lines.append("Haz: " + clip(hazard))
                if sc: lines.append("SC: " + clip(sc))
                if lpis: lines.append("LPIS: " + clip(lpis))

            # Measure box size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            text_sizes = [cv2.getTextSize(l, font, font_scale, thickness)[0] for l in lines]
            max_width = max(w for (w, h) in text_sizes)
            line_height = max(h for (w, h) in text_sizes) + 6
            total_height = line_height * len(lines) + 6

            box_top = max(0, y1 - total_height - 10)
            box_left = x1

            cv2.rectangle(
                img,
                (box_left, box_top),
                (box_left + max_width + 16, box_top + total_height),
                (0, 0, 0),
                thickness=-1,
            )

            cv2.rectangle(
                img,
                (box_left, box_top),
                (box_left + max_width + 16, box_top + total_height),
                (0, 255, 255),
                thickness=1,
            )

            # Draw lines
            y_text = box_top + 4 + line_height - 4
            for line, (tw, th) in zip(lines, text_sizes):
                cv2.putText(
                    img,
                    line,
                    (box_left + 8, y_text),
                    font,
                    font_scale,
                    (0, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )
                y_text += line_height

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Layout: video left, panel right
col_video, col_panel = st.columns([2.3, 1])

with col_video:
    st.markdown(
        "<div class='spectre-panel'><b>Instructions:</b> On iPhone, rear camera will be "
        "used by default. You can switch cameras from the sidebar.</div>",
        unsafe_allow_html=True,
    )

    webrtc_ctx = webrtc_streamer(
        key="spectre-agentic-ar-v2",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=ARVideoProcessor,
        media_stream_constraints={
            "video": {"facingMode": {"exact": facing_mode}},  # 🔥 THIS IS THE IMPORTANT PART
            "audio": False
        },
        async_processing=True,
    )

with col_panel:
    st.markdown("### Intelligence Panel")
    panel = st.empty()

    intel = None
    if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.video_processor:
        vp = webrtc_ctx.video_processor
        if vp.last_label and vp.last_agents:
            intel = {
                "Object": vp.last_label,
                "Sustainability": vp.last_agents.get("SUSTAINABILITY"),
                "Supply Chain": vp.last_agents.get("SUPPLY_CHAIN"),
                "Econometrics": vp.last_agents.get("ECONOMETRICS"),
                "Hazard": vp.last_agents.get("HAZARD"),
                "LPIS/GEO": vp.last_agents.get("LPIS_GEO"),
            }

    if intel:
        panel.markdown(
            f"""
            <div class='spectre-panel'>
              <div><b>Detected object:</b> {intel["Object"]}</div>
              <hr style="border: 0; border-top: 1px solid #1b2436; margin: 0.5rem 0;" />
              <div><b> Sustainability</b><br>{intel["Sustainability"]}</div>
              <div style="margin-top:0.5rem;"><b>📦 Supply chain</b><br>{intel["Supply Chain"]}</div>
              <div style="margin-top:0.5rem;"><b>🧮 Econometrics</b><br>{intel["Econometrics"]}</div>
              <div style="margin-top:0.5rem;"><b>🔍 Hazard</b><br>{intel["Hazard"]}</div>
              <div style="margin-top:0.5rem;"><b>📡 LPIS / Geo</b><br>{intel["LPIS/GEO"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        panel.markdown(
            "<div class='spectre-panel'>No object locked yet. "
            "Hold the camera steady toward the object.</div>",
            unsafe_allow_html=True,
        )
