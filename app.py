import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
from backend.detector import Detector
from backend.agentic_reasoner import AgenticReasoner

st.set_page_config(
    page_title="Spectre Agentic AR – Irish Object Lens",
    layout="wide"
)

# --- Spectre UI Styling ---
with open("ui/spectre_theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    "<h1 class='spectre-title'>Spectre Agentic AR – Irish Object Lens</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class='spectre-subtitle'>
    Point your iPhone or webcam at any object. The app will detect it, classify it,
    and overlay an explanation pop-up above the object in real time.
    </p>
    """,
    unsafe_allow_html=True
)

# Sidebar controls
st.sidebar.header("Agentic AR Controls")
run_reasoner = st.sidebar.checkbox("Enable Agentic Reasoner (GPT-4o-mini)", value=False)
max_fps = st.sidebar.slider("Processing FPS", 1, 15, 5)
confidence_threshold = st.sidebar.slider("Detection confidence", 0.2, 0.9, 0.5, 0.05)

detector = Detector(confidence_threshold=confidence_threshold)
reasoner = AgenticReasoner(enabled=run_reasoner)


class ARVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_label = None
        self.last_box = None
        self.last_explanation = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Only process every Nth frame to control API / CPU load
        process_interval = max(1, int(15 / max_fps))

        if self.frame_count % process_interval == 0:
            label, box = detector.detect_single(img)
            if label is not None:
                self.last_label = label
                self.last_box = box

                if run_reasoner:
                    self.last_explanation = reasoner.explain(label)
            else:
                self.last_label = None
                self.last_box = None
                self.last_explanation = None

        # Draw overlay if we have a recent detection
        if self.last_label and self.last_box is not None:
            x1, y1, x2, y2 = self.last_box

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Prepare text block
            lines = [self.last_label]
            if self.last_explanation:
                # Keep only short snippet for overlay
                snippet = self.last_explanation.split(".")[0][:140]
                lines.append(snippet)

            text = " | ".join(lines)

            # Put a filled rectangle above the box
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_width, text_height = text_size
            box_top = max(0, y1 - text_height - 10)

            cv2.rectangle(
                img,
                (x1, box_top),
                (x1 + text_width + 10, box_top + text_height + 8),
                (0, 0, 0),
                thickness=-1,
            )

            cv2.putText(
                img,
                text,
                (x1 + 5, box_top + text_height + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.markdown(
    "<div class='spectre-panel'><b>Instructions:</b> On iPhone, open this app in Safari, "
    "give camera permission, and point it at objects around you. The overlay will appear "
    "as small squares with captions above the detected object.</div>",
    unsafe_allow_html=True,
)

webrtc_streamer(
    key="spectre-agentic-ar",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=ARVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
