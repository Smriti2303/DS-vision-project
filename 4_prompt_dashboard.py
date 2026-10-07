"""
STEP 4 — Wrap detection in a simple web dashboard with a live prompt box.

WHAT THIS DOES:
  A small web page where someone can TYPE what to detect (e.g. "person
  wearing glasses") into a text box, hit Enter, and immediately see the
  video re-annotated for that new prompt — no code editing, no restart.
  This is the actual "prompt-based dashboard" your director asked for.

HOW TO RUN:
  streamlit run 4_prompt_dashboard.py
  This opens a browser tab automatically (usually http://localhost:8501)

HOW IT WORKS UNDER THE HOOD:
  - Streamlit re-runs this script top-to-bottom every time you interact
    with the page, but @st.cache_resource keeps the model loaded in memory
    so it doesn't reload from scratch on every prompt change (that would
    be slow).
  - Every time you submit a new prompt, model.set_classes() is called
    again with your new text — this is the same "no retraining" mechanic
    from Steps 1-3, just wired up to a text input instead of hardcoded.

NEXT STEP AFTER THIS WORKS:
  Change VIDEO_SOURCE below from 0 (webcam) to your RTSP URL once you've
  confirmed Step 3 works — same one-line swap as before.
"""

import streamlit as st
import cv2
from ultralytics import YOLOWorld

VIDEO_SOURCE = 0   # 0 = webcam. Swap for an RTSP URL string once ready.

st.set_page_config(page_title="Prompt-Based Detection", layout="wide")
st.title("Prompt-Based Object Detection")
st.caption("Type what you want to detect below — no retraining needed.")


@st.cache_resource
def load_model():
    """Loads the model once and keeps it in memory across reruns."""
    return YOLOWorld("yolov8s-world.pt")


model = load_model()

# --- The prompt box ---
prompt_text = st.text_input(
    "What should the camera look for? (comma-separated for multiple things)",
    value="person, phone, cup",
)
prompts = [p.strip() for p in prompt_text.split(",") if p.strip()]

if prompts:
    model.set_classes(prompts)   # <-- re-prompts the model live, no retraining

run = st.checkbox("Start camera", value=False)
frame_placeholder = st.empty()
log_placeholder = st.sidebar.empty()

if run:
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    detection_log = []

    while run:
        ok, frame = cap.read()
        if not ok:
            st.error("Couldn't read from the video source. Check VIDEO_SOURCE / RTSP URL.")
            break

        results = model.predict(frame, verbose=False)
        annotated = results[0].plot()   # draws boxes + labels on the frame
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(annotated_rgb, channels="RGB")

        # Keep a short rolling log of what's been detected, for the sidebar
        for box in results[0].boxes:
            label = model.names[int(box.cls[0])]
            if label not in detection_log[-5:]:
                detection_log.append(label)
        log_placeholder.write("Recent detections:\n" + "\n".join(detection_log[-10:]))

    cap.release()
else:
    st.info("Check 'Start camera' above to begin.")
