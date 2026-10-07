"""
BACKEND API v2.1 - three ways to tell the camera what to detect:
  1. Text prompt   - "person, forklift, helmet"
  2. Voice prompt  - handled in the browser (Lovable), which transcribes
                     speech to text and sends it here as a normal text prompt.
  3. Visual prompt - upload a photo of a product; the camera then looks for
                     that specific product instead of a word.

WHAT CHANGED IN v2.1:
  - Model swapped from yoloe-11s-seg.pt to yoloe-26n-seg.pt (YOLOE-26 nano).
    Same YOLOE class and same set_classes() / visual-prompt API.
  - generate_frames() rewritten to be safer:
      * opens the webcam with DirectShow on Windows (CAP_DSHOW)
      * always releases the camera when the browser tab closes
      * gives up cleanly if the camera stops delivering frames
      * prints how long each detection takes, so slowness is visible
      * conf=0.1 so weak detections (like a phone) are not hidden
      * masks=False so only boxes are drawn, not blue segmentation overlays

HOW TO RUN:
  1. Copy .env.example to .env
  2. pip install -r requirements.txt
  3. uvicorn main:app --host 0.0.0.0 --port 8000
     (add --reload only while editing code; reload resets the prompt and
      can leave the webcam locked)

NOTE: the active prompt resets to ["person"] every time the server starts.
"""

import os
import time
import tempfile
import threading
from collections import deque
from datetime import datetime, timezone

import cv2
import httpx
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

load_dotenv()

VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")
VIDEO_SOURCE = int(VIDEO_SOURCE) if VIDEO_SOURCE.isdigit() else VIDEO_SOURCE
CAMERA_ID = os.getenv("CAMERA_ID", "camera-1")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Detection settings - easy to tweak in one place
IMG_SIZE = 480          # smaller = faster, bigger = more accurate
CONF_THRESHOLD = 0.1    # lower shows weaker detections (more false boxes)
DETECT_EVERY_N = 3      # run the model on every Nth frame

app = FastAPI(title="Prompt-Based Detection API v2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading YOLOE model (this happens once, at server startup)...")
# "26n" = YOLOE-26 nano, the smallest and fastest. Try "yoloe-26s-seg.pt"
# for better accuracy if your machine can handle it.
model = YOLOE("yoloe-26n-seg.pt")
current_prompts = ["person"]
current_mode = "text"   # "text" or "visual" - dashboard can display this
try:
    model.set_classes(current_prompts)
except Exception as e:
    print(f"WARNING: couldn't set initial prompt yet ({e}). "
          f"Server is still starting - fix network access and call /api/set_prompt.")

recent_detections = deque(maxlen=100)
prompts_lock = threading.Lock()


class PromptRequest(BaseModel):
    camera_id: str = CAMERA_ID
    prompts: list[str]


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "current_prompts": current_prompts,
        "mode": current_mode,
        "camera_id": CAMERA_ID,
    }


@app.post("/api/set_prompt")
def set_prompt(req: PromptRequest):
    """TEXT prompt (also used by voice prompts, after the browser
    transcribes speech to text)."""
    global current_prompts, current_mode
    with prompts_lock:
        current_prompts = req.prompts
        model.set_classes(current_prompts)
        current_mode = "text"
    return {"status": "updated", "mode": current_mode, "prompts": current_prompts}


@app.post("/api/set_visual_prompt")
async def set_visual_prompt(
    camera_id: str = Form(default=CAMERA_ID),
    file: UploadFile = File(...),
):
    """VISUAL prompt - upload a clean, cropped photo of a single product.
    The whole image is treated as the bounding box, since the user already
    cropped it to just the product."""
    global current_prompts, current_mode

    contents = await file.read()
    tmp_path = os.path.join(tempfile.gettempdir(), f"product_ref_{camera_id}.jpg")
    with open(tmp_path, "wb") as f:
        f.write(contents)

    img = Image.open(tmp_path)
    w, h = img.size
    visual_prompts = {
        "bboxes": np.array([[0, 0, w, h]]),
        "cls": np.array([0]),
    }

    with prompts_lock:
        # This call both (a) runs a detection on the reference image itself,
        # and (b) bakes this visual class into the model, so every later
        # model.predict(frame) call in the live stream looks for this product.
        model.predict(
            tmp_path,
            refer_image=tmp_path,
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
        )
        current_prompts = [f"product photo: {file.filename}"]
        current_mode = "visual"

    return {"status": "updated", "mode": current_mode, "prompts": current_prompts}


def push_to_supabase(label: str, confidence: float, prompt_used: list[str]):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        httpx.post(
            f"{SUPABASE_URL}/rest/v1/detections",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "camera_id": CAMERA_ID,
                "label": label,
                "confidence": confidence,
                "prompt_used": ",".join(prompt_used),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            timeout=2.0,
        )
    except Exception as e:
        print(f"Supabase push failed (non-fatal, continuing): {e}")


def generate_frames():
    # DirectShow backend is more reliable for webcams on Windows
    if isinstance(VIDEO_SOURCE, int):
        cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(VIDEO_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

    frame_counter = 0
    last_annotated = None
    failures = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures > 20:
                    print("Camera stopped giving frames, closing stream.")
                    break
                time.sleep(0.1)
                continue
            failures = 0

            frame_counter += 1
            if frame_counter % DETECT_EVERY_N == 0 or last_annotated is None:
                t0 = time.time()
                with prompts_lock:
                    # Same call for text and visual mode.
                    results = model.predict(
                        frame, verbose=False, imgsz=IMG_SIZE, conf=CONF_THRESHOLD
                    )
                print(f"detection took {time.time() - t0:.2f}s")

                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    label = (
                        model.names[cls_id]
                        if cls_id in model.names
                        else current_prompts[0]
                    )
                    confidence = float(box.conf[0])
                    recent_detections.append({
                        "label": label,
                        "confidence": round(confidence, 2),
                        "time": datetime.now(timezone.utc).isoformat(),
                    })
                    push_to_supabase(label, confidence, current_prompts)

                last_annotated = results[0].plot(masks=False)

            ok, buffer = cv2.imencode(".jpg", last_annotated)
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        # Always free the webcam when the browser tab closes
        cap.release()


@app.get("/api/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/detections/recent")
def recent():
    return {"detections": list(recent_detections)}