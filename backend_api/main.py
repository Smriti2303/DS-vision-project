"""
BACKEND API v3 - works with ANY camera source:
  VIDEO_SOURCE=0                      laptop webcam
  VIDEO_SOURCE=1  (or 2, 3 ...)       USB camera, e.g. the ESP32-S3 board in USB-webcam mode
  VIDEO_SOURCE=http://<ip>:81/stream  ESP32-S3 board streaming over Wi-Fi (MJPEG)
  VIDEO_SOURCE=C:\\path\\video.mp4     a recorded video file (backup for demos)

WHAT CHANGED IN v3 (same endpoints, so the dashboard keeps working):
  - ONE background thread reads the camera and always keeps the NEWEST frame.
    This removes the lag you get from network streams, and opening the video
    in two browser tabs no longer fights over the camera.
  - If the camera stops, the thread reconnects by itself.
  - If no frames arrive, the video shows a "no camera frames" message
    instead of hanging forever.
  - Boxes are drawn on the live frame, so the video stays smooth between
    detections.
  - Supabase logging is limited to once every 2 seconds per label.
  - New endpoint: GET /api/camera_status  (is the camera alive, and its fps)

Optional settings in backend_api\\.env :
  VIDEO_SOURCE=0
  CAM_WIDTH=640
  CAM_HEIGHT=480
  CAM_FOURCC=MJPG        (only if a USB camera is slow; leave empty otherwise)

HOW TO RUN:
  uvicorn main:app --host 0.0.0.0 --port 8000
NOTE: the active prompt resets to ["person"] every time the server starts.
"""

import os
import time
import tempfile
import threading
from collections import deque
from contextlib import asynccontextmanager
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

_src = os.getenv("VIDEO_SOURCE", "0").strip()
VIDEO_SOURCE = int(_src) if _src.isdigit() else _src
CAM_WIDTH = int(os.getenv("CAM_WIDTH", "640"))
CAM_HEIGHT = int(os.getenv("CAM_HEIGHT", "480"))
CAM_FOURCC = os.getenv("CAM_FOURCC", "").strip()
CAMERA_ID = os.getenv("CAMERA_ID", "camera-1")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Detection settings - easy to tweak in one place
IMG_SIZE = 480          # smaller = faster, bigger = more accurate
CONF_THRESHOLD = 0.1    # lower shows weaker detections (more false boxes)
DETECT_EVERY_N = 3      # run the model on every Nth new frame


# ---------------------------------------------------------------------------
# Camera reader: one thread, always keeps the newest frame
# ---------------------------------------------------------------------------
class FrameGrabber:
    def __init__(self, source):
        self.source = source
        self.cap = None
        self.frame = None
        self.seq = 0
        self.fps = 0.0
        self.last_frame_time = 0.0
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        # A recorded video file would otherwise play far too fast
        self.is_file = isinstance(source, str) and os.path.isfile(source)

    def _open(self):
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.source)
        if CAM_FOURCC and len(CAM_FOURCC) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAM_FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep latency low
        return cap

    def _run(self):
        failures = 0
        count = 0
        t0 = time.time()
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                print(f"Opening video source: {self.source}")
                self.cap = self._open()
                if not self.cap.isOpened():
                    print("Could not open the video source, retrying in 1 second...")
                    self.cap = None
                    time.sleep(1.0)
                    continue
            ok, frame = self.cap.read()
            if not ok or frame is None:
                failures += 1
                if failures > 30:
                    print("Camera stopped giving frames - reconnecting...")
                    self.cap.release()
                    self.cap = None
                    failures = 0
                time.sleep(0.05)
                continue
            failures = 0
            now = time.time()
            with self.lock:
                self.frame = frame
                self.seq += 1
                self.last_frame_time = now
            count += 1
            if self.is_file:
                time.sleep(1.0 / 30.0)    # play recorded videos at normal speed
            if now - t0 >= 1.0:
                self.fps = count / (now - t0)
                count = 0
                t0 = now
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def get(self):
        with self.lock:
            if self.frame is None:
                return None, self.seq
            return self.frame.copy(), self.seq


grabber = FrameGrabber(VIDEO_SOURCE)


@asynccontextmanager
async def lifespan(app):
    grabber.start()
    yield
    grabber.stop()


app = FastAPI(title="Prompt-Based Detection API v3", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading YOLOE model (this happens once, at server startup)...")
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
last_push = {}


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


@app.get("/api/camera_status")
def camera_status():
    age = time.time() - grabber.last_frame_time if grabber.last_frame_time else None
    return {
        "source": str(VIDEO_SOURCE),
        "reader_running": grabber.running,
        "has_frame": grabber.frame is not None,
        "seconds_since_last_frame": round(age, 2) if age is not None else None,
        "camera_fps": round(grabber.fps, 1),
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


def log_detection(label: str, confidence: float):
    recent_detections.append({
        "label": label,
        "confidence": round(confidence, 2),
        "time": datetime.now(timezone.utc).isoformat(),
    })
    now = time.time()
    if now - last_push.get(label, 0) > 2.0:      # at most once per 2 s per label
        last_push[label] = now
        push_to_supabase(label, confidence, current_prompts)


def message_frame(text: str):
    """A plain picture with a message, shown when no camera frames arrive."""
    img = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, CAM_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else b""


def mjpeg_chunk(jpeg_bytes: bytes) -> bytes:
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
    )


def generate_frames():
    grabber.start()
    last_seq = -1
    frame_counter = 0
    last_result = None
    started = time.time()
    last_msg = 0.0

    while True:
        frame, seq = grabber.get()

        if frame is None:
            # No camera frames yet: show a message instead of hanging
            if time.time() - started > 3 and time.time() - last_msg > 1.0:
                last_msg = time.time()
                yield mjpeg_chunk(message_frame("Waiting for camera frames..."))
            time.sleep(0.05)
            continue

        if seq == last_seq:
            # Camera has no new frame yet; if it has been silent too long, say so
            if time.time() - grabber.last_frame_time > 3 and time.time() - last_msg > 1.0:
                last_msg = time.time()
                yield mjpeg_chunk(message_frame("Camera stopped - reconnecting..."))
            time.sleep(0.005)
            continue

        last_seq = seq
        frame_counter += 1

        if frame_counter % DETECT_EVERY_N == 0 or last_result is None:
            t0 = time.time()
            with prompts_lock:
                results = model.predict(
                    frame, verbose=False, imgsz=IMG_SIZE, conf=CONF_THRESHOLD
                )
            print(f"detection took {time.time() - t0:.2f}s")
            last_result = results[0]

            for box in last_result.boxes:
                cls_id = int(box.cls[0])
                label = (
                    model.names[cls_id]
                    if cls_id in model.names
                    else current_prompts[0]
                )
                log_detection(label, float(box.conf[0]))

        # Draw the latest boxes on the CURRENT live frame (smooth video)
        try:
            annotated = last_result.plot(img=frame, masks=False)
        except Exception:
            annotated = frame

        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            continue
        yield mjpeg_chunk(buffer.tobytes())


@app.get("/api/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/detections/recent")
def recent():
    return {"detections": list(recent_detections)}
