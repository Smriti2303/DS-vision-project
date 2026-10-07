"""
STEP 3 — Run the SAME model on a real RTSP camera stream.

WHAT THIS DOES:
  Only ONE thing changes from Step 2: the video source. Instead of your
  webcam (source=0), we point at an RTSP URL. Everything else — the model,
  the prompt-setting, the detection loop — is identical. This is the
  moment your director's "self-detecting RTSP" ask becomes real.

WHAT AN RTSP URL LOOKS LIKE:
  rtsp://username:password@camera-ip-address:554/stream-path
  Example:  rtsp://admin:admin123@192.168.1.64:554/Streaming/Channels/101
  You'll need to get this exact URL from whoever manages the factory's
  cameras/NVR system — every camera brand formats it slightly differently.

BEFORE RUNNING THIS ON A REAL FACTORY CAMERA:
  Test the RTSP URL works at all first, completely separate from AI:
      ffplay rtsp://your-camera-url-here
  or open it in VLC (Media > Open Network Stream). If you can't see the
  video there, the AI code won't work either — fix the camera connection
  first, that's not a code problem.

HOW TO RUN:
  1. Edit RTSP_URL below with your real camera URL
  2. python 3_rtsp_stream.py
"""

from ultralytics import YOLOWorld

RTSP_URL = "rtsp://username:password@camera-ip:554/stream-path"  # <- EDIT THIS
PROMPTS = ["person", "helmet", "forklift"]                        # <- and this

print("Loading model...")
model = YOLOWorld("yolov8s-world.pt")
model.set_classes(PROMPTS)

print(f"Connecting to RTSP stream: {RTSP_URL}")
print("Press 'q' in the video window to quit.")

# stream=True is important for live video — it processes frame by frame
# instead of trying to load the whole video into memory at once.
model.predict(source=RTSP_URL, show=True, stream=True)
