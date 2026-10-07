"""
STEP 1 — Run YOLO-World on a single photo with your own text prompt.

WHAT THIS DOES:
  Loads a pretrained open-vocabulary detector, tells it (in plain English)
  what to look for, and draws boxes around anything it finds — with NO
  training and NO labeled dataset.

WHY THIS IS THE STEP TO START WITH:
  It's the fastest way to see the "self detection" concept work with your
  own eyes, using a single image, before adding video/webcam/RTSP complexity.

HOW TO RUN:
  1. Put any photo in this same folder and name it "my_test_image.jpg"
     (or edit IMAGE_PATH below to point at any image on your computer)
  2. python 1_test_on_image.py
  3. A window will pop up showing the image with detection boxes drawn on it.
     (First run downloads two things automatically: the YOLO-World model
     weights (~25MB) and a CLIP text-understanding model (~350MB) — this
     needs an unrestricted internet connection the FIRST time only. After
     that, both are cached locally on your machine and it works offline.)
"""

from ultralytics import YOLOWorld

# ---- SETTINGS: change these two lines to experiment ----
IMAGE_PATH = "my_test_image.jpg"          # <- path to your test photo
PROMPTS = ["face", "person", "eye makeup"]  # <- type ANY objects here, in plain English
# ----------------------------------------------------------

print(f"Loading YOLO-World model...")
model = YOLOWorld("yolov8s-world.pt")   # small/fast version — good for a laptop

print(f"Setting custom prompt: {PROMPTS}")
model.set_classes(PROMPTS)              # <-- THIS is the "no retraining" moment

print(f"Running detection on {IMAGE_PATH} ...")
results = model.predict(IMAGE_PATH, save=True)

print("\nDetections found:")
for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    label = model.names[class_id]
    print(f"  - {label}  (confidence: {confidence:.2f})")

results[0].show()   # pops open a window with boxes drawn on the image
print("\nDone. Annotated image also saved under a new 'runs/detect/' folder.")
print("Now try changing PROMPTS at the top of this file and re-running —")
print("that's the entire 'self detection' idea in one line of code.")
