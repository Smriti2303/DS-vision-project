"""
STEP 2 — Run the SAME model live on your laptop's webcam.

WHAT THIS DOES:
  Identical logic to Step 1, except the source is now a live camera feed
  instead of a single photo. This is the exact same code pattern you'll
  use for a real RTSP factory camera in Step 3 — a webcam is just a free,
  easy stand-in to test with first.

WHY THIS STEP MATTERS:
  Debugging "is my detection logic correct?" is much easier on a webcam
  you can point at anything in the room, than on a flaky factory RTSP
  stream. Get this working first.

HOW TO RUN:
  python 2_webcam_live.py
  A live window will open. Press "q" to quit.
  Try pointing your webcam at different objects and see what gets boxed.
"""

import cv2
from ultralytics import YOLOWorld

PROMPTS = ["person", "phone", "cup", "bottle"]   # <- change this to whatever you want to test

print("Loading model...")
model = YOLOWorld("yolov8s-world.pt")
model.set_classes(PROMPTS)

print("Opening webcam — press 'q' in the video window to quit.")

# source=0 means "default webcam". show=True opens a live annotated window.
# stream=True makes predict() return a generator — it does NOT run anything
# on its own, so we MUST loop over it below for frames to actually be
# processed and the window to actually appear.
results_generator = model.predict(source=0, show=True, stream=True)

frame_count = 0
for result in results_generator:
    frame_count += 1
    # 'q' key check — lets you quit the live window cleanly
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"Quit requested. Processed {frame_count} frames.")
        break

cv2.destroyAllWindows()
print("Webcam session ended.")