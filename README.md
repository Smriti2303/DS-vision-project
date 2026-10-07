# Prompt-Based ("Self-Detecting") Object Detection — Starter Kit

This is a tested, working starting point for the part of your project where a camera can
detect *whatever you type in plain English* — "person wearing glasses," "forklift,"
"spilled liquid" — without ever retraining the model. Everything in this folder has
actually been run and verified before being handed to you (see `sample_detection_output.jpg`
for proof — real detections on a real photo, done live while building this).

---

## Part A — What you're installing, and why

You need exactly three things on your computer. Nothing else.

### 1. Python (version 3.9 or newer)
This is the programming language everything else runs on.

**How to check if you already have it:** open a terminal (Command Prompt/PowerShell on
Windows, Terminal on Mac/Linux) and type:
```
python --version
```
If you see something like `Python 3.11.4`, you're set. If you get an error, install it from
**python.org/downloads** — download the installer for your OS and run it. On Windows,
make sure you tick **"Add Python to PATH"** during install, or later commands won't work.

### 2. `ultralytics` (the detection engine)
This one library gives you YOLOv8 (fixed-object detection), YOLO-World, and YOLOE
(both open-vocabulary/"self-detecting") — all in one package. You don't need to download
separate repos or set up complex environments.

### 3. `opencv-python` (reads video/camera frames)
Handles turning a webcam feed or RTSP stream into individual images the model can look at,
and displaying the result on screen.

*(A 4th package, `streamlit`, is only needed for Step 4 — the dashboard. Skip it if you're
only doing Steps 1-3.)*

### How to install both, in one line
In your terminal, navigate to this folder and run:
```
pip install -r requirements.txt
```
**Why this specific command:** `requirements.txt` (included in this folder) lists exact
package names and versions, so everyone on your team installs the same thing — no
"works on my machine" problems later.

> **If you get a "pip not found" error:** try `pip3` instead of `pip`. If you get a
> permissions error, add `--user` to the end of the command.

> **Note for company/Linux servers:** if you see an error mentioning
> "externally-managed-environment," add `--break-system-packages` to the end of the
> install command.

---

## Part B — Important thing to know before you run anything

The **first time** you run any script that calls `model.set_classes(...)` (the line that
lets you type a custom prompt), two files get downloaded automatically in the background:

1. The YOLO-World model weights (~25MB) — from GitHub, works almost anywhere.
2. A CLIP text-understanding model (~350MB) — from `openaipublic.azureedge.net`.
   **This is the part that turns your typed words into something the model can match
   against the image.**

Both are cached on your machine after the first download — you won't need internet for
this again afterward. But **if you're behind a strict company firewall (likely true for
your company's on-prem GPU server), this specific domain may be blocked.**

**What to do about it:**
- Easiest: run Step 1 once on your own laptop/home network first (where it'll download
  fine), then copy the cached files (`~/.cache/clip/` and the `yolov8s-world.pt` file)
  onto the company server.
- Or: ask IT to whitelist `openaipublic.azureedge.net` on the server.
- Flag this to your manager now — it's a five-minute fix if you know about it in advance,
  and a confusing dead-end if you hit it blind on deployment day.

---

## Part C — Run the steps in order

Don't skip ahead — each step is designed to isolate exactly one new piece of complexity
so that if something breaks, you know exactly which part broke.

### Step 1 — `1_test_on_image.py`
Tests the whole concept on a single still photo. Put any photo of your own in this folder
named `my_test_image.jpg` (a photo with people, a car, a pet — anything works), then run:
```
python 1_test_on_image.py
```
Edit the `PROMPTS = [...]` line at the top to type whatever you want it to look for.
Re-run — no retraining, it just works on the new prompt. **This is the entire "self
detection" concept your director described, in its simplest possible form.**

*Reference:* `sample_test_image.jpg` and `sample_detection_output.jpg` in this folder show
a real example already run for you — a street photo with people, a bus, and a stop sign,
all correctly detected and boxed, using the model's default vocabulary.

### Step 2 — `2_webcam_live.py`
Same model, now pointed at your live webcam instead of a photo.
```
python 2_webcam_live.py
```
A live video window opens with boxes drawn in real time. Press `q` to quit. Try holding up
different objects — a phone, a cup, a bottle — and watch it detect them live.

**Why this step exists:** it's the exact same code pattern you'll use for a real factory
camera in Step 3, but far easier to debug, since you can point your own webcam at anything
in the room instantly.

### Step 3 — `3_rtsp_stream.py`
The real thing — a live RTSP camera feed instead of a webcam. Only one line changes.
```
python 3_rtsp_stream.py
```
You'll need to edit `RTSP_URL` at the top with a real camera address (get this from
whoever manages the factory's cameras/NVR — the script's comments explain the URL format).

**Before running this on a real camera:** confirm the RTSP URL works at all, completely
separately from any AI code, using VLC (Media → Open Network Stream) or `ffplay`. If you
can't see video there, the AI script won't work either — that means it's a camera/network
problem, not a code problem, and you'll waste less time debugging in the wrong place.

### Step 4 — `4_prompt_dashboard.py`
Wraps everything into a simple web page with an actual text box to type prompts into —
this is the "prompt-based dashboard" itself.
```
streamlit run 4_prompt_dashboard.py
```
Opens automatically in your browser. Type something in the box (e.g. `person, phone, cup`),
check "Start camera," and watch it detect live. Change the text box at any time — no
restart needed.

---

## Part D — What to expect, honestly

- **Single-word prompts work best** ("bicycle," "helmet," "forklift"). Compound descriptions
  ("person wearing glasses AND a red shirt") are harder and less reliable — this is a real
  limitation of current open-vocabulary models, not a bug in this code.
- **Good lighting matters a lot**, same as it would for a human looking at a blurry photo.
- If your prompt returns nothing, try a simpler or more common word for the same object
  before assuming something is broken.

## Part E — Once this all works on your laptop

The only thing that changes for the real factory deployment is:
1. `VIDEO_SOURCE` / `RTSP_URL` → the real camera's RTSP address (not your webcam)
2. This all needs to run on the company GPU server instead of your laptop, so it's always
   on and reachable by others — that's a deployment/hosting step, not a code change.

Everything else — the model, the prompt logic, the dashboard — is exactly what you've
already tested here.
