# Push-up Tracker

Counts push-ups from a **front-facing** webcam. Uses **OpenCV** for video, **MediaPipe Pose** for landmarks, and a fused **depth score** (shoulders, elbows, chest, and arm angle) with smoothing and calibration.

## Setup

```bash
cd /pushup-tracker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Accurate pose model (recommended)
curl -L -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

## Run

```bash
python -m pushup_tracker.main
```

1. **Face the camera** — phone/laptop on the floor in front of you; chest, arms, and hands in frame.
2. **Hold the top** — arms locked at the top while the bar fills (~1.5 s).
3. **Full reps** — chest low at the bottom, full lockout at the top.

**Keys:** `q` quit · `r` reset · `c` recalibrate (hold top again)

The preview is mirrored by default. A large rep count appears after calibration. The bottom **depth bar** shows how close you are to the bottom (blue) vs top (green); markers show where reps register.

## Tips if counting is still off

- Recalibrate with **`c`** while holding a solid top position.
- Frame your **whole upper body** — if wrists leave the frame, detection fails.
- Use the **full** pose model (`pose_landmarker_full.task`), not lite.
- Good lighting on your torso helps.
- Go **deeper** — shallow reps may not cross the down threshold.

## How it works

Multiple signals are smoothed and combined into one depth value (0% = bottom, 100% = top). After calibration at the top:

- Enter **DOWN** when depth stays below ~38% for several frames.
- Count a rep when depth returns above ~72% with enough range and depth.

This debouncing and minimum depth cut down false counts from pose jitter.
