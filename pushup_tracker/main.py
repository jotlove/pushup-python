"""Webcam push-up counter — front-facing camera with large on-screen rep count."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

from pushup_tracker.counter import CounterState, Phase, PushUpCounter
from pushup_tracker.draw import draw_pose
from pushup_tracker.metrics import DepthEstimator, extract_signals

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL = MODELS_DIR / "pose_landmarker_full.task"
FALLBACK_MODEL = MODELS_DIR / "pose_landmarker_lite.task"


def resolve_model(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if DEFAULT_MODEL.is_file():
        return DEFAULT_MODEL
    return FALLBACK_MODEL


def draw_hud(
    frame,
    reps: int,
    phase: Phase,
    depth_pct: Optional[float],
    counter: PushUpCounter,
    range_span: float,
) -> None:
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cx = w // 2

    if counter.is_calibrating:
        progress = counter.calibration_progress
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        label = "HOLD TOP POSITION"
        cv2.putText(frame, label, (cx - 175, h // 2 - 50), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            "Arms locked — chest up — face the camera",
            (cx - 260, h // 2),
            font,
            0.55,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        bar_w = 320
        bx1 = cx - bar_w // 2
        bx2 = cx + bar_w // 2
        by = h // 2 + 50
        cv2.rectangle(frame, (bx1, by), (bx2, by + 14), (60, 60, 60), -1)
        fill = int(bx1 + progress * (bx2 - bx1))
        cv2.rectangle(frame, (bx1, by), (fill, by + 14), (0, 200, 120), -1)
        return

    # Large rep count
    rep_text = str(reps)
    scale = min(w, h) / 180.0
    thickness = max(4, int(scale * 2.5))
    (tw, th), _ = cv2.getTextSize(rep_text, font, scale, thickness)
    cy = int(h * 0.2)
    tx, ty = cx - tw // 2, cy + th // 2
    cv2.putText(
        frame, rep_text, (tx + 3, ty + 3), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA
    )
    cv2.putText(frame, rep_text, (tx, ty), font, scale, (0, 255, 128), thickness, cv2.LINE_AA)
    cv2.putText(
        frame, "PUSH-UPS", (cx - 70, cy - th - 14), font, 0.65, (220, 220, 220), 2, cv2.LINE_AA
    )

    phase_text = "DOWN" if phase is Phase.DOWN else "UP"
    phase_color = (80, 180, 255) if phase is Phase.DOWN else (180, 255, 180)
    cv2.putText(
        frame, phase_text, (cx - 35, cy + th + 34), font, 0.9, phase_color, 2, cv2.LINE_AA
    )

    if depth_pct is not None:
        bar_y = h - 40
        cv2.rectangle(frame, (0, bar_y - 10), (w, h), (25, 25, 25), -1)

        bar_x1, bar_x2 = 16, w - 16
        by = bar_y - 18
        cv2.rectangle(frame, (bar_x1, by), (bar_x2, by + 12), (50, 50, 50), -1)

        down_x = int(bar_x1 + counter.down_depth * (bar_x2 - bar_x1))
        up_x = int(bar_x1 + counter.up_depth * (bar_x2 - bar_x1))
        cv2.line(frame, (down_x, by - 2), (down_x, by + 14), (100, 100, 255), 1)
        cv2.line(frame, (up_x, by - 2), (up_x, by + 14), (100, 255, 100), 1)

        fill_x = int(bar_x1 + depth_pct * (bar_x2 - bar_x1))
        color = (80, 160, 255) if phase is Phase.DOWN else (0, 210, 110)
        cv2.rectangle(frame, (bar_x1, by), (fill_x, by + 12), color, -1)

        cv2.putText(
            frame,
            f"Depth {depth_pct * 100:.0f}%   range {range_span:.3f}",
            (16, h - 10),
            font,
            0.5,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track push-ups facing the camera (front view)."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--model", type=Path, default=None, help="Pose model .task file")
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal flip (mirror is on by default)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = resolve_model(args.model)

    if not model_path.is_file():
        print(f"Model not found: {model_path}", file=sys.stderr)
        print(
            "Download a model, e.g.:\n"
            "  curl -L -o models/pose_landmarker_full.task "
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
            file=sys.stderr,
        )
        return 1

    counter = PushUpCounter()
    depth_estimator = DepthEstimator()
    mirror = not args.no_mirror

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    model_name = model_path.name
    print(f"Push-up tracker (front view) — model: {model_name}")
    print("  1. Face the camera with hands and chest in frame.")
    print("  2. Hold the TOP position while it calibrates (~1.5 s).")
    print("  3. Do full push-ups — go deep, lock out at the top.")
    print("  Keys: q quit | r reset | c recalibrate")

    frame_idx = 0
    depth_pct: Optional[float] = None
    range_span = 0.0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / fps)
            frame_idx += 1

            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            depth_pct = None

            if results.pose_landmarks:
                landmarks = results.pose_landmarks[0]
                if not counter.is_calibrating:
                    draw_pose(frame, landmarks)

                raw = extract_signals(landmarks)
                if raw is not None:
                    if counter.is_calibrating:
                        combined = depth_estimator.observe_calibration(raw)
                        counter.update(None, 0.0, combined)
                        if counter.calibration_progress >= 1.0:
                            counter.finish_calibration(depth_estimator)
                    else:
                        combined, depth_pct = depth_estimator.update(raw)
                        range_span = depth_estimator.range_span
                        counter.update(depth_pct, range_span, combined)

            draw_hud(frame, counter.reps, counter.phase, depth_pct, counter, range_span)
            cv2.imshow("Push-up Tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                counter.reset()
                depth_estimator.reset_range()
            if key == ord("c"):
                counter.reset()
                depth_estimator.reset_range()

    cap.release()
    cv2.destroyAllWindows()
    print(f"Session total: {counter.reps} push-ups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
