"""
main.py — YOLOv8 + Supervision ByteTrack Vehicle Detection
"""


import cv2
import json
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import supervision as sv
from homography import HomographyCalibrator
from utils import draw_annotations
import torch
from collections import deque

# ---------------- USER CONFIG ----------------
MODEL_NAME = "yolov8s.pt"
CALIBRATION_FILE = "sample_calibration.json"
FRAME_RESIZE = (960, 540)
FRAME_SKIP = 1
# ---------------------------------------------

# Auto-detect best available device


if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"[INFO] Running inference on {DEVICE.upper()}")



def process_video(video_source, output_path, progress_callback=None):
    """
    Processes a video or live stream using YOLOv8 + ByteTrack,
    computes smoothed vehicle speed, direction, and saves annotated output.
    """

    # Load model & tracker
    model = YOLO("yolov8s.pt")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    tracker = sv.ByteTrack()

    # Load homography calibration
    with open("sample_calibration.json", "r") as f:
        calib = json.load(f)
    hom = HomographyCalibrator(np.array(calib["image_points"]), np.array(calib["world_points"]))

    # Open input source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise SystemExit(f" Cannot open video source: {video_source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # fallback for live camera or RTSP
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    if FRAME_RESIZE:
        frame_w, frame_h = FRAME_RESIZE

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None

    track_history = {}
    speed_history = {}

    processed_frames = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
    start_time = time.time()

    print("[INFO] Processing started...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frames += 1
            if FRAME_SKIP > 1 and processed_frames % FRAME_SKIP != 0:
                continue

            if FRAME_RESIZE:
                frame = cv2.resize(frame, FRAME_RESIZE)

            # YOLOv8 inference
            results = model(frame, verbose=False, half=True)[0]
            detections = sv.Detections.from_ultralytics(results)

            # Keep only relevant vehicle classes
            detections = detections[np.isin(detections.class_id, [1, 2, 3, 5, 7])]

            tracked = tracker.update_with_detections(detections)

            annotations = []
            log_rows = []

            for xyxy, cls, track_id in zip(tracked.xyxy, tracked.class_id, tracked.tracker_id):
                if track_id is None:
                    continue

                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = int((x1 + x2) / 2), int(y2)
                label = model.names[int(cls)]

                # Convert image to world coordinates
                world_pt = np.array(hom.image_to_world(np.array([cx, cy])))

                if track_id in track_history:
                    prev_pt = track_history[track_id]
                    dx, dy = world_pt - np.array(prev_pt)
                    dist_m = np.sqrt(dx**2 + dy**2)

                    # avoid jitter
                    if dist_m < 0.05:
                        dist_m = 0.0

                    # Fixed delta time for stability
                    dt = 1.0 / fps
                    speed_kmph = (dist_m / dt) * 3.6

                    # Rolling average smoothing
                    if track_id not in speed_history:
                        speed_history[track_id] = deque(maxlen=5)
                    speed_history[track_id].append(speed_kmph)
                    smooth_speed = sum(speed_history[track_id]) / len(speed_history[track_id])

                    direction = get_direction(dx, dy)

                else:
                    smooth_speed = 0.0
                    direction = "N/A"

                track_history[track_id] = world_pt

                annotations.append({
                    "box": (x1, y1, x2, y2),
                    "id": int(track_id),
                    "label": label,
                    "speed": smooth_speed,
                    "direction": direction
                })

                log_rows.append([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    track_id,
                    label,
                    round(smooth_speed, 2),
                    direction
                ])

           
    

            # Draw annotations
            annotated = draw_annotations(frame.copy(), annotations)

            # Initialize output writer lazily
            if out is None:
                out_fps = fps / FRAME_SKIP
                out = cv2.VideoWriter(output_path, fourcc, out_fps, (frame_w, frame_h))
                print(f"[INFO] Output FPS set to {out_fps:.2f}")

            out.write(annotated)

            # Optional progress callback
            if progress_callback and processed_frames % 10 == 0 and total_frames > 0:
                progress = int((processed_frames / total_frames) * 100)
                progress_callback("processing", progress)

        elapsed = time.time() - start_time
        real_fps = processed_frames / elapsed if elapsed > 0 else fps
        print(f"[INFO] Processed {processed_frames} frames in {elapsed:.1f}s ({real_fps:.2f} FPS)")
        print(f"[INFO] Annotated video saved to {output_path}")

    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()


def get_direction(dx, dy):
    """Estimate cardinal direction of motion."""
    if abs(dx) > abs(dy):
        return "East" if dx > 0 else "West"
    else:
        return "South" if dy > 0 else "North"

if __name__ == "__main__":
    process_video("video.mp4", "data/output_annotated.mp4", "data/vehicle_log.csv")
