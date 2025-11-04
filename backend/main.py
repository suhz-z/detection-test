"""
main.py — YOLOv8 + Supervision ByteTrack Vehicle Detection
Generates annotated video + CSV (no streaming).
"""

import os
import cv2
import csv
import json
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import supervision as sv
from homography import HomographyCalibrator
from utils import draw_annotations
import torch

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



def process_video(video_source, output_path, csv_path):
    """Processes a video using YOLOv8 + Supervision ByteTrack."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Initialize CSV log
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "id", "label", "speed_kmph", "direction"])

    # Load YOLOv8 model + ByteTrack tracker
    model = YOLO(MODEL_NAME)
    model.to(DEVICE)
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()

    # Load homography calibration
    with open(CALIBRATION_FILE, "r") as f:
        calib = json.load(f)
    hom = HomographyCalibrator(
        np.array(calib["image_points"]), np.array(calib["world_points"])
    )

    # Open video
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {video_source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if FRAME_RESIZE:
        frame_w, frame_h = FRAME_RESIZE

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    track_history = {}
    prev_time = time.time()
    frame_idx = 0

    print("[INFO] Processing started...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video.")
                break

            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue

            if FRAME_RESIZE:
                frame = cv2.resize(frame, FRAME_RESIZE)

            now = time.time()
            dt = now - prev_time if (now - prev_time) > 0 else (1.0 / fps)
            prev_time = now

            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)

            # Keep only vehicle classes
            vehicle_classes = [1, 2, 3, 5, 7]  # bicycle, car, motorbike, bus, truck
            mask = np.isin(detections.class_id, vehicle_classes)
            detections = detections[mask]

            tracked = tracker.update_with_detections(detections)
            annotations, log_rows = [], []

            for xyxy, cls, track_id in zip(
                tracked.xyxy, tracked.class_id, tracked.tracker_id
            ):
                if track_id is None:
                    continue

                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = int((x1 + x2) / 2), int(y2)
                label = model.names[int(cls)]

                world_pt = hom.image_to_world(np.array([cx, cy]))
                if track_id in track_history:
                    prev_pt = track_history[track_id]
                    dx, dy = world_pt - prev_pt
                    dist_m = np.sqrt(dx ** 2 + dy ** 2)
                    speed_kmph = (dist_m / dt) * 3.6
                    direction = get_direction(dx, dy)
                else:
                    speed_kmph = 0.0
                    direction = "N/A"

                track_history[track_id] = world_pt

                annotations.append(
                    {
                        "box": (x1, y1, x2, y2),
                        "id": int(track_id),
                        "label": label,
                        "speed": speed_kmph,
                        "direction": direction,
                    }
                )

                log_rows.append(
                    [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        track_id,
                        label,
                        round(speed_kmph, 2),
                        direction,
                    ]
                )

            # Write to CSV
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerows(log_rows)

            annotated = draw_annotations(frame.copy(), annotations)
            out.write(annotated)

        print(f"[INFO] Finished. Video saved at {output_path}")
        print(f"[INFO] CSV saved at {csv_path}")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()


def get_direction(dx, dy):
    if abs(dx) > abs(dy):
        return "Eastbound" if dx > 0 else "Westbound"
    else:
        return "Southbound" if dy > 0 else "Northbound"


if __name__ == "__main__":
    process_video("video.mp4", "data/output_annotated.mp4", "data/vehicle_log.csv")
