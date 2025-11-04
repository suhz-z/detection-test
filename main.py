"""
main.py — Optimized Vehicle Detection, Speed & Direction Estimation
✅ GPU acceleration
✅ Frame resizing (for speed)
✅ Skip every other frame (optional)
✅ Improved class filtering to prevent car↔bike confusion
✅ Flask-compatible (yields frames for streaming)
"""

import time
import json
import cv2
import csv
import numpy as np
import os
from datetime import datetime
from detector import YOLODetector
from tracker import Sort
from homography import HomographyCalibrator
from utils import draw_annotations

# ---------- USER CONFIG ----------
VIDEO_SOURCE = 'video.mp4'        # Replace with your file or RTSP stream
MODEL_NAME = 'yolov8s.pt'         # Use larger model for better accuracy
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.5
CALIBRATION_FILE = 'sample_calibration.json'
OUTPUT_VIDEO = 'output_annotated.mp4'

FRAME_RESIZE = (960, 540)         # Resize input frames for speed (None = original)
FRAME_SKIP = 1                    # Process every n-th frame (1 = process every frame)
DEVICE = 'cuda'                   # 'cuda' for GPU, 'cpu' for CPU
CSV_PATH = 'data/vehicle_log.csv'

MAX_AGE = 30
MIN_HITS = 3
# --------------------------------


def process_frame():
    """
    Generator for Flask — streams annotated frames and logs data to CSV.
    Uses YOLOv8 with GPU, frame resizing, and optional frame skipping.
    """
    os.makedirs('data', exist_ok=True)
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'id', 'label', 'speed_kmph', 'direction'])

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {VIDEO_SOURCE!r}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resize values for consistent scaling
    if FRAME_RESIZE:
        frame_w, frame_h = FRAME_RESIZE

    # Calibration
    with open(CALIBRATION_FILE, 'r') as f:
        calib = json.load(f)
    hom = HomographyCalibrator(np.array(calib['image_points']),
                               np.array(calib['world_points']))

    # Detector + Tracker
    detector = YOLODetector(MODEL_NAME, conf_thres=CONFIDENCE_THRESHOLD, device=DEVICE)
    tracker = Sort(max_age=MAX_AGE, min_hits=MIN_HITS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_w, frame_h))

    frame_idx = 0
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or cannot read frame.")
                break

            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue  # Skip frames for faster processing

            # Resize for speed
            if FRAME_RESIZE:
                frame = cv2.resize(frame, FRAME_RESIZE)

            dt = 1.0 / fps

            # YOLO detection (GPU accelerated)
            dets = detector.detect(frame)

            # Filter out weak/irrelevant classes
            dets = [d for d in dets if d[5] in ['car', 'bus', 'truck', 'motorbike', 'bicycle']]

            dets_for_tracker = [[d[0], d[1], d[2], d[3], d[4]] for d in dets]
            dets_np = np.array(dets_for_tracker) if dets_for_tracker else np.empty((0, 5))
            tracks = tracker.update(dets_np)

            annotations, log_rows = [], []

            for t in tracks:
                x1, y1, x2, y2, tid = t
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cx, cy = int((x1 + x2) / 2), int(y2)

                world_pt = hom.image_to_world(np.array([cx, cy]))
                tracker.add_world_position(tid, frame_idx, world_pt)

                speed_kmph = tracker.get_speed_kmph(tid, dt)
                direction = tracker.get_direction(tid)

                # Find best-matching detection (highest IoU overlap)
                matched_label = 'vehicle'
                best_iou = 0
                for d in dets:
                    dx1, dy1, dx2, dy2, score, cls = d[:6]
                    iou = tracker.iou([dx1, dy1, dx2, dy2], [x1, y1, x2, y2])
                    if iou > 0.3 and score > CONFIDENCE_THRESHOLD and iou > best_iou:
                        matched_label = cls
                        best_iou = iou

                annotations.append({
                    'box': (x1, y1, x2, y2),
                    'id': int(tid),
                    'label': matched_label,
                    'speed': speed_kmph,
                    'direction': direction
                })

                log_rows.append([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    tid, matched_label, round(speed_kmph, 2), direction
                ])

            with open(CSV_PATH, 'a', newline='') as f:
                csv.writer(f).writerows(log_rows)

            out_frame = draw_annotations(frame.copy(), annotations)
            out.write(out_frame)

            # Encode for Flask stream
            ret, buffer = cv2.imencode('.jpg', out_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Flask stream ended. Saved annotated video to {OUTPUT_VIDEO}")
