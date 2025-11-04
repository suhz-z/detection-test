"""tracker.py - Lightweight SORT-like tracker with history for speed estimation.
This implementation is intentionally simple and explained inline for learning.
For production, consider using DeepSORT, ByteTrack or Norfair for robustness.
"""

import numpy as np
from collections import OrderedDict
import math

class Track:
    def __init__(self, bbox, track_id):
        # bbox: [x1,y1,x2,y2] in pixel coords (floats)
        self.bbox = np.array(bbox, dtype=float)
        self.id = track_id
        self.hits = 1
        self.age = 0
        # history stores tuples: (frame_idx, (world_x_m, world_y_m))
        self.history = []

    def update_bbox(self, bbox):
        self.bbox = np.array(bbox, dtype=float)
        self.hits += 1
        self.age = 0

    def add_world_pos(self, frame_idx, world_pt):
        # world_pt: (x_m, y_m)
        self.history.append((frame_idx, tuple(world_pt)))
        # keep only last N positions to limit memory and smooth speed
        if len(self.history) > 30:
            self.history.pop(0)

    def speed_kmph(self, dt=1.0):
        # Compute speed using the last two world positions.
        if len(self.history) < 2 or dt <= 0:
            return 0.0
        (_, p1), (_, p2) = (self.history[-2], self.history[-1])
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist_m = math.hypot(dx, dy)
        speed_ms = dist_m / dt
        return speed_ms * 3.6  # convert m/s -> km/h

    def direction(self):
        # Coarse direction from last displacement (returns string or None)
        if len(self.history) < 2:
            return None
        (_, p1), (_, p2) = (self.history[-2], self.history[-1])
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = math.degrees(math.atan2(dy, dx))
        # Convert angle to coarse labels
        if abs(angle) < 45:
            return 'right'
        if abs(angle) > 135:
            return 'left'
        if angle > 0:
            return 'down'
        return 'up'

class Sort:
    def __init__(self, max_age=30, min_hits=3):
        # max_age: how many frames to keep a track without updates before deleting
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = OrderedDict()
        self._next_id = 1

    def update(self, detections):
        """Naive IoU-based matching:
        detections: N x 5 array-like [[x1,y1,x2,y2,score], ...]
        Returns array: [[x1,y1,x2,y2,track_id], ...]
        """
        dets = detections.tolist() if hasattr(detections, 'tolist') else []
        assigned = set()

        # For each detection, try to match with existing tracks by IoU
        for d in dets:
            best_iou = 0.0
            best_tid = None
            for tid, tr in self.tracks.items():
                iou = self.iou(d[:4], tr.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_iou > 0.3 and best_tid is not None:
                # update matched track
                self.tracks[best_tid].update_bbox(d[:4])
                assigned.add(best_tid)
            else:
                # create new track
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(d[:4], tid)
                assigned.add(tid)

        # Age non-updated tracks and delete old ones
        to_delete = []
        for tid, tr in list(self.tracks.items()):
            if tid not in assigned:
                tr.age += 1
            else:
                tr.age = 0
            if tr.age > self.max_age:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        # Build output array
        out = []
        for tid, tr in self.tracks.items():
            out.append([*tr.bbox.tolist(), tr.id])
        return np.array(out)

    @staticmethod
    def iou(b1, b2):
        # Compute IoU between two boxes [x1,y1,x2,y2]
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        inter = w * h
        a1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        a2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    # Helper API used by main.py
    def get_speed_kmph(self, track_id, dt):
        tr = self.tracks.get(track_id, None)
        if tr is None:
            return 0.0
        return tr.speed_kmph(dt)

    def get_direction(self, track_id):
        tr = self.tracks.get(track_id, None)
        if tr is None:
            return None
        return tr.direction()

    def add_world_position(self, track_id, frame_idx, world_pt):
        # Helper to add a world position to a track's history
        tr = self.tracks.get(track_id, None)
        if tr is None:
            return
        tr.add_world_pos(frame_idx, world_pt)
