"""
detector.py — YOLOv8-based vehicle detector (optimized for GPU + accuracy)
"""

from ultralytics import YOLO
import numpy as np
import torch


class YOLODetector:
    def __init__(self, model_path='yolov8s.pt', conf_thres=0.45, device='cuda', imgsz=960):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to YOLOv8 model (e.g., yolov8s.pt)
            conf_thres: Confidence threshold for detections
            device: 'cuda' for GPU, 'cpu' for CPU
            imgsz: Inference image size (smaller = faster)
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.conf_thres = conf_thres
        self.imgsz = imgsz

        # COCO vehicle-related classes
        self.vehicle_class_ids = {
            2: 'car',
            3: 'motorbike',
            5: 'bus',
            7: 'truck',
            1: 'bicycle'
        }

        # Optional: half precision on GPU for speed
        if self.device == 'cuda':
            self.model.model.half()
            print("[INFO] Using GPU (FP16 mode).")
        else:
            print("[INFO] Using CPU (FP32 mode).")

    def detect(self, frame):
        """
        Run YOLO inference on a frame.
        Returns a list of [x1, y1, x2, y2, score, label].
        """
        # Resize for speed if specified
        img = frame if self.imgsz is None else \
            self._resize_preserve_aspect(frame, self.imgsz)

        # Run inference
        results = self.model.predict(
            img, conf=self.conf_thres, imgsz=self.imgsz,
            device=self.device, verbose=False
        )

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                score = float(box.conf[0])
                if cls_id in self.vehicle_class_ids:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    label = self.vehicle_class_ids[cls_id]
                    detections.append([
                        int(x1), int(y1), int(x2), int(y2),
                        score, label
                    ])

        return detections

    @staticmethod
    def _resize_preserve_aspect(frame, target_w):
        """
        Resize frame preserving aspect ratio to target width.
        """
        h, w = frame.shape[:2]
        if w <= target_w:
            return frame
        scale = target_w / w
        new_h = int(h * scale)
        return np.ascontiguousarray(
            cv2.resize(frame, (target_w, new_h))
        )
