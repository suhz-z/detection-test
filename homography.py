"""homography.py - Calibration helper to map image pixels -> world meters using homography.
The homography assumes a flat road plane. Provide at least 4 non-collinear correspondences.
"""

import numpy as np
import cv2

class HomographyCalibrator:
    def __init__(self, image_pts, world_pts):
        # image_pts: Nx2 pixels, world_pts: Nx2 meters
        image_pts = np.array(image_pts, dtype=np.float32)
        world_pts = np.array(world_pts, dtype=np.float32)
        assert image_pts.shape[0] >= 4 and world_pts.shape[0] >= 4, 'Need >=4 correspondences'
        # Compute homography from image -> world plane (meters)
        self.H, _ = cv2.findHomography(image_pts, world_pts, method=0)

    def image_to_world(self, pt):
        # Convert (x,y) pixel -> (x_m, y_m)
        px = np.array([pt[0], pt[1], 1.0], dtype=float)
        wp = self.H.dot(px)
        wp = wp / wp[2]
        return (float(wp[0]), float(wp[1]))
