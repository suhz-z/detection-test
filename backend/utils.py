"""utils.py - Small drawing utilities used by main.py for visualization."""

import cv2

def draw_annotations(img, annotations):
    for a in annotations:
        x1,y1,x2,y2 = a['box']
        tid = a['id']
        label = a.get('label', 'vehicle')
        speed = a.get('speed', 0.0)
        direction = a.get('direction', '')
        # Draw bounding box
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        # Compose text (ID, label, speed, direction)
        txt = f"ID {tid} {label} {speed:.1f} km/h {direction or ''}"
        cv2.putText(img, txt, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    return img
