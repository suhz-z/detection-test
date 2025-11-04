import cv2


# RGB color definitions (you can tweak them)
CLASS_COLORS = {
    "car": (0, 255, 0),          # Green
    "truck": (0, 128, 255),      # Orange-blue
    "bus": (255, 0, 0),          # Red
    "motorbike": (255, 255, 0),  # Yellow
    "bicycle": (255, 0, 255),    # Magenta
    "vehicle": (200, 200, 200),  # Default gray
}
  # if CLASS_COLORS is in same file, skip this import

def draw_annotations(frame, annotations):
 
    for ann in annotations:
        x1, y1, x2, y2 = ann['box']
        tid = ann['id']
        label = ann['label'].lower()
        speed = ann.get('speed', 0.0)
        direction = ann.get('direction', "N/A")

        # Pick color based on label
        color = CLASS_COLORS.get(label, (255, 255, 255))  # default white if unknown

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Text: ID + Label + Speed
        text = f"ID {tid} | {label} | {speed:.1f} km/h | {direction}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame
