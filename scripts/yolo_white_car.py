import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Paths
VIDEO_PATH = '/home/shaurya/Downloads/vid2.mp4'
MODEL_PATH = 'yolo11m-obb.pt'
OUTPUT_VIDEO_PATH = '/home/shaurya/Detect-Track/outputs/white_cars_only.mp4'

# Load model
print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH).to(device).eval()
print("Model loaded.")

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

def is_white_region(image, mask_pts, sat_thresh=60, val_thresh=180, fraction_thresh=0.2):
    """Detect visually white-ish regions using HSV filtering for top-down cars."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [mask_pts], 255)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    sat_vals = s[mask == 255]
    val_vals = v[mask == 255]

    if len(sat_vals) == 0:
        return False

    white_like = np.logical_and(sat_vals < sat_thresh, val_vals > val_thresh)
    white_fraction = np.sum(white_like) / len(sat_vals)

    return white_fraction > fraction_thresh


frame_idx = 0
while cap.isOpened() and frame_idx < 180:
    ret, frame = cap.read()
    if not ret:
        break

    print(f"Processing frame {frame_idx}...")
    results = model(frame)[0]

    # Filter for "car" class (assuming class index is 2 or known)
    class_ids = results.obb.cls.cpu().numpy().astype(int)
    obb_boxes = results.obb.xywhr.cpu().numpy() if hasattr(results.obb, "xywhr") else []

    for i, (cx, cy, w, h, angle) in enumerate(obb_boxes):
        if class_ids[i] != 2:  # adjust this if "car" has a different class index
            continue
        
        # Convert to polygon
        rect = ((cx, cy), (w, h), angle)
        box_pts = cv2.boxPoints(rect).astype(int)

        # Check color
        if is_white_region(frame, box_pts):
            cv2.drawContours(frame, [box_pts], 0, (0, 255, 0), 2)
            cx_text, cy_text = np.mean(box_pts, axis=0).astype(int)
            cv2.putText(frame, "white car", (cx_text, cy_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    out_writer.write(frame)
    frame_idx += 1

cap.release()
out_writer.release()
print(f"✅ Done! Saved only white car detections to {OUTPUT_VIDEO_PATH}")
