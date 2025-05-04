import cv2
import torch
from ultralytics import YOLO

# Paths
VIDEO_PATH = '/home/shaurya/Downloads/vid2.mp4'
MODEL_PATH = 'yolo11m-obb.pt'
OUTPUT_VIDEO_PATH = '/home/shaurya/Detect-Track/outputs/yolo_annotated.mp4'

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

frame_idx = 0
while cap.isOpened() and frame_idx < 180:
    ret, frame = cap.read()
    if not ret:
        break

    print(f"Processing frame {frame_idx}...")
    results = model(frame)[0]


    # Check if model outputs rotated boxes
    if hasattr(results.boxes, 'xywhr'):
        boxes = results.boxes.xywhr.cpu().numpy()  # (cx, cy, w, h, angle)
        for cx, cy, w, h, angle in boxes:
            rect = ((cx, cy), (w, h), angle)
            box_pts = cv2.boxPoints(rect)
            box_pts = box_pts.astype(int)
            cv2.drawContours(frame, [box_pts], 0, (0, 255, 0), 2)
    else:
        # Fallback to axis-aligned boxes if needed
        for x1, y1, x2, y2 in results.obb.xyxy.cpu().numpy():
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    out_writer.write(frame)
    frame_idx += 1

cap.release()
out_writer.release()
print(f"✅ Done! Saved annotated video to {OUTPUT_VIDEO_PATH}")