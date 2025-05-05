import sys
import os
sys.path.append(os.path.abspath("./external/ByteTrack"))

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from shapely.geometry import Polygon
from types import SimpleNamespace

# Paths
VIDEO_PATH = '/home/shaurya/Downloads/vid2.mp4'
MODEL_PATH = 'yolo11m-obb.pt'
OUTPUT_VIDEO_PATH = '/home/shaurya/Detect-Track/outputs/yolo_obb_bytetrack.mp4'

# Load model
print("Loading YOLO-OBB model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH).to(device).eval()
print("Model loaded.")

# Init ByteTrack
args = SimpleNamespace(
    track_thresh=0.4,
    match_thresh=0.8,
    track_buffer=60,
    frame_rate=60,
    use_byte=True,
    mot20=False,
    conf_thres=0.3,
    low_conf_thres=0.1
)

tracker = BYTETracker(args, frame_rate=args.frame_rate)

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

def obb_to_polygon(box):
    """Convert rotated box (cx, cy, w, h, angle) to polygon"""
    rect = ((box[0], box[1]), (box[2], box[3]), box[4])
    points = cv2.boxPoints(rect)
    return Polygon(points), points.astype(int)

frame_idx = 0
while cap.isOpened() and frame_idx < 180:
    ret, frame = cap.read()
    if not ret:
        break

    print(f"Processing frame {frame_idx}...")
    results = model(frame)[0]

    # Extract axis-aligned boxes for ByteTrack
    obb_xyxy = results.obb.xyxy.cpu().numpy() if results.obb.xyxy is not None else np.empty((0, 4))
    confs = results.obb.conf.cpu().numpy() if results.obb.conf is not None else np.empty((0,))
    dets = np.hstack((obb_xyxy, confs.reshape(-1, 1))) if len(confs) > 0 else np.empty((0, 5))

    # Run ByteTrack
    online_targets = tracker.update(dets, [height, width], (height, width))

    # Get rotated boxes
    obbs = results.obb.xywhr.cpu().numpy() if hasattr(results.obb, "xywhr") else []

    # Convert rotated boxes to polygons for matching
    obb_polygons = [obb_to_polygon(obb) for obb in obbs]

    # Draw
    for track in online_targets:
        tlwh = track.tlwh  # axis-aligned
        tid = track.track_id
        track_box = [tlwh[0], tlwh[1], tlwh[0]+tlwh[2], tlwh[1]+tlwh[3]]
        track_poly = Polygon([[track_box[0], track_box[1]], [track_box[2], track_box[1]],
                              [track_box[2], track_box[3]], [track_box[0], track_box[3]]])

        # Match with closest rotated box (IoU)
        best_iou = 0
        best_poly = None
        for poly, pts in obb_polygons:
            iou = track_poly.intersection(poly).area / track_poly.union(poly).area
            if iou > best_iou:
                best_iou = iou
                best_poly = pts

        # Draw the best matching rotated box
        if best_poly is not None:
            cv2.drawContours(frame, [best_poly], 0, (0, 255, 0), 2)
            cx, cy = np.mean(best_poly, axis=0).astype(int)
            cv2.putText(frame, f'ID {tid}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    out_writer.write(frame)
    frame_idx += 1

cap.release()
out_writer.release()
print(f"✅ Saved annotated tracking video with rotation to: {OUTPUT_VIDEO_PATH}")