import sys
import os

import torch
import numpy as np
import cv2
import random
from ultralytics import YOLO

# Paths
VIDEO_PATH = "/home/shaurya/Detect-Track/data/videos/bg_vid.mp4"
MODEL_PATH = "yolo11m-obb.pt"

NUM_SAMPLES = 5
OUTPUT_DIR = "/home/shaurya/Detect-Track/outputs/yolo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Model
print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)
model.to(device)
model.eval()
print("Model loaded.")

# Load Video
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_indices = sorted(random.sample(range(frame_count), NUM_SAMPLES))

print(f"Sampling {NUM_SAMPLES} frames out of {frame_count}...")

current_idx = 0
sample_idx = 0

while cap.isOpened() and sample_idx < NUM_SAMPLES:
    ret, frame = cap.read()
    if not ret:
        break

    if current_idx == frame_indices[sample_idx]:
        #Processing
        frame = cv2.resize(frame, (1024, 1024)) 

        print(f"Running YOLO on frame {current_idx}...")
        results = model(frame)[0]

        # Annotate
        annotated_frame = results.plot()
        out_path = os.path.join(OUTPUT_DIR, f"{current_idx}.jpg")
        cv2.imwrite(out_path, annotated_frame)
        print(f"Saved to {out_path}")

        sample_idx += 1

    current_idx +=1 

cap.release()
print("✅ YOLO frame sampling test completed.")


