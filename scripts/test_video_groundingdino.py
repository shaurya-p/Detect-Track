import warnings
warnings.filterwarnings("ignore")

import sys
import os
import random
import cv2
import torch

import numpy as np
import supervision as sv

# Load groundingdino
GROUNDINGDINO_PATH = os.path.join(os.path.dirname(__file__), "..", "external", "GroundingDINO")
sys.path.append(GROUNDINGDINO_PATH)
from groundingdino.util.inference import load_model, predict

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
CONFIG_PATH = "external/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"
VIDEO_PATH = "/home/shaurya/Detect-Track/data/videos/bg_vid.mp4"
OUTPUT_DIR = "/home/shaurya/Detect-Track/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
NUM_SAMPLES = 5
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.10
CAPTION = "car, person, bike, truck"

# Load Model
print("Loading GroundingDINO...")
model = load_model(CONFIG_PATH, WEIGHTS_PATH)
model = model.to(device)
model.eval()
print("Model loaded.")

# Open Video
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'{frame_count} frames found in the video')
frame_indices = random.sample(range(frame_count), NUM_SAMPLES)
frame_indices.sort()

current_idx = 0
sample_idx = 0

while sample_idx < NUM_SAMPLES and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if current_idx == frame_indices[sample_idx]:
        # Processing
        print(f"prcessing frame {current_idx}...")
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        MAX_SIZE = 800  # you can adjust this based on your memory
        h, w, _ = img_rgb.shape
        scale = MAX_SIZE / max(h, w)
        if scale < 1.0:
            img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(device)

        # Inference
        boxes, scores, labels = predict(
            model,
            img_tensor,
            caption = CAPTION,
            box_threshold = BOX_THRESHOLD,
            text_threshold = TEXT_THRESHOLD
        )

        # Visualize
        h, w, _ = frame.shape
        scale_tensor = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
        boxes_xyxy = boxes * scale_tensor

        annotator = sv.BoxAnnotator()
        detections = sv.Detections(
            xyxy = boxes_xyxy.cpu().detach().numpy(),
            confidence = scores.cpu().detach().numpy(),
            class_id=np.zeros(len(labels), dtype=int)
        )
        labels = [str(label) for label in labels]
        annotated_frame = annotator.annotate(scene=frame, detections=detections, labels=labels)

        # Save outputs
        output_path = os.path.join(OUTPUT_DIR, f"sample_{sample_idx}.jpg")
        cv2.imwrite(output_path, annotated_frame)

        print(f"Saved {output_path}")
        sample_idx += 1

    current_idx += 1

cap.release()

