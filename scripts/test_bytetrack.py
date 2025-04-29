import sys
import os
sys.path.append(os.path.abspath("./external/ByteTrack"))

from yolox.tracker.byte_tracker import BYTETracker

import numpy as np
import cv2

from types import SimpleNamespace

# Dummy frame rate
frame_rate = 30

# Create dummy tracker config
tracker_args = {
    "track_thresh": 0.5,
    "match_thresh": 0.8,
    "track_buffer": 30,
    "min_box_area": 10,
    "mot20": False,
}



# Initialize tracker
tracker = BYTETracker(SimpleNamespace(**tracker_args))
# tracker = BYTETracker(tracker_args, frame_rate)

# Create a dummy detection list
# format: (x, y, w, h, score, class_id)
detections = np.array([
    [100, 150, 50, 80, 0.9, 0],
    [400, 300, 60, 90, 0.85, 0],
])

# Create a dummy frame
frame = np.zeros((600, 800, 3), dtype=np.uint8)

# Prepare detections for tracker
detections_for_tracker = []
for det in detections:
    x, y, w, h, score, cls_id = det
    detections_for_tracker.append(
        np.array([x, y, x + w, y + h, score])
    )

# Convert to numpy array
detections_for_tracker = np.array(detections_for_tracker)

# Dummy timestamp
timestamp = 0

# Update tracker
online_targets = tracker.update(
    detections_for_tracker,
    frame.shape[:2],  # height, width
    frame.shape[:2],  # height, width
)


# Draw results
from utils.visualize import plot_tracking

tlwhs = []
obj_ids = []

for target in online_targets:
    tlwh = target.tlwh
    tid = target.track_id
    tlwhs.append(tlwh)
    obj_ids.append(tid)

# Plot
out_img = plot_tracking(frame, tlwhs, obj_ids, frame_id=timestamp)

# Save output
cv2.imwrite("outputs/test_bytetrack_result.jpg", out_img)
print("✅ ByteTrack dummy test completed. Output saved to outputs/test_bytetrack_result.jpg")
