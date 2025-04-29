import warnings
warnings.filterwarnings("ignore")

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'external', 'GroundingDINO')))

import torch
from groundingdino.models import build_model
from groundingdino.util.inference  import load_model, predict
import numpy as np

# -------------------
# 1. Setup
# -------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG_PATH = "external/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"

print("Loading GroundingDINO...")
model = load_model(CONFIG_PATH, WEIGHTS_PATH)
model.to(device)
model.eval()

# -------------------
# 2. Create a Dummy Image
# -------------------

dummy_img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

# Convert to torch tensor
img_tensor = torch.from_numpy(dummy_img).float().permute(2,0,1) / 255.0 
img_tensor = img_tensor.to(device)

# -------------------
# 3. Run Prediction
# -------------------
print("running inference on image...")
caption = "a person, a car, a dog"  
box_threshold = 0.3
text_threshold = 0.25

boxes, scores, labels = predict(
    model, 
    img_tensor, 
    caption=caption,
    box_threshold=box_threshold,
    text_threshold=text_threshold
)

# -------------------
# 4. Display Results
# -------------------

print(f"Found {len(boxes)} bounding boxes.")
for i, (box, score) in enumerate(zip(boxes, scores)):
    box = box.cpu().detach().numpy()
    score = float(score.cpu())
    print(f"Box {i}: {box}, Score: {score:.4f}")

print("✅ GroundingDINO test completed.")