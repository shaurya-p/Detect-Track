import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'external', 'GroundingDINO')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'external', 'ByteTrack')))

import torch
import cv2
import groundingdino
import yolox

print(torch.cuda.is_available())