# ~/Detect-Track/utils/visualize.py

import cv2
import numpy as np

def plot_tracking(image, tlwhs, obj_ids=None, frame_id=0):
    """
    Draw bounding boxes with object IDs on the image.

    Args:
        image: np.ndarray - input image
        tlwhs: list of bounding boxes (x, y, w, h)
        obj_ids: list of object IDs corresponding to tlwhs
        frame_id: int - frame number
    Returns:
        image: np.ndarray - annotated image
    """
    im = np.ascontiguousarray(np.copy(image))

    for idx, tlwh in enumerate(tlwhs):
        x, y, w, h = tlwh
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        color = (255, 0, 0)
        obj_id = int(obj_ids[idx]) if obj_ids is not None else 0

        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
        cv2.putText(im, str(obj_id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(im, f'Frame: {frame_id}', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    return im
