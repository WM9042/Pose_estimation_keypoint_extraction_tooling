import cv2
import numpy as np

def preprocess_frame(frame: np.ndarray, input_size: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Converts BGR to RGB and resizes the frame."""
    frame_resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), input_size, interpolation=cv2.INTER_AREA)
    bbox = np.array([0, 0, frame_resized.shape[1], frame_resized.shape[0]], dtype=np.float32)
    return frame_resized, bbox
