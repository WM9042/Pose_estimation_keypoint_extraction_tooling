import numpy as np
import torch
from pathlib import Path

def tensor_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)}")


def validate_path(path: str, description: str = "Path", is_file: bool = True):
    p = Path(path)
    
    if not p.exists():
        raise FileNotFoundError(f"{description} does not exist: {p}")
    
    if is_file and not p.is_file():
        raise IsADirectoryError(f"{description} is a directory, expected a file: {p}")
    
    if not is_file and not p.is_dir():
        raise NotADirectoryError(f"{description} is a file, expected a directory: {p}")


def process_keypoints(keypoints, scores, target_landmark_count=133):
    """Processes keypoints and scores into a consistent (num_keypoints, 3) format."""
    keypoints, scores = tensor_to_numpy(keypoints), tensor_to_numpy(scores)

    # Ensure correct shape: remove batch dimension if present
    if keypoints.ndim == 3:
        keypoints = keypoints[0]
    if scores.ndim == 2:
        scores = scores[0]

    # Ensure scores are a column vector
    scores = scores.reshape(-1, 1)

    # Combine keypoints and scores
    keypoints = np.hstack([keypoints, scores])

    # Ensure correct shape with padding
    num_missing = max(0, target_landmark_count - keypoints.shape[0])
    if num_missing > 0:
        padding = np.zeros((num_missing, keypoints.shape[1]))
        keypoints = np.vstack([keypoints, padding])

    return keypoints[:target_landmark_count]

