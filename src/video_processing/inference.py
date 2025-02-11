import numpy as np
from .utils import process_keypoints
from mmpose.apis.inference import inference_topdown

def run_pose_inference(pose_model, frame, bbox, target_landmark_count):
    """Runs pose estimation model and extracts keypoints."""
    results = inference_topdown(pose_model, frame, [bbox], bbox_format="xyxy")

    if isinstance(results, list) and results:
        pred_instance = getattr(results[0], "pred_instances", None)
        if pred_instance and hasattr(pred_instance, "keypoints"):
            return process_keypoints(pred_instance.keypoints, pred_instance.keypoint_scores, target_landmark_count)

    return np.zeros((target_landmark_count, 3), dtype=np.float32)
