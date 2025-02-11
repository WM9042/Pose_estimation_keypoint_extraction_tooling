import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from .video_reader import open_video_capture
from .preprocessing import preprocess_frame
from .inference import run_pose_inference

def extract_keypoints_from_video(video_path: Path, pose_model, target_landmark_count=133, input_size=(192, 256)):
    """Extracts skeleton keypoints from a video using a pose estimation model."""
    with open_video_capture(video_path) as cap:
        if cap is None:
            return None
        
        keypoints_list = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for _ in tqdm(range(total_frames), desc=f"Processing {video_path.name}", unit="frame"):
            ret, frame = cap.read()
            if not ret or frame is None:
                tqdm.write(f"Warning: Failed to read frame from {video_path}. Skipping remaining frames.")
                break

            try:
                frame_resized, bbox = preprocess_frame(frame, input_size)
                keypoints = run_pose_inference(pose_model, frame_resized, bbox, target_landmark_count)
                keypoints_list.append(keypoints)

            except (ValueError, RuntimeError) as e:
                tqdm.write(f"Error processing frame in {video_path}: {e}")
                continue
            except Exception as e:
                raise RuntimeError(f"Unexpected error processing {video_path}") from e

        return np.stack(keypoints_list, axis=0) if keypoints_list else None

def process_videos_in_folder(input_folder: str, output_folder: str, pose_model, split_dict, 
                             target_landmark_count=133, input_size=(192, 256)):
    """Processes all videos in the input folder and extracts skeleton data."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Ensure train/val/test folders exist
    for split in ["train", "val", "test"]:
        (output_folder / split).mkdir(parents=True, exist_ok=True)


    
    video_files = [f for f in input_folder.rglob("*") if f.suffix.lower() in {".mp4", ".avi", ".mov"}]

    with tqdm(total=len(video_files), desc="Processing videos", unit="video") as overall_pbar:
        for video_path in video_files:
            video_id = video_path.stem  
            split_folder = output_folder / split_dict.get(video_id, "train")
            output_path = split_folder / f"{video_id}.npy"

            if output_path.exists():
                tqdm.write(f"Skipping {video_id}: Already processed.")
                continue

           
            keypoints_array = extract_keypoints_from_video(video_path, pose_model, target_landmark_count, input_size)
            if keypoints_array is not None:
                np.save(output_path, keypoints_array)
                tqdm.write(f"Saved keypoints: {output_path}")

            overall_pbar.update(1)
