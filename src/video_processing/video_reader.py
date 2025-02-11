import cv2
from pathlib import Path
from contextlib import contextmanager

class VideoCaptureError(Exception):
    pass

@contextmanager
def open_video_capture(video_path: Path):
    """Context manager for safely opening and releasing a video file."""
    if not isinstance(video_path, Path):
        raise TypeError(f"Expected video_path to be a Path, got {type(video_path).__name__}")

    try:
        cap = cv2.VideoCapture(str(video_path))
    except Exception as e:
        raise VideoCaptureError(f"Unexpected error while opening video: {e}")

    try:
        if not cap.isOpened():
            raise VideoCaptureError(f"Error: Unable to open video file '{video_path}'.")
        yield cap
    finally:
        cap.release()
