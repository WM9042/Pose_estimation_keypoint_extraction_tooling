import torch
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from mmpose.apis.inference import init_model


def setup_pose_model(config_file: str, checkpoint_file: str):
    """Initializes the MMPOSE model."""
    
      
    if not Path(config_file).exists():
        tqdm.write(f"[ERROR] Config file not found: {config_file}")
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not Path(checkpoint_file).exists():
        tqdm.write(f"[ERROR] Checkpoint file not found: {checkpoint_file}")
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    
    device = "cuda" if torch.cuda.is_available() else "cpu" #Checks cuda enabled GPU and if not found uses cpu
    tqdm.write(f"[INFO] Initializing model on {device}...")

    try: 
        pose_model = init_model(config_file, checkpoint_file, device=device)
        tqdm.write("[INFO] Model loaded successfully.")
        return pose_model
    except FileNotFoundError as e:
        tqdm.write(f"[ERROR] File not found: {e}")
        raise
    except RuntimeError as e:
        tqdm.write(f"[ERROR] Runtime error during model initialization: {e}")
        raise
    except Exception as e:
        tqdm.write(f"[ERROR] Unexpected error initializing model: {e}")
        return None  
