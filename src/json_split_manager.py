import json
import re
from pathlib import Path
from tqdm import tqdm

def load_json_file(json_path: str) -> dict | None:
    """Loads a JSON file and returns its content, or None if invalid."""
    p = Path(json_path)
    try:
        with Path(json_path).open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        tqdm.write(f"Error: JSON file not found - {json_path}. Skipping...")
        return None
    except json.JSONDecodeError as e:
        tqdm.write(f"Error: Invalid JSON format in {json_path}. Skipping file. Details: {e}")
        return None

def load_split_info(train_json: str, val_json: str, test_json: str) -> dict[str, str]:
    """Parses dataset split JSON files and returns a mapping of video IDs to dataset splits."""
    split_dict = {}
    video_id_pattern = re.compile(r"(?<=v=).{11}")

    def process_json(json_file: str, split: str):
        data = load_json_file(json_file)
        if data is None:
            return
        
        for item in data:
            url = item.get("url")
            if url:
                match = video_id_pattern.search(url)
                if match:
                    video_id = match.group().strip()
                    if len(video_id) == 11:  # YouTube IDs are always 11 characters
                        split_dict[video_id] = split
                    else:
                        tqdm.write(f"Warning: Extracted invalid video ID '{video_id}' from {json_file}")
                else:
                    tqdm.write(f"Warning: Could not extract video ID from {json_file}")


    for json_file, split in zip([train_json, val_json, test_json], ["train", "val", "test"]):
        process_json(json_file, split)

    return split_dict
