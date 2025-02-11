import argparse
from .utils import validate_path


def parse_arguments():
    """Parses command-line arguments and validates paths."""
    parser = argparse.ArgumentParser(
        description="Extract skeleton data from raw videos using a heatmap-based model with JSON splits."
    )
    parser.add_argument("--raw_videos_dir", type=str, required=True, help="Directory containing raw video files.")
    parser.add_argument("--skeleton_output_dir", type=str, required=True, help="Directory to save output skeleton data.")
    parser.add_argument("--train_json", type=str, required=True, help="Path to train.json.")
    parser.add_argument("--val_json", type=str, required=True, help="Path to val.json.")
    parser.add_argument("--test_json", type=str, required=True, help="Path to test.json.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the model configuration file.")
    parser.add_argument("--checkpoint_file", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--target_landmarks", type=int, default=133, help="Number of landmarks to extract.")
    parser.add_argument("--input_size", type=int, nargs=2, default=(192, 256), help="Model input size.")


    args = parser.parse_args()

    validate_path(args.config_file, "Config file", is_file=True)
    validate_path(args.checkpoint_file, "Checkpoint file", is_file=True)
    validate_path(args.videos_dir, "Video directory", is_file=False)
    validate_path(args.output_dir, "Output directory", is_file=False)
    validate_path(args.train_json, "Training split JSON", is_file=True)
    validate_path(args.val_json, "Validation split JSON", is_file=True)
    validate_path(args.test_json, "Test split JSON", is_file=True)

    return args
