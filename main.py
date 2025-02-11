import sys
from src.arg_parser import parse_arguments
from src.json_split_manager import load_split_info
from src.model_init import setup_pose_model
from src.video_processing.video_pipeline import process_videos_in_folder



def main():
    try:
        args = parse_arguments()
        
        split_dict = load_split_info(args.train_json, args.val_json, args.test_json)
        pose_model = setup_pose_model(args.config_file, args.checkpoint_file)

        process_videos_in_folder(
            input_folder=args.videos_dir,
            output_folder=args.output_dir,
            pose_model=pose_model,
            split_dict=split_dict,
            target_landmark_count=args.target_landmarks,
            input_size=tuple(args.input_size),
        )
    except FileNotFoundError as e:
        print(f"Error: Missing file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
