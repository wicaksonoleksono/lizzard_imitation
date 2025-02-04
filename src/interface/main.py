"""
Interface: Entry point (CLI or script) that ties everything together.
"""

import argparse
from application.training import train_pose_model
from application.inference import run_inference_and_convert_2d_to_3d

def main():
    parser = argparse.ArgumentParser(description="Lizard Leg Tracking Pipeline")
    parser.add_argument('--mode', type=str, required=True, choices=['train','infer'],
                        help="Mode to run: train or infer")
    parser.add_argument('--config', type=str, help="Path to DLC config.yaml")
    parser.add_argument('--video', type=str, nargs='+', help="Path(s) to video(s)")
    args = parser.parse_args()

    if args.mode == 'train':
        if not args.config or not args.video:
            print("Please provide --config and --video for training.")
            return
        train_pose_model(
            config_path=args.config,
            project_name="LizardLegProject",
            experimenter="ExperimenterName",
            videos=args.video,
            working_directory="."
        )

    elif args.mode == 'infer':
        if not args.config or not args.video:
            print("Please provide --config and --video for inference.")
            return
        
        # Example parameters
        segment_lengths = [5.0, 4.0]   # L1, L2 in 'some' unit
        initial_params = [0.0, 1.57, 1.57]  # z_armpit, theta1, theta2 in radians

        results_3d = run_inference_and_convert_2d_to_3d(
            config_path=args.config,
            video_list=args.video,
            segment_lengths=segment_lengths,
            initial_params=initial_params
        )

        for vid, frames_3d in results_3d.items():
            print(f"Video: {vid}, 3D frames computed: {len(frames_3d)}")
            # Do something with frames_3d, e.g. save to file, analyze stats, etc.

if __name__ == "__main__":
    main()
