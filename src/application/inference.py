"""
Application: High-level use case for running inference on new videos.
"""

import numpy as np
from domain.pose_estimation import Pose2D, Pose3D
from domain.lizard_kinematics import process_frame
from infrastructure.dlc_integration import DLCIntegration

def run_inference_and_convert_2d_to_3d(config_path, video_list, segment_lengths, initial_params):
    """
    1) Runs DLC inference on each video.
    2) Retrieves the 2D keypoints.
    3) Runs the 2D→3D conversion using domain logic (process_frame).
    """
    dlc_service = DLCIntegration(config_path)

    # 1. Analyze videos with the trained model
    dlc_service.analyze_videos(video_list, videotype='mp4')

    # Optionally create labeled videos for quick check
    dlc_service.create_labeled_video(video_list, videotype='mp4')

    # 2. Suppose each 'video' has an associated .h5 file with 2D poses
    #    We'll assume a 1:1 mapping (video_name -> video_nameDLC.h5)
    #    Adjust as needed for your naming convention.
    results = {}
    for video_path in video_list:
        h5_result_path = video_path.replace('.mp4','DLC_res.h5')  # example
        frames_2d = dlc_service.get_pose2d_from_h5(h5_result_path)

        frames_3d = []
        for frame_keypoints in frames_2d:
            # Convert list -> Pose2D object
            pose_2d = Pose2D(np.array(frame_keypoints))
            
            # In your domain, define which keypoints are "armpit" etc.
            # Example: armpit = pose_2d.get_keypoint(0)
            # We'll assume base2d is the "armpit" in 2D
            base2d = pose_2d.get_keypoint(0)
            
            # Observed 2D for all joints
            observed_2d = pose_2d.as_numpy()

            # Run domain logic to get 3D
            optimized_params, estimated_3d = process_frame(
                observed_2d,
                base2d,
                segment_lengths,
                initial_params
            )
            frames_3d.append(estimated_3d)

        # Store the 3D results
        results[video_path] = frames_3d

    return results
