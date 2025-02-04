"""
Infrastructure: Integrate DeepLabCut functionalities 
(training, inference, etc.) with the domain logic.
"""

import deeplabcut
import os

class DLCIntegration:
    """
    Infrastructure service for interacting with DeepLabCut.
    """
    def __init__(self, config_path):
        """
        config_path: path to the DeepLabCut config.yaml
        """
        self.config_path = config_path

    def create_new_project(self, project_name, experimenter, videos, working_directory="."):
        """
        Example: create a new DeepLabCut project.
        """
        deeplabcut.create_new_project(
            project_name,
            experimenter,
            videos,
            working_directory=working_directory,
            copy_videos=True
        )
        print("DLC Project created successfully!")

    def label_frames(self):
        """
        Example placeholder for labeling frames with DLC's GUI.
        """
        deeplabcut.label_frames(self.config_path)

    def check_labels(self):
        """
        Check the labeled frames in the folder with DLC's labeling GUI.
        """
        deeplabcut.check_labels(self.config_path)

    def create_training_dataset(self):
        """
        Create training dataset from labeled frames.
        """
        deeplabcut.create_training_dataset(self.config_path)

    def train_network(self):
        """
        Train the DLC network.
        """
        deeplabcut.train_network(self.config_path)

    def evaluate_network(self):
        """
        Evaluate the trained DLC network.
        """
        deeplabcut.evaluate_network(self.config_path)

    def analyze_videos(self, video_list, videotype='mp4'):
        """
        Analyze videos with the trained DLC model.
        """
        deeplabcut.analyze_videos(self.config_path, video_list, videotype=videotype)

    def create_labeled_video(self, video_list, videotype='mp4'):
        """
        Create labeled videos for quick visualization.
        """
        deeplabcut.create_labeled_video(self.config_path, video_list, videotype=videotype)
    
    def get_pose2d_from_h5(self, analyzed_video_path):
        """
        Load the 2D pose estimation results from the DLC output (e.g., an .h5 file).
        
        Returns a dictionary or list of domain.Pose2D objects.
        """
        import pandas as pd
        df = pd.read_hdf(analyzed_video_path)
        
        # Suppose your config has 3 keypoints: armpit, elbow, feet
        # We'll parse them into a Pose2D for each frame
        frames_pose2d = []
        for frame_idx in range(len(df.index)):
            # Extract x,y for each keypoint
            # The multi-index columns are something like ('DLC_resnet50LizardJan30shuffle1_500000', 'armpit', 'x')
            # So adapt accordingly.
            # This is just a conceptual example:
            armpit_x = df.iloc[frame_idx][('DLC_network','armpit','x')]
            armpit_y = df.iloc[frame_idx][('DLC_network','armpit','y')]
            elbow_x  = df.iloc[frame_idx][('DLC_network','elbow','x')]
            elbow_y  = df.iloc[frame_idx][('DLC_network','elbow','y')]
            feet_x   = df.iloc[frame_idx][('DLC_network','feet','x')]
            feet_y   = df.iloc[frame_idx][('DLC_network','feet','y')]

            keypoints = [
                [armpit_x, armpit_y],
                [elbow_x,  elbow_y ],
                [feet_x,   feet_y  ]
            ]
            frames_pose2d.append(keypoints)
        
        return frames_pose2d
