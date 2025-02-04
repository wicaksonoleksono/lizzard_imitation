"""
Domain: Abstraction for pose estimation outputs.
This module provides domain-level structures or logic for managing pose data.
"""

import numpy as np

class Pose2D:
    """
    Value Object representing 2D keypoints for a single frame.
    """
    def __init__(self, keypoints):
        """
        keypoints: np.array of shape (num_joints, 2)
        """
        self.keypoints = keypoints

    def get_keypoint(self, idx):
        return self.keypoints[idx]

    def as_numpy(self):
        return self.keypoints


class Pose3D:
    """
    Value Object representing 3D keypoints for a single frame.
    """
    def __init__(self, keypoints_3d):
        """
        keypoints_3d: np.array of shape (num_joints, 3)
        """
        self.keypoints_3d = keypoints_3d

    def as_numpy(self):
        return self.keypoints_3d
