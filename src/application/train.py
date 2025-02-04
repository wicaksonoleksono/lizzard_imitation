"""
Application: High-level use case for training the DeepLabCut model.
"""

from infrastructure.dlc_integration import DLCIntegration

def train_pose_model(config_path, project_name, experimenter, videos, working_directory="."):
    """
    Orchestrates the steps needed to train a new DLC model from scratch.
    """
    dlc_service = DLCIntegration(config_path)

    # 1. Create a new DLC project
    dlc_service.create_new_project(project_name, experimenter, videos, working_directory)

    # 2. Label frames (GUI step)
    # In practice, you'd manually label frames with DLC's GUI. We'll call:
    # dlc_service.label_frames()

    # 3. Check labels
    # dlc_service.check_labels()

    # 4. Create the training dataset
    dlc_service.create_training_dataset()

    # 5. Train the network
    dlc_service.train_network()

    # 6. Evaluate the network
    dlc_service.evaluate_network()

    print("Training pipeline completed!")
