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
def fine_tune_pose_model(config_path, project_name, experimenter, videos, working_directory="."):
    dlc_service = DLCIntegration(config_path)

    # 1. If you already have a project, you might not need to create_new_project again.
    #    If it’s truly new data, you can create or clone an existing project.

    # 2. Label new frames or import existing labels (depending on your data).
    # dlc_service.label_frames()
    # dlc_service.check_labels()

    # 3. Create or update the training dataset with new data
    dlc_service.create_training_dataset()

    # 4. Train the network, but using a config that points to an existing model
    dlc_service.train_network()

    # 5. Evaluate to see improvements
    dlc_service.evaluate_network()

    print("Fine-tuning pipeline completed!")
