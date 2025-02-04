"""
Domain: Lizard kinematics and 2D→3D conversion logic.
"""

import numpy as np
from scipy.optimize import minimize

def forward_kinematics(params, base2d, segment_lengths):
    """
    Given a set of parameters, compute the 3D positions of the joints.
    
    Parameters
    ----------
    params : list or np.array
        A vector of parameters, for example: [z_armpit, theta1, theta2].
    base2d : np.array
        The (x,y) coordinate of the 'armpit' (2D).
    segment_lengths : list
        [L1, L2] for the two limb segments.

    Returns
    -------
    positions : np.array (3, 3)
        3D positions of the armpit, elbow, and feet.
    """
    z_armpit, theta1, theta2 = params
    L1, L2 = segment_lengths

    pos_armpit = np.array([base2d[0], base2d[1], z_armpit])
    
    # Armpit → Elbow
    x_elbow = pos_armpit[0] + L1 * np.cos(theta1)
    y_elbow = pos_armpit[1]  # no lateral change assumed
    z_elbow = pos_armpit[2] + L1 * np.sin(theta1)
    pos_elbow = np.array([x_elbow, y_elbow, z_elbow])
    
    # Elbow → Feet
    x_feet = pos_elbow[0] + L2 * np.cos(theta2)
    y_feet = pos_elbow[1]   # no lateral change assumed
    z_feet = pos_elbow[2] + L2 * np.sin(theta2)
    pos_feet = np.array([x_feet, y_feet, z_feet])
    
    return np.stack([pos_armpit, pos_elbow, pos_feet], axis=0)

def objective_function(params, observed_2d, base2d, segment_lengths):
    """
    Compute the error between the projected 3D joints and the observed 2D data.
    Add a penalty term to enforce a constraint on the armpit joint angle.
    """
    positions_3d = forward_kinematics(params, base2d, segment_lengths)
    projected_2d = positions_3d[:, :2]

    # Reprojection error
    reprojection_error = np.sum((projected_2d - observed_2d)**2)

    # Constraint: armpit→elbow angle near 90° (for example)
    desired_theta1 = np.pi / 2
    angle_penalty = 1000 * (params[1] - desired_theta1)**2

    total_error = reprojection_error + angle_penalty
    return total_error

def process_frame(observed_2d, base2d, segment_lengths, initial_params):
    """
    Given a single frame of 2D data, estimate the missing parameters.
    
    Returns
    -------
    optimized_params : np.array
        The best-fit parameters [z_armpit, theta1, theta2].
    estimated_3d : np.array
        The resulting 3D joint positions from forward kinematics.
    """
    result = minimize(
        objective_function,
        initial_params,
        args=(observed_2d, base2d, segment_lengths),
        method='L-BFGS-B'
    )
    optimized_params = result.x
    estimated_3d = forward_kinematics(optimized_params, base2d, segment_lengths)
    return optimized_params, estimated_3d
