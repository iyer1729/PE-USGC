import numpy as np

def calculate_velocity(landmarks, step_size):
    """Calculate velocity from landmarks."""
    velocities = (landmarks[1:] - landmarks[:-1]) / (.033 * step_size)
    return np.vstack((velocities, velocities[-1]))  # Pad to match the shape

def calculate_acceleration(velocities, step_size):
    """Calculate acceleration from velocities."""
    accelerations = (velocities[1:] - velocities[:-1]) / (.033 * step_size)
    return np.vstack((accelerations, accelerations[-1]))  # Pad to match the shape
