import os
import pandas as pd
import numpy as np
from .kinematics import calculate_acceleration, calculate_velocity

def load_data(dataset_path, step_size=6):
    """Load and combine data from all participants with a step size, default to 5 FPS."""
    combined_features = []
    labels = []
    for participant in range(1, 25):
        participant_id = f'S{participant:02d}'
        for task, label in zip(['Chopping', 'Sawing', 'Slicing'], [0, 1, 2]):
            file_path = os.path.join(dataset_path, participant_id, f'{task}.csv')
            df = pd.read_csv(file_path)
            df = df.iloc[::step_size]
            landmarks = df.iloc[:, 1:].values
            velocities = calculate_velocity(landmarks, step_size)
            accelerations = calculate_acceleration(velocities, step_size)
            features = np.hstack((landmarks, velocities, accelerations))
            combined_features.append(features)
            labels.extend([label] * features.shape[0])
    return np.vstack(combined_features), np.array(labels)
