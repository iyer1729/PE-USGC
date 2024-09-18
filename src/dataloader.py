import os
import pandas as pd
import numpy as np
from .kinematics import calculate_acceleration, calculate_velocity

task_landmark_modes = {
    'Chopping': [11, 13, 15, 21, 17, 19],
    'Grating': [11, 13, 15, 21, 17, 19],
    'Mixing': [11, 13, 15, 21, 17, 19],
    'Lifting': [12, 14, 16, 22, 18, 20],
    'Oven': [12, 14, 16, 22, 18, 20],
    'Peeling': [11, 13, 15, 21, 17, 19],
    'Rolling': [11, 13, 15, 21, 17, 19],
    'Sauteing': [11, 13, 15, 21, 17, 19],
    'Sawing': [11, 13, 15, 21, 17, 19],
    'Slicing': [11, 13, 15, 21, 17, 19],
    'Stirring': [11, 13, 15, 21, 17, 19],
    'Whisking': [11, 13, 15, 21, 17, 19]
}

def get_landmark_indices_to_retain(landmark_indices, landmarks_per_point=3):
    indices_to_retain = []
    for idx in landmark_indices:
        start_idx = idx * landmarks_per_point
        indices_to_retain.extend([start_idx, start_idx + 1, start_idx + 2])
    return indices_to_retain

def load_data(dataset_path, step_size=6, retain_only_obscured=False, participant_group='s'):
    """Load and combine data from selected participants ('s' or 'w') with a step size, defaulting to 5 FPS."""
    combined_features = []
    labels = []

    if participant_group == 's':
        max_landmarks = 33
        participant_range = range(1, 25)
        participant_prefix = 'S'
    elif participant_group == 'w':
        max_landmarks = 11
        participant_range = range(1, 11)
        participant_prefix = 'W'
    else:
        raise ValueError("participant_group must be 's' or 'w'")

    for participant in participant_range:
        participant_id = f'{participant_prefix}{participant:02d}'
        for task, label in zip(
                ['Chopping', 'Grating', 'Lifting', 'Mixing', 'Oven', 'Peeling', 'Rolling', 'Sauteing', 'Sawing',
                 'Slicing', 'Stirring', 'Whisking'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]):
            file_path = os.path.join(dataset_path, participant_id, f'{task}.csv')
            df = pd.read_csv(file_path)
            df = df.iloc[::step_size]

            if retain_only_obscured and participant_group == 's':
                obscured_landmarks = task_landmark_modes.get(task, [])
                indices_to_retain = get_landmark_indices_to_retain(obscured_landmarks)
                landmarks = df.iloc[:, 1:max_landmarks * 3 + 1].values  # Adjust for all 3 columns per landmark
                landmarks = landmarks[:, indices_to_retain]
            else:
                landmarks = df.iloc[:, 1:max_landmarks * 3 + 1].values

            velocities = calculate_velocity(landmarks, step_size)
            accelerations = calculate_acceleration(velocities, step_size)
            features = np.hstack((landmarks, velocities, accelerations))
            combined_features.append(features)
            labels.extend([label] * features.shape[0])

    return np.vstack(combined_features), np.array(labels)