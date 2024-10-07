import os
import pandas as pd
import numpy as np
from .kinematics import calculate_acceleration, calculate_velocity

import os
import numpy as np
import pandas as pd

def load_data(dataset_path, step_size=6, participant_group='s'):
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
        for task, label in zip(['Chopping', 'Sawing', 'Slicing'], [0, 1, 2]):
            file_path = os.path.join(dataset_path, participant_id, f'{task}.csv')
            df = pd.read_csv(file_path)
            df = df.iloc[::step_size]
            landmarks = df.iloc[:, 1:max_landmarks + 1].values  # Adjust slicing to match the number of landmarks
            velocities = calculate_velocity(landmarks, step_size)
            accelerations = calculate_acceleration(velocities, step_size)
            features = np.hstack((landmarks, velocities, accelerations))
            combined_features.append(features)
            labels.extend([label] * features.shape[0])

    return np.vstack(combined_features), np.array(labels)
