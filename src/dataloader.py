import os
import pandas as pd
import numpy as np
from .kinematics import calculate_acceleration, calculate_velocity

import os
import numpy as np
import pandas as pd

def load_data(dataset_path, step_size=1, participant_group='s'):
    """Load and combine data from selected participants ('s' or 'w') with a step size, defaulting to 5 FPS."""
    combined_features = []
    labels = []

    if participant_group == 's':
        max_landmarks = 33
        participant_range = range(1, 23)
        participant_prefix = 'S'
    elif participant_group == 'w':
        max_landmarks = 11
        participant_range = range(1, 11)
        participant_prefix = 'W'
    else:
        raise ValueError("participant_group must be 's' or 'w'")

    for participant in participant_range:
        participant_id = f'{participant:03d}'
        for task, label in zip(['Diving-Side', 'Golf-Swing-Back', 'Golf-Swing-Front', 'Golf-Swing-Side', 'Kicking-Front', 'Kicking-Side', 'Lifting', 'Riding-Horse', 'Run-Side', 'SkateBoarding-Front', 'Swing-Bench', 'Swing-SideAngle', 'Walk-Front'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
            file_path = os.path.join(dataset_path, participant_id, f'{task}_posture.csv')
            if not os.path.isfile(file_path):
                continue
            df = pd.read_csv(file_path)
            df = df.iloc[::step_size]
            landmarks = df.iloc[:, 1:max_landmarks + 1].values  # Adjust slicing to match the number of landmarks
            velocities = calculate_velocity(landmarks, step_size)
            accelerations = calculate_acceleration(velocities, step_size)
            features = np.hstack((landmarks, velocities, accelerations))
            combined_features.append(features)
            labels.extend([label] * features.shape[0])

    return np.vstack(combined_features), np.array(labels)

