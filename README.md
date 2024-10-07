# PE-USGC: Posture Estimation-based Unsupervised Spatial Gaussian Clustering

## Overview
**PE-USGC** is an algorithm designed for the supervised classification of near-duplicate human motion using an unsupervised clustering approach. It leverages spatial Gaussian clustering and posture estimation to categorize and classify human motion data more effectively.

## YouTube
https://www.youtube.com/watch?v=1yVW5Dxf3lw

## Features
- **Posture Estimation**: Extracts and analyzes human posture features from input data.
- **Unsupervised Spatial Clustering**: Uses Gaussian clustering to identify similar posture groups.
- **Supervised Classification**: Classifies near-duplicate human motions based on clustered features.

## Repository Structure
- `data/` - Sample datasets for motion and posture analysis.
- `models/` - Pre-trained models used in clustering and classification.
- `main.py` - Main script to run clustering and classification pipeline.
- `src/` - Source code for PE-USGC algorithm implementation.
  - `dataloader.py` - Script responsible for loading and preprocessing motion data.
  - `kinematics.py` - Provides utility functions for kinematic analysis and feature extraction from motion data.
  - `train.py` - Contains functions and classes to train supervised models on clustered data.
  - `visualize.py` - Includes methods for visualizing clustering results, motion patterns, and classification outcomes.

## File Descriptions
### `src/dataloader.py`
This module handles the loading and preprocessing of motion data. It includes methods for data augmentation, filtering, and normalization to ensure compatibility with the posture estimation and clustering algorithms.

### `src/kinematics.py`
The kinematics module provides helper functions for extracting kinematic features from motion data, such as joint angles and velocities. These features are used in the posture estimation and clustering process.

### `main.py`
The entry point of the repository, `main.py` initiates the complete PE-USGC pipeline. It integrates data loading, clustering, and classification processes and outputs the final results.

### `src/train.py`
This script contains the training logic for the supervised classification models. It includes hyperparameter tuning, cross-validation, and model evaluation using the clustered posture data.

### `src/visualize.py`
The visualization module provides tools to display clustering and classification results. It includes functions for plotting Gaussian clusters, visualizing motion trajectories, and generating performance metrics.

## Getting Started
To run the PE-USGC algorithm:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/iyer1729/PE-USGC.git
    ```
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the main script**:
    ```bash
    python main.py
    ```

## Dependencies
- Python 3.7 or higher
- Numpy
- Pandas
- Matplotlib
- Scikit-learn
- Joblib

## License
This project is licensed under the MIT License.

## Contact
For any questions or feedback, please contact Hari Iyer (hniyer@asu.edu).
