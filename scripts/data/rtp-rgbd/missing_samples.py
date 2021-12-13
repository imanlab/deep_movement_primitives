"""
Check for missing samples in the RTP-RGBD dataset.
"""

import os

from utils.data import check_missing_samples

DATASET_DIR = "data/rtp-rgbd/"
DATA_DIRS = ["color", "depth", "pointclouds", "trajectories", "trajectories_task", "color_img"]

if __name__ == "__main__":
    dirs = [os.path.join(DATASET_DIR, d) for d in DATA_DIRS]
    check_missing_samples(dirs)
