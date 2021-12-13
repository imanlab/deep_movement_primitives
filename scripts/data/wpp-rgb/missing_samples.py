"""
Check for missing samples in the RTP-RGBD dataset.
"""

import os

from utils.data import check_missing_samples

DATASET_DIR = "data/wpp-rgb/"
DATA_DIRS = ["trajectories", "img_resized"]

if __name__ == "__main__":
    dirs = [os.path.join(DATASET_DIR, d) for d in DATA_DIRS]
    check_missing_samples(dirs)
