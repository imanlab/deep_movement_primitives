"""
Check for missing samples in the RTP-RGB dataset.
"""

import os

from utils.data import check_missing_samples

DATASET_DIR = "data/rtp-rgb"
DATA_DIRS = ["color_img", "trajectories"]

if __name__ == "__main__":
    dirs = [os.path.join(DATASET_DIR, d) for d in DATA_DIRS]
    check_missing_samples(dirs)
