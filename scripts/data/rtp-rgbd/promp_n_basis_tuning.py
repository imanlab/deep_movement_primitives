"""
Use the ProMPTuner class to find a good choice for the n_basis parameter,
considering the trajectories in the RTP-RGBD dataset.
"""

import numpy as np

from utils.promp import ProMP, ProMPTuner
from utils.data import get_valid_ids, load_array_from_json


TRAJ_DIR = "data/rtp-rgbd/trajectories"
JOINT_KEY = "joint_position"
N_BASIS = 8
N_DOF = 7
N_T = 150


def main():
    promp = ProMP(N_BASIS, N_DOF, N_T)

    # Load the trajectories.
    id_list = get_valid_ids([TRAJ_DIR])
    trajectories = load_array_from_json(TRAJ_DIR, JOINT_KEY, id_list, slicer=np.s_[..., 0:N_DOF])

    # Tune the n_basis parameter.
    promp_tuner = ProMPTuner(trajectories, promp)
    promp_tuner.tune_n_basis(2, 15, 1, show=True)


if __name__ == "__main__":
    main()
