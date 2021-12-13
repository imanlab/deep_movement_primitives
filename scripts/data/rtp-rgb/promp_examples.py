"""
Show the original trajectories overlayed to the reconstructed ProMP trajectories.

This gives an idea of how good the ProMP is at reconstructing the trajectory.
"""

import matplotlib.pyplot as plt
import numpy as np

from utils.promp import ProMP
from utils.data import get_valid_ids, load_array_from_json


TRAJ_DIR = "data/rtp-rgb/trajectories"
N_BASIS = 8
N_DOF = 7
N_T = 150


def main():
    # Load trajectories.
    id_list = get_valid_ids([TRAJ_DIR])
    trajectories = load_array_from_json(TRAJ_DIR, "joint_position", id_list, slicer=np.s_[..., 0:N_DOF])

    # Initalize the ProMP
    promp = ProMP(N_BASIS, N_DOF, N_T)
    t_rec = np.linspace(0, 1, N_T)
    for traj in trajectories:
        t_orig = np.linspace(0, 1, traj.shape[0])
        # Reconstruct the trajectory.
        traj_rec = promp.trajectory_from_weights(promp.weights_from_trajectory(traj))
        # Show a comparison between original and reconstructed trajectories.
        cols = 4
        rows = N_DOF // cols + 1
        _, axs = plt.subplots(rows, cols)
        for i in range(N_DOF):
            axs[i // cols, i % cols].plot(t_orig, traj[:, i], t_rec, traj_rec[:, i])
            axs[i // cols, i % cols].legend(("Original", "Reconstructed"))
        plt.show()


if __name__ == "__main__":
    main()
