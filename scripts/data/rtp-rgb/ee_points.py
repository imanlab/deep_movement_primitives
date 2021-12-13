"""
Show the distribution of the final trajectory points on the (x, y) plane
among the trajectories in the RTP-RGB dataset.
"""

import os
import json
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


TRAJ_DIR = "data/rtp-rgb/trajectories/"
N_JOINTS = 7
JOINT_POS_KEY = "joint_position"
TMATRIX_KEY = "transf_matrix"


def main():
    last_ee_coords = {}
    idx_test = {
        "1": np.array([190, 140, 186, 63, 38, 21, 137, 168, 270, 308]),
        "2": np.array([266, 267, 482, 478, 481, 445, 493, 65, 420]),
        "3": np.array([286, 234, 432, 304, 463, 150, 487, 471, 272, 476]),
        "4": np.array([59, 58, 88, 45, 90, 73, 176, 97, 94, 238])
    }
    id_by_regions = {
        "1": list(map(lambda i: str(i+1).zfill(3), idx_test["1"])),
        "2": list(map(lambda i: str(i+1).zfill(3), idx_test["2"])),
        "3": list(map(lambda i: str(i+1).zfill(3), idx_test["3"])),
        "4": list(map(lambda i: str(i+1).zfill(3), idx_test["4"]))
    }
    id_lists = {}

    # Load all trajectories.
    for traj_file in natsorted(os.listdir(TRAJ_DIR)):
        filename, _ = os.path.splitext(traj_file)

        traj_file_path = os.path.join(TRAJ_DIR, traj_file)
        # Load xyz position of the last point of the trajectory from the transformation matrix.
        tmatrix = np.array(json.load(open(traj_file_path))[TMATRIX_KEY])
        pos_xyz_last = tmatrix[-1, 0:3, 3]

        # Get the region each point belongs to, according tho the ID.
        for region, ids in id_by_regions.items():
            if filename in ids:
                region_id = region
                break
        else:
            region_id = "train"
        # Store all ID by region.
        if region_id not in id_lists.keys():
            id_lists[region_id] = []
        id_lists[region_id].append(filename)

        # Save the (x,y) coordinates by region.
        if region_id not in last_ee_coords.keys():
            last_ee_coords[region_id] = []
        last_ee_coords[region_id].append(pos_xyz_last[[0, 1]])

    # Colors of each region. Last one is for "training".
    regions_colors = ["red", "green", "blue", "purple", "orange"]

    # Draw patches to highlight the regions.
    d1 = mpatches.Circle((0.425, -0.03), 0.07, alpha=0.1, facecolor=regions_colors[0])
    d2 = mpatches.Circle((0.550, +0.08), 0.05, alpha=0.1, facecolor=regions_colors[1])
    d3 = mpatches.Circle((0.550, -0.10), 0.04, alpha=0.1, facecolor=regions_colors[2])
    d4 = mpatches.Circle((0.400, -0.23), 0.07, alpha=0.1, facecolor=regions_colors[3])

    regions = list(sorted(set(last_ee_coords.keys())))
    for region, color in zip(regions, regions_colors):
        # Plot all final points with the color of their region.
        last_ee_coords[region] = np.array(last_ee_coords[region])
        plt.scatter(last_ee_coords[region][:, 0], last_ee_coords[region][:, 1], c=color, label=f"Region {region}")
        # Annotate each point with its ID.
        for id, x, y in zip(id_lists[region], last_ee_coords[region][:, 0], last_ee_coords[region][:, 1]):
            plt.annotate(id, (x, y))

    plt.legend([f"Reg. {region}" for region in regions])

    plt.gca().add_patch(d1)
    plt.gca().add_patch(d2)
    plt.gca().add_patch(d3)
    plt.gca().add_patch(d4)
    plt.show()

    # Repeat, but now plot all points the same color.
    regions = list(sorted(set(last_ee_coords.keys())))

    # Draw patches to highlight the regions.
    d1 = mpatches.Circle((0.425, -0.03), 0.07, alpha=0.1, facecolor=regions_colors[0])
    d2 = mpatches.Circle((0.550, +0.08), 0.05, alpha=0.1, facecolor=regions_colors[1])
    d3 = mpatches.Circle((0.550, -0.10), 0.04, alpha=0.1, facecolor=regions_colors[2])
    d4 = mpatches.Circle((0.400, -0.23), 0.07, alpha=0.1, facecolor=regions_colors[3])

    for region in regions:
        # Plot all final points with the color of their region.
        last_ee_coords[region] = np.array(last_ee_coords[region])
        plt.scatter(last_ee_coords[region][:, 0], last_ee_coords[region][:, 1], c="orange", label=f"Region {region}")
        # Annotate each point with its ID.
        for id, x, y in zip(id_lists[region], last_ee_coords[region][:, 0], last_ee_coords[region][:, 1]):
            plt.annotate(id, (x, y))

    plt.legend([f"Reg. {region}" for region in regions])

    plt.gca().add_patch(d1)
    plt.gca().add_patch(d2)
    plt.gca().add_patch(d3)
    plt.gca().add_patch(d4)
    plt.show()


if __name__ == "__main__":
    main()
