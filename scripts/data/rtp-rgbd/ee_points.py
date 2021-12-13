import os
import json
import numpy as np
import matplotlib.pyplot as plt


TRAJ_DIR = "data/rtp-rgbd/trajectories"
N_JOINTS = 7
JOINT_POS_KEY = "joint_position"
CART_POS_KEY = "base_to_eef_position"

id_by_region = {}
last_ee_coords = {}

for traj_file in os.listdir(TRAJ_DIR):

    # Get the ID of the sample.
    id, _ = os.path.splitext(traj_file)

    # Load joint position.
    traj_file_path = os.path.join(TRAJ_DIR, traj_file)
    # Load true cartesian positions.
    cart_pos = np.array(json.load(open(traj_file_path))[CART_POS_KEY])
    last_cart = cart_pos[-1, :]

    # Get the region from the filename.
    region = traj_file[0]
    # Save the (x,y) coordinates of the last point.
    if region not in last_ee_coords.keys():
        last_ee_coords[region] = []
    last_ee_coords[region].append(last_cart[[0, 1]])

    # Save the ID.
    if region not in id_by_region.keys():
        id_by_region[region] = []
    id_by_region[region].append(id)

colors = ["red", "green", "blue", "purple"]
markers = ["*", "o", "<", ">"]

regions = list(sorted(set(last_ee_coords.keys())))
for region, color, marker in zip(regions, colors, markers):
    last_ee_coords[region] = np.array(last_ee_coords[region])

    plt.scatter(last_ee_coords[region][:, 0], last_ee_coords[region][:, 1], c=color, label=f"Region {region}", marker=marker)
    # Annotate each point with its ID.
    for id, x, y in zip(id_by_region[region], last_ee_coords[region][:, 0], last_ee_coords[region][:, 1]):
        plt.annotate(id, (x, y))

plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend([f"Reg. {region}" for region in regions])
plt.show()
