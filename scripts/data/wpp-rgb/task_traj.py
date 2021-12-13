import json
import os
import numpy as np

from utils.data import get_valid_ids, load_array_from_json
from utils.kinematics.np import joint_to_task_trajectories


N_DOF = 7
TRAJ_DIR = "data/wpp-rgb/trajectories"


def main():

    id_list = get_valid_ids([TRAJ_DIR])
    joint_trajectories = load_array_from_json(TRAJ_DIR, "joint_position", id_list, slicer=np.s_[..., 0:N_DOF])
    task_trajectories = [joint_to_task_trajectories(joint_traj) for joint_traj in joint_trajectories]

    for id, task_traj in zip(id_list, task_trajectories):
        src_path = os.path.join(TRAJ_DIR, id + ".json")
        dst_path = src_path
        with open(src_path, "r") as f:
            json_data = json.load(f)
        json_data["ee_pose_wrt_base"] = task_traj.tolist()

        json.dump(json_data, open(dst_path, "w"), indent=4, sort_keys=True)
    return


if __name__ == "__main__":
    main()
