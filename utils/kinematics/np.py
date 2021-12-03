"""
NumPy implementation of various functions to work with the kinematics of the robot.

The robot is a Franka Emika Panda robotic arm with 7 dof.
"""

import numpy as np
from numpy import cos, sin, sqrt
from scipy.spatial.transform import Rotation


def euler_to_quaternion(r: float, p: float, y: float) -> np.ndarray:
    """Convert (roll, pitch, yaw) -> quaternion.

    Args:
        r (float): Roll [rad].
        p (float): Pitch [rad].
        y (float): Yaw [rad].

    Returns:
        np.ndarray: Orientation quaternion.
    """
    ci = cos(r/2.0)
    si = sin(r/2.0)
    cj = cos(p/2.0)
    sj = sin(p/2.0)
    ck = cos(y/2.0)
    sk = sin(y/2.0)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk
    return np.array([cj*sc-sj*cs, cj*ss + sj*cc, cj*cs - sj*sc, cj*cc + sj*ss])


def T_rpyxyz(r: float, p: float, y: float, X: float, Y: float, Z: float) -> np.ndarray:
    """Calculate the rototranslation matrix for a given pose.

    Inputs are roll, pitch, yaw and (x, y, z) position coordinates.
    See: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Args:
        r (float): Roll [rad].
        p (float): Pitch [rad].
        y (float): Yaw [rad].
        X (float): X coordinate.
        Y (float): Y coordinate.
        Z (float): Z coordinate.

    Returns:
        np.ndarray: The 4x4 rototranslation matrix describing the pose.
    """

    q = euler_to_quaternion(r, p, y)
    qi = q[0]
    qj = q[1]
    qk = q[2]
    qr = q[3]

    norm = sqrt(qi*qi + qj*qj + qk*qk + qr*qr)
    qi = qi / norm
    qj = qj / norm
    qk = qk / norm
    qr = qr / norm

    T = np.array([
        [1 - 2 * (qj*qj + qk*qk), 2 * (qi*qj - qk*qr), 2 * (qi*qk + qj*qr), X],
        [2 * (qi*qj + qk*qr), 1 - 2 * (qi*qi + qk*qk), 2 * (qj*qk - qi*qr), Y],
        [2 * (qi*qk - qj*qr), 2 * (qj*qk + qi*qr), 1 - 2 * (qi*qi + qj*qj), Z],
        [0, 0, 0, 1]
    ])
    return T


def T_rz(theta: float) -> np.ndarray:
    """Calculate the rototranslation matrix for a pure rotation around the z axis.

    See: https://en.wikipedia.org/wiki/Rotation_matrix

    Args:
        theta (float): Rotation angle [rad].

    Returns:
        np.ndarray: The 4x4 rototranslation matrix describing the rotation.
    """
    T = np.array([
        [cos(theta), -sin(theta), 0, 0],
        [sin(theta), cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return T


def rpyxyz_from_T(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a rototranslation matrix into roll pith yaw and cartesian position.

    Args:
        T (np.ndarray): Rototranslation matrix, with size (4, 4).

    Returns:
        tuple[np.ndarray, np.ndarray]: [roll, pitch, yaw], [x, y, z]
    """

    r = list(reversed(Rotation.from_matrix(T[0:3, 0:3]).as_euler("ZYX")))
    t = T[0:3, 3]

    return np.ndarray(r), np.ndarray(t)


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """Compute the transformation matrix of the end-effector for a given configuration (MoveIt frame).

    Values from URDF file https://github.com/StanfordASL/PandaRobot.jl/blob/master/deps/Panda/panda.urdf

    Args:
        q (np.ndarray): Vector of joint positions with shape (n_joints, ).

    Returns:
        np.ndarray: The 4x4 rototranslation matrix describing the kinematic chain from the base to the EE.
    """

    # Link 0 through 7 are panda robots links. Each one corresponds to a DOF of the robot arm.
    T01 = np.matmul(T_rpyxyz(0, 0, 0, 0, 0, 0.333), T_rz(q[0]))
    T12 = np.matmul(T_rpyxyz(-1.57079632679, 0, 0, 0, 0, 0), T_rz(q[1]))
    T23 = np.matmul(T_rpyxyz(1.57079632679, 0, 0, 0, -0.316, 0), T_rz(q[2]))
    T34 = np.matmul(T_rpyxyz(1.57079632679, 0, 0, 0.0825, 0, 0), T_rz(q[3]))
    T45 = np.matmul(T_rpyxyz(-1.57079632679, 0, 0, -0.0825, 0.384, 0), T_rz(q[4]))
    T56 = np.matmul(T_rpyxyz(1.57079632679, 0, 0, 0, 0, 0), T_rz(q[5]))
    T67 = np.matmul(T_rpyxyz(1.57079632679, 0, 0, 0.088, 0, 0), T_rz(q[6]))
    # The following links, instead are rigidly conencted (no associated DOF).
    # Link 8 is a dummy link, rigidly connected to link 7.
    # Link hand is the panda hand, rigidly connected to link 8.
    # Link leftf is the panda left finger, rigidly connected to link hand (it could have the left finger DOF, but it's set to 0).
    # Link xela is the center of the contact patch of the XELA sensor, rigidly connected to link leftf.
    T78 = T_rpyxyz(0, 0, -np.pi/4, 0, 0, 0.107)
    T8_cs = T_rpyxyz(0, 0, -np.pi/4, 0, 0, 0)
    Tcs_hand = T_rpyxyz(0, 0, 0, 0, 0, 0.005)
    Thand_leftf = T_rpyxyz(0, 0, 0, 0, 0, 0.0584)
    Tleftf_xela = T_rpyxyz(0, 0, 0, 0.015, 0.01, 0.067)

    T8_xela = np.matmul(T8_cs, np.matmul(Tcs_hand, np.matmul(Thand_leftf, Tleftf_xela)))
    T0_xela = np.matmul(T01, np.matmul(T12, np.matmul(T23, np.matmul(T34, np.matmul(T45, np.matmul(T56, np.matmul(T67, np.matmul(T78, T8_xela))))))))
    return T0_xela


def T_matrix_to_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 4x4 transformation matrix in xyz position vector and orientation quaternion.

    Args:
        T (np.ndarray): The transformation matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: The position vector and the orientation quaternion.
    """

    rotation_matrix = T[0:3, 0:3]
    position_vec = T[0:3, 3]
    rotation = Rotation.from_matrix(rotation_matrix)
    rotation_quat = rotation.as_quat()

    return position_vec, rotation_quat


def joint_to_task_trajectories(traj_joint: np.ndarray) -> np.ndarray:
    """Convert a joint space trajectory into a task space trajectory.

    Args:
        traj_joint (np.ndarray): The joint trajectory, with shape (n_t, n_joints).

    Returns:
        np.ndarray: The task space trajectory (position vector concat orientation quaternion), with shape (n_t, 7)
    """

    n_traj_samples = traj_joint.shape[0]
    # Position vector.
    traj_ee_pose = np.zeros((n_traj_samples, 7))
    for t in range(n_traj_samples):
        T_ee_pose = forward_kinematics(traj_joint[t, :])
        ee_pos, ee_ori = T_matrix_to_pose(T_ee_pose)
        traj_ee_pose[t, :] = np.hstack((ee_pos, ee_ori))

    return traj_ee_pose
