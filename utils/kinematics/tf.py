"""
NumPy implementation of various functions to work with the kinematics of the robot.

The robot is a Franka Emika Panda robotic arm with 7 dof.
"""

import numpy as np
import tensorflow as tf
from tensorflow_graphics.geometry.transformation import quaternion

from utils.kinematics.np import T_rpyxyz as T_rpyxyz_np


def T_rpyxyz(r: float, p: float, y: float, X: float, Y: float, Z: float) -> tf.Tensor:
    """Calculate the transformation matrix for a given pose as a tensor.

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
        tf.Tensor: The transformation matrix, with shape (4, 4).
    """
    return tf.cast(tf.convert_to_tensor(T_rpyxyz_np(r, p, y, X, Y, Z)), tf.float32)


def make_rotation(alpha: np.ndarray, axis: str) -> np.ndarray:
    """Compute the transformation matrix corresponding to a rotation around the given axis.

    Source: https://stackoverflow.com/a/58979616/3047294

    Args:
        alpha (np.ndarray): The angles of rotation, with shape (A1, ..., An, n_samples).
        axis (str): The axis of rotation. Must be one of "x", "y" or "z".

    Raises:
        ValueError: If an invalid axis is provided.

    Returns:
        np.ndarray: The transformation matrices corresponding to the rotations, with shape (A1, ..., An, n_samples, 4, 4).
    """
    return make_rotation_sincos(tf.math.sin(alpha), tf.math.cos(alpha), axis)


def make_rotation_sincos(sin: tf.Tensor, cos: tf.Tensor, axis: str) -> tf.Tensor:
    """Compute the transformation matrix corresponding to a rotation around the given axis.

    Source: https://stackoverflow.com/a/58979616/3047294

    Args:
        sin (tf.Tensor): Sine of the rotation angles, with shape (A1, ..., An, n_samples).
        cos (tf.Tensor): Cosine of the rotation angles, with shape (A1, ..., An, n_samples).
        axis (str): The axis of rotation. Must be one of "x", "y" or "z".

    Raises:
        ValueError: If an invalid axis is provided.

    Returns:
        tf.Tensor: The transformation matrices corresponding to the rotation, with shape (A1, ..., An, n_samples, 4, 4).
    """

    axis = axis.strip().lower()
    zeros = tf.zeros_like(sin)
    ones = tf.ones_like(sin)
    if axis == 'x':
        rot = tf.stack([
            tf.stack([ones,  zeros, zeros], -1),
            tf.stack([zeros,   cos,  -sin], -1),
            tf.stack([zeros,   sin,   cos], -1),
        ], -2)
    elif axis == 'y':
        rot = tf.stack([
            tf.stack([cos,   zeros,   sin], -1),
            tf.stack([zeros,  ones, zeros], -1),
            tf.stack([-sin,  zeros,   cos], -1),
        ], -2)
    elif axis == 'z':
        rot = tf.stack([
            tf.stack([cos,    -sin, zeros], -1),
            tf.stack([sin,     cos, zeros], -1),
            tf.stack([zeros, zeros,  ones], -1),
        ], -2)
    else:
        raise ValueError('Invalid axis {!r}.'.format(axis))
    last_row = tf.expand_dims(tf.stack([zeros, zeros, zeros], -1), -2)
    last_col = tf.expand_dims(tf.stack([zeros, zeros, zeros, ones], -1), -1)
    return tf.concat([tf.concat([rot, last_row], -2), last_col], -1)


def forward_kinematics(q: tf.Tensor) -> tf.Tensor:
    """Computes the forward kinematic chain given the joint state of the robot.

    Args:
        q (tf.Tensor): The joint state of the robot, with shape (n_batch, n_t, n_joints).

    Returns:
        tf.Tensor: The transformation matrices for the pose of the EE at each time instant, with shape (n_batch, n_t, 4, 4).
    """
    # Create transformation matrices with shape (n_batch, n_t, n_joints, 4, 4).
    rots = make_rotation(q, "z")
    # TODO: Store these matrices in memory (only need to compute them one time).
    T01 = T_rpyxyz(0, 0, 0, 0, 0, 0.333)
    T12 = T_rpyxyz(-1.57079632679, 0, 0, 0, 0, 0)
    T23 = T_rpyxyz(1.57079632679, 0, 0, 0, -0.316, 0)
    T34 = T_rpyxyz(1.57079632679, 0, 0, 0.0825, 0, 0)
    T45 = T_rpyxyz(-1.57079632679, 0, 0, -0.0825, 0.384, 0)
    T56 = T_rpyxyz(1.57079632679, 0, 0, 0, 0, 0)
    T67 = T_rpyxyz(1.57079632679, 0, 0, 0.088, 0, 0)
    T78 = T_rpyxyz(0, 0, -np.pi/4, 0, 0, 0.107)
    T8_cs = T_rpyxyz(0, 0, -np.pi/4, 0, 0, 0)
    Tcs_hand = T_rpyxyz(0, 0, 0, 0, 0, 0.005)
    Thand_leftf = T_rpyxyz(0, 0, 0, 0, 0, 0.0584)
    Tleftf_xela = T_rpyxyz(0, 0, 0, 0.015, 0.01, 0.067)

    T8_xela = tf.matmul(T8_cs, tf.matmul(Tcs_hand, tf.matmul(Thand_leftf, Tleftf_xela)))

    # TODO: Find a better way to express this mess. tf.scan() might help.
    T0_xela = tf.matmul(
        tf.matmul(T01, rots[:, :, 0, :, :]),
        tf.matmul(
            tf.matmul(T12, rots[:, :, 1, :, :]),
            tf.matmul(
                tf.matmul(T23, rots[:, :, 2, :, :]),
                tf.matmul(
                    tf.matmul(T34, rots[:, :, 3, :, :]),
                    tf.matmul(
                        tf.matmul(T45, rots[:, :, 4, :, :]),
                        tf.matmul(
                            tf.matmul(T56, rots[:, :, 5, :, :]),
                            tf.matmul(
                                tf.matmul(T67, rots[:, :, 6, :, :]),
                                tf.matmul(T78, T8_xela)
                            )))))))
    return T0_xela


def joint_to_task_trajectories(traj_joint: tf.Tensor) -> tf.Tensor:
    """Convert a joint space trajectory into a task space trajectory.

    Args:
        traj_joint (tf.Tensor): Robot joint states with shape (n_batch, n_t, n_joints).

    Returns:
        tf.Tensor: Poses with shape (n_batch, n_t, 7).
    """

    # Compute the forward kinematic chain from joint states to ee pose as transformation matrices.
    T_ee = forward_kinematics(tf.convert_to_tensor(traj_joint))
    # Extract position and orientation.
    vec_pos_ee = T_ee[..., 0:3, 3]
    quat_ori_ee = quaternion.from_rotation_matrix(T_ee[..., 0:3, 0:3])
    # Return the pose as vector concat quaternion.
    pose_ee = tf.concat((vec_pos_ee, quat_ori_ee), axis=-1)
    return pose_ee
