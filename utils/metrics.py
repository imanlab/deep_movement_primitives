import abc

import tensorflow as tf
from tensorflow.python.keras.metrics import MeanMetricWrapper

from utils.kinematics.tf import joint_to_task_trajectories
from utils.promp import ProMP


class PrompTrajMeanMetric(abc.ABC, MeanMetricWrapper):
    """Custom base class for a metric taking ProMP weights as input and computing
    the metric value on the corresponding trajectory.

    The metric is averaged on all batches of each epoch.
    """

    def __init__(self, promp: ProMP, promp_space: str, name: str, **kwargs):
        """

        Args:
            promp (ProMP): A initialized ProMP instance to compute weights and trajectories.
            promp_space (str): The space where ProMP trajectories live. Must be either "joint" or "task".
            name (str): The name of the metric
        """
        # MeanMetricWrapper is used to compute the mean on the batches.
        super().__init__(fn=self.metric_fn, name=name, **kwargs)
        # ProMP variables to compute the trajectory from the weights.
        self.promp = promp
        assert promp_space in ["joint", "task"], "The 'promp_space' parameter must be either 'joint' or 'task'."
        self.promp_space = promp_space

    @abc.abstractmethod
    def metric_fn(self, promp_true, promp_pred):
        """Compute the metric value on a batch of data.

        Args:
            promp_true: Ground truth tensor of the ProMP weights with shape (n_batch, n_dof*n_basis).
            promp_pred: Predicted tensor of the ProMP weights with shape (n_batch, n_dof*n_basis).

        Returns:
            The value of the metric on this batch.
        """
        pass


class RmseJointsPromp(PrompTrajMeanMetric):
    """RMSE of the joint trajectories from a batch of ProMP weights of joint trajectories.

    The value is averaged on all batches of each epoch.
    """

    def __init__(self, promp: ProMP, promp_space: str, name: str = "rmse_joints", **kwargs):
        super().__init__(promp=promp, promp_space=promp_space, name=name, **kwargs)

    def metric_fn(self, promp_true, promp_pred):
        """Compute the RMSE value on a batch of data.

        Args:
            promp_true: Ground truth tensor of the ProMP weights with shape (n_batch, n_dof*n_basis).
            promp_pred: Predicted tensor of the ProMP weights with shape (n_batch, n_dof*n_basis).

        Returns:
            The MSE value on this batch.
        """

        # Compute trajectories from weights.
        traj_true = self.promp.trajectory_from_weights_tf(promp_true)
        traj_pred = self.promp.trajectory_from_weights_tf(promp_pred)
        # Convert to joint space.
        if self.promp_space == "task":
            # Convert from task to joint space.
            raise NotImplementedError("The method to convert from task to joint space has not been implemented yet!")
        elif self.promp_space == "joint":
            # Trajectories are already in joint space.
            pass
        else:
            raise ValueError(f"'{self.promp_space}' is not a valid value for 'promp_space'. It must be either 'joint' or 'task'.")
        # Compute the RMSE on joint trajectories.
        rmse_joints = tf.sqrt(tf.reduce_mean(tf.square(traj_true - traj_pred), axis=1))
        # Average over joints and batch samples.
        rmse_joints = tf.reduce_mean(rmse_joints)

        return rmse_joints


class MseJointsPromp(PrompTrajMeanMetric):
    """MSE of the joint trajectories from a batch of ProMP weights of joint trajectories.

    The value is averaged on all batches of each epoch.
    """

    def __init__(self, promp: ProMP, promp_space: str, name: str = "mse_joints", **kwargs):
        super().__init__(promp=promp, promp_space=promp_space, name=name, **kwargs)

    def metric_fn(self, promp_true, promp_pred):
        """Compute the MSE value on a batch of data.

        Args:
            promp_true: Ground truth tensor of the ProMP weights with shape (n_batch, n_dof*n_basis).
            promp_pred: Predicted tensor of the ProMP weights with shape (n_batch, n_dof*n_basis).

        Returns:
            The MSE value on this batch.
        """

        # Compute trajectories from weights.
        traj_true = self.promp.trajectory_from_weights_tf(promp_true)
        traj_pred = self.promp.trajectory_from_weights_tf(promp_pred)
        # Convert to joint space.
        if self.promp_space == "task":
            # Convert from task to joint space.
            raise NotImplementedError("The method to convert from task to joint space has not been implemented yet!")
        elif self.promp_space == "joint":
            # Trajectories are already in joint space.
            pass
        else:
            raise ValueError(f"'{self.promp_space}' is not a valid value for 'promp_space'. It must be either 'joint' or 'task'.")
        # Compute the RMSE on joint trajectories.
        mse_joints = tf.reduce_mean(tf.square(traj_true - traj_pred), axis=1)
        # Average over joints and batch samples.
        mse_joints = tf.reduce_mean(mse_joints)

        return mse_joints


class EuclideanDistanceEePosFinalPromp(PrompTrajMeanMetric):
    """Euclidean distance of the ee final position from a batch of ProMP weights.

    The value is averaged on all batches of each epoch.
    """

    def __init__(self, promp: ProMP, promp_space: str, name: str = "ed_ee_pos_final", **kwargs):
        super().__init__(promp=promp, promp_space=promp_space, name=name, **kwargs)

    @tf.autograph.experimental.do_not_convert
    def metric_fn(self, promp_true, promp_pred):
        """Compute the Euclidean Distance for the final trajectory point on a batch of data.

        Args:
            promp_true: Ground truth tensor of the ProMP weights with shape (n_batch, n_dof*n_basis).
            promp_pred: Predicted tensor of the ProMP weights with shape (n_batch, n_dof*n_basis).

        Returns:
            The ED value on the final trajectory points of this batch.
        """

        # Compute trajectories from weights.
        traj_true = self.promp.trajectory_from_weights_tf(promp_true)
        traj_pred = self.promp.trajectory_from_weights_tf(promp_pred)
        # Select the final trajectory points.
        traj_final_true = tf.expand_dims(traj_true[:, -1, :], axis=1)
        traj_final_pred = tf.expand_dims(traj_pred[:, -1, :], axis=1)
        # Convert to joint space.
        if self.promp_space == "joint":
            # Convert the trajectories from joint to task space.
            traj_final_true = joint_to_task_trajectories(traj_final_true)
            traj_final_pred = joint_to_task_trajectories(traj_final_pred)
        elif self.promp_space == "task":
            # Trajectories are already in task space.
            pass
        else:
            raise ValueError(f"'{self.promp_space}' is not a valid value for 'promp_space'. It must be either 'joint' or 'task'.")
        # Select the final positions.
        ee_pos_final_true = traj_final_true[:, :, 0:3]
        ee_pos_final_pred = traj_final_pred[:, :, 0:3]
        # Compute the euclidean distance.
        ed_ee_pos_final = tf.linalg.norm(ee_pos_final_true - ee_pos_final_pred, axis=2)
        # Average results on batch samples.
        ed_ee_pos_final = tf.reduce_mean(ed_ee_pos_final)

        return ed_ee_pos_final


class RmseEePosPromp(PrompTrajMeanMetric):
    """RMSE of the ee position trajectories from a batch of ProMP weights.

    The value is averaged on all batches of each epoch.
    """

    def __init__(self, promp: ProMP, promp_space: str, name: str = "rmse_ee_pos", **kwargs):
        super().__init__(promp=promp, promp_space=promp_space, name=name, **kwargs)

    @tf.autograph.experimental.do_not_convert
    def metric_fn(self, promp_true, promp_pred):
        """Compute the RMSE value on a batch of data.

        Args:
            promp_true: Ground truth tensor of the ProMP weights with shape (n_batch, n_dof*n_basis).
            promp_pred: Predicted tensor of the ProMP weights with shape (n_batch, n_dof*n_basis).

        Returns:
            The MSE value on this batch.
        """

        # Compute joint trajectories from weights.
        traj_true = self.promp.trajectory_from_weights_tf(promp_true)
        traj_pred = self.promp.trajectory_from_weights_tf(promp_pred)
        # Convert to task space.
        if self.promp_space == "joint":
            # Convert the trajectories from joint to task space.
            traj_true = joint_to_task_trajectories(traj_true)
            traj_pred = joint_to_task_trajectories(traj_pred)
        elif self.promp_space == "task":
            # Trajectories are already in task space.
            pass
        else:
            raise ValueError(f"'{self.promp_space}' is not a valid value for 'promp_space'. It must be either 'joint' or 'task'.")
        # Extract the xyz position vector from the pose.
        ee_pos_traj_true = traj_true[:, :, 0:3]
        ee_pos_traj_pred = traj_pred[:, :, 0:3]
        # Compute the RMSE on the position trajectories.
        rmse_ee_pos = tf.sqrt(tf.reduce_mean(tf.square(ee_pos_traj_true - ee_pos_traj_pred), axis=1))
        # Average over the joints and the batch samples.
        rmse_ee_pos = tf.reduce_mean(rmse_ee_pos)

        return rmse_ee_pos


class RmseEeOriAnglePromp(PrompTrajMeanMetric):
    """RMSE of the ee orientation angle trajectories from a batch of ProMP weights.

    The value is averaged on all batches of each epoch.
    """

    def __init__(self, promp: ProMP, promp_space: str, name: str = "rmse_ee_ori_angle", **kwargs):
        super().__init__(promp=promp, promp_space=promp_space, name=name, **kwargs)

    @tf.autograph.experimental.do_not_convert
    def metric_fn(self, promp_true, promp_pred):
        """Compute the RMSE value on a batch of data.

        Args:
            promp_true: Ground truth tensor of the ProMP weights with shape (n_batch, n_dof*n_basis).
            promp_pred: Predicted tensor of the ProMP weights with shape (n_batch, n_dof*n_basis).

        Returns:
            The MSE value on this batch.
        """

        # Compute joint trajectories from weights.
        traj_true = self.promp.trajectory_from_weights_tf(promp_true)
        traj_pred = self.promp.trajectory_from_weights_tf(promp_pred)
        # Convert to task space.
        if self.promp_space == "joint":
            # Convert the trajectories from joint to task space.
            traj_true = joint_to_task_trajectories(traj_true)
            traj_pred = joint_to_task_trajectories(traj_pred)
        elif self.promp_space == "task":
            # Trajectories are already in task space.
            pass
        else:
            raise ValueError(f"'{self.promp_space}' is not a valid value for 'promp_space'. It must be either 'joint' or 'task'.")
        # Extract the orientation quaternions.
        ee_ori_traj_true = traj_true[:, :, 3:7]
        ee_ori_traj_pred = traj_pred[:, :, 3:7]
        # Normalize quaternions.
        ee_ori_traj_true, _ = tf.linalg.normalize(ee_ori_traj_true, axis=2)
        ee_ori_traj_pred, _ = tf.linalg.normalize(ee_ori_traj_pred, axis=2)
        # Compute the angle error between prediction and ground truth.
        # Reference: https://math.stackexchange.com/a/90098
        angle_error = tf.acos(tf.minimum(1.0, 2 * tf.square(tf.reduce_sum(ee_ori_traj_true * ee_ori_traj_pred, axis=2)) - 1))
        # Compute the RMSE on the trajectory.
        rmse_angle = tf.sqrt(tf.reduce_mean(tf.square(angle_error), axis=1))
        # Average over the batch samples.
        rmse_angle = tf.reduce_mean(rmse_angle)

        return rmse_angle
