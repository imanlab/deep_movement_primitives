"""This module collects all the loss functions."""

import tensorflow as tf

from utils.promp import ProMP


def get_joint_loss(promp: ProMP):
    """Obtain the loss function evaluating the RMSE on the joint trajectories from the full ProMP weights (in TensorFlow).

    Args:
        promp (ProMP): An initialized ProMP instance.

    Returns:
        The loss function
    """

    def joint_loss(promp_true, promp_pred):
        """RMSE loss on the joint trajectory from the full ProMP weights.

        Args:
            promp_true: Ground truth full ProMP weights with shape (n_batch, n_basis * n_dof).
            promp_pred: Predicted full ProMP weights with shape (n_batch, n_basis * n_dof).

        Returns:
            The resulting loss.
        """
        traj_true = promp.trajectory_from_weights_tf(promp_true)
        traj_pred = promp.trajectory_from_weights_tf(promp_pred)
        # RMSE on joint trajectories.
        loss = tf.sqrt(tf.reduce_mean(tf.square(traj_true - traj_pred), axis=1))
        # Average over batches and joints.
        loss = tf.reduce_mean(loss)
        return loss

    return joint_loss


def get_dmp_weight_loss(n_dmps, alpha=1):
    """
    Loss in weight space, as defined by Rok Pahic et al. (https://doi.org/10.1016/j.neunet.2020.04.010) eq. (4).

    n_dmps: Number of dmps (corresponds to the number of degree of freedom).
    alpha: Relative weight of the goal loss with respect to the weights loss.
    """

    def dmp_weight_loss(dmp_params_true, dmp_params_pred):
        # dmp_params [(n_dmps+1)*n_bfs] = concatenate(dmp_weights [n_dmps*n_bfs], dmp_goal[n_dmps])
        dmp_weights_true, dmp_goal_true = dmp_params_true[:, :-n_dmps], dmp_params_true[:, -n_dmps:]
        dmp_weights_pred, dmp_goal_pred = dmp_params_pred[:, :-n_dmps], dmp_params_pred[:, -n_dmps:]

        loss = tf.sqrt(tf.reduce_mean(tf.square(dmp_weights_true - dmp_weights_pred))) + alpha * tf.sqrt(tf.reduce_mean(tf.square(dmp_goal_true - dmp_goal_pred)))
        return loss

    return dmp_weight_loss
