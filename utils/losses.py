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
