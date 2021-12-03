"""This module collects classes and functions related to data output."""

import os
from datetime import datetime
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class Logger:
    """A custom logger for saving info to a text file and optionally printing it to stdout."""

    def __init__(self, log_dir: str, log_name: str = "log.txt") -> None:
        """A custom logger for saving info to a text file and optionally printing it to stdout.

        Args:
            log_dir (str): Directory where the log is saved.
            log_name (str, optional): Name of the log file. Defaults to "log.txt".
        """

        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, log_name)

        ensure_dirs([log_dir])

        self.log(f"Log file '{self.log_file}' created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", "w")

    def log(self, text: str, write_mode: str = "a"):
        """Write a string of text to the log file.

        A new line is appended afterwards.

        Args:
            text (str): The text to be logged.
            write_mode (str, optional): Specifies the mode in which the file is open. Defaults to "a" (append).
        """
        with open(self.log_file, write_mode) as f:
            f.write(text + "\n")

    def print(self, text: str):
        """Print a string to stdout and log it.

        Args:
            text (str): The string to be printed and logged.
        """
        print(text)
        self.log(text)


def plot_metric(
    data,
    legend_labels: list[str] = None, title: str = None,
    ylabel: str = None, xlabel: str = "epoch",
    yscale: str = "linear",
    save_path: str = None, show: bool = False
):
    """Plot the history of a metric during a training episode.

    Args:
        data: History of the metric values with shape (n_samples,) or (n_channels, n_samples).
        legend_labels (list[str], optional): Legend labels. Defaults to None.
        title (str, optional): Title of the graph. Defaults to None.
        ylabel (str, optional): Label of the y axis. Defatuls to None.
        xlabel (str, optional): Label of the x axis. Defaults to "epoch".
        yscale (str, optional): The y scale of the plot. Defaults to "linear".
        save_path (str, optional): If specified, the file where the plot image is to be saved. Defaults to None.
        show (bool, optional): If True, displays the plot in a window, halting execution. Defaults to False.
    """

    data = np.array(data)
    epochs = range(1, data.shape[-1]+1)

    fig = plt.figure()
    plt.plot(epochs, np.transpose(data))
    plt.yscale(yscale)
    plt.grid(True, which='both')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend_labels:
        plt.legend(legend_labels)

    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)


def ensure_dirs(dir_paths: list[str]):
    """Create these directories if they do not exist yet.

    Args:
        dir_paths (list[str]): List of directory paths to create.
    """
    for dir_path in dir_paths:
        os.makedirs(dir_path, exist_ok=True)


def plot_joint_traj(file_path: str, traj_true: np.ndarray, traj_pred: np.ndarray = None, columns: int = 4, title: str = None, show: bool = False):
    """Plot the trajectories of all joints.

    Args:
        file_path (str): Path of the image file to save.
        traj_true (np.ndarray): Ground truth of trajectory data with shape (samples, dof).
        traj_pred (np.ndarray, optional): Predicted trajectory data with shape (samples, dof). Defaults to None.
        columns (int, optional): Number of columns for the subplot grid. Defaults to 4.
        title (str, optional): Title of the graph. Defaults to None.
        show (bool, optional): If True, the plot is showed in a window, halting execution as long as it remains open. Defaults to False.
    """
    assert traj_pred is None or traj_true.shape == traj_pred.shape, "Trajectory ground truth and prediction do not have the same shape!"

    n_joints = traj_true.shape[1]
    rows = int(np.ceil(n_joints / columns))

    fig, axarr = plt.subplots(rows, columns)
    fig.tight_layout()
    for i in range(0, n_joints):
        row = i // columns
        col = i % columns

        plt.sca(axarr[row, col])
        plt.plot(traj_true[:, i], 'k', label=f'q{i+1} true')
        if traj_pred is not None:
            plt.plot(traj_pred[:, i], 'b--', label=f'q{i+1} pred')
        plt.legend()

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(file_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_task_traj(file_path: str, traj_true: np.ndarray, traj_pred: np.ndarray = None, columns: int = 4, title: str = None, show: bool = False):
    """Plot the trajectories of the EE pose (3 cartesian coordinates + 4 quaternion elements).

    Args:
        file_path (str): Path of the image file to save.
        traj_true (np.ndarray): Ground truth of trajectory data with shape (samples, 7).
        traj_pred (np.ndarray, optional): Predicted trajectory data with shape (samples, 7). Defaults to None.
        columns (int, optional): Number of columns for the subplot grid. Defaults to 4.
        title (str, optional): Title of the graph. Defaults to None.
        show (bool, optional): If True, the plot is showed in a window, halting execution as long as it remains open. Defaults to False.
    """
    assert traj_pred is None or traj_true.shape == traj_pred.shape, "Trajectory ground truth and prediction do not have the same shape!"

    n_dof = traj_true.shape[1]
    rows = int(np.ceil(n_dof / columns))

    labels = ["X", "Y", "Z", "x", "y", "z", "w"]

    fig, ax_arr = plt.subplots(rows, columns, squeeze=False)
    fig.tight_layout()
    for i in range(0, n_dof):
        # Plot (X, Y, Z) on the first row and (x, y, z, w) on the second row.
        row = (i if i < 3 else i+1) // columns
        col = (i if i < 3 else i+1) % columns

        plt.sca(ax_arr[row, col])
        plt.plot(traj_true[:, i], 'k', label=f'{labels[i]} true')
        if traj_pred is not None:
            plt.plot(traj_pred[:, i], 'b--', label=f'{labels[i]} pred')
        plt.legend()

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(file_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_2d_traj(file_path: str, traj_true: np.ndarray, traj_pred: np.ndarray = None, title: str = None, show: bool = False):
    """Plot the trajectory of the EE projected on the XY plane.

    Args:
        file_path (str): Path of the image file to save.
        traj_true (np.ndarray): Ground truth of trajectory data with shape (samples, 2).
        traj_pred (np.ndarray, optional): Predicted trajectory data with shape (samples, 2). Defaults to None.
        title (str, optional): Title of the graph. Defaults to None.
        show (bool, optional): If True, the plot is showed in a window, halting execution as long as it remains open. Defaults to False.
    """
    assert traj_pred is None or traj_true.shape == traj_pred.shape, "Trajectory ground truth and prediction do not have the same shape!"

    plt.figure()

    plt.plot(traj_true[:, 0], traj_true[:, 1], 'k', label="True")
    plt.plot(traj_true[0, 0], traj_true[0, 1], 'ko')
    plt.plot(traj_true[-1, 0], traj_true[-1, 1], 'ks')
    if traj_pred is not None:
        plt.plot(traj_pred[:, 0], traj_pred[:, 1], 'b', label="Pred")
        plt.plot(traj_pred[0, 0], traj_pred[0, 1], 'bo')
        plt.plot(traj_pred[-1, 0], traj_pred[-1, 1], 'bs')
        plt.legend()

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    if title is not None:
        plt.title(title)

    plt.savefig(file_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_3d_traj(file_path: str, traj_true: np.ndarray, traj_pred: np.ndarray = None, title: str = None, show: bool = False):
    """Plot the 3D trajectory of the EE projected in XYZ space.

    Args:
        file_path (str): Path of the image file to save.
        traj_true (np.ndarray): Ground truth of trajectory data with shape (samples, 2).
        traj_pred (np.ndarray, optional): Predicted trajectory data with shape (samples, 2). Defaults to None.
        title (str, optional): Title of the graph. Defaults to None.
        show (bool, optional): If True, the plot is showed in a window, halting execution as long as it remains open. Defaults to False.
    """
    assert traj_pred is None or traj_true.shape == traj_pred.shape, "Trajectory ground truth and prediction do not have the same shape!"

    plt.figure(figsize=plt.figaspect(1))
    ax = plt.axes(projection="3d")

    ax.plot(traj_true[:, 0], traj_true[:, 1], traj_true[:, 2], 'k', label="True")
    ax.plot(traj_true[0, 0], traj_true[0, 1], traj_true[0, 2], 'ko')
    ax.plot(traj_true[-1, 0], traj_true[-1, 1], traj_true[-1, 2], 'ks')
    if traj_pred is not None:
        ax.plot(traj_pred[:, 0], traj_pred[:, 1], traj_pred[:, 2], 'b', label="Pred")
        ax.plot(traj_pred[0, 0], traj_pred[0, 1], traj_pred[0, 2], 'bo')
        ax.plot(traj_pred[-1, 0], traj_pred[-1, 1], traj_pred[-1, 2], 'bs')
        plt.legend()

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    # Set equal axis.
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

    if title is not None:
        plt.title(title)

    plt.savefig(file_path)

    if show:
        plt.show()
    else:
        plt.close()


def get_bounding_box(task_traj: np.ndarray) -> np.ndarray:
    """Get the bounding box of a cartesian trajectory.
    If more than one trajectory the returned value is the average of all bounding boxes.

    Args:
        task_traj (np.ndarray): Cartesian trajectory, with shape (A1, ...AN, n_samples, 3)

    Returns:
        np.ndarray: [[xmin, ymin, zmin], [xmax, ymaxc, zmax]]
    """

    lower_bounds = np.amin(task_traj, axis=-2)[..., 0:3]
    lower_bounds = np.mean(lower_bounds, axis=tuple(range(0, len(lower_bounds.shape) - 1)))

    upper_bounds = np.amax(task_traj, axis=-2)[..., 0:3]
    upper_bounds = np.mean(upper_bounds, axis=tuple(range(0, len(upper_bounds.shape) - 1)))

    return np.vstack((lower_bounds, upper_bounds))


class CustomFitOutput(tf.keras.callbacks.Callback):
    """Limit the metrics reported in the console output of model.fit()."""

    def __init__(self, metrics: list[str], time_epoch: bool = True, epochs: int = None, decimal_digits: int = 4):
        """Limit the metrics reported in the console output of model.fit().

        Args:
            metrics (list[str]): Names of the metrics to output.
            time_epoch (bool, optional): If True, log the time each epoch required to be trained. Defaults to true.
            epochs (int, optional): Number of total epochs that will be run. Defaults to None.
            decimal_digits (int, optional): Number of decimal digits to display. Defaults to 4.
        """
        super().__init__()
        self.metrics = metrics
        self.time_epoch = time_epoch
        self.epochs = epochs
        self.decimal_digits = decimal_digits

    def on_epoch_begin(self, epoch, logs=None):
        if self.time_epoch:
            self.epoch_start_time = timer()

    def on_epoch_end(self, epoch, logs=None):
        # Current epoch.
        output = "Epoch "
        if self.epochs is not None:
            num_digits = len(str(self.epochs))
            output += f"{epoch:0{num_digits}d}/{self.epochs}"
        else:
            output += f"{epoch}"

        # Epoch time.
        if self.time_epoch:
            output += f" - Time: {timer() - self.epoch_start_time:.1f} s"

        # Metric values.
        for name, log_val in logs.items():
            if name in self.metrics:
                output += f" - {name}: {log_val:.{self.decimal_digits}f}"

        print(output)


def evaluate_metrics(
    model: tf.keras.Model, X_eval: np.ndarray, y_eval: np.ndarray,
    excluded_metrics: list[str] = None, save_path: str = None
) -> list[dict[str, float]]:
    """Evaluate all the metrics on the test dataset provided.

    Args:
        model (tf.keras.Model): The model to be evaluated.
        X_eval (np.ndarray): Input data.
        y_eval (n.ndarray): Output ground truth.
        excluded_metrics (list[str], optional): Metrics in this list will not be reported. Defaults to ["loss"].
        save_path (str, optional): If specified, the text file where to store the metric outputs. Defaults to None.

    Returns:
        list[dict]: The list of metrics each in the form of a dictionary {"name": value}.
    """

    if excluded_metrics is None:
        # The loss is included in the metric list, but it has no value since we are evaluating, so we remove it
        excluded_metrics = ["loss"]

    metrics = model.evaluate(X_eval, y_eval, return_dict=True, verbose=0)
    # Sort metrics by "name".
    metrics: dict[str, float] = dict(sorted(metrics.items()))
    # Remove unwanted metrics.
    metrics = {m_name: m_val for m_name, m_val in metrics.items() if m_name not in excluded_metrics}

    # Output the metrics.
    if save_path:
        with open(save_path, "w") as file:
            for m_name, m_val in metrics.items():
                file.write(f"{m_name} {m_val}\n")
    for m_name, m_val in metrics.items():
        print(f"{m_name} {m_val}")

    return metrics
