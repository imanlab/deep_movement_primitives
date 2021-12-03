import os

import numpy as np
import tensorflow as tf

from utils.data import (get_train_val_test_idx_by_region,
                        get_valid_ids, load_array_from_json,
                        load_encoded_images)
from utils.experiments import Config, Experiment
from utils.kinematics.np import joint_to_task_trajectories
from utils.losses import get_joint_loss
from utils.metrics import (EuclideanDistanceEePosFinalPromp, MseJointsPromp,
                           RmseEeOriAnglePromp, RmseEePosPromp,
                           RmseJointsPromp)
from utils.models import get_encoder_model, get_annotated_model
from utils.output import (CustomFitOutput, ensure_dirs, evaluate_metrics, plot_2d_traj, plot_3d_traj,
                          plot_joint_traj, plot_metric, plot_task_traj)
from utils.promp import ProMP


class Experiment08(Experiment):
    """CNN model predicting full ProMP weights of WPP using the WPP-RGB dataset.

    Task: WPP

    Dataset: WPP-RGB

    Input: RGB images from home position with a circle on the target position + pixel coordinates of the target point

    Output: full ProMP weights of WPP trajectory in joint space

    Model: (RGB image -> Encoder -> bottleneck image -> CNN -> feature vector) + target coordinates -> dense network -> full ProMP weights

    Optimizer: Adam

    Training loss: RMSE on joint trajectories.

    Metrics:
    - RMSE on trajectory in joint space.
    - RMSE on the cartesian trajectory of the EE.
    - RMSE on the orientation angle of the EE.
    - Euclidean distance on final trajectory point in cartesian space.

    Each configuration has its own test set, evaluated separately.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        # Variables.
        self.promp = ProMP(n_basis=self.cfg.PROMP["n_basis"], n_dof=self.cfg.PROMP["n_dof"], n_t=self.cfg.PROMP["n_t"])
        self.encoder = get_encoder_model(tf.keras.models.load_model(self.cfg.AUTOENCODER_MODEL_PATH))

        print("Loading data...")

        # Load samples ID.
        id_list = get_valid_ids([self.cfg.IMAGE_DIR])

        # Configuration of each sample.
        config_list = np.array(list(map(lambda id: id[0], id_list)))
        # Palpation path of each sample.
        # palp_path_list = np.array(list(map(lambda id: id[2], id_list)))

        self.configs = list(sorted(set(config_list)))
        self.idx_train, self.idx_val, self.idx_test = get_train_val_test_idx_by_region(
            config_list, val_frac=self.cfg.VAL_FRAC, test_size=self.cfg.TEST_SIZE,
            random_state=self.cfg.RANDOM_STATE, shuffle=True
        )

        # Load images.
        self.encoded_images = load_encoded_images(self.cfg.IMAGE_DIR, id_list, self.encoder, self.cfg.IMAGE_RESHAPE)

        # Load trajectories.
        trajectories = load_array_from_json(self.cfg.TRAJ_DIR, "joint_position", id_list, slicer=np.s_[..., 0:self.cfg.PROMP["n_dof"]])
        # Load end points.
        self.end_points = np.array(load_array_from_json(self.cfg.TRAJ_DIR, "image_2d_point_resized_end", id_list))

        # Convert each trajectory into ProMP weights.
        self.promp_weights = np.zeros((len(trajectories), self.promp.n_basis * self.promp.n_dof))
        for i, trajectory in enumerate(trajectories):
            self.promp_weights[i, :] = self.promp.weights_from_trajectory(trajectory)

        print("Done!")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.cfg.LR)
        self.loss = get_joint_loss(self.promp)
        self.metrics_test = [
            MseJointsPromp(self.promp, promp_space=self.cfg.PROMP["space"], name="mse_joints"),
            RmseJointsPromp(self.promp, promp_space=self.cfg.PROMP["space"], name="rmse_joints"),
            EuclideanDistanceEePosFinalPromp(self.promp, promp_space=self.cfg.PROMP["space"], name="ed_ee_pos_final"),
            RmseEePosPromp(self.promp, promp_space=self.cfg.PROMP["space"], name="rmse_ee_pos"),
            RmseEeOriAnglePromp(self.promp, promp_space=self.cfg.PROMP["space"], name="rmse_ee_ori_angle")
        ]
        # Some metrics can be computationally expensive during the training phase, especially those who require joint -> task space conversion.
        # They can be removed here to speed up the training at the cost of not having the metrics' history.
        # If you do remove any metric, remember to remove all the references as well (e.g. when they are plotted at the end of training).
        self.metrics_train = [
            RmseJointsPromp(self.promp, promp_space=self.cfg.PROMP["space"], name="rmse_joints"),
            EuclideanDistanceEePosFinalPromp(self.promp, promp_space=self.cfg.PROMP["space"], name="ed_ee_pos_final"),
            RmseEePosPromp(self.promp, promp_space=self.cfg.PROMP["space"], name="rmse_ee_pos"),
            RmseEeOriAnglePromp(self.promp, promp_space=self.cfg.PROMP["space"], name="rmse_ee_ori_angle")
        ]

    def get_model(self) -> tf.keras.Model:
        """Load the uninitialized model for this experiment.

        Returns:
            tf.keras.Model: The model.
        """

        model = get_annotated_model(
            output_size=self.promp.n_basis*self.promp.n_dof, annotation_size=2,
            activation=self.cfg.ACTIVATION, l1_reg=self.cfg.L1_REG, l2_reg=self.cfg.L2_REG
        )

        return model

    def get_data(self, indices: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Get the input and output data for the model.

        Args:
            indices (np.ndarray): The list of indices to slice the dataset.

        Returns:
            tuple[tuple[np.ndarray, np.ndarray], np.ndarray]: Input and output data for the model.
        """
        X = (self.encoded_images[indices], self.end_points[indices])
        y = self.promp_weights[indices]

        return X, y

    def train(self) -> None:
        self.logger.log(f"Training model '{self.cfg.MODEL_NAME}'.")

        # Create necessary folders.
        ensure_dirs([self.cfg.LOSS_DIR])

        callbacks = [
            CustomFitOutput(["loss", "val_loss", "val_rmse_ee_pos", "val_rmse_ee_ori_angle"], epochs=self.cfg.EPOCHS)
        ]
        if self.cfg.ES["enable"]:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=self.cfg.ES["min_delta"], patience=self.cfg.ES["patience"],
                                                              verbose=1, restore_best_weights=True)
            callbacks.append(early_stopping)

        model = self.get_model()
        model.summary(print_fn=self.logger.print)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics_train)

        # Load the data.
        X_train, y_train = self.get_data(self.idx_train)
        X_val, y_val = self.get_data(self.idx_val)

        # Training.
        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=self.cfg.EPOCHS, batch_size=self.cfg.BATCH_SIZE, callbacks=callbacks, shuffle=True, verbose=0
        )
        model.save_weights(self.cfg.MODEL_WEIGHTS_PATH)

        # Plot the loss history.
        plot_metric(
            [history.history["loss"], history.history["val_loss"]],
            ["train", "val"], ylabel="loss",
            save_path=os.path.join(self.cfg.LOSS_DIR, "loss.png")
        )
        plot_metric(
            [history.history["loss"], history.history["val_loss"]],
            ["train", "val"], ylabel="loss", yscale="log",
            save_path=os.path.join(self.cfg.LOSS_DIR, "loss_log.png")
        )
        # Plot the history of metrics.
        plot_metric(
            [history.history["ed_ee_pos_final"], history.history["val_ed_ee_pos_final"]],
            ["train", "val"], ylabel="Euclidean distance of EE position [m]", title="Euclidean distance of final EE cartesian position",
            save_path=os.path.join(self.cfg.LOSS_DIR, "ed_ee_pos_final.png")
        )
        plot_metric(
            [history.history["rmse_ee_pos"], history.history["val_rmse_ee_pos"]],
            ["train", "val"], ylabel="RMSE of EE position [m]", title="RMSE on EE cartesian position trajectories",
            save_path=os.path.join(self.cfg.LOSS_DIR, "ee_pos_rmse.png")
        )
        plot_metric(
            [history.history["rmse_ee_ori_angle"], history.history["val_rmse_ee_ori_angle"]],
            ["train", "val"], ylabel="RMSE of EE orientation angle [rad]", title="RMSE on EE orientation angle trajectories",
            save_path=os.path.join(self.cfg.LOSS_DIR, "ee_ori_angle_rmse.png")
        )

        # Save the actual history values.
        np.save(os.path.join(self.cfg.LOSS_DIR, 'train_history'), history.history)

    def eval(self) -> dict[str, float]:
        """Evaluate a saved instance of the model.

        Returns:
            dict[str, float]: A dictionary containing metric names and values.
        """

        # Load the model.
        model = self.get_model()
        # expect_partial() Nprevents errors from missing custom losses/metrics.
        model.load_weights(self.cfg.MODEL_WEIGHTS_PATH).expect_partial()
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics_test)

        # X_test, y_test = self.get_data(self.idx_test)
        # metrics = self.evaluate_batch(model, X_test, y_test, self.cfg.MODEL_NAME)

        # Evaluate each configuration.
        metrics: dict[str, float] = {}
        for configuration in self.configs:
            X_test, y_test = self.get_data(self.idx_test[configuration])
            new_metrics = self.evaluate_batch(model, X_test, y_test, f"{self.cfg.MODEL_NAME}_config_{configuration}")
            # Add configuration to the metrics' names.
            new_metrics = {f"{m_name} config_{configuration}": m_val for m_name, m_val in new_metrics.items()}
            metrics.update(new_metrics)

        return dict(sorted(metrics.items()))

    def evaluate_batch(self, model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray, name: str) -> dict[str, float]:
        """Evaluate a batch of data.

        Args:
            model (tf.keras.Model): The model to evaluate.
            X_test (np.ndarray): Input data.
            y_test (np.ndarray): Output data.
            name (str): Name of the run.

        Returns:
            dict[str, float]: Dictionary containing metric names and values.
        """
        self.logger.print(f"Evaluating '{name}'.")

        # Create necessary folders.
        PLOT_JOINT_DIR = os.path.join(self.cfg.EXP_DIR, "output", name, 'joints')
        PLOT_EE_POSE_DIR = os.path.join(self.cfg.EXP_DIR, "output", name, 'ee_pose')
        PLOT_EE_POS_2D_DIR = os.path.join(self.cfg.EXP_DIR, "output", name, 'ee_pos_2d')
        PLOT_EE_POS_3D_DIR = os.path.join(self.cfg.EXP_DIR, "output", name, 'ee_pos_3d')
        OUT_METRICS_DIR = os.path.join(self.cfg.EXP_DIR, "output", name)
        ensure_dirs([PLOT_JOINT_DIR, PLOT_EE_POSE_DIR, PLOT_EE_POS_2D_DIR, PLOT_EE_POS_3D_DIR, OUT_METRICS_DIR])

        # Use the model to make a prediction on the test data.
        y_test_pred = model.predict(X_test)

        # Compute and plot the trajectory of each sample.
        joint_trajectories = []
        ee_pose_trajectories = []
        for promp_true, promp_pred in zip(y_test, y_test_pred):
            # Joint trajectory.
            traj_joint_true = self.promp.trajectory_from_weights(promp_true)
            traj_joint_pred = self.promp.trajectory_from_weights(promp_pred)

            traj_ee_pose_true = joint_to_task_trajectories(traj_joint_true)
            traj_ee_pose_pred = joint_to_task_trajectories(traj_joint_pred)

            joint_trajectories.append({'true': traj_joint_true, 'pred': traj_joint_pred})
            ee_pose_trajectories.append({'true': traj_ee_pose_true, 'pred': traj_ee_pose_pred})

        # Plot the trajectories
        for i, (joint_traj, ee_pose_traj) in enumerate(zip(joint_trajectories, ee_pose_trajectories)):
            joint_plot_file_path = os.path.join(PLOT_JOINT_DIR, f'traj_{i+1}.png')
            plot_joint_traj(joint_plot_file_path, joint_traj["true"], joint_traj["pred"])
            ee_pose_plot_file_path = os.path.join(PLOT_EE_POSE_DIR, f'traj_{i+1}.png')
            plot_task_traj(ee_pose_plot_file_path, ee_pose_traj["true"], ee_pose_traj["pred"])
            ee_pos_2d_plot_file_path = os.path.join(PLOT_EE_POS_2D_DIR, f'traj_{i+1}.png')
            plot_2d_traj(ee_pos_2d_plot_file_path, ee_pose_traj["true"][:, 0:2], ee_pose_traj["pred"][:, 0:2])
            ee_pos_3d_plot_file_path = os.path.join(PLOT_EE_POS_3D_DIR, f'traj_{i+1}.png')
            plot_3d_traj(ee_pos_3d_plot_file_path, ee_pose_traj["true"][:, 0:3], ee_pose_traj["pred"][:, 0:3])

        # Save the trajectories.
        np.save(os.path.join(PLOT_JOINT_DIR, 'trajectories.npy'), joint_trajectories)
        np.save(os.path.join(PLOT_EE_POSE_DIR, 'trajectories.npy'), ee_pose_trajectories)

        # Evaluate all the metrics.
        metrics = evaluate_metrics(model, X_test, y_test, save_path=os.path.join(OUT_METRICS_DIR, "metrics.txt"))

        return metrics


if __name__ == "__main__":
    from utils.experiments import ExperimentArgumentParser

    from .config import ConfigExp

    parser = ExperimentArgumentParser("Experiment CNN WPP-RGB: CNN model predicting full ProMP weights of WPP using the WPP-RGB dataset.")
    args = parser.parse_args()

    cfg = ConfigExp(args.name)
    exp = Experiment08(cfg)

    if args.command == "train":
        exp.train()
    elif args.command == "eval":
        exp.eval()
    elif args.command == "run" and args.multi is None:
        exp.train()
        exp.eval()
    elif args.command == "run" and args.multi is not None:
        exp.train_eval_multi(args.multi, args.name, ConfigExp)
