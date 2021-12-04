import os

import numpy as np
import tensorflow as tf

from utils.data import (get_train_val_test_idx_by_region, get_valid_ids,
                        load_array_from_json, load_encoded_images)
from utils.experiments import Config, Experiment
from utils.kinematics.np import joint_to_task_trajectories
from utils.losses import get_dmp_weight_loss
from utils.models import get_encoder_model, get_convolutional_model
from utils.output import (CustomFitOutput, ensure_dirs,
                          plot_2d_traj, plot_3d_traj, plot_joint_traj,
                          plot_metric, plot_task_traj)
from utils.pydmps import DMPs_discrete


class Experiment04(Experiment):
    """CNN model predicting full DMP weights of RTP using the RTP-RGBD dataset.

    Task: RTP

    Dataset: RTP-RGBD

    Input: RGB images from home position

    Output: full DMP weights of RTP trajectory

    Model: RGB image -> Encoder -> bottleneck image -> CNN -> DMP weights

    Optimizer: Adam

    Training loss: RMSE on DMP weights

    Metrics:
    - MSE on trajectory in joint space.
    - Euclidean distance on final trajectory point in cartesian space.

    Each region has its own test set, evaluated separately.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        # Variables.
        self.dmp = DMPs_discrete(n_dmps=self.cfg.DMP["n_dof"], n_bfs=self.cfg.DMP["n_basis"])
        self.encoder = get_encoder_model(tf.keras.models.load_model(self.cfg.AUTOENCODER_MODEL_PATH))

        print("Loading data...")

        # Load samples ID.
        id_list = get_valid_ids([self.cfg.IMAGE_DIR])
        # Get samples region.
        region_list = region_list = np.array(list(map(lambda id: id[0], id_list)))
        # Unique regions.
        self.regions = list(sorted(set(region_list)))
        self.idx_train, self.idx_val, self.idx_test = get_train_val_test_idx_by_region(region_list, val_frac=self.cfg.VAL_FRAC, test_size=self.cfg.TEST_SIZE,
                                                                                       random_state=self.cfg.RANDOM_STATE, shuffle=True)

        # Load images.
        self.encoded_images = load_encoded_images(self.cfg.IMAGE_DIR, id_list, self.encoder, self.cfg.IMAGE_RESHAPE)

        # Load trajectories.
        trajectories = load_array_from_json(self.cfg.TRAJ_DIR, "joint_position", id_list, slicer=np.s_[..., 0:self.cfg.DMP["n_dof"]])

        dmp_weights = np.zeros((len(trajectories), self.dmp.n_dmps * self.dmp.n_bfs))
        dmp_goal = np.zeros((len(trajectories), self.dmp.n_dmps))
        for i, trajectory in enumerate(trajectories):
            # Impose the trajectory.
            self.dmp.imitate_path(y_des=np.transpose(trajectory), plot=False)
            # Get the weights and the goal.
            dmp_weights[i, :] = np.matrix.flatten(self.dmp.w).squeeze()  # [n_dmps*n_bfs]
            dmp_goal[i, :] = self.dmp.goal  # [n_dmps]
        # Stack all DMP parameters in a single numpy array.
        self.dmp_params = np.hstack((dmp_weights, dmp_goal))  # [(n_dmps+1)*n_bfs]

        print("Done!")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.cfg.LR)
        self.loss = get_dmp_weight_loss(self.cfg.DMP["n_dof"], self.cfg.DMP["alpha"])

    def get_model(self) -> tf.keras.Model:
        """Load the uninitialized model for this experiment.

        Returns:
            tf.keras.Model: The model.
        """

        model = get_convolutional_model(
            output_size=self.dmp.n_dmps * (self.dmp.n_bfs + 1),
            activation=self.cfg.ACTIVATION, l1_reg=self.cfg.L1_REG, l2_reg=self.cfg.L2_REG
        )

        return model

    def get_data(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get the input and output data for the model.

        Args:
            indices (np.ndarray): The list of indices to slice the dataset.

        Returns:
            tuple[np.ndarray, np.ndarray]: Input and output data for the model.
        """
        X = self.encoded_images[indices]
        y = self.dmp_params[indices]

        return X, y

    def train(self) -> None:
        self.logger.log(f"Training model '{self.cfg.MODEL_NAME}'.")

        # Create necessary folders.
        ensure_dirs([self.cfg.LOSS_DIR])

        callbacks = [
            CustomFitOutput(["loss", "val_loss"], epochs=self.cfg.EPOCHS)
        ]
        if self.cfg.ES["enable"]:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=self.cfg.ES["min_delta"], patience=self.cfg.ES["patience"],
                                                              verbose=1, restore_best_weights=True)
            callbacks.append(early_stopping)

        model = self.get_model()
        model.summary(print_fn=self.logger.print)
        model.compile(optimizer=self.optimizer, loss=self.loss)

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

        # Save the actual history values.
        np.save(os.path.join(self.cfg.LOSS_DIR, 'train_history'), history.history)

    def eval(self) -> dict[str, float]:
        """Evaluate a saved instance of the model.

        Returns:
            dict[str, float]: A dictionary containing metric names and values.
        """
        metrics: dict[str, float] = {}

        # Load the model.
        model = self.get_model()
        # expect_partial() prevents errors from missing custom losses/metrics.
        model.load_weights(self.cfg.MODEL_WEIGHTS_PATH).expect_partial()
        model.compile(optimizer=self.optimizer, loss=self.loss)

        # Evaluate each region.
        for region in self.regions:
            X_test, y_test = self.get_data(self.idx_test[region])
            new_metrics = self.evaluate_batch(model, X_test, y_test, f"{self.cfg.MODEL_NAME}_region_{region}")
            # Add region prefix to the metrics.
            new_metrics = {f"{m_name} region_{region}": m_val for m_name, m_val in new_metrics.items()}
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
        mse_joint = 0.0
        ed_ee_pos_final = 0.0
        for dmp_true, dmp_pred in zip(y_test, y_test_pred):
            # Trajectory in joint space.
            self.dmp.w = np.reshape(dmp_true[:-self.dmp.n_dmps], (self.dmp.n_dmps, self.dmp.n_bfs))
            self.dmp.goal = dmp_true[-self.dmp.n_dmps:]
            traj_joint_true, _, _ = self.dmp.rollout()

            self.dmp.w = np.reshape(dmp_true[:-self.dmp.n_dmps], (self.dmp.n_dmps, self.dmp.n_bfs))
            self.dmp.goal = dmp_pred[-self.dmp.n_dmps:]
            traj_joint_pred, _, _ = self.dmp.rollout()

            traj_ee_pose_true = joint_to_task_trajectories(traj_joint_true)
            traj_ee_pose_pred = joint_to_task_trajectories(traj_joint_pred)

            joint_trajectories.append({'true': traj_joint_true, 'pred': traj_joint_pred})
            ee_pose_trajectories.append({'true': traj_ee_pose_true, 'pred': traj_ee_pose_pred})

            # Metrics must be evaluated manually because pydmps has no TF implementation.
            mse_joint += np.mean((traj_joint_true - traj_joint_pred)**2)
            ed_ee_pos_final += np.linalg.norm(traj_ee_pose_true[-1, 0:3] - traj_ee_pose_pred[-1, 0:3])

        mse_joint /= y_test.shape[0]
        ed_ee_pos_final /= y_test.shape[0]

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

        metrics = {
            "mse_joint": mse_joint,
            "ed_ee_pos_final": ed_ee_pos_final
        }

        with open(os.path.join(OUT_METRICS_DIR, "metrics.txt"), "w") as file:
            for m_name, m_val in metrics.items():
                file.write(f"{m_name} {m_val}\n")
        for m_name, m_val in metrics.items():
            print(f"{m_name} {m_val}")

        return metrics


if __name__ == "__main__":
    from utils.experiments import ExperimentArgumentParser

    from .config import ConfigExp

    parser = ExperimentArgumentParser("Experiment DMP RTP-RGBD: CNN model predicting full DMP weights of RTP using the RTP-RGBD dataset.")
    args = parser.parse_args()

    cfg = ConfigExp(args.name)
    exp = Experiment04(cfg)

    if args.command == "train":
        exp.train()
    elif args.command == "eval":
        exp.eval()
    elif args.command == "run" and args.multi is None:
        exp.train()
        exp.eval()
    elif args.command == "run" and args.multi is not None:
        exp.train_eval_multi(args.multi, args.name, ConfigExp)
