import os

from utils.experiments import Config


class ConfigExp(Config):

    def __init__(self, model_name: str = None) -> None:
        super().__init__(model_name=model_name)

        self.EXP_DIR = "experiments/e03_rtp-rgbd_fc/"

        # Data.
        self.DATA_DIR = "data/rtp-rgbd/"
        self.IMAGE_DIR = os.path.join(self.DATA_DIR, "color_img/")
        self.IMAGE_RESHAPE = (256, 256, 3)
        self.TRAJ_DIR = os.path.join(self.DATA_DIR, "trajectories/")

        # Input.
        self.AUTOENCODER_MODEL_PATH = "experiments/e00_autoencoder/models/rtp-rgbd"

        # Output.
        self.PLOT_DIR = os.path.join(self.EXP_DIR, "plots", self.MODEL_NAME)
        self.LOSS_DIR = os.path.join(self.EXP_DIR, "loss", self.MODEL_NAME)
        self.LOG_DIR = os.path.join(self.EXP_DIR, "logs", self.MODEL_NAME)
        self.MODEL_DIR = os.path.join(self.EXP_DIR, "models")
        self.MODEL_WEIGHTS_PATH = os.path.join(self.MODEL_DIR, self.MODEL_NAME, "weights")

        # Hyperparameters.
        self.BATCH_SIZE = 10
        self.EPOCHS = 150
        self.LR = 1e-3
        self.VAL_FRAC = 0.3
        self.TEST_SIZE = 10
        self.RANDOM_STATE = 42
        self.ACTIVATION = "linear"
        # Kernel regularization.
        self.L1_REG = 0
        self.L2_REG = 0
        # Early stopping
        self.ES = {
            "enable": True,
            "min_delta": 0,
            "patience": 15
        }
        # ProMP data.
        self.PROMP = {
            "n_basis": 8,
            "n_dof": 7,
            "n_t": 150,
            "space": "joint"
        }
