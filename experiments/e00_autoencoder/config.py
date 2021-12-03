import os

from utils.experiments import Config


class ConfigExp(Config):

    def __init__(self, model_name: str = None) -> None:
        super().__init__(model_name=model_name)

        self.EXP_DIR = "experiments/e_00_autoencoder"

        # Data.
        self.DATA_DIR = "data/rtp-rgbd/"
        self.IMAGE_DIR = os.path.join(self.DATA_DIR, "color_img/")

        # self.DATA_DIR = "data/rtp-rgbd/"
        # self.IMAGE_DIR = os.path.join(self.DATA_DIR, "color_img/")

        # self.DATA_DIR = "data/wpp-rgb/"
        # self.IMAGE_DIR = os.path.join(self.DATA_DIR, "img_resized/")

        self.IMAGE_RESHAPE = (256, 256, 3)

        # Output.
        self.LOSS_DIR = os.path.join(self.EXP_DIR, "loss", self.MODEL_NAME)
        self.LOG_DIR = os.path.join(self.EXP_DIR, "logs", self.MODEL_NAME)
        self.MODEL_DIR = os.path.join(self.EXP_DIR, "models")
        self.MODEL_PATH = os.path.join(self.MODEL_DIR, self.MODEL_NAME)

        # Hyperparameters.
        self.BATCH_SIZE = 32
        self.EPOCHS = 400
        self.LR = 1e-3
        self.VAL_FRAC = 0.3
        self.TEST_SIZE = 10
        self.RANDOM_STATE = 42
        self.ACTIVATION = "linear"
        self.OPTIMIZER = "adam"
        self.LOSS = "mse"
        # Kernel regularization.
        self.L1_REG = 0
        self.L2_REG = 0
        # Early stopping
        self.ES = {
            "enable": True,
            "min_delta": 0,
            "patience": 15
        }
