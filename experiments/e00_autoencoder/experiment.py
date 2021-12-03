import os

import imageio
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils.data import (AutoencoderImageGenerator, get_valid_ids,
                        load_image_batch)
from utils.experiments import Config, Experiment
from utils.models import get_autoencoder_model
from utils.output import ensure_dirs, plot_metric


class ExperimentAutoencoder(Experiment):
    """Autoencoder.

    This autoencoder can be trained on a dataset of images: it will produce a bottleneck with shape (32, 32, 3)
    which can be used in the other models as input.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        print("Preparing data...")
        id_list = get_valid_ids([self.cfg.IMAGE_DIR])
        self.id_train, self.id_test = train_test_split(id_list, test_size=self.cfg.TEST_SIZE, random_state=self.cfg.RANDOM_STATE, shuffle=True)
        print("Done!")

    def get_model(self) -> tf.keras.Model:
        """Load the uninitialized model for this experiment.

        Returns:
            tf.keras.Model: The model.
        """

        return get_autoencoder_model()

    def get_data(self, indices: np.ndarray) -> None:
        # Not needed for this experiment.
        pass

    def train(self) -> None:
        self.logger.log(f"Training model '{self.cfg.MODEL_NAME}'.")

        # Create necessary folders.
        ensure_dirs([self.cfg.LOSS_DIR])

        # Load model.
        autoencoder = self.get_model()
        autoencoder.compile(optimizer=self.cfg.OPTIMIZER, loss=self.cfg.LOSS)
        autoencoder.summary(print_fn=self.logger.print)

        callbacks = []
        if self.cfg.ES["enable"]:
            es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=self.cfg.ES["patience"],
                                                  min_delta=self.cfg.ES["min_delta"], verbose=1, restore_best_weights=True)
            callbacks.append(es)

        # Load data.
        train_img_generator = AutoencoderImageGenerator(self.cfg.IMAGE_DIR, self.id_train, self.cfg.BATCH_SIZE, self.cfg.IMAGE_RESHAPE)

        # Training.
        history = autoencoder.fit(train_img_generator, epochs=self.cfg.EPOCHS, callbacks=callbacks, shuffle=True)

        autoencoder.save(self.cfg.MODEL_PATH)

        # Plot the training loss history.
        plot_metric(
            history.history["loss"],
            ["train"], ylabel="loss",
            save_path=os.path.join(self.cfg.LOSS_DIR, "loss.png")
        )
        plot_metric(
            history.history["loss"],
            ["train"], ylabel="loss", yscale="log",
            save_path=os.path.join(self.cfg.LOSS_DIR, "loss_log.png")
        )

        # Save the actual history values.
        np.save(os.path.join(self.cfg.LOSS_DIR, 'train_history'), history.history)

    def eval(self) -> dict[str, float]:
        """Evaluate a saved instance of the model.

        Returns:
            dict[str, float]: A dictionary containing metric names and values.
        """

        # Load model.
        autoencoder = tf.keras.models.load_model(os.path.join(self.cfg.MODEL_DIR, self.cfg.MODEL_NAME))
        encoder = tf.keras.Model(autoencoder.get_layer("input_image").output, autoencoder.get_layer("bottleneck").output)

        # Load data.
        input_img = load_image_batch(self.cfg.IMAGE_DIR, self.id_test, resize_shape=self.cfg.IMAGE_RESHAPE)

        return self.evaluate_batch(encoder, autoencoder, input_img, self.cfg.MODEL_NAME)

    def evaluate_batch(self, encoder: tf.keras.Model, autoencoder: tf.keras.Model, X_test: np.ndarray, name: str) -> dict[str, float]:
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
        BOTTLENECK_DIR = os.path.join(self.cfg.EXP_DIR, "output", name, 'bottleneck')
        RECONSTRUCTED_DIR = os.path.join(self.cfg.EXP_DIR, "output", name, 'reconstructed')
        ensure_dirs([BOTTLENECK_DIR, RECONSTRUCTED_DIR])

        # Evaluation.
        reconstructed_images = autoencoder.predict(X_test)
        bottleneck_images = encoder.predict(X_test)

        for i, (img_orig, img_bott, img_rec) in enumerate(zip(X_test, bottleneck_images, reconstructed_images)):
            imageio.imwrite(os.path.join(BOTTLENECK_DIR, f"encoded_{i+1}.jpg"), (img_bott * 255).astype(np.uint8))
            img_comparison = np.hstack((img_orig, img_rec))
            imageio.imwrite(os.path.join(RECONSTRUCTED_DIR, f"reconstructed_{i+1}.jpg"), (img_comparison * 255).astype(np.uint8))

        # No metrics.
        return {}


if __name__ == "__main__":
    from utils.experiments import ExperimentArgumentParser

    from .config import ConfigExp

    parser = ExperimentArgumentParser("Experiment autoencoder: CNN autoencoder.")
    args = parser.parse_args()

    cfg = ConfigExp(args.name)
    exp = ExperimentAutoencoder(cfg)

    if args.command == "train":
        exp.train()
    elif args.command == "eval":
        exp.eval()
    elif args.command == "run" and args.multi is None:
        exp.train()
        exp.eval()
    elif args.command == "run" and args.multi is not None:
        print("Argument '--multi' is not accepted for this experiment.")
