"""This module collects the base elements to build a new experiment."""

import argparse
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Type

import numpy as np
import tensorflow as tf

from utils.output import Logger, ensure_dirs


class Config(ABC):
    """Configuration of an experiment."""

    def __init__(self, model_name: str) -> None:
        """Configuration of an experiment.

        Args:
            model_name (str): The name of the model to load/run.
        """

        self.MODEL_NAME = model_name


class Experiment(ABC):
    """Base class to describe an experiment."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = Logger(self.cfg.LOG_DIR)
        self.logger.log("Configuration: \n" + json.dumps(vars(cfg), indent=2))

    @abstractmethod
    def get_model(self) -> tf.keras.Model:
        """Load the uninitialized model for this experiment.

        Returns:
            tf.keras.Model: The model.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_data(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get the input and output data for the model.

        Args:
            indices (np.ndarray): The list of indices to slice the dataset.

        Returns:
            tuple[np.ndarray, np.ndarray]: Input and output data for the model.
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self) -> None:
        """Train a new instance of the model."""
        raise NotImplementedError()

    @abstractmethod
    def eval(self) -> dict[str, float]:
        """Evaluate a saved instance of the model.

        Returns:
            dict[str, float]: A dictionary containing metric names and values.
        """
        raise NotImplementedError()

    def train_eval_multi(self, multi: int, name: str, config_class: Type[Config]) -> None:
        """Run multiple training and evaluations, then aggregate the output metrics.

        Args:
            multi (int): Number of runs.
            name (str): Base name of the experiments (actual names will be f'{base_name}_{i}')
            config_class (Type[Config]): Config class of the experiment.
        """
        OUT_METRICS_DIR = os.path.join(self.cfg.EXP_DIR, "output", name, "metrics")
        ensure_dirs([OUT_METRICS_DIR])
        metrics_values = {}
        for i in range(multi):
            # Update the configuration with the new name.
            self.cfg = config_class(f"{name}_{i}")
            # New train and evaluation run.
            self.train()
            metrics = self.eval()
            # Store all metrics.
            for m_name, m_val in metrics.items():
                if m_name not in metrics_values.keys():
                    metrics_values[m_name] = []
                metrics_values[m_name].append(m_val)

        # Output aggregated metrics.
        self.logger.print(f"Number of trials: {multi}")
        self.logger.print("Name: mean stddev")
        for m_name, m_values in metrics_values.items():
            self.logger.print(f"{m_name} {np.mean(m_values)} {np.std(m_values)}")

        metric_path = os.path.join(OUT_METRICS_DIR, "metrics.npy")
        np.save(metric_path, metrics_values)


class ExperimentArgumentParser(argparse.ArgumentParser):
    """Argument parser to run the module of an experiment."""

    COMMANDS = {
        "run": "  execute both training and evaluation",
        "train": "execute the training",
        "eval": " execute the evaluation"
    }

    def __init__(self, description, *args, **kwargs):
        super().__init__(description=description, formatter_class=argparse.RawTextHelpFormatter, *args, **kwargs)

        self.add_argument("command", metavar="command", default=list(self.COMMANDS.keys())[0], choices=self.COMMANDS.keys(),
                          help="Command. Must be one of:\n" + "\n".join([f"  {name}: {descr}" for name, descr in self.COMMANDS.items()]))
        self.add_argument(
            "-n",
            "--name",
            nargs="?",
            help="Name of the model to run/train/save.\nDefaults to the current date and time for 'run' and 'train', while it's required by 'eval'."
        )
        self.add_argument(
            "-m",
            "--multi",
            nargs="?",
            const=10,
            type=int,
            help="Number of times this experiment will run, providing average and standard deviation values for each metric in the end.\n"
                 "If not specified the experiment will run only one time and no aggreagate metrics will be computed.\n"
                 "If specified without a value, it will default to 10 times.\n"
                 "This argument only affects the 'run' command."
        )

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        if args.command == "eval" and args.name is None:
            self.error('command="eval" requires NAME.')
        elif args.command in ["train", "run"] and args.name is None:
            args.name = datetime.now().strftime("%Y%m%d_%H%M%S")
        return args
