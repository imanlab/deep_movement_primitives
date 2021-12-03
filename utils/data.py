"""This module collects various functions related to data.

Reading from file, data pipelines and so on.
"""

import json
import os

import numpy as np
import tensorflow as tf
from natsort import natsorted
from skimage.io import imread
from skimage.transform import resize


def normalize_negative_one(img, two_d=True):
    normalized_input = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    if two_d:
        return 2*normalized_input - 1
    else:
        return 2*normalized_input - 1, np.amin(img), np.amax(img)


def reverse_normalize_negative_one(img, min, max):
    po = (img + 1) / 2
    reversed_input = ((max - min) * po) + min
    return reversed_input


def load_image(img_path: str, resize_shape: tuple[int] = None, normalize: bool = False) -> np.ndarray:
    """Load an image from file.

    Args:
        img_path (str): Path of the image.
        resize_shape (tuple[int], optional): If specified, reshape all images to this shape. Defaults to None.
        normalize (bool, optional): If True, normalize the image in [-1, 1]. Defaults to False.

    Returns:
        np.ndarray: The image data.
    """

    img = imread(img_path, pilmode='RGB')
    if resize_shape:
        img = resize(img, resize_shape)
    if normalize:
        img = normalize_negative_one(img)
    return img


def load_image_batch(img_dir: str, id_list: list[str], img_ext: str = ".png", resize_shape: tuple[int] = None, normalize: bool = False) -> np.ndarray:
    """Load a batch of images from a directory.

    Args:
        img_dir (str): Path of the directory.
        id_list (list[str]): List of image ids.
        img_ext (str, optional): Extension of the image files. Defaults to ".png".
        resize_shape (tuple[int], optional): If specified, reshape all images to this shape. Defaults to None.
        normalize (bool, optional): If True, normalize the image in [-1, 1]. Defaults to False.

    Returns:
        np.ndarray: The image data with shape (samples, width, height, channels).
    """

    image_data = []
    for id in id_list:
        img = load_image(os.path.join(img_dir, id + img_ext), resize_shape, normalize)
        image_data.append(img)

    return np.array(image_data)


def load_encoded_image(img_path: str, encoder: tf.keras.Model, resize_shape: tuple[int] = None, normalize: bool = False) -> np.ndarray:
    """Load an image an pass it through an encoder model.

    Args:
        img_path (str): Path of the image.
        encoder (tf.keras.Model): The encoder model.
        resize_shape (tuple[int], optional): If specified, reshape all images to this shape. Defaults to None.
        normalize (bool, optional): If True, normalize the image in [-1, 1]. Defaults to False.

    Returns:
        np.ndarray: The image data with shape (width, height, channels).
    """
    return np.squeeze(encoder(np.expand_dims(load_image(img_path, resize_shape, normalize), axis=0)))


def load_encoded_images(img_dir: str, id_list: np.ndarray, encoder: tf.keras.Model, resize_shape: tuple[int] = None, normalize: bool = False) -> np.ndarray:
    """Load multiple images by ID from a directory an pass them through an encoder model.

    Args:
        img_dir (str): Path of the directory.
        id_list (np.ndarray): List of image id
        encoder (tf.keras.Model): The encoder model.
        resize_shape (tuple[int], optional): If specified, reshape all images to this shape. Defaults to None.
        normalize (bool, optional): If True, normalize the image in [-1, 1]. Defaults to False.

    Returns:
        np.ndarray: The image data with shape (samples, width, height, channels).
    """
    return np.array([load_encoded_image(os.path.join(img_dir, id + ".png"), encoder, resize_shape, normalize) for id in id_list])


def get_valid_ids(dir_paths: list[str]) -> np.ndarray:
    """Obtain the list of all valid IDs in the dataset.

    An ID is valid if a file with that ID exists in all specified folders.

    Args:
        dir_paths (list[str]): List of directory paths to check.

    Returns:
        np.ndarray: The naturally sorted list of valid IDs.
    """

    id_sets = [{os.path.splitext(filename)[0] for filename in os.listdir(dir_path)} for dir_path in dir_paths]

    valid_ids = set(id_sets[0])
    for id_set in id_sets[1:]:
        valid_ids.intersection_update(id_set)

    return np.array(natsorted(valid_ids))


def check_missing_samples(dir_paths: list[str]):
    """Check that all directories in a dataset contain the same samples.

    All directories in the list must contain files with the same IDs.

    In this example ID "sample_001" is missing from folder2.
        folder1:
            sample_001.png
            sample_002.png
        folder2:
            sample_002.json


    Args:
        dir_paths (list[str]): [description]
    """

    assert len(dir_paths) >= 2, "A minimum of two directories is required"

    id_sets = []
    for d in dir_paths:
        id_set = set(os.listdir(d))
        id_sets.append(id_set)

    # IDs of all samples in the dataset.
    all_ids = set()
    for id_set in id_sets:
        all_ids.update(id_set)

    # Check that each directory contains all samples.
    for d, id_set in zip(dir_paths, id_sets):
        missing = all_ids - id_set
        if len(missing) > 0:
            print(f"Missing samples in '{d}':")
            print(missing)
        else:
            print(f"No missing samples in '{d}'.")


def get_train_val_test_idx(N_samples: int, val_frac: float = None, test_frac: float = None,
                           val_size: int = None, test_size: int = None, random_state: int = None,
                           shuffle: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the indices corresponding to each sample split between training, validation and testing.

    Args:
        n_samples (int): The total number of samples.
        val_frac (float, optional): Fraction of samples to use for validation. Defaults to None.
        test_frac (float, optional): Fraction of samples to use for testing. Defaults to None.
        val_size (int, optional): Number of samples to use for validation. This overrides val_frac. Defaults to None.
        test_size (int, optional): Number of samples to use for testing. This overrides test_frac. Defaults to None.
        random_state (int, optional): A random seed to initialize the RNG. Defaults to None.
        shuffle (bool, optional): If True, the samples are shuffled. Defaults to True.

    Returns:
        np.ndarray, np.ndarray, np.ndarray: The lists of indices corresponding to training, validation and testing samples.
    """

    rng = np.random.default_rng(random_state)

    # Explicit cast to int type since default is float and indices must be integers.
    idx_train = np.array([], dtype=np.int64)
    idx_val = np.array([], dtype=np.int64)

    # Validation samples.
    if val_size is not None:
        N_val = val_size
    elif val_frac is not None:
        N_val = np.floor(val_frac * N_samples).astype(np.int)
    else:
        N_val = 0
    # Test samples.
    if test_size is not None:
        N_test = test_size
    elif test_frac is not None:
        N_test = np.floor(test_frac * N_samples).astype(np.int)
    else:
        N_test = 0

    idx_all = np.arange(N_samples)
    # Extract N_test test samples.
    idx_test = rng.choice(idx_all, N_test, replace=False)
    idx_train = np.append(idx_train, np.setdiff1d(idx_all, idx_test))

    # Extract N_val validation samples from the whole dataset.
    N_val = int(np.ceil(len(idx_train) * val_frac))
    idx_val = rng.choice(idx_train, N_val, replace=False)

    # N_train samples from the whole dataset are what is left.
    idx_train = np.setdiff1d(idx_train, idx_val)

    if shuffle:
        # Shuffle the samples to mix the different regions.
        rng.shuffle(idx_train)
        rng.shuffle(idx_val)
        rng.shuffle(idx_test)

    return idx_train, idx_val, idx_test


def get_train_val_test_idx_by_region(region_list: np.ndarray, val_frac: float = None, test_frac: float = None,
                                     val_size: int = None, test_size: int = None, random_state: int = None,
                                     shuffle: bool = True) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Get the indices corresponding to each sample split between training, validation and testing.

    Test samples are extracted for each region, training and validation from the whole dataset.

    Args:
        region_list (np.ndarray): List of regions, one for each sample.
        val_frac (float, optional): Fraction of samples to use for validation. Defaults to None.
        test_frac (float, optional): Fraction of samples to use for testing. Defaults to None.
        val_size (int, optional): Number of samples to use for validation. This overrides val_frac. Defaults to None.
        test_size (int, optional): Number of samples to use for testing. This overrides test_frac. Defaults to None.
        random_state (int, optional): A random seed to initialize the RNG. Defaults to None.
        shuffle (bool, optional): If True, the samples are shuffled. Defaults to True.

    Returns:
        np.ndarray, np.ndarray, dict[str, np.ndarray]: The lists of indices corresponding to training, validation and testing samples. \
                                                       For testing a dict is returned, with one item for each region.
    """

    rng = np.random.default_rng(random_state)
    # Unique regions.
    regions = list(sorted(set(region_list)))
    # Total number of samples.
    N_tot = region_list.shape[0]
    # Validation samples.
    if val_size is not None:
        N_val = val_size
    elif val_frac is not None:
        N_val = np.floor(val_frac * N_tot).astype(np.int)
    else:
        N_val = 0
    # Test samples.
    if test_size is not None:
        N_test = test_size
    elif test_frac is not None:
        N_test = np.floor(test_frac * N_tot).astype(np.int)
    else:
        N_test = 0

    # Explicit cast to int type since default is float and indices must be integers.
    idx_train = np.array([], dtype=np.int64)
    idx_val = np.array([], dtype=np.int64)
    idx_test = {}

    # Extract N_test test samples from each region.
    for region in regions:
        idx_region = np.where(region_list == region)[0]
        idx_test[region] = rng.choice(idx_region, N_test, replace=False)
        idx_train = np.append(idx_train, np.setdiff1d(idx_region, idx_test[region]))

    # Extract N_val validation samples from the whole dataset.
    N_val = int(np.ceil(len(idx_train) * val_frac))
    idx_val = rng.choice(idx_train, N_val, replace=False)

    # N_train samples from the whole dataset are what is left.
    idx_train = np.setdiff1d(idx_train, idx_val)

    if shuffle:
        # Shuffle the samples to mix the different regions.
        rng.shuffle(idx_train)
        rng.shuffle(idx_val)
        for region in regions:
            rng.shuffle(idx_test[region])

    return idx_train, idx_val, idx_test


def get_train_val_test_idx_by_config_and_palp_path(
    config_list: np.ndarray,
    palp_path_list: np.ndarray,
    test_frac_by_palp_path: dict[str, float],
    val_frac: float = None,
    random_state: int = None,
    shuffle: bool = True
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Get the indices corresponding to each sample split between training, validation and testing.

    First, testing samples are extracted according to the values specified in test_frac_by_palp_path (each palpation path might be split differently).
    The all samples that are not used for testing are randomly split between training and validation.

    Ttest samples are alos separated by configuration.

    Args:
        config_list (np.ndarray): List of configurations, one for each sample.
        palp_path_list (np.ndarray): List of palpation path indexes, one for each sample.
        test_frac_by_palp_path (dict[str, float]): Fraction of samples to use for testing for each palpation path.
        val_frac (float, optional): Fraction of samples to use for validation. Defaults to None.
        random_state (int, optional): A random seed to initialize the RNG. Defaults to None.
        shuffle (bool, optional): If True, the samples are shuffled. Defaults to True.

    Returns:
        np.ndarray, np.ndarray, dict[str, np.ndarray]: The lists of indices corresponding to training, validation and testing samples. \
                                                       For testing a dict is returned, with one item for each region.
    """

    rng = np.random.default_rng(random_state)

    # Unique configurations.
    configurations = list(sorted(set(config_list)))
    # Unique palpation paths.
    # palp_paths = list(sorted(set(palp_path_list)))

    # Explicit cast to int type since default is float and indices must be integers.
    idx_train = np.array([], dtype=np.int64)
    idx_val = np.array([], dtype=np.int64)
    idx_test = {}

    for config in configurations:
        idx_test[config] = []
        for palp_path, test_frac in test_frac_by_palp_path.items():
            if test_frac is None:
                # If test_frac is None samples from this palpation path should be discarded completely (i.e. not used for training nor testing).
                continue
            # Extract test samples.
            idx_config = np.where((config_list == config) & (palp_path_list == palp_path))[0]
            N_test = np.floor(test_frac * len(idx_config)).astype(np.int)
            new_idx_test = rng.choice(idx_config, N_test, replace=False)
            idx_test[config].extend(new_idx_test)
            # All remaining samples will be either train or validation.
            idx_train = np.append(idx_train, np.setdiff1d(idx_config, new_idx_test))

    # Extract N_val validation samples from the whole dataset.
    N_val = int(np.ceil(len(idx_train) * val_frac))
    idx_val = rng.choice(idx_train, N_val, replace=False)

    # N_train samples from the whole dataset are what is left.
    idx_train = np.setdiff1d(idx_train, idx_val)

    if shuffle:
        # Shuffle the samples to mix the different configurations.
        rng.shuffle(idx_train)
        rng.shuffle(idx_val)
        for config in configurations:
            rng.shuffle(idx_test[config])

    return idx_train, idx_val, idx_test


def load_array_from_json(dir_path: str, json_key: str, id_list: list[str], slicer: tuple[slice] = None) -> list[np.ndarray]:
    """Load (a slice of) a Numpy array from multiple JSON files in a folder.

    Can be used, for example, to load joint states from multiple trajectory files.

    Args:
        dir_path (str): The path of the directory containing the files.
        json_keys (list[str]): The keys to be read from the JSON files.
        id_list (list[str]): The ids of the files to be read.
        slicer (tuple[slice], optional): If specified, slice every array accoroding to this slicer. Defaults to None.

    Returns:
        list[np.ndarray]: An array for each key.
    """

    data = []
    for id in id_list:
        file_path = os.path.join(dir_path, f"{id}.json")
        with open(file_path, "r") as file:
            json_data = json.load(file)[json_key]
            arr = np.array(json_data)
            if slicer:
                arr = arr[slicer]
            data.append(arr)

    return data


class AutoencoderImageGenerator(tf.keras.utils.Sequence):
    """Load images from a directory."""

    def __init__(self, image_dir: str, image_ids: list[str], batch_size: int, img_reshape: tuple[int]):
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.batch_size = batch_size
        self.img_reshape = img_reshape

    def __len__(self):
        """Total number of batches."""
        return (np.ceil(len(self.image_ids) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        """Load a batch of data"""
        batch_ids = self.image_ids[idx * self.batch_size:(idx+1) * self.batch_size]
        img_batch_data = load_image_batch(self.image_dir, batch_ids, resize_shape=self.img_reshape)

        # Sequence.__getitem__() returns (input, output).
        return img_batch_data, img_batch_data
