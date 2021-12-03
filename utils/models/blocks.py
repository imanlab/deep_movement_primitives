"""Useful blocks of layers, which can be used by different models."""

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Layer


def encoder_block(input_layer: Layer, output_name: str = "bottleneck") -> Layer:
    """Encoder part of a convolutional autoencoder."""

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same', name=output_name)(x)

    return x


def decoder_block(input_layer: Layer, output_name: str = "reconstructed") -> Layer:
    """Decoder part of a convolutional autoencoder."""

    x = layers.Conv2D(3, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same', name=output_name)(x)

    return x


def fully_connected_block(
    input_layer: Layer, neurons: list[int], output_size: int,
    activation: str, output_activation: str,
    l1_reg: float, l2_reg: float
) -> Layer:
    """Fully connected layers.

    Args:
        input_layer (Layer): Input layer of the model.
        neurons (list[int]): List of the sizes of all hidden layers.
        output_size (int): Size of the output layer.
        activation (str): Activation of the hidden layers.
        output_activation (str): Activation of the output layer.
        l1_reg (float): L1 regularization value.
        l2_reg (float): L2 regularization value.

    Returns:
        Layer: The functional block of layers, called on the input layer.
    """

    reg_l1_l2 = tf.keras.regularizers.l1_l2(l1_reg, l2_reg)

    x = input_layer
    for dense_size in neurons:
        x = layers.Dense(dense_size, activation=activation, kernel_regularizer=reg_l1_l2)(x)
    output_layer = layers.Dense(output_size, activation=output_activation)(x)

    return output_layer


def convolutional_block(input_layer: Layer, l1_reg: float, l2_reg: float) -> Layer:
    """Convolution block, input is (32, 32, 3).

    Args:
        input_layer (Layer): Input layer of the model.
        l1_reg (float): L1 regularization value.
        l2_reg (float): L2 regularization value.

    Returns:
        Layer: The functional block of layers, called on the input layer.
    """
    l1_l2_reg = tf.keras.regularizers.l1_l2(l1_reg, l2_reg)

    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=l1_l2_reg)(input_layer)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(16, (3, 3), padding="same", kernel_regularizer=l1_l2_reg)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(8, (3, 3), padding="same", kernel_regularizer=l1_l2_reg)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(4, (3, 3), padding="same", kernel_regularizer=l1_l2_reg)(x)
    output_layer = layers.Flatten(name="feature_vec")(x)

    return output_layer
