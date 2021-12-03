from tensorflow.keras import Model
import tensorflow.keras.layers as layers

from .blocks import convolutional_block, decoder_block, encoder_block, fully_connected_block


def get_autoencoder_model(name: str = "autoencoder") -> Model:
    """Autoencoder model.

    Input and output have shape (256, 256, 3).
    The bottleneck layer has shape (32, 32, 3).

    Args:
        name (str, optional): The name of the model. Defaults to "autoencoder".

    Returns:
        Model: The autoencoder model.
    """

    input_layer = layers.Input(shape=(256, 256, 3), name="image_input")
    bottleneck_layer = encoder_block(input_layer)
    output_layer = decoder_block(bottleneck_layer)

    return Model(inputs=input_layer, outputs=output_layer, name=name)


def get_encoder_model(autoencoder: Model, bottleneck_layer_name: str = "bottleneck", name: str = "encoder") -> Model:
    """Extract the encoder sub-model from an autoencoder model.

    Args:
        autoencoder (Model): The autoencoder model.
        bottleneck_layer_name (str, optional): The name of the bottleneck layer in the autoencoder model. Defaults to "bottleneck".
        name (str, optional): The name of the resulting model. Defaults to "encoder".

    Returns:
        Model: The encoder model.
    """

    bottleneck_layer = autoencoder.get_layer(bottleneck_layer_name).output
    return Model(inputs=autoencoder.inputs, outputs=bottleneck_layer, name=name)


def get_fully_connected_model(output_size: int, activation: str = "relu", l1_reg: float = 0.0, l2_reg: float = 0.0, name: str = "fc_model") -> Model:
    """The fully connected model predicting ProMP weights from the encoded images.

    Args:
        output_size (int): Size of the output layer.
        activation (str, optional): Activation of the hidden layers. Defaults to "relu".
        l1_reg (float, optional): L1 regularization value. Defaults to 0.0.
        l2_reg (float, optional): L2 regularization value. Defaults to 0.0.
        name (str, optional): Name of the model. Defaults to "fc_model".

    Returns:
        Model: The fully connected model.
    """

    input_layer = layers.Input(shape=(32, 32, 3), name="encoded_image_input")
    x = layers.Flatten(name="feature_vec")(input_layer)
    x = fully_connected_block(
        input_layer=x,
        neurons=[2816, 2560, 2304, 2048, 1792, 1536, 1280, 1024, 768, 512, 256, 128, 64],
        output_size=output_size, output_activation="linear",
        activation=activation,
        l1_reg=l1_reg, l2_reg=l2_reg
    )

    return Model(inputs=input_layer, outputs=x, name=name)


def get_convolutional_model(output_size: int, activation: str = "relu", l1_reg: float = 0.0, l2_reg: float = 0.0, name: str = "conv_model") -> Model:
    """The convolutional model predicting ProMP weights from the encoded images.

    Args:
        output_size (int): Size of the output layer.
        activation (str, optional): Activation of the hidden layers. Defaults to "relu".
        l1_reg (float, optional): L1 regularization value. Defaults to 0.0.
        l2_reg (float, optional): L2 regularization value. Defaults to 0.0.
        name (str, optional): Name of the model. Defaults to "fc_model".

    Returns:
        Model: The convolutional model.
    """

    input_layer = layers.Input(shape=(32, 32, 3), name="encoded_image_input")
    x = convolutional_block(input_layer, l1_reg=l1_reg, l2_reg=l2_reg)
    output_layer = fully_connected_block(x, [64], output_size, activation=activation, output_activation="linear", l1_reg=l1_reg, l2_reg=l2_reg)

    return Model(inputs=input_layer, outputs=output_layer, name=name)


def get_annotated_model(
    output_size: int, annotation_size: int,
    activation: str = "relu", l1_reg: float = 0.0, l2_reg: float = 0.0,
    name: str = "cnn_annotated_model"
) -> Model:
    """Model used to predict ProMP weights towards a given point, provided in the annotation.

    Args:
        output_size (int): Size of the output layer.
        annotation_size (int): Size of the annotation input.
        activation (str, optional): Activation of the hidden layers. Defaults to "relu".
        l1_reg (float, optional): L1 regularization value. Defaults to 0.0.
        l2_reg (float, optional): L2 regularization value. Defaults to 0.0.
        name (str, optional): Name of the model. Defaults to "cnn_annotated_model".

    Returns:
        Model: The annotated model.
    """

    input_layer = layers.Input(shape=(32, 32, 3), name="encoded_image_input")
    x = convolutional_block(input_layer, l1_reg=l1_reg, l2_reg=l2_reg)

    # Annotation input layer.
    annotation_input_layer = layers.Input(shape=(annotation_size, ), name="annotation")
    x = layers.Concatenate(name="annotated_feature_vec")((x, annotation_input_layer))
    output_layer = fully_connected_block(x, [64], output_size, activation=activation, output_activation="linear", l1_reg=l1_reg, l2_reg=l2_reg)

    return Model(inputs=[input_layer, annotation_input_layer], outputs=output_layer, name=name)


def get_residual_model(base_model: Model, name: str = None) -> Model:
    """Expand a base model to predict residual ProMP weights instead of full ProMP weights.

    The base model predicts the residual weights, the expanded model takes the mean weights as
    an additional input and outputs the full predicted ProMP weights.

    Args:
        base_model (Model): The base model, predicting residual ProMP weights.
        name (str, optional): The name of the model. Defaults to base_model.name + "_residual".

    Returns:
        Model: The expanded model, predicting residual ProMP weights.
    """

    if name is None:
        name = base_model.name + "_residual"

    # Add an input (mean weights) with same size of the base model output.
    input_size = base_model.output.shape[1]
    mean_weights_layer = layers.Input(shape=(input_size,), name="mean_weights")

    # The base model predicts the residual weights. Adding the mean component the output returns the full weight vector.
    residual_weights_layer = base_model.output
    output_layer = layers.Add(name="full_weights")((mean_weights_layer, residual_weights_layer))

    expanded_inputs = base_model.inputs + [mean_weights_layer]

    return Model(inputs=expanded_inputs, outputs=output_layer, name=name)
