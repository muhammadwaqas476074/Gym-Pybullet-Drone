import os
from typing import Iterable, Dict

import kerasncp as kncp
from kerasncp.tf import LTCCell, WiredCfcCell
from tensorflow import keras, concat

from node_cell import *
from tf_cfc import CfcCell, MixedCfcCell
from tf_cfc import LTCCell as CFCLTCCell

import tensorflow as tf

IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

DROPOUT = 0.1
DEFAULT_CFC_CONFIG = {
    "clipnorm": 1,
    "backbone_activation": "silu",
    "backbone_dr": 0.1,
    "forget_bias": 1.6,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 1e-06
}
DEFAULT_NCP_SEED = 22222
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class ModelParams:
    # dataclasses can't have non-default follow default
    seq_len: int = field(default=False, init=True)
    image_shape: Tuple[int, int, int] = IMAGE_SHAPE
    augmentation_params: Optional[Dict] = None
    batch_size: Optional[int] = None
    single_step: bool = False
    no_norm_layer: bool = False


@dataclass
class NCPParams(ModelParams):
    seed: int = DEFAULT_NCP_SEED

def model(checkpoint_path: str, single_step: bool):
    """
    Convenience function that calls load_model_from weights as above but tries to infer reasonable default params if not
    known
    """
    if 'ncp' in checkpoint_path:
        params = NCPParams(seq_len=64, single_step=single_step)
    elif 'mixedcfc' in checkpoint_path:
        params = CTRNNParams(seq_len=64, rnn_sizes=[128], ct_network_type="mixedcfc", single_step=single_step)
    elif 'lstm' in checkpoint_path:
        params = LSTMParams(seq_len=64, rnn_sizes=[128], single_step=single_step)
    elif "tcn" in checkpoint_path:
        params = TCNParams(seq_len=64, nb_filters=128, kernel_size=2, dilations=[1, 2, 4, 8, 16, 32])
    else:
        raise ValueError(f"Unable to infer model name from path {checkpoint_path}")

    return load_model_from_weights(params, checkpoint_path)

def load_model_from_weights(params: ModelParams, checkpoint_path: str, load_name_ok: bool = False):
    """
    Convenience function that loads weights from checkpoint_path into model_skeleton
    """
    model_skeleton = get_skeleton(params)
    if load_name_ok:
        try:
            model_skeleton.load_weights(checkpoint_path)
        except ValueError:
            # different number of weights from file and model. Assume normalization layer in model but not file
            # rename conv layers starting at 5
            print("Model had incorrect number of layers. Attempting to load from layer names")
            conv_index = 5
            dense_index = 1
            for layer in model_skeleton.layers:
                if isinstance(layer, Conv2D):
                    layer._name = f"conv2d_{conv_index}"
                    conv_index += 1
                elif isinstance(layer, Dense):
                    layer._name = f"dense_{dense_index}"
                    dense_index += 1
            model_skeleton.load_weights(checkpoint_path, by_name=True)
    else:
        model_skeleton.load_weights(checkpoint_path)

    return model_skeleton

def get_skeleton(params: ModelParams):
    """
    Returns a new model with randomized weights according to the parameters in params
    """
    if isinstance(params, NCPParams) or "NCPParams" in params.__class__.__name__:
        model_skeleton = generate_ncp_model(**asdict(params))

    else:
        raise ValueError(f"Could not parse param type {params.__class__}")
    return model_skeleton

def generate_ncp_model(seq_len,
                       image_shape,
                       augmentation_params=None,
                       batch_size=None,
                       seed=DEFAULT_NCP_SEED,
                       single_step: bool = False,
                       no_norm_layer: bool = False,
                       ):
    inputs_image, inputs_value, x = generate_network_trunk(
        seq_len,
        image_shape,
        augmentation_params=augmentation_params,
        batch_size=batch_size,
        single_step=single_step,
        no_norm_layer=no_norm_layer,
    )

    # Setup the network
    wiring = kncp.wirings.NCP(
        inter_neurons=18,  # Number of inter neurons
        command_neurons=12,  # Number of command neurons
        motor_neurons=4,  # Number of motor neurons
        sensory_fanout=6,  # How many outgoing synapses has each sensory neuron
        inter_fanout=4,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=4,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=6,  # How many incoming syanpses has each motor neuron,
        seed=seed,  # random seed to generate connections between nodes
    )

    rnn_cell = LTCCell(wiring)

    if single_step:
        inputs_state = tf.keras.Input(shape=(rnn_cell.state_size,))
        # wrap output states in list since want output to just be ndarray, not list of 1 el ndarray
        motor_out, [output_states] = rnn_cell(x, inputs_state)
        ncp_model = keras.Model([inputs_image, inputs_value, inputs_state], [motor_out, output_states])
    else:
        x = keras.layers.RNN(rnn_cell,
                             batch_input_shape=(batch_size,
                                                seq_len,
                                                x.shape[-1]),
                             return_sequences=True)(x)

        ncp_model = keras.Model([inputs_image, inputs_value], [x])

    return ncp_model




def generate_augmentation_layers(x, augmentation_params: Dict, single_step: bool):
    # translate -> rotate -> zoom -> noise
    trans = augmentation_params.get('translation', None)
    rot = augmentation_params.get('rotation', None)
    zoom = augmentation_params.get('zoom', None)
    noise = augmentation_params.get('noise', None)

    if trans is not None:
        x = wrap_time(keras.layers.experimental.preprocessing.RandomTranslation(
            height_factor=trans, width_factor=trans), single_step)(x)

    if rot is not None:
        x = wrap_time(keras.layers.experimental.preprocessing.RandomRotation(rot), single_step)(x)

    if zoom is not None:
        x = wrap_time(keras.layers.experimental.preprocessing.RandomZoom(
            height_factor=zoom, width_factor=zoom), single_step)(x)

    if noise:
        x = wrap_time(keras.layers.GaussianNoise(stddev=noise), single_step)(x)

    return x


def generate_normalization_layers(x, single_step: bool):
    rescaling_layer = keras.layers.experimental.preprocessing.Rescaling(1. / 255)

    normalization_layer = keras.layers.experimental.preprocessing.Normalization(
        mean=[0.41718618, 0.48529191, 0.38133072],
        variance=[.057, .05, .061])

    x = rescaling_layer(x)
    x = wrap_time(normalization_layer, single_step)(x)
    return x


def wrap_time(layer, single_step: bool):
    """
    Helper function that wraps layer in a timedistributed or not depending on the arguments of this function
    """
    if not single_step:
        return keras.layers.TimeDistributed(layer)
    else:
        return layer


def generate_network_trunk(seq_len,
                           image_shape,
                           augmentation_params: Dict = None,
                           batch_size=None,
                           single_step: bool = False,
                           no_norm_layer: bool = False, ):
    """
    Generates CNN image processing backbone used in all recurrent models. Uses Keras.Functional API

    returns input to be used in Keras.Model and x, a tensor that represents the output of the network that has shape
    (batch [None], seq_len, num_units) if single step is false and (batch [None], num_units) if single step is true.
    Input has shape (batch, h, w, c) if single step is True and (batch, seq, h, w, c) otherwise

    """

    if single_step:
        inputs_image = keras.Input(shape=image_shape, name="input_image")
        inputs_value = keras.Input(shape=(2,), name="input_vector")
    else:
        inputs_image = keras.Input(batch_input_shape=(batch_size, seq_len, *image_shape), name="input_image")
        inputs_value = keras.Input(batch_input_shape=(batch_size, seq_len, 2), name="input_vector")

    xi = inputs_image
    xp = inputs_value

    if not no_norm_layer:
        xi = generate_normalization_layers(xi, single_step)

    if augmentation_params is not None:
        xi = generate_augmentation_layers(xi, augmentation_params, single_step)

    # Conv Layers
    xi = wrap_time(keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'), single_step)(
        xi)
    xi = wrap_time(keras.layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'), single_step)(
        xi)
    xi = wrap_time(keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'), single_step)(
        xi)
    xi = wrap_time(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'), single_step)(
        xi)
    xi = wrap_time(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu'), single_step)(
        xi)

    xi = wrap_time(keras.layers.Flatten(), single_step)(xi)
    xi = wrap_time(keras.layers.Dense(units=128, activation='linear'), single_step)(xi)
    xi = wrap_time(keras.layers.Dropout(rate=DROPOUT), single_step)(xi)

    xp = wrap_time(keras.layers.Dense(units=128, activation='relu'), single_step)(xp)
    xp = wrap_time(keras.layers.Dropout(rate=DROPOUT), single_step)(xp)

    # x = wrap_time(keras.layers.Concatenate(axis=-1), single_step)([xi, xp])
    # concatenate xi and xp using tf.concat along the last axis
    #x = wrap_time(keras.layers.Lambda(lambda y: tf.concat(y, axis=-1)), single_step)([xi, xp])
    x = tf.concat([xi, xp], axis=-1)

    return inputs_image, inputs_value, x