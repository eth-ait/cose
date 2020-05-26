"""Functions building neural network components."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from common.constants import Constants as C
from smartink.util.utils import err_unknown_type


class Activations(object):
  """Factory class for tensorflow activation functions."""

  @classmethod
  def get(cls, type_str):
    """Creates the activation function, given its type.

    Args:
      type_str:

    Returns:
    """
    # Check if the activation is already callable.
    if callable(type_str):
      return type_str

    # Check if the activation is a built-in or custom function.
    if type_str == C.RELU:
      return tf.nn.relu
    elif type_str == C.ELU:
      return tf.nn.elu
    elif type_str == C.TANH:
      return tf.nn.tanh
    elif type_str == C.SIGMOID:
      return tf.nn.sigmoid
    elif type_str == C.SOFTPLUS:
      return tf.nn.softplus
    elif type_str == C.SOFTMAX:
      return tf.nn.softmax
    elif type_str == C.LRELU:
      return lambda x: tf.nn.leaky_relu(x, alpha=1. / 3.)
    elif type_str == C.CLRELU:
      with tf.compat.v1.name_scope("ClampedLeakyRelu"):

        def clamped_leaky_relu(x):
          return tf.clip_by_value(tf.nn.leaky_relu(x, alpha=1. / 3.), -3.0, 3.0)

        return clamped_leaky_relu
    elif type_str is None:
      return None
    else:
      err_unknown_type(type_str)


class RNNUtils(object):
  """Factory class for RNN cells/layers."""

  @classmethod
  def get_cells(cls, type_str, units, layers=1):
    """Creates a cell.

    Args:
      type_str:
      units:
      layers:

    Returns:
    """
    cells = []

    for _ in range(layers):
      if type_str == C.LSTM:
        cells.append(tf.keras.layers.LSTMCell(units))
      elif type_str == C.GRU:
        cells.append(tf.keras.layers.GRUCell(units))
      else:
        err_unknown_type(type_str)
    return cells

  @classmethod
  def get_initial_state(cls, cell, inputs):
    """Generates initial state of a rnn cell.

    Cell's call method expects the state in list type.
    GRUCell, however, generates tensor.
    Args:
      cell:
      inputs:

    Returns:
    """
    state_0 = cell.get_initial_state(inputs=inputs)
    if isinstance(cell, tf.keras.layers.GRUCell):
      return [state_0]
    else:
      return state_0

  @classmethod
  def get_initial_states(cls, cell_list, inputs):
    """Generates initial state of a stack of rnn cells.

    Cell's call method expects the state in list type.
    GRUCell, however, generates tensor.
    Args:
      cell_list (list): of cells.
      inputs:

    Returns:
    """
    states = []
    for cell in cell_list:
      states.append(RNNUtils.get_initial_state(cell, inputs))
    return states

  @classmethod
  def get_initial_states_layer(cls, layer_list, inputs):
    """Generates initial state of a stack of rnn layers.

    Args:
      layer_list (list): of cells.
      inputs:

    Returns:
    """
    states = []
    for rnn_layer in layer_list:
      states.append(rnn_layer.get_initial_state(inputs))
    return states

  @classmethod
  def set_initial_states(cls, state_ops, values):
    """Creates a dictionary of rnn state ops and corresponding values.

    Args:
      state_ops: list of rnn state ops, nested if it is lstm cell.
      values: list of rnn state in numpy.

    Returns:
      Dictionary with state:value pairs.
    """
    out_dict = dict()
    for idx, state_op in enumerate(state_ops):
      out_dict[state_op[0]] = values[idx][0]
      if len(state_op) == 2:  # LSTM
        out_dict[state_op[1]] = values[idx][1]
    return out_dict

  @classmethod
  def get_rnn_layer(cls,
                    type_str,
                    units,
                    return_sequences,
                    return_state,
                    stateful,
                    name,
                    recurrent_dropout=0.0):
    """Generates an RNN layer.

    Args:
      type_str:
      units:
      return_sequences:
      return_state:
      stateful:
      name:
      recurrent_dropout:

    Returns:
    """
    cell_cls = None
    if type_str == C.LSTM:
      cell_cls = tf.keras.layers.LSTM
    elif type_str == C.GRU:
      cell_cls = tf.keras.layers.GRU
    else:
      err_unknown_type(type_str)

    return cell_cls(
        units=units,
        return_sequences=return_sequences,
        return_state=return_state,
        stateful=stateful,
        recurrent_dropout=recurrent_dropout,
        name=name)


class DenseLayers(tf.keras.Model):
  """Creates fully connected layers."""

  def __init__(self,
               layer_units,
               hidden_activation=tf.keras.activations.relu,
               output_activation=None,
               prefix=""):
    """
    Args:
      layer_units: list of units per layer.
      hidden_activation:
      prefix: name prefix
    """

    super(DenseLayers, self).__init__()
    self.layer_units = layer_units
    self.prefix = prefix
    if prefix and not prefix.endswith("_"):
      self.prefix += "_"

    self.initializer = tf.compat.v1.random_normal_initializer(stddev=0.001)

    self.dense_layers = tf.keras.Sequential()
    for idx, units in enumerate(layer_units[:-1]):
      self.dense_layers.add(
          tf.keras.layers.Dense(
              units=units,
              activation=hidden_activation,
              name=prefix + "hidden_" + str(idx)))

    self.dense_layers.add(
        tf.keras.layers.Dense(
            kernel_initializer=self.initializer,
            units=layer_units[-1],
            activation=output_activation,
            name=prefix + "out"))

  def call(self, inputs, training=None, **kwargs):
    return self.dense_layers(inputs)
