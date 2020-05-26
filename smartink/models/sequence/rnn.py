"""RNN wrapper for sequences."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from common.constants import Constants as C  # pylint: disable=g-import-not-at-top
from smartink.models.base_model import BaseModel  # pylint: disable=g-import-not-at-top
from smartink.models.common.building_blocks import RNNUtils  # pylint: disable=g-import-not-at-top
from smartink.models.common.output import OutputModelDeterministic  # pylint: disable=g-import-not-at-top
from smartink.models.common.output import OutputModelNormal  # pylint: disable=g-import-not-at-top
from smartink.models.common.output import OutputModelNormal2DDense  # pylint: disable=g-import-not-at-top
from smartink.models.common.output import OutputModelGMMDense  # pylint: disable=g-import-not-at-top


class RNN(BaseModel):
  """Uni- or bi-directional RNN model."""

  def __init__(self,
               cell_type,
               cell_units,
               cell_layers,
               bidirectional=False,
               return_state=True,
               return_sequences=True,
               output_size=0,
               config_loss=None,
               name="rnn",
               run_mode=C.RUN_STATIC,
               **kwargs):
    """Constructor.

    Args:
      cell_type (str): 'lstm' or 'gru'.
      cell_units: number of cell units.
      cell_layers: number of encoder/decoder rnn cells.
      bidirectional:
      return_state:
      return_sequences:
      output_size: encoder/decoder rnn cell/output size.
      config_loss: loss configuration.
      name:
      run_mode: eager, static or estimator.
      **kwargs:

    Raises:
      ValueError: if run_mode is eager and tf.executing_eagerly() is False.
    """
    super(RNN, self).__init__(
        config_loss=config_loss, run_mode=run_mode, **kwargs)

    self.cell_units = cell_units
    self.cell_layers = cell_layers
    self.cell_type = cell_type
    self.bidirectional = bidirectional
    self.return_state = return_state
    self.return_sequences = return_sequences

    self.output_size = output_size

    self.output_layer = None
    self._rnn_layer = tf.keras.Sequential()

    for i in range(self.cell_layers - 1):
      rnn_layer = RNNUtils.get_rnn_layer(
          self.cell_type,
          self.cell_units,
          return_state=False,
          return_sequences=True,
          stateful=False,
          name=name + "_" + str(i))

      if self.bidirectional:
        rnn_layer = tf.keras.layers.Bidirectional(
            rnn_layer, merge_mode="concat")
      self._rnn_layer.add(rnn_layer)

    rnn_layer = RNNUtils.get_rnn_layer(
        self.cell_type,
        self.cell_units,
        return_state=return_state,
        return_sequences=return_sequences,
        stateful=False,
        name=name)
    if self.bidirectional:
      rnn_layer = tf.keras.layers.Bidirectional(rnn_layer, merge_mode="concat")
    self._rnn_layer.add(rnn_layer)

    # Deterministic or probabilistic outputs.
    if output_size > 0:
      if config_loss is not None:
        if config_loss["loss_type"] == C.NLL_NORMAL:
          self.output_layer = OutputModelNormal(self.output_size, logvar=True)
        elif config_loss["loss_type"] == C.NLL_BINORMAL:
          self.output_layer = OutputModelNormal2DDense(sigma_activation=tf.keras.activations.exponential)
        elif config_loss["loss_type"] == C.NLL_GMM:
          self.output_layer = OutputModelGMMDense(
              out_units=self.output_size,
              num_components=config_loss["num_components"],
              sigma_activation=tf.keras.activations.exponential)
        else:
          self.output_layer = OutputModelDeterministic(self.output_size, 0, 0)
      else:
        self.output_layer = OutputModelDeterministic(self.output_size, 0, 0)

  def call(self, inputs, training=None, **kwargs):
    """Call method.

    Given a sequence, predicts next time-step t+1 for each input step t.
    Args:
      inputs: [batch_size, seq_len, feature_size]
      training: whether in training mode or not.
      **kwargs:

    Returns:
      [batch_size, seq_len, output_size]
    """
    mask = None
    input_seq = inputs["input_seq"]
    seq_len = inputs.get("seq_len", None)
    if seq_len is not None:
      mask = tf.sequence_mask(seq_len)

    rnn_hidden = self._rnn_layer(input_seq, mask=mask, training=training)

    if self.output_layer is not None:
      return self.output_layer(rnn_hidden, training=training)
    else:
      return rnn_hidden
  
  @classmethod
  def get_model_tags(cls, config, config_loss=None):
    """Generates a string summarizing experiment parameters.

    Args:
      config:
      config_loss

    Returns:
    """
    rnn = "{}_{}x{}".format(config.cell_type,
                            config.cell_layers,
                            config.cell_units)
  
    return dict(model=rnn, model_name="RNN")
