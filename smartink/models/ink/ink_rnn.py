"""
RNN model of Alex Graves.
"""

import tensorflow as tf

from common.constants import Constants as C
from smartink.models.base_model import BaseModel
from smartink.models.common.output import OutputModelDeterministic
from smartink.models.common.output import OutputModelNormal
from smartink.models.common.output import OutputModelNormal2DDense
from smartink.models.common.output import OutputModelGMMDense
from smartink.models.common.building_blocks import RNNUtils


class InkRNN(BaseModel):
  def __init__(self,
               config_rnn,
               config_loss=None,
               name="ink_rnn",
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
    """
    super(InkRNN, self).__init__(
        config_loss=config_loss, run_mode=run_mode, **kwargs)
    
    self.config_rnn = config_rnn
    self.cell_units = self.config_rnn["cell_units"]
    self.cell_layers = self.config_rnn["cell_layers"]
    self.cell_type = self.config_rnn["cell_type"]
    self.config_loss = config_loss
    self.pen_threshold = 0.5

    self._rnn_layer = RNNUtils.get_rnn_layer(
        self.cell_type,
        self.cell_units,
        return_state=True,
        return_sequences=True,
        stateful=False,
        name=name)

    self._rnn_layer2 = None
    if self.cell_layers > 1:
      self._rnn_layer2 = RNNUtils.get_rnn_layer(
          self.cell_type,
          self.cell_units,
          return_state=True,
          return_sequences=True,
          stateful=False,
          name=name)

    self._rnn_layer3 = None
    if self.cell_layers > 2:
      self._rnn_layer3 = RNNUtils.get_rnn_layer(
          self.cell_type,
          self.cell_units,
          return_state=True,
          return_sequences=True,
          stateful=False,
          name=name)
    
    # Pen, stroke and end-of-sequence outputs.
    self.out_eos = True
    if "eos" in config_loss:
      self.out_eos = tf.keras.layers.Dense(1, name="out_eos",
                                           kernel_regularizer=self.kernel_regularizer,
                                           bias_regularizer=self.kernel_regularizer)
    
    self.out_pen = tf.keras.layers.Dense(1, name="out_pen")

    # Build output model depending on the loss type.
    if self.config_loss["stroke"]["loss_type"] == C.NLL_NORMAL:
      self.out_stroke = OutputModelNormal(
          out_units=2, hidden_units=0, hidden_layers=0)
    elif self.config_loss["stroke"]["loss_type"] == C.NLL_BINORMAL:
      self.out_stroke = OutputModelNormal2DDense(
          sigma_activation=tf.keras.activations.exponential)
    elif self.config_loss["stroke"]["loss_type"] == C.NLL_GMM:
      self.out_stroke = OutputModelGMMDense(
          out_units=2,
          num_components=self.config_loss["stroke"]["num_components"],
          sigma_activation=tf.keras.activations.exponential)
    else:
      self.out_stroke = OutputModelDeterministic(
          out_units=2, hidden_units=0, hidden_layers=0)
    

  def call(self, inputs, training=None, **kwargs):
    """Call method."""
    out_dict = dict()
    final_inputs = []
    shifted_inputs = tf.concat([tf.zeros_like(inputs[C.INP_ENC][:, 0:1]), inputs[C.INP_ENC][:, :-1]], axis=1)
    shifted_len = tf.minimum(tf.reduce_max(inputs[C.INP_SEQ_LEN]), inputs[C.INP_SEQ_LEN] + 1)
    state = inputs.get("rnn_state", None)
    mask = tf.sequence_mask(shifted_len)
    rnn_out, state_h, state_c = self._rnn_layer(shifted_inputs, mask=mask, training=training, initial_state=state)
    
    out_dict["rnn_state"] = [state_h, state_c]
    final_inputs.append(rnn_out)
    
    if self._rnn_layer2 is not None:
      state2 = inputs.get("rnn_state2", None)
      rnn_inp = tf.concat([rnn_out, inputs[C.INP_ENC]], axis=-1)
      rnn_out, state_h2, state_c2 = self._rnn_layer2(rnn_inp, mask=mask,
                                                     training=training,
                                                     initial_state=state2)
      out_dict["rnn_state2"] = [state_h2, state_c2]
      final_inputs.append(rnn_out)
      
    if self._rnn_layer3 is not None:
      state3 = inputs.get("rnn_state3", None)
      rnn_inp = tf.concat([rnn_out, inputs[C.INP_ENC]], axis=-1)
      rnn_out, state_h3, state_c3 = self._rnn_layer3(rnn_inp, mask=mask,
                                                     training=training,
                                                     initial_state=state3)
      out_dict["rnn_state3"] = [state_h3, state_c3]
      final_inputs.append(rnn_out)

    rnn_out = tf.concat(final_inputs, axis=-1)
    pen_logits = self.out_pen(rnn_out)
    stroke_logits = self.out_stroke(rnn_out)

    # Calculate pen-up probability from the logits.
    pen_prob = tf.nn.sigmoid(pen_logits)
    pen_binary = tf.compat.v1.where(
        tf.greater(pen_prob,
                   tf.fill(tf.shape(input=pen_prob), self.pen_threshold)),
        tf.fill(tf.shape(input=pen_prob), 1.0),
        tf.fill(tf.shape(input=pen_prob), 0.0))

    stroke_sample = self.out_stroke.draw_sample(stroke_logits, greedy=True)
    
    out_dict["stroke"]=stroke_sample
    out_dict["stroke_logits"]=stroke_logits
    out_dict["pen_logits"]=pen_logits
    out_dict["pen_prob"]=pen_prob
    out_dict["pen"]=pen_binary
    return out_dict

  @classmethod
  def get_model_tags(cls, config, config_loss=None):
    """Generates a string summarizing experiment parameters.

    Args:
      config:
      config_loss

    Returns:
    """
    if config_loss["stroke"]["loss_type"] == C.NLL_NORMAL:
      output = "normal"
    elif config_loss["stroke"]["loss_type"] == C.NLL_BINORMAL:
      output = "binormal"
    elif config_loss["stroke"]["loss_type"] == C.NLL_GMM:
      output = "gmm"
    else:
      output = config_loss["stroke"]["loss_type"]
      
    rnn = "{}_{}x{}".format(config.cell_type,
                            config.cell_layers,
                            config.cell_units)
    return dict(encoder=rnn, latent="", decoder="", output=output,
                model_name="InkRNN")