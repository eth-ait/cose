import random
import numpy as np
import tensorflow as tf
from smartink.models.base_model import BaseModel
from smartink.models.common.building_blocks import RNNUtils
from common.constants import Constants as C


class UnconditionalSketchRNN(BaseModel):
  def __init__(self,
               config_rnn,
               config_loss,
               run_mode=C.RUN_ESTIMATOR,
               **kwargs):
    """Constructor.

    Args:
      config_encoder:
      config_embedding:
      config_rnn:
      config_loss:
      run_mode: eager, static or estimator.
      **kwargs:

    Raises:
      ValueError: if run_mode is eager and tf.executing_eagerly() is False.
      Exception: if # layers > 1 and dynamic_h0 is True.
    """
    super(UnconditionalSketchRNN, self).__init__(
        config_loss=config_loss, run_mode=run_mode, **kwargs)
    
    self.pen_threshold = 0.5
    self.config_rnn = config_rnn
    
    self.cell_units = self.config_rnn["cell_units"]
    self.cell_type = self.config_rnn["cell_type"]
    self.recurrent_dropout = self.config_rnn.get("rec_dropout_rate", 0.0)
    self.decoder_input_drop_rate = self.config_rnn.get("dropout_rate", 0)
    self.loss_prefix = self.config_loss.get("prefix", "")
    self.num_mixture = self.config_loss.get("n_gmm_components", 1)

    self.n_out = (3 + self.num_mixture*6)
    self.decoder_rnn = RNNUtils.get_rnn_layer(
          self.cell_type,
          self.cell_units,
          return_state=True,
          return_sequences=True,
          stateful=False,
          recurrent_dropout=self.recurrent_dropout,
          name="decoder_rnn")
    self.decoder_output_layer = tf.keras.layers.Dense(self.n_out)
    
  def call(self, inputs, training=False, **kwargs):
    seq_len = inputs.get("seq_len", None)
    mask = None
    if seq_len is not None:
      mask = tf.sequence_mask(seq_len)
    
    rnn_out = self.decoder_rnn(inputs["decoder_inputs"], mask=mask, training=training)
    outputs = self.decoder_output_layer(rnn_out[0])
    
    mixture_outputs = self.get_mixture_coef(tf.reshape(outputs, [-1, self.n_out]))
    # [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits] = out
    out = dict()
    out["raw"] = mixture_outputs
    out["state"] = [rnn_out[1], rnn_out[2]]
    return out

  def call_step(self, input_step, state=None, training=False):
    rnn_out = self.decoder_rnn(input_step,
                               initial_state=state,
                               training=training)
    outputs = self.decoder_output_layer(rnn_out[0])
  
    mixture_outputs = self.get_mixture_coef(
      tf.reshape(outputs, [-1, self.n_out]))
    # [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits] = out
    out = dict()
    out["raw"] = mixture_outputs
    out["state"] = [rnn_out[1], rnn_out[2]]
    return out

  def get_mixture_coef(self, output):
    """Returns the tf slices containing mdn dist params."""
    # This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850.
    z = output
    # z_pen_logits = z[:, :, 0:3]  # pen states
    z_pen_logits = z[:, 0:3]  # pen states
    # z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(z[:, :, 3:], 6, -1)
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(z[:, 3:], 6, -1)
  
    # process output z's into MDN parameters
    # softmax all the pi's and pen states:
    z_pi = tf.keras.activations.softmax(z_pi)
    z_pen = tf.keras.activations.softmax(z_pen_logits)
  
    # exponentiate the sigmas and also make corr between -1 and 1.
    z_sigma1 = tf.keras.activations.exponential(z_sigma1)
    z_sigma2 = tf.keras.activations.exponential(z_sigma2)
    z_corr = tf.keras.activations.tanh(z_corr)
  
    r = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, z_pen_logits]
    return r

  def get_lossfunc(self, z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr,
                   z_pen_logits, x1_data, x2_data, pen_data, training=False):
    """Returns a loss fn based on eq #26 of http://arxiv.org/abs/1308.0850."""
    # This represents the L_R only (i.e. does not include the KL loss term).
  
    result0 = self.tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
    epsilon = 1e-6
    # result1 is the loss wrt pen offset (L_s in equation 9 of
    # https://arxiv.org/pdf/1704.03477.pdf)
    result1 = tf.multiply(result0, z_pi)
    result1 = tf.reduce_sum(result1, 1, keepdims=True)
    result1 = -tf.math.log(result1 + epsilon)  # avoid log(0)
  
    fs = 1.0 - pen_data[:, 2]  # use training data for this
    fs = tf.reshape(fs, [-1, 1])
    # Zero out loss terms beyond N_s, the last actual stroke
    result1 = tf.multiply(result1, fs)
  
    # result2: loss wrt pen state, (L_p in equation 9)
    result2 = tf.nn.softmax_cross_entropy_with_logits(labels=pen_data,
                                                      logits=z_pen_logits)
    result2 = tf.reshape(result2, [-1, 1])
    if not training:  # eval mode, mask eos columns
      result2 = tf.multiply(result2, fs)
  
    return result1, result2

  def tf_2d_normal(self, x1, x2, mu1, mu2, s1, s2, rho):
    """Returns result of eq # 24 of http://arxiv.org/abs/1308.0850."""
    norm1 = tf.subtract(x1, mu1)
    norm2 = tf.subtract(x2, mu2)
    s1s2 = tf.multiply(s1, s2)
    
    # eq 25
    z = (tf.square(tf.divide(norm1, s1)) + tf.square(tf.divide(norm2, s2)) -
         2*tf.divide(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
    neg_rho = 1 - tf.square(rho)
    result = tf.exp(tf.divide(-z, 2*neg_rho))
    denom = 2*np.pi*tf.multiply(s1s2, tf.sqrt(neg_rho))
    result = tf.divide(result, denom)
    return result
  
  def loss(self, predictions, targets, seq_len=None, prefix="", training=False):
    batch_size = tf.shape(targets["seq_len"])[0]
    max_seq_len = tf.cast(tf.reduce_max(targets["seq_len"]), tf.int32)
    
    # reshape target data so that it is compatible with prediction shape
    stroke_5 = tf.reshape(targets["stroke_5"], [-1, 5])
    
    [x1_data, x2_data, eos_data, eoc_data, cont_data] = tf.split(stroke_5, 5, -1)
    pen_data = tf.concat([eos_data, eoc_data, cont_data], 1)

    # o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits = [tf.reshape(pred, [batch_size*max_seq_len, -1]) for pred in predictions["raw"]]
    o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits = predictions["raw"]
    loss_stroke, loss_pen = self.get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen_logits, x1_data, x2_data, pen_data, training)

    mask_ = tf.cast(tf.sequence_mask(targets["seq_len"]), tf.float32)
    loss_stroke = tf.reshape(loss_stroke, [batch_size, -1]) * mask_
    loss_pen = tf.reshape(loss_pen, [batch_size, -1]) * mask_
    
    loss_dict = dict()
    loss_dict[self.loss_prefix + "_stroke"] = tf.reduce_mean(loss_stroke)
    loss_dict[self.loss_prefix + "_pen"] = tf.reduce_mean(loss_pen)
    loss_dict["loss"] = tf.reduce_mean(loss_stroke+loss_pen)
    return loss_dict

  def sample_sketchrnn(self, state=None, seq_len=250, temperature=1.0, greedy_mode=False):
    """Samples a sequence from a pre-trained model."""
  
    def adjust_temp(pi_pdf, temp):
      pi_pdf = np.log(pi_pdf)/temp
      pi_pdf -= pi_pdf.max()
      pi_pdf = np.exp(pi_pdf)
      pi_pdf /= pi_pdf.sum()
      return pi_pdf
  
    def get_pi_idx(x, pdf, temp=1.0, greedy=False):
      """Samples from a pdf, optionally greedily."""
      if greedy:
        return np.argmax(pdf)
      pdf = adjust_temp(np.copy(pdf), temp)
      accumulate = 0
      for i in range(0, pdf.size):
        accumulate += pdf[i]
        if accumulate >= x:
          return i
      return -1
  
    def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
      if greedy:
        return mu1, mu2
      mean = [mu1, mu2]
      s1 *= temp*temp
      s2 *= temp*temp
      cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
      x = np.random.multivariate_normal(mean, cov, 1)
      return x[0][0], x[0][1]
  
    prev_x = np.zeros((1, 1, 5), dtype=np.float32)
    prev_x[0, 0, 2] = 1  # initially, we want to see beginning of new stroke
    prev_state = state
  
    strokes = np.zeros((seq_len, 5), dtype=np.float32)
    mixture_params = []
  
    greedy = greedy_mode
    temp = temperature
  
    for i in range(seq_len):
      out = self.call_step(prev_x, state=prev_state)
      [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits] = out["raw"]
      next_state = out["state"]
    
      idx = get_pi_idx(random.random(), o_pi[0], temp, greedy)
      idx_eos = get_pi_idx(random.random(), o_pen[0], temp, greedy)
      eos = [0, 0, 0]
      eos[idx_eos] = 1
    
      next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx],
                                            o_sigma1[0][idx],
                                            o_sigma2[0][idx],
                                            o_corr[0][idx], np.sqrt(temp),
                                            greedy)
    
      strokes[i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]
      
      params = [o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0], o_pen[0]]
      mixture_params.append(params)
    
      prev_x = np.array([[[next_x1, next_x2, eos[0], eos[1], eos[2]]]], dtype=np.float32)  # shape = (1, 1, 5)
      prev_state = next_state
    return strokes, mixture_params

  def to_stroke3(self, stroke5, drop_last=True):
    stroke3 = np.concatenate([stroke5[:, 0:2], stroke5[:, 3:4]], axis=-1)

    if drop_last and not np.any(np.where(stroke5[:, 4] == 1)[0]):
      penups = np.where(stroke5[:, 3] == 1)[0]
      if np.any(penups):
        last_penup = penups[-1]
        stroke3 = stroke3[:last_penup]
    
    ink_sample = dict(stroke=np.expand_dims(stroke3[:, 0:2], axis=0),
                      pen=np.expand_dims(stroke3[:, 2:3], axis=0),
                      seq_len=np.array([stroke3.shape[0]]))
    return ink_sample, stroke3
    
  
  @classmethod
  def get_model_tags(cls, config, config_loss=None):
    """Generates a string summarizing experiment parameters.

    Args:
      config:
      config_loss

    Returns:
    """
    output = "gmm" + str(config_loss.n_gmm_components)
    rnn = "{}_{}x{}".format(config.cell_type,
                            config.cell_layers,
                            config.cell_units)
    decoder = ""
    if config.rec_dropout_rate > 0:
      decoder = "rdrop" + str(config.rec_dropout_rate)
    
    return dict(encoder=rnn, latent="", decoder=decoder, output=output,
                model_name="USketchRNN")