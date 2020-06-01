"""Models for strokes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf

import smartink.util.utils as utils
from common.constants import Constants as C
from smartink.util.utils import err_unknown_type
from smartink.models.base_model import BaseModel
from smartink.models.common.building_blocks import RNNUtils
from smartink.models.common.output import OutputModelDeterministic
from smartink.models.common.output import OutputModelNormal
from smartink.models.common.output import OutputModelNormal2DDense
from smartink.models.common.output import OutputModelGMMDense


class InkSeq2Seq(BaseModel):
  """A sequence to sequence model.

  The encoder and decoder networks are created by
  stacking RNN layers. Hence, it supports fast RNN
  layer implementations backed by cuDNN.
  The cuDNN variant is much faster, but it only works
  with a GPU. A model trained on GPU can't be restored
  on CPU.
  """
  
  def __init__(self,
               config_encoder,
               config_embedding,
               config_decoder,
               config_loss,
               run_mode=C.RUN_ESTIMATOR,
               **kwargs):
    """Constructor.

    Args:
      config_encoder:
      config_embedding:
      config_decoder:
      config_loss:
      run_mode: eager, static or estimator.
      **kwargs:

    Raises:
      ValueError: if run_mode is eager and tf.executing_eagerly() is False.
      Exception: if # layers > 1 and dynamic_h0 is True.
    """
    super(InkSeq2Seq, self).__init__(
        config_loss=config_loss, run_mode=run_mode, **kwargs)
    
    self.pen_threshold = 0.5
    self.config_encoder = config_encoder
    self.config_embedding = config_embedding
    self.config_decoder = config_decoder

    self.n_cell_units = self.config_encoder["cell_units"]
    self.n_cell_layers = self.config_encoder["cell_layers"]
    self.cell_type = self.config_encoder["cell_type"]
    self.bidirectional_encoder = self.config_encoder["bidirectional_encoder"]
    self.recurrent_dropout = self.config_encoder.get("rec_dropout_rate", 0.0)
    
    self.n_latent_units = self.config_embedding["latent_units"]
    self.use_vae = self.config_embedding["use_vae"]
    
    self.decoder_drop_rate = self.config_decoder.get("dropout_rate", 0)
    self.repeat_vae_sample = self.config_decoder.get("repeat_vae_sample", False)
    self.embedding_only = not self.config_decoder.get("autoregressive", False)
    self.dynamic_h0 = self.config_decoder.get("dynamic_h0", False)
    self.latent_prefix = ""
    
    if self.dynamic_h0 and self.n_cell_layers != 1:
      raise Exception("# rnn layers must be 1 for dynamic h0.")
    
    # RNN layer containers.
    self.encoder_rnn = list()
    self.decoder_rnn = list()
    
    # Encoder network:
    enc_rnn_units = self.n_cell_units
    if self.bidirectional_encoder:
      enc_rnn_units = enc_rnn_units//2
    for idx in range(self.n_cell_layers):
      rnn_layer = RNNUtils.get_rnn_layer(
          self.cell_type,
          enc_rnn_units,
          return_state=True,
          return_sequences=True,
          stateful=False,
          name="encoder_rnn_" + str(idx + 1),
          recurrent_dropout=self.recurrent_dropout)
      
      if self.bidirectional_encoder:
        rnn_layer = tf.keras.layers.Bidirectional(
            rnn_layer, merge_mode="concat")
      self.encoder_rnn.append(rnn_layer)
    
    # Deterministic or stochastic embedding.
    if self.use_vae:
      self.net_embedding = OutputModelNormal(
          out_units=self.n_latent_units,
          prefix=self.latent_prefix,
          sigma_activation=None,
          logvar=True)
    else:
      self.net_embedding = OutputModelDeterministic(
          out_units=self.n_latent_units,
          hidden_units=self.n_latent_units*2,
          hidden_layers=0,
          prefix=self.latent_prefix)
    
    # Decoder network:
    # Embedding (+input) -> Dense -> RNN layers -> Reconstruction.
    self.decoder_inp_dense = tf.keras.Sequential(name="decoder_inp")
    
    # RNN state is required for autoregressive prediction.
    for idx in range(self.n_cell_layers):
      rnn_layer = RNNUtils.get_rnn_layer(
          self.cell_type,
          self.n_cell_units,
          return_state=True,
          return_sequences=True,
          stateful=False,
          name="decoder_rnn_" + str(idx + 1))
      self.decoder_rnn.append(rnn_layer)
    
    # Pen and stroke outputs.
    if config_loss["pen"]["eval_only"]:
      self.decoder_out_pen = None
    else:
      self.decoder_out_pen = tf.keras.layers.Dense(1, activation=None, name="out_pen")
    
    # Build output model depending on the loss type.
    if self.config_loss["stroke"]["loss_type"] == C.NLL_NORMAL:
      self.decoder_out_stroke = OutputModelNormal(
          out_units=2, hidden_units=0, hidden_layers=0)
    elif self.config_loss["stroke"]["loss_type"] == C.NLL_BINORMAL:
      self.decoder_out_stroke = OutputModelNormal2DDense(
          sigma_activation=tf.keras.activations.exponential)
    elif self.config_loss["stroke"]["loss_type"] == C.NLL_GMM:
      self.decoder_out_stroke = OutputModelGMMDense(
          out_units=2,
          num_components=self.config_loss["stroke"]["num_components"],
          sigma_activation=tf.keras.activations.exponential)
    else:
      self.decoder_out_stroke = OutputModelDeterministic(out_units=2,
                                                         hidden_units=0,
                                                         hidden_layers=0)
    self.decoder_inp_dropout = None
    if self.decoder_drop_rate > 0:
      self.decoder_inp_dropout = tf.keras.layers.Dropout(self.decoder_drop_rate)
    
    # Provides access to the sample op if repeat_vae_sample is True.
    if self.dynamic_h0:
      init_ = tf.compat.v1.random_normal_initializer(stddev=0.001)
      dense_ = tf.keras.layers.Dense(
          self.n_cell_units*2,
          activation=tf.keras.activations.tanh,
          kernel_initializer=init_)
      self.decoder_state_nn = dense_
    else:
      self.decoder_state_nn = None

    # Variables for static mode. They are assigned in call method.
    # TODO(eaksan) We can get rid of them if autoregressive sampling is no
    #  longer required in static (graph) mode.
    self.op_encoder_inputs = None
    self.op_decoder_inputs = None
    self.op_input_seq_len = None
    self.op_embedding = None
    # self.op_decoder_initial_state = None
    self.op_embedding_sample = None
  
  def call(self, inputs, output_len=None, training=None, **kwargs):
    """Encoder and decoder functionality.

    Given an input sequence, calculates the embedding and reconstructs the
    sequence or makes a prediction autoregressively.

    If decoder_inputs is passed, then the decoder is fed with the input
    embedding and the corresponding decoder_inputs step.

    If the decoder_inputs is None and output_len passed, then the decoder
    is fed with its own predictions at the next step.

    The length of the output sequence is determined by either the length
    of the decoder_inputs or the output_len.
    Args:
      inputs (dict): expected to contain inputs for the encoder and decoder,
        and seq len ops.
      output_len (int): length of output sequence. If None, it is determined
        from the decoder input sequence.
      training: whether in training mode or not.
      **kwargs:

    Returns:
      [batch_size, seq_len, feature_size]
    """
    self.op_encoder_inputs = inputs[C.INP_ENC]
    self.op_decoder_inputs = inputs[C.INP_DEC] if output_len is None else None
    self.op_input_seq_len = inputs[C.INP_SEQ_LEN]
    
    assert not (self.op_decoder_inputs is None and
                output_len is None), "Output length is undetermined."
    
    self.op_embedding = self.call_encode(self.op_encoder_inputs,
                                         self.op_input_seq_len, training)
    
    # We need the embedding distribution in case of VAE. Hence, not passing an
    # embedding sample, but the embedding predictions.
    out_dict = self.call_decode(self.op_embedding,
                                self.op_decoder_inputs,
                                output_len,
                                None,
                                training)
    
    out_dict["embedding"] = self.op_embedding
    out_dict["embedding_sample"] = self.op_embedding_sample
    return out_dict
  
  def call_encode(self, inputs, input_seq_len, training):
    """Calculates the stroke embedding.

    Args:
      inputs:
      input_seq_len:
      training:

    Returns:
      embedding of size [batch_size, 1, latent_size]
    """
    rnn_layer = self.encoder_rnn[0]
    # non_zero_seq_len = tf.where(input_seq_len == 0, 1, input_seq_len)
    non_zero_seq_len = tf.compat.v1.where(input_seq_len == 0, tf.ones_like(input_seq_len), input_seq_len)
    encoder_rnn = rnn_layer(inputs, mask=tf.sequence_mask(non_zero_seq_len), training=training)
    
    if self.bidirectional_encoder:
      embedding_last_step = tf.concat([encoder_rnn[1], encoder_rnn[3]], axis=-1)
    else:
      embedding_last_step = encoder_rnn[1]

    embedding = self.net_embedding(embedding_last_step, training=training)
    return embedding
  
  def call_decode(self,
                  embedding,
                  decoder_inputs=None,
                  output_len=None,
                  decoder_rnn_state=None,
                  training=None):
    """Reconstructs stroke sequence given an embedding.

    If embedding_only is True, then decoder_inputs is not used.
    Args:
      embedding: (batch_size, n_latent_units)
      decoder_inputs:
      output_len:
      decoder_rnn_state:
      training:

    Returns:
      A dictionary of stroke, pen logits, pen probability and binary pen.
    """
    if isinstance(embedding, dict):
      self.op_embedding_sample = self.net_embedding.draw_sample(embedding)
    else:
      self.op_embedding_sample = embedding
      
    embedding_sample = tf.expand_dims(self.op_embedding_sample, axis=1)
    
    if decoder_rnn_state is None:
      op_decoder_initial_state = RNNUtils.get_initial_states_layer(
          self.decoder_rnn, embedding_sample)
      if self.dynamic_h0:
        decoder_state = self.decoder_state_nn(embedding_sample[:, 0])
        op_decoder_initial_state[0] = tf.split(decoder_state, 2, axis=-1)
      decoder_rnn_state = op_decoder_initial_state
    if decoder_inputs is not None:
      output_len = tf.shape(input=decoder_inputs)[1]
    
    # Prepare decoder input.
    if not self.use_vae or self.repeat_vae_sample or not isinstance(embedding, dict):
      # Use the same latent sample in all decoder steps.
      embedding_seq = tf.tile(embedding_sample, (1, output_len, 1))
    else:
      # Draw a new latent sample per decoding step.
      assert isinstance(embedding, dict), "Latent distribution is required."
      embedding_seq = dict()
      mu_ = tf.expand_dims(embedding[self.latent_prefix + C.MU], axis=1)
      embedding_seq[self.latent_prefix + C.MU] = tf.tile(mu_,
                                                         (1, output_len, 1))
      sigma_ = tf.expand_dims(embedding.get(self.latent_prefix + C.SIGMA, None), axis=1)
      if sigma_ is not None:
        embedding_seq[self.latent_prefix + C.SIGMA] = tf.tile(sigma_,
                                                              (1, output_len, 1))
      embedding_seq = self.net_embedding.draw_sample(embedding_seq)
    
    if self.embedding_only:
      emb_dec_input = embedding_seq
    else:
      if self.decoder_inp_dropout is not None:
        decoder_inputs = self.decoder_inp_dropout(decoder_inputs,
                                                  training=training)
      emb_dec_input = tf.concat([embedding_seq, decoder_inputs], axis=-1)
    
    # Running decoder.
    decoder_state = list()
    decoder_hidden = [self.decoder_inp_dense(emb_dec_input)]
    for idx, rnn_layer in enumerate(self.decoder_rnn):
      decoder_hidden = rnn_layer(
          decoder_hidden[0],
          initial_state=decoder_rnn_state[idx],
          training=training)
      decoder_state.append(decoder_hidden[1:])
    
    stroke_logits = self.decoder_out_stroke(decoder_hidden[0])
    
    if self.decoder_out_pen is not None:
      pen_logits = self.decoder_out_pen(decoder_hidden[0])
      # Calculate pen-up probability from the logits.
      pen_prob = tf.nn.sigmoid(pen_logits)
      pen_binary = tf.compat.v1.where(
          tf.greater(pen_prob, tf.fill(tf.shape(input=pen_prob), self.pen_threshold)),
          tf.fill(tf.shape(input=pen_prob), 1.0), tf.fill(tf.shape(input=pen_prob), 0.0))
    else:
      pen_logits = tf.ones_like(stroke_logits["mu"][:, :, 0:1])
      pen_prob = tf.random.uniform(tf.shape(pen_logits))
      pen_binary = tf.cast(tf.greater(pen_prob, 0.5), dtype=tf.float32)
    
    stroke_sample = self.decoder_out_stroke.draw_sample(stroke_logits, greedy=True)
    
    return dict(
        stroke=stroke_sample,
        stroke_logits=stroke_logits,
        pen_logits=pen_logits,
        pen_prob=pen_prob,
        pen=pen_binary,
        decoder_state=decoder_state)
  
  def decode_sequence(self,
                      embedding,
                      seq_len,
                      decoder_input=None):
  
    max_steps = tf.reduce_max(input_tensor=seq_len)
    
    if self.embedding_only or decoder_input is not None:
      decoded_seq = self.call_decode(embedding,
                                     decoder_inputs=decoder_input,
                                     output_len=max_steps,
                                     training=False)
      decoded_seq["seq_len"] = self.estimate_seq_len(decoded_seq, seq_len)
      return decoded_seq
    
    if isinstance(embedding, dict):
      embedding = self.net_embedding.draw_sample(embedding)
    embedding = tf.expand_dims(embedding, axis=1)
      
    if decoder_input is None:
      decoder_input_t = tf.zeros((embedding.shape[0], 1, 3))
    else:
      decoder_input_t = decoder_input[:, 0:1]
    
    state_t = RNNUtils.get_initial_states_layer(self.decoder_rnn,
                                                embedding)
    if self.dynamic_h0:
      decoder_state = self.decoder_state_nn(embedding[:, 0])
      state_t[0] = tf.split(decoder_state, 2, axis=-1)
    
    stop_signal = False
    step = 1
    stroke_logits = None
    pen, pen_prob, pen_logits, stroke_samples = list(), list(), list(), list()
    embedding_t = embedding
    while not stop_signal:
      if step == max_steps:
        stop_signal = True

      if isinstance(embedding, dict) and not self.repeat_vae_sample:
        embedding_t = self.net_embedding.draw_sample(embedding)
      
      if self.embedding_only:
        emb_dec_input_t = embedding_t
      else:
        if self.decoder_inp_dropout is not None:
          decoder_input_t = self.decoder_inp_dropout(decoder_input_t, training=False)
        emb_dec_input_t = tf.concat([embedding_t, decoder_input_t], axis=-1)
      
      decoder_hidden_t = self.decoder_inp_dense(emb_dec_input_t)
      for idx, rnn_layer in enumerate(self.decoder_rnn):
        decoder_rnn_t = rnn_layer(decoder_hidden_t, initial_state=state_t[idx])
        state_t[idx] = decoder_rnn_t[1:]
        decoder_hidden_t = decoder_rnn_t[0]
      
      stroke_t = self.decoder_out_stroke(decoder_hidden_t)
      pen_logits_t = self.decoder_out_pen(decoder_hidden_t)
      
      # Calculate pen-up probability from the logits.
      pen_t_prob = tf.nn.sigmoid(pen_logits_t)
      pen_t_binary = tf.compat.v1.where(
          tf.greater(pen_t_prob,
                     tf.fill(tf.shape(input=pen_t_prob), self.pen_threshold)),
          tf.fill(tf.shape(input=pen_t_prob), 1.0),
          tf.fill(tf.shape(input=pen_t_prob), 0.0))
      
      # Deterministic or probabilistic.
      stroke_t_sample = self.decoder_out_stroke.draw_sample_np(
          stroke_t, greedy=True)
      
      if decoder_input is None:
        decoder_input_t = tf.concat((stroke_t_sample, pen_t_binary), axis=-1)
      else:
        decoder_input_t = decoder_input[:, step:step + 1]
      
      stroke_logits = utils.dict_append(stroke_logits, stroke_t)
      stroke_samples.append(stroke_t_sample)
      pen_logits.append(pen_logits_t)
      pen.append(pen_t_binary)
      pen_prob.append(pen_t_prob)
      step += 1
    
    out_dict = dict()
    out_dict["stroke_logits"] = utils.tf_dict_concatenate(stroke_logits, axis=1)
    out_dict["stroke"] = tf.concat(stroke_samples, axis=1)
    out_dict["pen_logits"] = tf.concat(pen_logits, axis=1)
    out_dict["pen_prob"] = tf.concat(pen_prob, axis=1)
    out_dict["pen"] = tf.concat(pen, axis=1)
    # out_dict["seq_len"] = seq_len
    out_dict["seq_len"] = self.estimate_seq_len(out_dict, seq_len)
    
    return out_dict

  def estimate_seq_len(self, sample_dict, filler=None):
    # Detect when the pen-up event occurs.
    seq_len = np.argmax(sample_dict["pen"][:, :, 0].numpy() == 1, axis=1)
    # If pen-up doesn't occur, set a proxy seq_len if passed.
    if filler is not None:
      seq_len = np.where(seq_len == 0, filler, seq_len)
    return seq_len

  def latent_walk(self, latent_start, latent_end, steps, output_len):
    interp_data = np.vstack([
        self.get_numpy_value(latent_start[0]),
        self.get_numpy_value(latent_end[0])
        ])
    interp = interp1d([0, steps - 1], interp_data, axis=0)
    
    embeddings = interp(range(steps))
    embeddings = {C.MU: tf.expand_dims(tf.cast(embeddings, tf.float32), axis=1)}
    out_dict = self.decode_sequence(embeddings, output_len)
    out_dict["embeddings"] = embeddings
    out_dict["seq_len"] = output_len
    return out_dict
  
  def get_config(self):
    base_config = super(InkSeq2Seq, self).get_config()
    return base_config

  @classmethod
  def get_model_tags(cls, config, config_loss):
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
  
    latent = "L{}".format(config.embedding.latent_units)
    if config.embedding.use_vae:
      latent += "_vae"
      if isinstance(config_loss.embedding_kld.weight, float):
        latent += "_w" + str(config_loss.embedding_kld.weight)
      else:
        latent += "_aw" + str(config_loss.embedding_kld.weight["values"][1])
  
    if config.encoder.name == "rnn":
      encoder = "{}_{}x{}".format(config.encoder.cell_type,
                                  config.encoder.cell_layers,
                                  config.encoder.cell_units)
      if config.encoder.bidirectional_encoder:
        encoder = "bi" + encoder

      if config.encoder.rec_dropout_rate > 0:
        encoder += "_rdrop{}".format(config.encoder.rec_dropout_rate)
    else:
      err_unknown_type(config.encoder["name"])

    decoder = ""
    if config.decoder.repeat_vae_sample:
      decoder += "rep_"
    if config.decoder.dropout_rate > 0:
      decoder += "ddrop_" + str(config.decoder.dropout_rate)
    if config.decoder.dynamic_h0:
      decoder += "dh0_"

    model_name = "Seq2Seq"
    if config.decoder.autoregressive:
      model_name += "_ar"
  
    return dict(encoder=encoder, latent=latent, decoder=decoder, output=output,
                model_name=model_name)
