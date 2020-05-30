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
from smartink.models.sequence.rnn import RNN
from smartink.models.sequence.transformer import TransformerEmbedding as Transformer
from smartink.models.common.building_blocks import Activations
from smartink.models.common.output import OutputModelDeterministic
from smartink.models.common.output import OutputModelNormal
from smartink.models.common.output import OutputModelNormal2DDense
from smartink.models.common.output import OutputModelGMMDense


class TEmbedding(BaseModel):
  """A temporal stroke model explicitly factoring out the temporal dimension."""

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
    super(TEmbedding, self).__init__(
        config_loss=config_loss, run_mode=run_mode, **kwargs)

    self.pen_threshold = 0.3
    self.config_encoder = config_encoder
    self.config_embedding = config_embedding
    self.config_decoder = config_decoder
    self.latent_prefix = ""
    
    self.regularize_decoder = self.config_decoder.get("regularizer_weight", 0) > 0
    self.kernel_regularizer = None
    if self.regularize_decoder:
      self.kernel_regularizer=tf.keras.regularizers.l2(self.config_decoder.get("regularizer_weight", 0))

    self.n_latent_units = self.config_embedding["latent_units"]
    self.use_vae = self.config_embedding["use_vae"]
    self.decoder_drop_rate = self.config_decoder.get("dropout_rate", 0)
    self.t_frequency_channels = self.config_decoder.get("t_frequency_channels", 0)

    # Encoder network
    self.net_encoder = None
    if self.config_encoder["name"] == "rnn":
      self.net_encoder = RNN(
          self.config_encoder["cell_type"],
          self.config_encoder["cell_units"],
          self.config_encoder["cell_layers"],
          self.config_encoder["bidirectional_encoder"],
          return_sequences=False,
          return_state=False,
          run_mode=run_mode,
          use_cudnn=self.config_encoder["use_cudnn"],
          name="encoder_rnn")
    elif self.config_encoder["name"] == "mlp":
      pass
    elif self.config_encoder["name"] == "cnn":
      pass
    elif self.config_encoder["name"] == "transformer":
      self.net_encoder = Transformer(
          num_layers=self.config_encoder["layers"],
          d_model=self.config_encoder["d_model"],
          num_heads=self.config_encoder["heads"],
          dff=self.config_encoder["hidden_units"],
          rate=self.config_encoder["dropout_rate"],
          scale=self.config_encoder["scale"],
          pos_encoding_len=self.config_encoder["pos_encoding"],
          autoregressive=self.config_encoder["autoregressive"],
          return_sequences=False,
          config_loss=None,
          run_mode=run_mode)
    else:
      err_unknown_type(self.config_encoder["name"])

    # Deterministic or stochastic stroke.
    if self.use_vae:
      self.net_embedding = OutputModelNormal(
          out_units=self.n_latent_units,
          prefix=self.latent_prefix,
          sigma_activation=None,
          logvar=True)
    else:
      self.net_embedding = OutputModelDeterministic(
          out_units=self.n_latent_units, prefix=self.latent_prefix)

    # Decoder network:
    self.net_decoder = tf.keras.Sequential(name="decoder")

    layer_units = self.config_decoder["hidden_units"]
    if len(layer_units) == 1:
      layer_units = layer_units*self.config_decoder["n_layers"]

    decoder_activation = Activations.get(self.config_decoder["activation"])

    for idx in range(self.config_decoder["layers"]):
      self.net_decoder.add(
          tf.keras.layers.Dense(
              layer_units[idx], activation=decoder_activation,
              kernel_regularizer=self.kernel_regularizer,
              bias_regularizer=self.kernel_regularizer))
      if self.decoder_drop_rate > 0:
        self.net_decoder.add(tf.keras.layers.Dropout(self.decoder_drop_rate))

    # Pen and stroke outputs.
    if config_loss["pen"]["eval_only"]:
      self.decoder_out_pen = None
    else:
      self.decoder_out_pen = tf.keras.layers.Dense(1, name="out_pen", kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.kernel_regularizer)

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
      self.decoder_out_stroke = OutputModelDeterministic(
          out_units=2, hidden_units=0, hidden_layers=0,
          kernel_regularizer=self.kernel_regularizer,
          bias_regularizer=self.kernel_regularizer)

    # Variables for static mode. They are assigned in call method.
    # TODO(eaksan) We can get rid of them if autoregressive sampling is no
    #  longer required in static (graph) mode.
    self.op_encoder_inputs = None
    self.op_input_seq_len = None
    self.op_embedding = None
    self.op_decoder_inputs = None
    self.op_embedding_sample = None

  def call(self, inputs, training=None, **kwargs):
    """Encoder and decoder functionality.

    Given an input sequence, calculates the stroke first. Then predicts a
    single step corresponding the real-valued time step.

    It doesn't reconstruct the entire sequence. Instead it only predicts one
    step. See decode_sequence method to get the entire sequence.
    Args:
      inputs (dict): expected to contain inputs for the encoder and decoder, and
        seq len ops.
      training: whether in training mode or not.
      **kwargs:

    Returns:
      [batch_size, seq_len, feature_size]
    """
    self.op_encoder_inputs = inputs[C.INP_ENC]
    self.op_decoder_inputs = inputs[C.INP_T]
    # self.op_input_seq_len = inputs[C.INP_SEQ_LEN]

    if len(inputs[C.INP_SEQ_LEN].shape) == 2:
      self.op_input_seq_len = inputs[C.INP_SEQ_LEN][:, 0]
    else:
      self.op_input_seq_len = inputs[C.INP_SEQ_LEN]
    
    self.op_embedding = self.call_encode(self.op_encoder_inputs,
                                         self.op_input_seq_len, training)
    
    self.op_embedding_sample = self.net_embedding.draw_sample(self.op_embedding)

    out_dict = self.call_decode(self.op_embedding_sample,
                                self.op_decoder_inputs, training)
    out_dict["embedding"] = self.op_embedding
    out_dict["embedding_sample"] = self.op_embedding_sample
    return out_dict

  def call_encode(self, inputs, input_seq_len, training):
    """Calculates the stroke stroke.

    Args:
      inputs:
      input_seq_len:
      training:

    Returns:
      stroke of size [batch_size, 1, latent_size]
    """
    inp_dict = {"input_seq": inputs, "seq_len": input_seq_len}
    # encoder_out = self.net_encoder(inputs, seq_len=input_seq_len, training=training)
    encoder_out = self.net_encoder(inp_dict, training=training)
    embedding = self.net_embedding(encoder_out, training=training)
    return embedding

  def call_decode(self, embedding, decoder_inputs, training=None):
    """Reconstructs stroke sequence given an stroke.

    Args:
      embedding:
      decoder_inputs: t value between 0 and 1.
      training:

    Returns:
      A dictionary of stroke, pen logits, pen probability and binary pen.
    """
    if isinstance(embedding, dict):
      embedding = self.net_embedding.draw_sample(embedding)
    
    # We may use multiple t samples for the same stroke vector. Hence,
    # tile and reshape the stroke vector accordingly.
    # if training:
    n_t_targets = tf.shape(input=decoder_inputs)[1]
    decoder_inputs = tf.reshape(decoder_inputs, [-1, 1])
    tiled = tf.tile(embedding[:, tf.newaxis, :], [1, n_t_targets, 1])
    embedding = tf.reshape(tiled, [-1, self.n_latent_units])
    
    if self.t_frequency_channels > 0:
      # decoder_inputs = self.frequency_encoding(decoder_inputs, self.t_frequency_channels)
      decoder_inputs = self.frequency_encoding_emb(decoder_inputs, embedding, self.t_frequency_channels)

    decoder_inp = tf.concat([decoder_inputs, embedding], axis=-1)
    # Running decoder.
    decoder_hidden = self.net_decoder(decoder_inp, training=training)
    stroke_logits = self.decoder_out_stroke(decoder_hidden)
    if self.decoder_out_pen is not None:
      pen_logits = self.decoder_out_pen(decoder_hidden)
      # Calculate pen-up probability from the logits.
      pen_prob = tf.nn.sigmoid(pen_logits)
      pen_binary = tf.compat.v1.where(
          tf.greater(pen_prob, tf.fill(tf.shape(input=pen_prob), self.pen_threshold)),
          tf.fill(tf.shape(input=pen_prob), 1.0), tf.fill(tf.shape(input=pen_prob), 0.0))
    else:
      pen_logits = tf.ones_like(stroke_logits["mu"][:, 0:1])
      pen_prob = tf.random.uniform(tf.shape(pen_logits))
      pen_binary = tf.cast(tf.greater(pen_prob, 0.5), dtype=tf.float32)

    stroke_sample = self.decoder_out_stroke.draw_sample(
        stroke_logits, greedy=True)

    return dict(
        stroke=stroke_sample,
        stroke_logits=stroke_logits,
        pen_logits=pen_logits,
        pen_prob=pen_prob,
        pen=pen_binary)

  @tf.function(input_signature=[tf.TensorSpec(shape=[None, 8], dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int32)])
  def serving_decode_strokes(self, embedding_sample, target_seq_len):
    """Decodes stroke embeddings into a sequence by mapping t in [0,1] to
    target_seq_len. Decoded stroke length will be the same for all.

    Args:
      embedding_sample: embedding sample shape (batch_size, latent_units).
      target_seq_len (): # of sequence steps.

    Returns (dict):
      with keys stroke, pen and more.
    """
    n_strokes = tf.shape(input=embedding_sample)[0]

    t_inp = tf.tile(tf.expand_dims(tf.linspace(0.0, 1.0, target_seq_len), axis=0), (n_strokes, 1))  # (batch_size, target_seq_len)
    decoded = self.call_decode(embedding_sample, t_inp, training=False)
    decoded_seq_len = tf.ones(n_strokes, dtype=tf.int32)*target_seq_len

    return dict(stroke=tf.reshape(decoded["stroke"], (n_strokes, -1, 2)),
                pen=tf.reshape(decoded["pen"], (n_strokes, -1, 1)),
                seq_len=decoded_seq_len)

  @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.int32)])
  def serving_encode_strokes(self, input_stroke, input_seq_len):
    """Encodes a stroke sequence into a fixed length embedding.

    Args:
      input_stroke: stroke sequence of shape (batch_size, seq_len, 3).
      input_seq_len: (batch_size)

    Returns:
      embedding vector with shape (batch_size, latent_units).
    """
    embedding = self.call_encode(input_stroke, input_seq_len, training=False)
    embedding_sample = self.net_embedding.draw_sample(embedding)
    return dict(embedding_sample=embedding_sample)

  @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.int32)])
  def serving_forward_pass(self, input_stroke, input_seq_len, target_seq_len):
    """Encodes and decodes a stroke sequence. Also works with batches if the
    decoded stroke length is the same for all.

    Args:
      input_stroke: stroke sequence of shape (batch_size, input_seq_len, 3).
      input_seq_len: (batch_size)
      target_seq_len: ()
    Returns:
      embedding sample, decoded stroke.
    """
    embedding = self.call_encode(input_stroke, input_seq_len, training=False)
    embedding_sample = self.net_embedding.draw_sample(embedding)

    n_strokes = tf.shape(input=embedding_sample)[0]
    # embedding_inp = tf.tile(embedding_sample, [target_seq_len, 1])
    t_inp = tf.tile(tf.expand_dims(tf.linspace(0.0, 1.0, target_seq_len), axis=0), (n_strokes, 1))  # (batch_size, target_seq_len)
    decoded = self.call_decode(embedding_sample, t_inp, training=False)
    
    # decoded = self.batch_stroke_to_single_diagram(decoded_out, n_strokes)
    decoded_seq_len = tf.ones_like(input_seq_len)*target_seq_len
    
    return dict(stroke=tf.reshape(decoded["stroke"], (n_strokes, -1, 2)),
                pen=tf.reshape(decoded["pen"], (n_strokes, -1, 1)),
                seq_len=decoded_seq_len,
                embedding_sample=embedding_sample)
  
  def decode_sequence(self, embedding, seq_len):
    """Decodes an stroke into a sequence by mapping t in [0,1] to seq_len.

    Args:
      embedding: stroke sample or dict of shape (1, latent_units).
      seq_len (np.array): # of sequence steps.

    Returns:
    """
    if isinstance(embedding, dict):
      embedding = self.net_embedding.draw_sample(embedding)

    n_strokes = tf.shape(input=embedding)[0]
    n_latent = tf.shape(input=embedding)[1]
    max_len = tf.reduce_max(input_tensor=seq_len)

    embedding_inp = tf.reshape(
        tf.tile(tf.expand_dims(embedding, 1), [1, max_len, 1]), (-1, n_latent))
    # TODO(aksan) TF compatible? Only works in eager mode right now.
    t_vals = []
    for sid in range(len(seq_len)):
      t_ = tf.expand_dims(tf.linspace(0.0, 1.0, seq_len[sid]), axis=1)
      t_ = tf.pad(tensor=t_, paddings=[[0, max_len - seq_len[sid]], [0, 0]])
      t_vals.append(t_)
    t_inp = tf.concat(t_vals, axis=0)

    decoded_out = self.call_decode(embedding_inp, t_inp, training=False)
    # Convert stroke batch into a diagram sample with padded strokes.
    decoded_seq = self.batch_stroke_to_single_diagram(decoded_out, n_strokes)
    decoded_seq["seq_len"] = seq_len
    return decoded_seq

  def latent_walk(self, latent_start, latent_end, steps, output_len):
    interp_data = np.vstack([
        self.get_numpy_value(latent_start[0]),
        self.get_numpy_value(latent_end[0])
    ])
    interp = interp1d([0, steps - 1], interp_data, axis=0)
    embeddings = tf.cast(interp(range(steps)), tf.float32)

    if isinstance(output_len, list):
      interp = interp1d([0, steps - 1], np.array(output_len), axis=0)
      seq_len = np.round(interp(range(steps))).astype(np.int32)
    else:
      seq_len = np.array([output_len] * steps)

    out_dict = self.decode_sequence(embeddings, seq_len)
    out_dict["embeddings"] = embeddings
    out_dict["seq_len"] = seq_len
    return out_dict

  def loss(self, predictions, targets, seq_len=None, prefix="", training=True):
    if not prefix:
      prefix = self.config_loss.get("prefix", "")
    output_losses = self.loss_fn(
        self.config_loss,
        predictions=predictions,
        targets=targets,
        seq_len=seq_len,
        prefix=prefix,
        run_mode=self.run_mode,
        training=training)
  
    if self.regularize_decoder:
      dec_all = self.net_decoder.losses
      dec_all.extend(self.decoder_out_stroke.losses)
      if self.decoder_out_pen is not None:
        dec_all.extend(self.decoder_out_pen.losses)
      dec_reg = tf.math.add_n(dec_all)
      output_losses["loss"] += dec_reg
      output_losses["decoder_l2"] = dec_reg
  
    return output_losses
  
  @classmethod
  def frequency_encoding(cls, inputs, n_layers):
    out = list()
    for l in range(n_layers):
      pi_constant = (np.power(2, l)*np.pi).astype(np.float32)
      out.append(tf.sin(pi_constant*inputs))
      out.append(tf.cos(pi_constant*inputs))
    return tf.concat(out, axis=-1)

  @classmethod
  def frequency_encoding_emb(cls, t, embeddings, n_layers):
    out = list()
    for l in range(n_layers):
      pi_constant = (np.power(2, l)*np.pi).astype(np.float32)
      out.append(tf.sin(pi_constant*t)*embeddings)
      out.append(tf.cos(pi_constant*t)*embeddings)
    return tf.concat(out, axis=-1)

  @classmethod
  def batch_stroke_to_single_diagram(cls, stroke_batch, n_strokes):
    """Converts a batch of strokes into a diagram sample.

    Reshapes entries of shape (n_strokes x padded_seq_len, feature_dim)
    in the given dictionary to (n_strokes, padded_seq_len, feature_dim).
    Works with 1-level of nested structure to handle model output dictionaries.

    Args:
      stroke_batch:
      n_strokes:

    Returns:
    """
    diagram = dict()
    for key_, val_ in stroke_batch.items():
      if isinstance(val_, dict):
        val_dict = dict()
        for val_key, val_val in val_.items():
          feature_size = tf.shape(input=val_val)[-1]
          val_dict[val_key] = tf.reshape(val_val, (n_strokes, -1, feature_size))
        diagram[key_] = val_dict
      else:
        feature_size = tf.shape(input=val_)[-1]
        diagram[key_] = tf.reshape(val_, (n_strokes, -1, feature_size))

    return diagram

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
      
    decoder = "{}x{}".format(config.decoder.layers,
                             config.decoder.hidden_units[0])

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
    elif config.encoder.name == "transformer":
      encoder = "TR_{}_{}x{}-head_{}-drop_{}".format(
          config.encoder.d_model, config.encoder.layers,
          config.encoder.hidden_units, config.encoder.heads,
          config.encoder.dropout_rate)
      if not config.encoder.autoregressive:
        encoder = "bi" + encoder
    else:
      err_unknown_type(config.encoder["name"])

    return dict(encoder=encoder, latent=latent, decoder=decoder, output=output,
                model_name="TEMB")
