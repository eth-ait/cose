"""CNN model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf

from common.constants import Constants as C  # pylint: disable=g-import-not-at-top
from smartink.models.stroke.t_emb import BaseModel  # pylint: disable=g-import-not-at-top
from smartink.models.common.output import OutputModelDeterministic  # pylint: disable=g-import-not-at-top
from smartink.models.common.output import OutputModelNormal  # pylint: disable=g-import-not-at-top
from smartink.models.common.output import OutputModelNormal2DDense  # pylint: disable=g-import-not-at-top
from smartink.models.common.output import OutputModelGMMDense  # pylint: disable=g-import-not-at-top


class CNNEmbedding(BaseModel):
  """Seq2seq by using CNN models on padded and fixed-length sequences."""

  def __init__(self,
               latent_units,
               filters,
               kernel_size=3,
               use_vae=True,
               config_loss=None,
               run_mode=C.RUN_ESTIMATOR,
               **kwargs):
    super(CNNEmbedding, self).__init__(
        config_loss=config_loss, run_mode=run_mode, **kwargs)

    self.pen_threshold = 0.1
    self.latent_units = latent_units
    self.filters = filters
    self.num_layers = len(self.filters)
    self.kernel_size = kernel_size
    self.use_vae = use_vae

    width, height, channels = 96, 3, self.filters[-1]

    self.encoder = tf.keras.Sequential()
    self.encoder.add(tf.keras.layers.InputLayer(input_shape=(width, height, 1)))
    for idx in range(self.num_layers):
      self.encoder.add(
          tf.keras.layers.Conv2D(
              filters=self.filters[idx],
              kernel_size=self.kernel_size,
              strides=(2, 1),
              padding="same",
              dilation_rate=(1, 1),
              activation=tf.keras.activations.relu,
              use_bias=True,
              kernel_initializer="glorot_uniform",
              bias_initializer="zeros"))

      # self.encoder.add(tf.keras.layers.MaxPool2D(pool_size=(2, 1),
      #                                            strides=None,
      #                                            padding='same'))
      width = width // 2

    self.encoder.add(tf.keras.layers.Flatten())
    self.encoder.add(tf.keras.layers.Dense(self.latent_units))

    print("Encoder CNN shape: ", str((width, height, channels)))

    # Deterministic or stochastic stroke.
    if self.use_vae:
      self.encoder_embedding = OutputModelNormal(
          out_units=self.latent_units,
          hidden_units=self.latent_units * 2,
          hidden_layers=0,
          sigma_activation=None,
          logvar=True)
    else:
      self.encoder_embedding = OutputModelDeterministic(
          out_units=self.num_latent_units,
          hidden_units=self.num_latent_units * 2,
          hidden_layers=0)

    self.decoder = tf.keras.Sequential()
    self.decoder.add(tf.keras.layers.Dense(self.latent_units))

    self.decoder.add(
        tf.keras.layers.Dense(
            units=width * height * channels, activation=tf.nn.relu))
    self.decoder.add(
        tf.keras.layers.Reshape(target_shape=(width, height, channels)))

    for idx in reversed(range(self.num_layers)):
      cnn_layer = tf.keras.layers.Conv2DTranspose(
          filters=self.filters[idx],
          kernel_size=self.kernel_size,
          strides=(2, 1),
          padding="same",
          activation=tf.keras.activations.relu,
          use_bias=True,
          kernel_initializer="glorot_uniform",
          bias_initializer="zeros")
      width *= 2
      self.decoder.add(cnn_layer)

    cnn_layer = tf.keras.layers.Conv2DTranspose(
        filters=self.latent_units,
        kernel_size=self.kernel_size,
        strides=(1, 1),
        padding="same",
        activation=tf.keras.activations.relu,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros")
    self.decoder.add(cnn_layer)

    channels = self.filters[0]
    self.decoder.add(
        tf.keras.layers.Reshape(
            target_shape=(width, height * self.latent_units)))

    print("Decoder CNN shape: ", str((width, height, channels)))

    # Outputs
    self.decoder_out_pen = tf.keras.layers.Dense(
        1, activation=None, name="out_pen")

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
          out_units=2, hidden_units=0, hidden_layers=0)

  def call(self, inputs, training=None, **kwargs):
    """Encoder and decoder functionality.

    Given an input sequence, calculates the stroke and reconstructs the
    sequence.

    Args:
      inputs (dict): expected to contain inputs for the encoder and decoder, and
        seq len ops.
      training: whether in training mode or not.
      **kwargs:

    Returns:
      [batch_size, seq_len, feature_size]
    """
    # encoder_inputs = tf.expand_dims(inputs[C.INP_ENC][:, :, 0:2], axis=-1)
    encoder_inputs = tf.expand_dims(inputs[C.INP_ENC], axis=-1)

    encoder_cnn = self.encoder(encoder_inputs, training=training)
    latent_space = self.encoder_embedding(encoder_cnn, training=training)

    latent_sample = self.encoder_embedding.draw_sample(
        latent_space, greedy=False)

    dec_output = self.decoder(latent_sample, training=training)

    stroke_logits = self.decoder_out_stroke(dec_output, training=training)
    pen_logits = self.decoder_out_pen(dec_output)

    # Calculate pen-up probability from the logits.
    pen_prob = tf.nn.sigmoid(pen_logits)
    pen_binary = tf.compat.v1.where(
        tf.greater(pen_prob, tf.fill(tf.shape(input=pen_prob), self.pen_threshold)),
        tf.fill(tf.shape(input=pen_prob), 1.0), tf.fill(tf.shape(input=pen_prob), 0.0))
    stroke_sample = self.decoder_out_stroke.draw_sample(
        stroke_logits, greedy=True)
    return dict(
        stroke=stroke_sample,
        stroke_logits=stroke_logits,
        pen_logits=pen_logits,
        pen_prob=pen_prob,
        pen=pen_binary,
        embedding=latent_space)

  def call_encode(self, inputs, input_seq_len, training):
    """Calculates the stroke stroke.

    Args:
      inputs:
      input_seq_len:
      training:
    Returns:
      stroke of size [batch_size, 1, latent_size]
    """
    # encoder_inputs = tf.expand_dims(inputs[:, :, 0:2], axis=-1)
    encoder_inputs = tf.expand_dims(inputs, axis=-1)

    encoder_cnn = self.encoder(encoder_inputs, training=training)
    embedding_seq = self.encoder_embedding(encoder_cnn, training=training)
    embedding = embedding_seq
    # stroke = dict()
    # stroke[self.latent_prefix + C.MU] = tf.expand_dims(
    #     embedding_seq[self.latent_prefix + C.MU], axis=1)
    # if self.use_vae:
    #   stroke[self.latent_prefix + C.SIGMA] = tf.expand_dims(
    #       embedding_seq[self.latent_prefix + C.SIGMA], axis=1)
    return embedding

  def call_decode(self, embedding, training=None):
    """Reconstructs stroke sequence given an stroke.

    Args:
      embedding:
      training:

    Returns:
      A dictionary of stroke, pen logits, pen probability and binary pen.
    """
    latent_sample = self.encoder_embedding.draw_sample(embedding)

    # Running decoder.
    decoder_out = self.decoder(latent_sample, training=training)

    stroke_logits = self.decoder_out_stroke(decoder_out)
    pen_logits = self.decoder_out_pen(decoder_out)

    # Calculate pen-up probability from the logits.
    pen_prob = tf.nn.sigmoid(pen_logits)
    pen_binary = tf.compat.v1.where(
        tf.greater(pen_prob, tf.fill(tf.shape(input=pen_prob), self.pen_threshold)),
        tf.fill(tf.shape(input=pen_prob), 1.0), tf.fill(tf.shape(input=pen_prob), 0.0))
    stroke_sample = self.decoder_out_stroke.draw_sample(
        stroke_logits, greedy=True)
    return dict(
        stroke=stroke_sample,
        stroke_logits=stroke_logits,
        pen_logits=pen_logits,
        pen_prob=pen_prob,
        pen=pen_binary)

  def autoregressive_decode_eager(self,
                                  embedding,
                                  output_len,
                                  decoder_input=None):
    return self.call_decode(embedding)

  def call_static(self,
                  session,
                  prediction_ops,
                  loss_ops,
                  target_ops,
                  autoregressive=True,
                  output_len=None):
    return session.run([prediction_ops, loss_ops, target_ops])

  def latent_walk(self, latent_start, latent_end, steps, output_len):
    interp_data = np.vstack([
        self.get_numpy_value(latent_start[0]),
        self.get_numpy_value(latent_end[0])
    ])
    interp = interp1d([0, steps - 1], interp_data, axis=0)

    embeddings = interp(range(steps))
    embeddings = {C.MU: tf.cast(embeddings, tf.float32)}
    out_dict = self.autoregressive_decode_eager(embeddings, output_len)
    return out_dict

  @classmethod
  def get_experiment_name(cls, config):
    """Generates a string summarizing experiment parameters.

    Args:
      config:

    Returns:
    """
    template = "{tag}EMB_{model}{latent}-{experiment}-{data}"

    data = config.data.data_name

    model = "CNN_{}_{}x{}@{}".format(
        config.model.latent_units,
        len(config.model.filters),
        config.model.filters[-1],
        config.model.kernel_size,
    )

    latent = ""
    if config.model.use_vae:
      latent += "-vae"
      if isinstance(config.loss.embedding_kld.weight, float):
        latent += "_w" + str(config.loss.embedding_kld.weight)
      else:
        latent += "_aw" + str(config.loss.embedding_kld.weight["values"][1])

    experiment = "B{}_LR{}".format(config.data.batch_size,
                                   config.experiment.learning_rate)

    return template.format(
        tag=config.experiment.tag + "_" if config.experiment.tag else "",
        experiment=experiment,
        model=model,
        latent=latent,
        data=data,
    )
