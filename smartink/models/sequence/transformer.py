"""Transformer model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from common.constants import Constants as C
from smartink.models.base_model import BaseModel
from smartink.models.common.building_blocks import DenseLayers
from smartink.models.common.output import OutputModelDeterministic
from smartink.models.common.output import OutputModelNormal2DDense
from smartink.models.common.output import OutputModelNormal
from smartink.models.common.output import OutputModelGMMDense


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
  """Embedding vectors based on the order in a sequence.

  Args:
    position: maximum length.
    d_model: stroke dimension.

  Returns:
  """
  angle_rads = get_angles(
      np.arange(position)[:, np.newaxis],
      np.arange(d_model)[np.newaxis, :], d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq, seq_len=None):
  if seq_len is not None:
    seq = 1 - tf.cast(tf.sequence_mask(seq_len), tf.float32)
  else:
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
  # TODO tf.linalg.band_part is not supported by TensorflowJS.
  # mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

  row_idx = tf.tile(tf.range(size)[:, tf.newaxis], [1, size])
  col_idx = tf.tile(tf.range(size)[tf.newaxis, :], [size, 1])
  mask = 1 - tf.compat.v1.where(row_idx >= col_idx, tf.ones((size, size)), tf.zeros((size, size)))
  return mask  # (seq_len, seq_len)


def create_masks(inp, tar, seq_len=None):
  """Creates padding and look-ahead masks given the inputs.

  Args:
    inp: encoder inputs (batch_size, seq_len)
    tar: decoder inputs (batch_size, seq_len)
    seq_len: list of sequence lengths for each sample in the batch (batch_size).

  Returns:
  """
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp, seq_len)

  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp, seq_len)

  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(input=tar)[1])
  dec_target_padding_mask = create_padding_mask(tar, seq_len)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  return enc_padding_mask, combined_mask, dec_padding_mask


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(dff, activation="relu"),
      # (batch_size, seq_len, d_model)
      tf.keras.layers.Dense(d_model)
  ])


def scaled_dot_product_attention(q, k, v, mask, rel_key_emb=None):
  """Calculate the attention weights.

  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k).
      Defaults to None.
    rel_key_emb:
    
  Returns:
    output, attention_weights
  """
    
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  if rel_key_emb is not None:
    q_expanded = tf.expand_dims(q, axis=2)
    q_rel = tf.matmul(q_expanded, rel_key_emb, transpose_b=True)
    q_rel_t = tf.transpose(q_rel, [2, 3, 0, 1, 4])
    
    q_len = tf.shape(q)[-2]
    idx = tf.transpose(tf.stack([tf.range(q_len), tf.range(q_len)]), [1,0])
    
    # selected_q_rel_t is of shape (batch_size, n_heads, q_len, q_len, k_len)
    # We need to select the slices on the diagonal of the 3rd and 4th dimensions.
    # which corresponds to eye(q_len) selected_q_rel_t[:, :, eye(q_len), :]
    # and result in selected_q_rel of shape (batch_size, n_heads, q_len, k_len).
    selected_q_rel_t = tf.gather_nd(q_rel_t, idx)
    selected_q_rel = tf.transpose(selected_q_rel_t, [1, 2, 0, 3])
    matmul_qk += selected_q_rel
    
  # scale matmul_qk
  dk = tf.cast(tf.shape(input=k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  """Multi-head attention layer."""

  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).

    Transpose the result such that the shape is
    (batch_size, num_heads, seq_len, depth).

    Args:
      x:
      batch_size:

    Returns:
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(a=x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask, rel_key_emb=None):
    batch_size = tf.shape(input=q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    # (batch_size, num_heads, seq_len_q, depth)
    q = self.split_heads(q, batch_size)
    # (batch_size, num_heads, seq_len_k, depth)
    k = self.split_heads(k, batch_size)
    # (batch_size, num_heads, seq_len_v, depth)
    v = self.split_heads(v, batch_size)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask, rel_key_emb)
    # (batch_size, seq_len_q, num_heads, depth)
    scaled_attention = tf.transpose(a=scaled_attention, perm=[0, 2, 1, 3])

    # (batch_size, seq_len_q, d_model)
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))
    # (batch_size, seq_len_q, d_model)
    output = self.dense(concat_attention)

    return output, attention_weights
  
  
class MultiHeadAttentionRelative(MultiHeadAttention):
  """Multi-head attention layer with relative positional encodings."""

  def __init__(self, d_model, num_heads, n_spatial_encodings):
    super(MultiHeadAttentionRelative, self).__init__(d_model, num_heads)
    
    self.n_spatial_encodings = n_spatial_encodings
    if self.n_spatial_encodings > 0:
      self.head_d_model = d_model//num_heads
      self.key_embedding_table = tf.Variable(initial_value=np.random.uniform(-0.01, 0.01, size=[self.n_spatial_encodings, self.d_model]), name="key_embeddings", trainable=True, dtype=tf.float32)
      # self.value_embedding_table = tf.Variable(initial_value=np.random.uniform(-0.5, 0.5, size=[self.n_spatial_encodings, self.d_model]), name="value_embeddings", trainable=True, dtype=tf.float32)

  def call(self, v, k, q, mask, rel_key_emb=None):
    # rel_key_emb and rel_val_emb are indices,
    
    batch_size = tf.shape(input=q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    # (batch_size, num_heads, seq_len_q, depth)
    q = self.split_heads(q, batch_size)
    # (batch_size, num_heads, seq_len_k, depth)
    k = self.split_heads(k, batch_size)
    # (batch_size, num_heads, seq_len_v, depth)
    v = self.split_heads(v, batch_size)

    if self.n_spatial_encodings > 0:
      # Get the embeddings and reshape into (batch_size, n_heads, seq_len, head_d_model)
      if rel_key_emb is not None:
        len1 = tf.shape(rel_key_emb)[1]
        len2 = tf.shape(rel_key_emb)[2]
        rel_key_emb = tf.gather(self.key_embedding_table, rel_key_emb)
        rel_key_emb = tf.transpose(tf.reshape(rel_key_emb, [batch_size, len1, len2, -1, self.head_d_model]), [0, 3, 1, 2, 4])

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask, rel_key_emb)
    
    # (batch_size, seq_len_q, num_heads, depth)
    scaled_attention = tf.transpose(a=scaled_attention, perm=[0, 2, 1, 3])

    # (batch_size, seq_len_q, d_model)
    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
    # (batch_size, seq_len_q, d_model)
    output = self.dense(concat_attention)

    return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
  """Transformer encoder layer.

  Applies multi-head attention operation on encoder inputs and updates the
  input embeddings.
  """

  def __init__(self, d_model, num_heads, dff, rate=0.1, n_spatial_encodings=0):
    super(EncoderLayer, self).__init__()
    
    if n_spatial_encodings > 0:
      self.mha = MultiHeadAttentionRelative(d_model, num_heads, n_spatial_encodings)
    else:
      self.mha = MultiHeadAttention(d_model, num_heads)
      
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask, rel_key_emb=None):
    # (batch_size, input_seq_len, d_model)
    attn_output, _ = self.mha(x, x, x, mask, rel_key_emb)
    attn_output = self.dropout1(attn_output, training=training)
    # (batch_size, input_seq_len, d_model)
    out1 = self.layernorm1(x + attn_output)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    # (batch_size, input_seq_len, d_model)
    out2 = self.layernorm2(out1 + ffn_output)
    return out2


class DecoderLayer(tf.keras.layers.Layer):
  """Transformer decoder layer.

  Applies self-attention on the decoder inputs and updates the embeddings by
  using the encoder embeddings.
  """

  def __init__(self, d_model, num_heads, dff, rate=0.1, n_spatial_encodings=0):
    super(DecoderLayer, self).__init__()
    
    if n_spatial_encodings > 0:
      self.mha1 = MultiHeadAttentionRelative(d_model, num_heads, n_spatial_encodings)
      self.mha2 = MultiHeadAttentionRelative(d_model, num_heads, n_spatial_encodings)
    else:
      self.mha1 = MultiHeadAttention(d_model, num_heads)
      self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training, look_ahead_mask, padding_mask, rel_key_emb=None):
    # enc_output.shape == (batch_size, input_seq_len, d_model)
    # (batch_size, target_seq_len, d_model)
    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask, rel_key_emb)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    # (batch_size, target_seq_len, d_model)
    attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask, rel_key_emb)
    attn2 = self.dropout2(attn2, training=training)
    # (batch_size, target_seq_len, d_model)
    out2 = self.layernorm2(attn2 + out1)

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    # (batch_size, target_seq_len, d_model)
    out3 = self.layernorm3(ffn_output + out2)
    return out3, attn_weights_block1, attn_weights_block2


class TransformerEncoder(tf.keras.layers.Layer):
  """Encoder block of transformer model.

  Assuming that the inputs are already in the stroke space.
  """

  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, n_spatial_encodings=0):
    super(TransformerEncoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.enc_layers = [
        EncoderLayer(d_model, num_heads, dff, rate, n_spatial_encodings) for _ in range(num_layers)
    ]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask, pos_encoding=None, scale=False, rel_key_emb=None, **kwargs):
    if scale:
      x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    if pos_encoding is not None:
      seq_len = tf.shape(input=x)[1]
      x += pos_encoding[:, :seq_len]

    x = self.dropout(x, training=training)
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask, rel_key_emb)

    return x  # (batch_size, input_seq_len, d_model)


class TransformerDecoder(tf.keras.layers.Layer):
  """Decoder block of transformer model.

  Assuming that the inputs are already in the stroke space.
  """

  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, n_spatial_encodings=0):
    super(TransformerDecoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.dec_layers = [
        DecoderLayer(d_model, num_heads, dff, rate, n_spatial_encodings) for _ in range(num_layers)
    ]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self,
           x,
           enc_output,
           training,
           look_ahead_mask,
           padding_mask,
           pos_encoding=None,
           scale=False,
           rel_key_emb=None,
           **kwargs):
    attention_weights = {}

    if scale:
      x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    if pos_encoding is not None:
      seq_len = tf.shape(input=x)[1]
      x += pos_encoding[:, :seq_len]

    x = self.dropout(x, training=training)
    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask,
                                             rel_key_emb)
      attention_weights["decoder_layer{}_block1".format(i + 1)] = block1
      attention_weights["decoder_layer{}_block2".format(i + 1)] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights


class TransformerSeq2Seq(BaseModel):
  """Transformer model in seq2seq concept.

  The encoder and decoder models operate on the entire sequence to predict the
  next step. Look-ahead masks are used to ensure that no future information is
  leaked.
  The encoder creates a context by using the steps until the current step.
  Similarly, the decoder uses the inputs steps until the current step to
  generate a query.
  """

  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               rate=0.1,
               config_loss=None,
               output_model=C.OUT_DETERMINISTIC,
               run_mode=C.RUN_ESTIMATOR,
               scale=False,
               pos_encoding_len=0,
               **kwargs):
    super(TransformerSeq2Seq, self).__init__(
        config_loss=config_loss, run_mode=run_mode, **kwargs)

    self.output_model = output_model
    self.pos_encoding_len = pos_encoding_len
    self.scale = scale
    self.encoder = TransformerEncoder(num_layers, d_model, num_heads, dff, rate)
    self.decoder = TransformerDecoder(num_layers, d_model, num_heads, dff, rate)

    self.pos_encoding = None
    if pos_encoding_len > 0:
      self.pos_encoding = positional_encoding(pos_encoding_len, d_model)

    if self.output_model == C.OUT_NORMAL:
      self.output_layer = OutputModelNormal(d_model, 0, 0, logvar=True)
    else:
      self.output_layer = OutputModelDeterministic(d_model, 0, 0)

  def call(self, inputs, seq_len=None, training=None, **kwargs):
    inp = inputs
    tar = inputs

    _, look_ahead_mask, _ = create_masks(inp[:, :, 0], tar[:, :, 0], seq_len)
    enc_padding_mask = look_ahead_mask
    dec_padding_mask = look_ahead_mask

    # (batch_size, inp_seq_len, d_model)
    enc_output = self.encoder(
        inp,
        training,
        enc_padding_mask,
        pos_encoding=self.pos_encoding,
        scale=self.scale)

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, _ = self.decoder(
        tar,
        enc_output,
        training,
        look_ahead_mask,
        dec_padding_mask,
        pos_encoding=self.pos_encoding,
        scale=self.scale)
    # (batch_size, tar_seq_len, target_vocab_size)
    return self.output_layer(dec_output)
  

class TransformerSeq2seqConditional(BaseModel):
  """Transformer model in seq2seq fashion where every step is also conditioned
  on some input.

  The encoder and decoder models operate on the entire sequence to predict the
  next step. Look-ahead masks are used to ensure that no future information is
  leaked.
  The encoder creates a context by using the steps until the current step.
  Similarly, the decoder uses the inputs steps until the current step to
  generate a query.
  """

  def __init__(self,
               output_size,
               num_layers,
               d_model,
               num_heads,
               dff,
               rate=0.1,
               config_loss=None,
               run_mode=C.RUN_ESTIMATOR,
               scale=False,
               pos_encoding_len=0,
               autoregressive=False,
               pooling="last_step",
               use_encoder=False,
               inp_target_dist_cond=False,
               inp_conditions=True,
               n_spatial_encodings=0,
               encoder_input_layer=None,
               encoder=None,
               **kwargs):
    super(TransformerSeq2seqConditional, self).__init__(
        config_loss=config_loss, run_mode=run_mode, **kwargs)

    self.output_size = output_size
    self.pos_encoding_len = pos_encoding_len
    self.scale = scale
    self.autoregressive = autoregressive
    self.use_encoder = use_encoder
    
    self.dec_input_layer = DenseLayers([d_model])

    self.encoder = None
    if self.use_encoder:
      # self.enc_input_layer = DenseLayers([d_model])
      # self.encoder = TransformerEncoder(num_layers, d_model, num_heads, dff, rate, n_spatial_encodings)
      self.enc_input_layer = encoder_input_layer
      self.encoder = encoder
      # self.encoder = TransformerEncoder(num_layers, d_model, num_heads, dff, rate)
    # self.decoder = TransformerDecoder(num_layers, d_model, num_heads, dff, rate, n_spatial_encodings)
    self.decoder = TransformerDecoder(num_layers, d_model, num_heads, dff, rate)
    self.decoder_out = DenseLayers([512,256], output_activation=tf.keras.activations.relu)

    self.pos_encoding = None
    if pos_encoding_len > 0:
      self.pos_encoding = positional_encoding(pos_encoding_len, d_model)
    
    if pooling == "last_step":
      self.pooling_layer = self.pool_last_step
    elif pooling == "mean":
      self.pooling_layer = self.pool_mean
    else:
      self.pooling_layer = None
    
    # Whether to concatenate the stroke positions (i.e., input_cond and target_cond)
    # with the model inputs (i.e., stroke embeddings) or not.
    self.inp_conditions = inp_conditions
    
    # Whether to concatenate the Euclidean distance between the target stroke
    # position and the start positions of the existing strokes.
    self.inp_target_dist_cond = inp_target_dist_cond
    
    # Relative positional encodings based on pairwise spatial distance.
    self.n_spatial_encodings = n_spatial_encodings
    if self.n_spatial_encodings > 0:
      self.dist_boundaries = np.arange(0, 1, 1./self.n_spatial_encodings)[1:].tolist()
    
    # Deterministic or stochastic outputs.
    self.output_layer = None
    if self.output_size > 0:
      if config_loss is not None:
        if config_loss["loss_type"] == C.NLL_NORMAL:
          self.output_layer = OutputModelNormal(self.output_size, logvar=True)
        elif config_loss["loss_type"] == C.NLL_BINORMAL:
          self.output_layer = OutputModelNormal2DDense(
              sigma_activation=tf.keras.activations.exponential)
        elif config_loss["loss_type"] == C.NLL_GMM:
          self.output_layer = OutputModelGMMDense(
              out_units=self.output_size,
              num_components=config_loss["num_components"],
              sigma_activation=tf.keras.activations.exponential)
        elif config_loss["loss_type"] == C.KLD:
          self.output_layer = OutputModelNormal(self.output_size, logvar=True)
        else:
          self.output_layer = OutputModelDeterministic(self.output_size, 0, 0)
      else:
        self.output_layer = OutputModelDeterministic(self.output_size, 0, 0)
  
  def pool_last_step(self, inp_, seq_len):
    # Get the last non-padded step.
    n_strokes = tf.shape(input=inp_)[0]
    batch_idx = tf.range(n_strokes)
    gather_idx = tf.stack([
        batch_idx,
        seq_len - 1
        ], axis=-1)
    pooled = tf.gather_nd(inp_, gather_idx)
    return tf.expand_dims(pooled, axis=1)
  
  def pool_mean(self, inp_, seq_len):
    # Take the average by ignoring the padded steps.
    seq_mask = tf.cast(tf.expand_dims(tf.sequence_mask(seq_len), axis=2), tf.float32)
    sum_ = tf.reduce_sum(seq_mask*inp_, axis=1)
    mean_ = sum_ / tf.cast(tf.expand_dims(seq_len, axis=1), tf.float32)
    return tf.expand_dims(mean_, axis=1)
  
  def frequency_encoding(self, inputs, n_layers):
    out = list()
    for l in range(n_layers):
      pi_constant = (np.power(2, l)*np.pi).astype(np.float32)
      out.append(tf.sin(pi_constant*inputs))
      out.append(tf.cos(pi_constant*inputs))
    return tf.concat(out, axis=-1)

  def distance_matrix_batch(self, array1, array2, norm_ord='euclidean'):
    """
    arguments:
        array1: the array, size: (batch_size, num_point, num_feature)
        array2: the samples, size: (batch_size, num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
            , it's size: (batch_size, num_point, num_point)
    """
    batch_size = tf.shape(input=array1)[0]
    num_point = tf.shape(input=array1)[1]
    num_features = tf.shape(input=array1)[2]
    expanded_array1 = tf.tile(array1, (1, num_point, 1))
    expanded_array2 = tf.reshape(
        tf.tile(tf.expand_dims(array2, 2),
                (1, 1, num_point, 1)),
        (batch_size, -1, num_features))
    if norm_ord == "diff":  # num_features must be 1.
      distances = expanded_array1 - expanded_array2
    else:
      distances = tf.norm(tensor=expanded_array1 - expanded_array2, axis=-1, ord=norm_ord)
    distances = tf.reshape(distances, (batch_size, num_point, num_point))
    return distances
  
  def normalized_pairwise_distances_l2(self, samples, seq_len):
    """
    Args:
      samples: (batch_size, seq_len, feature_dim)
      seq_len: in case samples are padded.
    Returns:
    """
    pairwise_dist = self.distance_matrix_batch(samples, samples)
    mask_ = tf.expand_dims((1-tf.cast(tf.sequence_mask(seq_len), tf.float32))*1e6, axis=-1)
    min_xy = tf.reduce_min(samples+mask_, axis=1)
    max_xy = tf.reduce_max(samples-mask_, axis=1)
    range_l2 = tf.norm(max_xy - min_xy, axis=1)
    return pairwise_dist/range_l2[:, tf.newaxis, tf.newaxis]
  
  def normalized_pairwise_distances_block(self, samples, seq_len):
    """
    Args:
      samples: (batch_size, seq_len, feature_dim)
      seq_len: in case samples are padded.
    Returns:
    """
    pairwise_dist_x = self.distance_matrix_batch(samples[:, :, 0:1], samples[:, :, 0:1], norm_ord="diff")
    pairwise_dist_y = self.distance_matrix_batch(samples[:, :, 1:2], samples[:, :, 1:2], norm_ord="diff")
    pairwise_dist = tf.concat([tf.expand_dims(pairwise_dist_x, axis=-1), tf.expand_dims(pairwise_dist_y, axis=-1)], axis=-1)
    
    mask_ = tf.expand_dims((1-tf.cast(tf.sequence_mask(seq_len), tf.float32))*1e6, axis=-1)
    min_xy = tf.reduce_min(samples+mask_, axis=1)
    max_xy = tf.reduce_max(samples-mask_, axis=1)
    range_block = max_xy - min_xy
    return pairwise_dist/(range_block[:, tf.newaxis, tf.newaxis, :])
    
  
  def call(self, inputs, training=False, **kwargs):
    seq_len = inputs["seq_len"]
    target_cond = inputs["target_cond"]
    input_cond = inputs["input_cond"]
    input_seq = inputs["input_seq"]
    enc_padding_mask = create_padding_mask(input_seq[:, :, 0], seq_len)
    
    using_end_pos = False
    if input_cond.shape[-1] == 4:
      print("Using End Position.")
      using_end_pos = True
      start_pos = input_cond[:, :, 0:2]
      end_pos = input_cond[:, :, 2:4]
      input_cond = start_pos
      
    if self.autoregressive:
      look_ahead_mask = create_look_ahead_mask(tf.shape(input=input_seq[:, :, 0])[1])
      encoder_mask = tf.maximum(enc_padding_mask, look_ahead_mask)
    else:
      encoder_mask = enc_padding_mask
    
    # Are we gonna concatenate the inputs with a given condition? In our case,
    # the inputs are the existing embeddings and the conditions can be the start
    # position of the given strokes and/or the start position of the target stroke.
    concat_cond = None
    if self.inp_conditions:
      concat_cond = input_cond
      
      if target_cond is not None:
        n_input = tf.shape(input=input_cond)[1]
        target_cond_inp = tf.tile(target_cond, [1, n_input, 1])
        if self.inp_target_dist_cond:
          target_dist_cond = tf.norm(input_cond - target_cond, axis=-1, keepdims=True)
          concat_cond = tf.concat([input_cond, target_cond_inp, target_dist_cond], axis=-1)
        else:
          concat_cond = tf.concat([input_cond, target_cond_inp], axis=-1)

    rel_dist_idx=None
    if self.n_spatial_encodings > 0:
      # If we use relative positional encodings, we don't want any conditions
      # appended to the inputs (for now).
      concat_cond = None
      # Calculate the Euclidean distances between the strokes, normalize and
      # fetch the corresponding (trainable) encodings. They are like the tokens
      # in NLP. Here we "tokenize" the L2-distance.
      normalized_dist = self.normalized_pairwise_distances_l2(input_cond, seq_len)
      rel_dist_idx = tf.raw_ops.Bucketize(input=normalized_dist, boundaries=self.dist_boundaries)
      
      if self.inp_conditions is False and self.inp_target_dist_cond and target_cond is not None:
        target_dist_cond = tf.norm(input_cond - target_cond, axis=-1, keepdims=True)
        concat_cond = target_dist_cond

    if using_end_pos:
      inp_pos_dist_cond = tf.norm(start_pos - end_pos, axis=-1, keepdims=True)
      if target_cond is not None:
        targ_pos_dist_cond = tf.norm(target_cond - end_pos, axis=-1, keepdims=True)
        concat_cond = tf.concat([concat_cond, end_pos, inp_pos_dist_cond, targ_pos_dist_cond], axis=-1)
      else:
        concat_cond = tf.concat([concat_cond, end_pos, inp_pos_dist_cond], axis=-1)
    
    dec_input_seq = input_seq
    if concat_cond is not None:
      dec_input_seq = tf.concat([input_seq, concat_cond], axis=-1)
    dec_input_seq = self.dec_input_layer(dec_input_seq)
    
    if self.use_encoder:
      enc_input_seq = input_seq
      # if concat_cond is not None:
      #   enc_input_seq = tf.concat([input_seq, concat_cond], axis=-1)
      if self.inp_conditions:
        enc_input_seq = tf.concat([input_seq, input_cond], axis=-1)
      
      enc_input_seq = self.enc_input_layer(enc_input_seq)
      tr_enc_out = self.encoder(
          enc_input_seq,
          training,
          encoder_mask,
          pos_encoding=self.pos_encoding,
          scale=self.scale,
          rel_key_emb=rel_dist_idx)
    else:
      tr_enc_out = dec_input_seq

    tr_dec_out, attn_weights = self.decoder(
        dec_input_seq,
        tr_enc_out,
        training,
        encoder_mask,
        encoder_mask,
        pos_encoding=self.pos_encoding,
        scale=self.scale,
        rel_key_emb=None)
    
    if self.pooling_layer is not None:
      tr_dec_out = self.pooling_layer(tr_dec_out, seq_len)
      
    representation = self.decoder_out(tr_dec_out[:, 0])

    # (batch_size, tar_seq_len, target_vocab_size)
    model_out = self.output_layer(representation)
    model_out["attention_weights"] = attn_weights
    return model_out


class TransformerAR(BaseModel):
  """Transformer model in autoregressive concept.

  The model consists of the decoder only. It operates on the entire sequence
  to predict the next step. Look-ahead masks are used to ensure that no
  future information is leaked.
  The encoder creates a context by using the steps until the current step.
  Similarly, the decoder uses the input steps until the current step to
  generate a query.
  """
  
  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               rate=0.1,
               output_size=0,
               config_loss=None,
               run_mode=C.RUN_ESTIMATOR,
               return_sequence=True,
               scale=False,
               pos_encoding_len=200,
               **kwargs):
    super(TransformerAR, self).__init__(
        config_loss=config_loss, run_mode=run_mode, **kwargs)
    self.pos_encoding_len = pos_encoding_len
    self.scale = scale
    self.return_sequence = return_sequence
    self.d_model = d_model
    self.output_size = output_size
    self.decoder = TransformerDecoder(num_layers, d_model, num_heads, dff, rate)
    
    self.pos_encoding = None
    if pos_encoding_len > 0:
      self.pos_encoding = positional_encoding(pos_encoding_len, d_model)
    
    self.input_layer = OutputModelDeterministic(d_model, 0, 0)
    
    # Deterministic or stochastic outputs.
    self.output_layer = None
    if self.output_size > 0:
      if config_loss is not None:
        if config_loss["loss_type"] == C.NLL_NORMAL:
          self.output_layer = OutputModelNormal(self.output_size, logvar=True)
        elif config_loss["loss_type"] == C.NLL_BINORMAL:
          self.output_layer = OutputModelNormal2DDense(
              sigma_activation=tf.keras.activations.exponential)
        elif config_loss["loss_type"] == C.NLL_GMM:
          self.output_layer = OutputModelGMMDense(
              out_units=self.output_size,
              num_components=config_loss["num_components"],
              sigma_activation=tf.keras.activations.exponential)
        elif config_loss["loss_type"] == C.KLD:
          self.output_layer = OutputModelNormal(self.output_size, logvar=True)
        else:
          self.output_layer = OutputModelDeterministic(self.output_size, 0, 0)
      else:
        self.output_layer = OutputModelDeterministic(self.output_size, 0, 0)
  
  def call(self, inputs, seq_len=None, training=None, **kwargs):
    inp = inputs
    tar = inputs
    
    tar = self.input_layer(tar)["mu"]
    
    _, look_ahead_mask, _ = create_masks(inp[:, :, 0], tar[:, :, 0], seq_len)
    dec_padding_mask = look_ahead_mask

    # dec_padding_mask = create_padding_mask(inputs[:, :, 0], seq_len)
    # look_ahead_mask = dec_padding_mask
    
    dec_output, _ = self.decoder(
        tar,
        tar,
        training,
        look_ahead_mask,
        dec_padding_mask,
        pos_encoding=self.pos_encoding,
        scale=self.scale)
    
    if not self.return_sequence:
      # Get the last non-padded step.
      n_strokes = tf.shape(input=dec_output)[0]
      batch_idx = tf.range(n_strokes)
      gather_idx = tf.stack([
          batch_idx,
          seq_len - 1
          ], axis=-1)
      dec_output = tf.gather_nd(dec_output, gather_idx)
    
    if self.output_layer is None:
      return dec_output
    else:
      return self.output_layer(dec_output)
  
  @classmethod
  def get_model_tags(cls, config, config_loss=None):
    """Generates a string summarizing experiment parameters.

    Args:
      config:
      config_loss:

    Returns:
    """
    model = "{}x{}_{}-head_{}-drop_{}".format(
        config.layers,
        config.d_model,
        config.hidden_units,
        config.heads,
        config.dropout_rate)
    
    return dict(model=model, model_name="TR")


class TransformerEmbedding(BaseModel):
  """Decoder-only transformer model with an stroke model.

  The inputs are first projected into an stroke space. It operates on
  the entire sequence to predict the next step. Look-ahead masks are optionally
  used to ensure that no future information is leaked.
  The encoder creates a context by using the steps until the current step.
  Similarly, the decoder uses the input steps until the current step to
  generate a query.
  """

  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               rate=0.1,
               return_sequences=True,
               output_size=0,
               config_loss=None,
               run_mode=C.RUN_STATIC,
               scale=False,
               pos_encoding_len=200,
               autoregressive=False,
               **kwargs):
    super(TransformerEmbedding, self).__init__(
        config_loss=config_loss, run_mode=run_mode, **kwargs)
    self.return_sequences = return_sequences
    self.autoregressive = autoregressive
    self.d_model = d_model
    self.pos_encoding_len = pos_encoding_len
    self.scale = scale
    self.pen_threshold = 0.5
    self.output_size = output_size

    self.init_embedding = tf.keras.layers.Dense(units=d_model, name="stroke")
    self.encoder = TransformerEncoder(num_layers, d_model, num_heads, dff, rate)

    self.pos_encoding = None
    if pos_encoding_len > 0:
      self.pos_encoding = positional_encoding(pos_encoding_len, d_model)

    # Deterministic or stochastic outputs.
    self.output_layer = None
    if self.output_size > 0:
      if config_loss is not None:
        if config_loss["loss_type"] == C.NLL_NORMAL:
          self.output_layer = OutputModelNormal(self.output_size, logvar=True)
        elif config_loss["loss_type"] == C.NLL_BINORMAL:
          self.output_layer = OutputModelNormal2DDense(
              sigma_activation=tf.keras.activations.exponential)
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
    input_seq = inputs["input_seq"]
    seq_len = inputs["seq_len"]
    enc_padding_mask = create_padding_mask(input_seq[:, :, 0], seq_len)
    if self.autoregressive:
      look_ahead_mask = create_look_ahead_mask(tf.shape(input=input_seq[:, :, 0])[1])
      combined_mask = tf.maximum(enc_padding_mask, look_ahead_mask)
    else:
      combined_mask = enc_padding_mask

    # (batch_size, inp_seq_len, d_model)
    init_embeddings = self.init_embedding(input_seq)

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    output = self.encoder(
        init_embeddings,
        training,
        combined_mask,
        pos_encoding=self.pos_encoding,
        scale=self.scale)

    if not self.return_sequences:
      output = self.fetch_last_step(output, seq_len, self.d_model)
    
    if self.output_layer is None:
      return output
    else:
      return self.output_layer(output)

  @classmethod
  def get_experiment_name(cls, config):
    """Generates a string summarizing experiment parameters.

    Args:
      config:

    Returns:
    """
    template = "{tag}EMB_{model}-{experiment}-{data}"

    data = config.data.data_name

    model = "TR_{}_{}x{}-head_{}-drop_{}".format(config.model.latent_units,
                                                 config.model.layers,
                                                 config.model.hidden_units,
                                                 config.model.heads,
                                                 config.model.dropout_rate)

    experiment = "B{}_LR{}".format(config.data.batch_size,
                                   config.experiment.learning_rate)

    return template.format(
        tag=config.experiment.tag + "_" if config.experiment.tag else "",
        experiment=experiment,
        model=model,
        data=data,
    )


class TransformerPredictive(BaseModel):
  """Decoder-only transformer model with an stroke model on strokes..

  The inputs are first projected into an stroke space. It operates on
  the entire sequence to predict the next step. Look-ahead masks are used
  to ensure that no future information is leaked.
  The encoder creates a context by using the steps until the current step.
  Similarly, the decoder uses the input steps until the current step to
  generate a query.
  """

  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               rate=0.1,
               config_loss=None,
               output_model=C.OUT_DETERMINISTIC,
               run_mode=C.RUN_ESTIMATOR,
               scale=False,
               pos_encoding_len=None,
               **kwargs):

    super(TransformerPredictive, self).__init__(
        config_loss=config_loss, run_mode=run_mode, **kwargs)
    self.output_model = output_model
    self.pos_encoding_len = pos_encoding_len
    self.scale = scale
    self.seq_len = pos_encoding_len
    self.pen_threshold = 0.5
    self.decoder_embedding_units = 16

    self.embedding_size = d_model
    self.encoder_embedding = OutputModelDeterministic(d_model, 0, 0)
    self.decoder_embedding = OutputModelDeterministic(
        self.seq_len * self.decoder_embedding_units, 0, 0)
    self.decoder = TransformerDecoder(num_layers, d_model, num_heads, dff, rate)

    self.pos_encoding = None
    if pos_encoding_len > 0:
      self.pos_encoding = positional_encoding(pos_encoding_len, d_model)

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

  def call(self, inputs, seq_len=None, training=None, **kwargs):
    inp = inputs[C.INP_ENC]
    # tar = inputs[C.INP_ENC]
    # seq_len = inputs[C.INP_SEQ_LEN]
    input_num_strokes = inputs[C.INP_NUM_STROKE]

    flat_strokes = tf.reshape(inp, (tf.shape(input=inp)[0], self.seq_len * 3))
    embedding = self.encoder_embedding(flat_strokes)

    diagram_embedding = self.batch_stroke_to_diagram(embedding,
                                                     input_num_strokes)

    inp = diagram_embedding["mu"]
    _, look_ahead_mask, _ = create_masks(inp[:, :, 0], inp[:, :, 0],
                                         input_num_strokes)
    dec_padding_mask = look_ahead_mask

    inp = tf.concat([tf.zeros_like(inp[:, 0:1]), inp[:, 0:-1]], axis=1)
    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, _ = self.decoder(
        inp,
        inp,
        training,
        look_ahead_mask,
        dec_padding_mask,
        pos_encoding=self.pos_encoding,
        scale=self.scale)

    # dec_output = tf.reshape(dec_output, (-1, self.seq_len,))
    dec_output = self.decoder_embedding(dec_output)["mu"]
    dec_output = tf.reshape(dec_output,
                            (-1, self.seq_len, self.decoder_embedding_units))

    stroke_logits = self.decoder_out_stroke(dec_output)
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
        pen=pen_binary)

  @classmethod
  def get_experiment_name(cls, config):
    """Generates a string summarizing experiment parameters.

    Args:
      config:

    Returns:
    """
    template = "{tag}EMB_{model}-{experiment}-{data}"

    data = config.data.data_name

    model = "TR_{}_{}x{}-head_{}-drop_{}".format(config.model.latent_units,
                                                 config.model.layers,
                                                 config.model.hidden_units,
                                                 config.model.heads,
                                                 config.model.dropout_rate)

    experiment = "B{}_LR{}".format(config.data.batch_size,
                                   config.experiment.learning_rate)

    return template.format(
        tag=config.experiment.tag + "_" if config.experiment.tag else "",
        experiment=experiment,
        model=model,
        data=data,
    )

  def batch_stroke_to_diagram(self, stroke_embedding, num_strokes):
    """Reshapes embeddings from batch of strokes to batch of diagrams.

    Args:
      stroke_embedding: Tensor of [num_diagrams x num_strokes, embedding_size]
        or dictionary of tensors with the same shape.
      num_strokes: [num_diagrams]

    Returns:
      Diagram as sequence of stroke embeddings [num_diagrams, num_strokes,
      embedding_size].
    """
    padded_num_strokes = tf.reduce_max(input_tensor=num_strokes)
    num_diagrams = tf.shape(input=num_strokes)[0]

    if isinstance(stroke_embedding, tf.Tensor):
      return tf.reshape(stroke_embedding,
                        [num_diagrams, padded_num_strokes, self.embedding_size])
    else:
      out_dict = dict()
      for key_, value_ in stroke_embedding.items():
        out_dict[key_] = tf.reshape(
            value_, [num_diagrams, padded_num_strokes, self.embedding_size])
      return out_dict

  def batch_diagram_to_stroke(self, diagram_embedding):
    """Reshapes embeddings from batch of diagrams to batch of strokes.

    Args:
      diagram_embedding: Tensor or dictionary of tensors with shape
        [num_diagrams, num_strokes, embedding_size]

    Returns:
      Batch of embeddings [num_diagrams x num_strokes, 1, embedding_size].
    """
    if isinstance(diagram_embedding, tf.Tensor):
      return tf.reshape(diagram_embedding, [-1, 1, self.embedding_size])
    else:
      out_dict = dict()
      for key_, value_ in diagram_embedding.items():
        out_dict[key_] = tf.reshape(value_, [-1, 1, self.embedding_size])
      return out_dict