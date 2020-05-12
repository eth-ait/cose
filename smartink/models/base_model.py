"""A base model class.

Providing an interface between the models and training/evaluation pipeline.
The base class implements the basic functionality as well.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import smartink.loss.nll as loss
from common.constants import Constants as C
from smartink.util.utils import err_unknown_type
from smartink.loss.reduce import reduce_mean_sequence
from smartink.loss.reduce import reduce_mean_step


class BaseModel(tf.keras.Model):
  """Implements basic functionality for models."""

  def __init__(self, config_loss, run_mode=C.RUN_ESTIMATOR, **kwargs):
    super(BaseModel, self).__init__(**kwargs)

    self.config_loss = config_loss
    self.run_mode = run_mode

    if self.run_mode == C.RUN_EAGER and not tf.executing_eagerly():
      raise ValueError("It is eager, passed " + self.run_mode)

  def loss(self, predictions, targets, seq_len=None, prefix="", training=True):
    if not prefix:
      prefix = self.config_loss.get("prefix", "")
    return self.loss_fn(
        self.config_loss,
        predictions=predictions,
        targets=targets,
        seq_len=seq_len if seq_len is not None else targets[C.BATCH_SEQ_LEN],
        prefix=prefix,
        run_mode=self.run_mode,
        training=training)

  @classmethod
  def loss_fn(cls,
              loss_config,
              predictions,
              targets,
              seq_len,
              prefix="",
              run_mode=C.RUN_STATIC,
              training=True):

    fixed_len_seq = loss_config.get("fixed_len_seq", 0)

    if prefix and not prefix.endswith("_"):
      prefix = prefix + "_"

    seq_len = seq_len
    loss_dict = dict()
    loss_metric_dict = dict()
    total_loss_ops = list()

    for loss_key, loss_term in loss_config.items():
      if not isinstance(loss_term, dict):
        continue
      loss_sequence = None
      loss_type = loss_term["loss_type"]
      loss_reduce = loss_term["reduce"]
      loss_weight = loss_term["weight"]

      # if not isinstance(loss_weight, float):
      #   global_step = tf.compat.v1.train.get_or_create_global_step()
      #   start, end, increment = loss_term["weight"]["values"]
      #
      #   loss_weight = end - (end - start) * increment**tf.cast(
      #       global_step, tf.float32)
      #
      #   if run_mode != C.RUN_EAGER:
      #     tf.compat.v1.summary.scalar(
      #         "training/kl_weight", loss_weight, collections=["training"])

      if not training:
        loss_weight = 1.0

      loss_objective = not loss_term.get("eval_only", False)
      loss_targets = targets.get(loss_term["target_key"], None)
      loss_predictions = predictions.get(loss_term["out_key"], None)

      seq_mask = None
      if seq_len is not None:
        if fixed_len_seq > 0:
          seq_mask = tf.expand_dims(
              tf.sequence_mask(seq_len, maxlen=fixed_len_seq, dtype=tf.float32),
              -1)
        else:
          seq_mask = tf.expand_dims(
              tf.sequence_mask(seq_len, dtype=tf.float32), -1)
      # Calculate loss with shape (batch_size, seq_len, 1)
      if loss_type == C.MSE:
        if isinstance(loss_targets, dict):
          loss_targets = loss_targets[C.MU]
        loss_sequence = tf.reduce_sum(input_tensor=tf.math.square(loss_targets - loss_predictions[C.MU]), axis=-1, keepdims=True)
      elif loss_type == C.L1:
        if isinstance(loss_targets, dict):
          loss_targets = loss_targets[C.MU]
        loss_sequence = tf.reduce_sum(input_tensor=tf.math.abs(loss_targets - loss_predictions[C.MU]), axis=-1, keepdims=True)
      elif loss_type == C.NLL_NORMAL:
        loss_sequence = -1*loss.logli_normal_diagonal(
            loss_targets, loss_predictions[C.MU], loss_predictions[C.SIGMA])
      elif loss_type == C.NLL_BINORMAL:
        loss_sequence = -1*loss.logli_normal_bivariate(
            loss_targets, loss_predictions[C.MU], loss_predictions[C.SIGMA],
            loss_predictions[C.RHO])
      elif loss_type == C.NLL_GMM:
        loss_sequence = -1*loss.logli_gmm_logsumexp(
            loss_targets, loss_predictions[C.MU], loss_predictions[C.SIGMA],
            loss_predictions[C.PI])
      elif loss_type == C.NLL_CENT_BINARY:
        loss_sequence = tf.nn.sigmoid_cross_entropy_with_logits(
            loss_targets,
            loss_predictions)
      elif loss_type == C.KLD_STANDARD:
        loss_sequence = loss.kld_normal_diagonal_standard_prior(
            loss_predictions[C.MU], loss_predictions[C.SIGMA])
      elif loss_type == C.KLD:
        loss_sequence = loss.kld_normal_diagonal(
            loss_predictions[C.MU],
            loss_predictions[C.SIGMA],
            loss_targets[C.MU],
            loss_targets[C.SIGMA],
            reduce_sum=False)
      elif loss_type == C.SNORM_L2:
        loss_sequence = tf.reduce_sum(input_tensor=tf.math.square(loss_predictions[C.MU]), axis=-1)
      else:
        err_unknown_type(loss_type)

      if len(loss_sequence.shape) == 3:
        # Mask the padded steps and calculate a scalar value.
        loss_sequence *= seq_mask
        loss_op = None
        if loss_reduce == C.R_MEAN_STEP:
          loss_op = reduce_mean_step(loss_sequence, seq_mask)
          # loss_op = tf.reduce_mean(loss_sequence)
        elif loss_reduce == C.R_MEAN_SEQUENCE:
          loss_op = reduce_mean_sequence(loss_sequence)
        else:
          err_unknown_type(loss_reduce)
      
      elif len(loss_sequence.shape) == 2:
          if seq_len is not None:
            nonzero_seq_len = tf.cast(tf.expand_dims(tf.compat.v1.where(seq_len > 0, tf.ones_like(seq_len), tf.zeros_like(seq_len)), axis=1), tf.float32)
            loss_op = loss_sequence * nonzero_seq_len
            loss_op = tf.reduce_sum(input_tensor=loss_op) / tf.reduce_sum(input_tensor=nonzero_seq_len)
          else:
            loss_op = tf.reduce_mean(input_tensor=loss_sequence)
      else:
        loss_op = tf.reduce_mean(input_tensor=loss_sequence)

      if loss_type == C.KLD_STANDARD:
        loss_op = tf.maximum(loss_op, 0.2)

      # The loss term is weighted only for the optimization. In order to enable
      # comparison in Tensorboard plots, we keep the loss unweighted.
      if len(loss_config) > 1:
        loss_dict[prefix + loss_key] = loss_op
      if loss_objective:
        total_loss_ops.append(loss_weight * loss_op)

      # tf.estimator requires the loss in tf.metrics.
      if run_mode == C.RUN_ESTIMATOR and len(loss_config) > 1:
        loss_metric_dict[prefix + loss_key] = tf.keras.metrics.Mean(loss_op)

    # Sum all loss terms to get the objective.
    loss_dict["loss"] = tf.math.add_n(
        total_loss_ops, name=prefix + "total_loss")
    loss_dict[prefix + "loss"] = loss_dict["loss"]

    if run_mode == C.RUN_ESTIMATOR:
      loss_metric_dict["loss"] = tf.keras.metrics.Mean(loss_dict["loss"])
      loss_metric_dict[prefix + "loss"] = loss_metric_dict["loss"]
      return loss_dict, loss_metric_dict
    else:
      return loss_dict

  def log_loss(self, loss_dict, prefix="", suffix=""):
    loss_format = prefix + "Total: {:.4f} \t"
    loss_entries = [self.get_numpy_value(loss_dict.get("loss", 0.0))]

    for loss_key in sorted(loss_dict.keys()):
      if loss_key != "loss":
        loss_format += "{}: {:.4f} \t"
        loss_entries.append(loss_key)
        loss_entries.append(self.get_numpy_value(loss_dict[loss_key]))
    loss_format += suffix
    print(loss_format.format(*loss_entries))

  @classmethod
  def get_numpy_value(cls, value):
    return value.numpy() if isinstance(value, tf.Tensor) else value

  @classmethod
  def fetch_last_step(cls, inputs, seq_len, feature_size):
    """Given a padded variable length sequence, fetches the last valid step.

    Args:
      inputs:
      seq_len:
      feature_size:

    Returns:
    """
    # TODO(eaksan) Looks like dark magic. Is there a clearer solution?
    seq_len_indices = tf.maximum(
        tf.tile(tf.expand_dims(seq_len - 1, -1), (1, feature_size)), 0)

    batch_indices = tf.range(tf.shape(input=inputs)[0])
    seq_indices = tf.range(feature_size)
    idx_grid = tf.meshgrid(batch_indices, seq_indices)

    gather_nd_idx = tf.stack([
        tf.transpose(a=idx_grid[0]),
        tf.transpose(a=idx_grid[1]),
        tf.cast(seq_len_indices, tf.int32)
    ],
                             axis=-1)

    last_step = tf.gather_nd(tf.transpose(a=inputs, perm=(0, 2, 1)), gather_nd_idx)
    return last_step
