"""Utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf


class TFSummary(object):
  """Housekeeping for tensorboard summaries.

  Provides functionality to create and update tf.summary ops.
  """

  def __init__(self, session, writer, collection):
    self.session = session
    self.collection = collection
    self.writer = writer

    # Merging summaries, evaluated by the model.
    self._summary_op = None

  @property
  def summary_op(self):
    if self._summary_op is None:
      self._summary_op = tf.compat.v1.summary.merge_all(self.collection)
    return self._summary_op

  def create_summaries(self, tag, ops, summary_type="scalar"):
    for key_, op_ in ops.items():
      self.create_summary(tag + key_, op_, summary_type)

  def create_summary(self, name, op, summary_type):
    if summary_type == "scalar":
      tf.compat.v1.summary.scalar(name, op, collections=[self.collection])

  def add_summary(self, summary, step):
    self.writer.add_summary(summary, step)


class TFSummaryAvg(TFSummary):
  """Housekeeping for tensorboard summaries.

  Provides functionality to create and update tf.summary ops.
  Reports average of multiple runs. This is useful in writing
  the result of multiple training or evaluation batches.
  """

  def __init__(self, session, writer, collection):
    super(TFSummaryAvg, self).__init__(session, writer, collection)
    self.summary_pl = dict()

  def create_summaries(self, tag, ops, summary_type="scalar"):
    for key_, _ in ops.items():
      self.summary_pl[key_] = tf.compat.v1.placeholder(
          dtype=tf.float32, name=self.collection + "_" + key_ + "_summary_pl")
      super(TFSummaryAvg,
            self).create_summary(tag + key_, self.summary_pl[key_],
                                 summary_type)

  def create_summary(self, name, op, summary_type):
    self.summary_pl[name] = tf.compat.v1.placeholder(
        dtype=tf.float32, name=self.collection + "_" + name + "_summary_pl")
    super(TFSummaryAvg, self).create_summary(name, self.summary_pl[name],
                                             summary_type)

  def add_summary(self, summary_values, step):
    feed_dict = dict()
    for key_, pl_ in self.summary_pl.items():
      feed_dict[pl_] = summary_values[key_]

    summary = self.session.run(self.summary_op, feed_dict=feed_dict)
    self.writer.add_summary(summary, step)


class AggregateAvg(object):
  """Bookkeeping of <key, value> pairs.

  The registered keys are expected to be passed later.
  """

  def __init__(self, key_list=None):
    super(AggregateAvg, self).__init__()
    self.steps = 0
    self.container = dict()
    if key_list is not None:
      self.reset_keys(key_list)

  def add(self, values):
    assert isinstance(values, dict), "<key, value> pairs expected."

    if not self.container:
      self.reset_keys(values)

    for key, value in self.container.items():
      if isinstance(value, list) or isinstance(value, np.ndarray):
        self.container[key].extend(values[key])
      else:
        self.container[key] = value + values[key]
    self.steps += 1

  def summary(self):
    summary_dict = dict()
    for key, value in self.container.items():
      if isinstance(value, list):
        summary_dict[key] = np.array(value).mean()
      else:
        summary_dict[key] = value/self.steps
    return summary_dict

  def summary_and_reset(self):
    summary_dict = dict()
    steps = self.steps
    for key, value in self.container.items():
      if isinstance(value, list):
        summary_dict[key] = np.array(value).mean()
        self.container[key] = list()
      else:
        summary_dict[key] = value / self.steps
        self.container[key] = 0.0
    self.steps = 0
    return summary_dict, steps

  def reset_keys(self, val_dict):
    for key, value in val_dict.items():
      if isinstance(value, list) or isinstance(value, np.ndarray):
        self.container[key] = list()
      else:
        self.container[key] = 0.0

  def reset(self):
    for key, value in self.container.items():
      if isinstance(value, list):
        self.container[key] = list()
      else:
        self.container[key] = 0.0
    self.steps = 0


class LearningRateScheduler(object):
  """Implements pre-defined learning rate schedulers."""

  def __init__(self, lr_type, initial_lr):
    self.lr_type = lr_type
    self.initial_lr = initial_lr
    self.min_learning_rate = 0.0001

  def __call__(self, global_step):
    if self.lr_type == "fixed":
      return self.initial_lr

    elif self.lr_type == "exponential":
      decay_steps = 1000
      decay_rate = 0.96
      return tf.maximum(
          tf.compat.v1.train.exponential_decay(
              self.initial_lr,
              global_step,
              decay_steps,
              decay_rate,
              staircase=False), self.min_learning_rate)

    elif self.lr_type == "sketch_rnn":
      decay_rate = 0.9999
      return ((self.initial_lr - self.min_learning_rate) *
              (decay_rate)**tf.cast(global_step, tf.float32) +
              self.min_learning_rate)

    elif self.lr_type == "transformer":
      float_global_step = tf.cast(global_step, tf.float32)
      # d_model = 128.0
      d_model = 1000.0
      warmup_steps = 4000.0
      arg1 = tf.math.rsqrt(float_global_step)
      arg2 = float_global_step * (warmup_steps**-1.5)
      return tf.math.rsqrt(d_model) * tf.math.minimum(arg1, arg2)
    else:
      err_unknown_type(self.lr_type)


def err_not_implemented(type_str=None):
  """Raises an NotImplementedError.

  Args:
    type_str:

  Raises:
    NotImplementedError
  """
  type_str = " :" + type_str + "." or "."
  raise NotImplementedError("Not implemented" + type_str)


def err_unknown_type(type_str=None):
  """Raises TypeError.

  Args:
    type_str:

  Raises:
    TypeError
  """
  type_str = " :" + type_str + "." or "."
  raise TypeError("Unknown type" + type_str)


def tf_repeat0(tensor, n):
  """Repeats 1D tensor.
  
  Args:
    tensor: of shape (M)
    n: number of copies.

  Returns:
    tensor of shape (Mn)
  """
  return tf.reshape(tf.tile(tf.expand_dims(tensor, axis=1), [1, n]), [-1])

def dict_tf_to_numpy(tf_dict):
  """Converts tf.Tensor type values to numpy in a dictionary..

  Args:
    tf_dict (dict):

  Returns:
  """
  np_dict = dict()
  for key_, value_ in tf_dict.items():
    if isinstance(value_, tf.Tensor):
      np_dict[key_] = value_.numpy()
    else:
      np_dict[key_] = value_
  return np_dict

def dict_slice(inp_dict, inp_slice):
  """Extracts the given slice from the entries in a dictionary.

  Args:
    inp_dict (dict):
  Returns:
  """
  pass


def dict_append(container, target):
  """Appends values in the target dict to the container dict with lists.

  Args:
    container (dict):
    target (dict):

  Returns:
  """
  if container is None:
    container = dict()
    for key_ in target.keys():
      container[key_] = list()

  for key_, value_ in target.items():
    container[key_].append(value_)
  return container


def np_dict_concatenate(container, axis=-1):
  """Concatenates values stored in a list.

  Entries in the lists are Numpy tensors.
  Args:
    container (dict):
    axis (int): concatenation dimension.

  Returns:
  """
  for key_, value_ in container.items():
    container[key_] = np.concatenate(value_, axis)
  return container


def tf_dict_concatenate(container, axis=-1):
  """Concatenates values stored in a list.

  Entries in the lists are Tensorflow tensors.
  Args:
    container (dict):
    axis (int): concatenation dimension.

  Returns:
  """
  for key_, value_ in container.items():
    container[key_] = tf.concat(value_, axis)
  return container


class ModelNotFoundError(Exception):
  """Raised when the loaded model is not found."""
  pass


class NotPredictiveModelError(Exception):
  """Raised when the loaded model is not a predictive model."""
  pass


class NotEmbeddingModelError(Exception):
  """Raised when the loaded model is not an embedding model."""
  pass