"""Dataset classes for diagram data in tfrecords.

Provides data I/O functionality in tf.data API, preprocessing and normalization
routines.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import os
import numpy as np
import tensorflow as tf

from smartink.data.base_dataset import Dataset
from common.constants import Constants as C


class TFRecordStroke(Dataset):
  """Dataset class for Google Diagram dataset.

  It is stored in TFRecords. Strokes in a diagram sample are padded and stored
  in a (num_strokes, max_stroke_length, ...) matrix. This class splits the
  diagram sample into individual strokes and operates on the stroke-level.
  Batches are composed of arbitrary strokes.
  TFRecordBatchDiagram and TFRecordSingleDiagram classes preserves the
  diagram structure in batches.
  """

  def __init__(self,
               data_path,
               meta_data_path,
               batch_size,
               pp_to_origin=True,
               pp_relative_pos=True,
               shuffle=True,
               normalize=False,
               run_mode=C.RUN_ESTIMATOR,
               fixed_len=False,
               **kwargs):
    # Sequences shorter or longer than the thresholds are discarded.
    self.min_length_threshold = kwargs.get("min_length_threshold", 4)
    self.max_length_threshold = kwargs.get("max_length_threshold", 200)
    # Sequences with less strokes are discarded.
    self.num_strokes_threshold = kwargs.get("num_strokes_threshold", 4)
    self.num_parallel_calls = kwargs.get("num_parallel_calls", 4)
    self.mask_pen = kwargs.get("mask_pen", False)
    self.scale_factor = kwargs.get("scale_factor", 0)
    self.affine_prob = kwargs.get("affine_prob", 0)
    self.resampling_factor = kwargs.get("resampling_factor", 1)
    self.resample_target = kwargs.get("resample_target", False)
    self.n_t_targets = kwargs.get("n_t_targets", 1)
    self.int_t_samples = kwargs.get("int_t_samples", False)
    self.concat_t_inputs = kwargs.get("concat_t_inputs", False)
    self.reverse_prob = kwargs.get("reverse_prob", 0)

    self.model_inp_size = 3
    if self.concat_t_inputs:
      self.model_inp_size += 1

    self.pp_to_origin = pp_to_origin
    self.pp_relative_pos = pp_relative_pos
    preprocessing = True if pp_to_origin or pp_relative_pos else False
    self.fixed_len = fixed_len

    super(TFRecordStroke,
          self).__init__(data_path, meta_data_path, batch_size, preprocessing,
                         shuffle, normalize, run_mode)

    # Normalization statistics, ignoring the timestamp.
    if self.normalize:
      self.undo_mean_channel = np.concatenate(
          [self.mean_channel[0:2], self.mean_channel[3:]], axis=0)
      self.undo_std_channel = np.concatenate(
          [self.std_channel[0:2], self.std_channel[3:]], axis=0)

  def get_next(self):
    inputs, targets = self.iterator.get_next()
    inputs[C.TARGET_T_INK] = tf.reshape(inputs[C.TARGET_T_INK], [-1, 3])
    targets[C.TARGET_T_INK] = tf.reshape(targets[C.TARGET_T_INK], [-1, 3])
    targets[C.TARGET_T_PEN] = tf.reshape(targets[C.TARGET_T_PEN], [-1, 1])
    targets[C.TARGET_T_STROKE] = tf.reshape(targets[C.TARGET_T_STROKE], [-1, 2])
    return inputs, targets
  
  def tf_data_transformations(self):
    """Loads the raw data and apply preprocessing.

    This method is also used in calculation of the dataset statistics
    (i.e., meta-data file).
    """
    self.tf_data = tf.data.TFRecordDataset.list_files(
        self.data_path, seed=self.seed, shuffle=self.shuffle)
    self.tf_data = self.tf_data.interleave(
        tf.data.TFRecordDataset,
        cycle_length=self.num_parallel_calls,
        block_length=1)
    
    self.tf_data = self.tf_data.map(
        functools.partial(self.parse_tfexample_fn),
        num_parallel_calls=self.num_parallel_calls)
    self.tf_data = self.tf_data.prefetch(self.batch_size * 2)

    # Converts batch of strokes into individual samples.
    self.tf_data = self.tf_data.interleave(
        tf.data.Dataset.from_tensor_slices,
        cycle_length=1 if self.shuffle else self.num_parallel_calls,
        block_length=16)
    self.tf_data = self.tf_data.map(
        functools.partial(self.pp_extract_from_padded),
        num_parallel_calls=self.num_parallel_calls)
    self.tf_data = self.tf_data.filter(functools.partial(self.__pp_filter))

    self.tf_data = self.tf_data.map(
        functools.partial(self.expand_to_batch),
        num_parallel_calls=self.num_parallel_calls)

  def tf_preprocessing(self):
    if self.reverse_prob > 0:
      self.tf_data = self.tf_data.map(
          functools.partial(self.pp_reverse),
          num_parallel_calls=self.num_parallel_calls)
    
    if self.scale_factor > 0:
      self.tf_data = self.tf_data.map(
          functools.partial(self.pp_random_scale),
          num_parallel_calls=self.num_parallel_calls)

    if self.resampling_factor > 1:
      self.tf_data = self.tf_data.map(
          functools.partial(self.pp_temporal_resampling),
          num_parallel_calls=self.num_parallel_calls)

    if self.affine_prob > 0:
      self.tf_data = self.tf_data.map(
          functools.partial(self.pp_random_affine_all),
          num_parallel_calls=self.num_parallel_calls)
    
    self.tf_data = self.tf_data.map(
        functools.partial(self.set_start_end_coord),
        num_parallel_calls=self.num_parallel_calls)

    if self.pp_to_origin:
      self.tf_data = self.tf_data.map(
          functools.partial(self.pp_translate_to_origin),
          num_parallel_calls=self.num_parallel_calls)
      
    if self.pp_relative_pos:
      self.tf_data = self.tf_data.map(
          functools.partial(self.pp_relative_position),
          num_parallel_calls=self.num_parallel_calls)

  def tf_data_normalization(self):
    # Applies normalization.
    self.tf_data = self.tf_data.map(
        functools.partial(
            self.normalize_zero_mean_unit_variance_channel, key="ink"))
    if self.resample_target and self.resampling_factor > 1:
      self.tf_data = self.tf_data.map(
          functools.partial(
              self.normalize_zero_mean_unit_variance_channel, key="target_ink"))

  def tf_data_to_model(self):
    """Converts the data into the format that a model expects.

    Creates input, target, sequence_length, etc.
    """

    def element_length_func(model_inputs, _):
      return tf.cast(model_inputs[C.INP_SEQ_LEN], tf.int32)

    if self.fixed_len:
      self.tf_data = self.tf_data.map(
          functools.partial(self.pp_pad_to_max_len),
          num_parallel_calls=self.num_parallel_calls)

    self.tf_data = self.tf_data.map(
        functools.partial(self.pp_get_t_targets),
        num_parallel_calls=self.num_parallel_calls)

    if self.concat_t_inputs:
      self.tf_data = self.tf_data.map(
          functools.partial(self.pp_concat_t_inputs),
          num_parallel_calls=self.num_parallel_calls)

    self.tf_data = self.tf_data.map(functools.partial(self.__to_model_batch))
    if self.shuffle:
      self.tf_data = self.tf_data.shuffle(self.batch_size * 10, seed=self.seed)

    if self.fixed_len:
      self.tf_data = self.tf_data.batch(self.batch_size)
    else:
      # self.tf_data = self.tf_data.padded_batch(
      #     batch_size=self.batch_size,
      #     padded_shapes=self.tf_data.output_shapes)
      self.tf_data = self.tf_data.apply(
          tf.data.experimental.bucket_by_sequence_length(
              element_length_func=element_length_func,
              bucket_batch_sizes=[self.batch_size] * 3,
              bucket_boundaries=[50, 120],
              pad_to_bucket_boundary=False))
    self.tf_data = self.tf_data.prefetch(2)

  def pp_pad_to_max_len(self, sample):
    padding = self.max_length_threshold - tf.shape(input=sample["ink"])[1]
    sample["ink"] = tf.pad(tensor=sample["ink"], paddings=[[0, 0], [0, padding], [0, 0]])
    return sample

  def create_meta_data(self):
    """Makes a pass on the whole dataset and calculate statistics."""

    stats = self.compute_statistics_online(self.tf_data, key="ink")

    # We don't want to normalize the pen event.
    stats[C.MEAN_CHANNEL][3] = 0.0
    stats[C.VAR_CHANNEL][3] = 1.0
    stats["threshold_min_length"] = self.min_length_threshold
    stats["threshold_max_length"] = self.max_length_threshold
    stats["threshold_num_strokes"] = self.num_strokes_threshold
    stats["pp_to_origin"] = self.pp_to_origin
    stats["pp_relative_pos"] = self.pp_relative_pos
    return stats

  def pp_concat_t_inputs(self, sample):
    """
    Concatenate (x,y,pen) tuples with the corresponding t in [0,1].

    Args:
      sample:
    Returns:
    """
    max_len = sample["stroke_length"]
    steps = tf.range(max_len)
    t = tf.cast(steps, tf.float32)/tf.cast(sample["stroke_length"] - 1, tf.float32)
    sample["ink"] = tf.concat([sample["ink"], t[tf.newaxis, :, tf.newaxis]], axis=2)
    return sample
  
  def pp_get_t_targets(self, sample):
    """Draw a random t from [0,1] and get the interpolated point in the sequence.

    Args:
      sample:

    Returns:
    """
    key_len = "stroke_length"
    key_ink = "ink"
    if self.resample_target:
      key_len = "target_stroke_length"
      key_ink = "target_ink"
    
    if self.int_t_samples:
      float32_seq_len = tf.cast(sample[key_len], tf.float32)
      t = tf.random.uniform([self.n_t_targets], minval=0, maxval=tf.reduce_max(input_tensor=float32_seq_len),
                            dtype=tf.float32)
      t = tf.floor(t)
      lower_idx = tf.cast(t, tf.int32)
      lower_points = tf.gather(sample[key_ink], lower_idx, axis=1)
      inter_points = tf.concat([lower_points[0, :, :-2], lower_points[0, :, -1:]], axis=-1)
      t = t/(float32_seq_len - 1)
    else:
      # Identify lower and upper points.
      t = tf.random.uniform([self.n_t_targets], minval=0, maxval=1,
                            dtype=tf.float32)
      len_t = t*tf.cast(sample[key_len] - 1, tf.float32)
      factor = (len_t - tf.floor(len_t))[tf.newaxis, :, tf.newaxis]
      lower_idx = tf.cast(tf.floor(len_t), tf.int32)
      upper_idx = tf.cast(tf.math.ceil(len_t), tf.int32)
      lower_points = tf.gather(sample[key_ink], lower_idx, axis=1)
      upper_points = tf.gather(sample[key_ink], upper_idx, axis=1)
  
      inter_points = factor * upper_points + (1 - factor) * lower_points
      max_pen = tf.maximum(lower_points[0, :, -1:], upper_points[0, :, -1:])
      inter_points = tf.concat([inter_points[0, :, :-2], max_pen], axis=-1)
    sample[C.INP_T] = t
    sample[C.TARGET_T_INK] = inter_points
    return sample
  
  def pp_random_affine(self, sample):
    """Applies a separate affine transformation to strokes in a diagram."""
    
    rot_prob = self.affine_prob
    scale_prob = self.affine_prob
    shear_prob = self.affine_prob / 2.0

    n_strokes = tf.shape(input=sample["ink"])[0]
    rot_angle = tf.random.uniform([n_strokes],
                                  minval=-np.pi/3,
                                  maxval=np.pi/3,
                                  dtype=tf.float32)
    rot_angle = tf.compat.v1.where(rot_prob > tf.random.uniform([n_strokes], maxval=1.0),
                         rot_angle,
                         tf.zeros_like(rot_angle))

    scale_xy = tf.random.uniform([n_strokes],
                                 minval=-2,
                                 maxval=2,
                                 dtype=tf.float32)
    scale_xy = tf.compat.v1.where(scale_prob > tf.random.uniform([n_strokes], maxval=1.0),
                        scale_xy,
                        tf.ones_like(scale_xy))
    
    shear_xy = tf.random.uniform([n_strokes],
                                 minval=-0.2,
                                 maxval=0.2,
                                 dtype=tf.float32)
    shear_xy = tf.compat.v1.where(shear_prob > tf.random.uniform([n_strokes], maxval=1.0),
                        shear_xy,
                        tf.zeros_like(shear_xy))
    
    affine_ = self.apply_affine(sample["ink"][:,:,0:2],
                                theta=rot_angle,
                                scale_x=scale_xy,
                                scale_y=scale_xy,
                                shear_x=shear_xy,
                                shear_y=shear_xy)
    augmented = tf.concat([affine_, sample["ink"][:, :, 2:]], axis=-1)
    sample["ink"] = augmented
    return sample

  def pp_random_affine_all(self, sample):
    """Applies the same affine transformation to all strokes in a diagram."""
  
    rot_prob = self.affine_prob
    scale_prob = self.affine_prob
    flip_prob = self.affine_prob
    shear_prob = self.affine_prob/3.0
  
    n_strokes = tf.shape(input=sample["ink"])[0]
    
    # Rotation
    rot_angle = tf.random.uniform([1],
                                  minval=-np.pi/2,
                                  maxval=np.pi/2,
                                  dtype=tf.float32)
    rot_angle = tf.compat.v1.where(rot_prob > tf.random.uniform([1], maxval=1.0),
                         rot_angle,
                         tf.zeros_like(rot_angle))
    rot_angle = tf.tile(rot_angle, [n_strokes])
    
    # Scale
    scale_xy = tf.random.uniform([1],
                                 minval=0.5,
                                 maxval=2.5,
                                 dtype=tf.float32)
    scale_xy = tf.compat.v1.where(scale_prob > tf.random.uniform([1], maxval=1.0),
                        scale_xy,
                        tf.ones_like(scale_xy))
    
    # Flip around x, y or both.
    scale_x = tf.compat.v1.where(flip_prob > tf.random.uniform([1], maxval=1.0),
                       scale_xy*-1,
                       scale_xy)
    scale_y = tf.compat.v1.where(flip_prob > tf.random.uniform([1], maxval=1.0),
                       scale_xy*-1,
                       scale_xy)

    scale_x = tf.tile(scale_x, [n_strokes])
    scale_y = tf.tile(scale_y, [n_strokes])
    
    # Shear
    shear_xy = tf.random.uniform([1],
                                 minval=-0.3,
                                 maxval=0.3,
                                 dtype=tf.float32)
    shear_xy = tf.compat.v1.where(shear_prob > tf.random.uniform([1], maxval=1.0),
                        shear_xy,
                        tf.zeros_like(shear_xy))
    shear_xy = tf.tile(shear_xy, [n_strokes])
    
    # Apply affine.
    affine_ = self.apply_affine(sample["ink"][:, :, 0:2],
                                theta=rot_angle,
                                scale_x=scale_x,
                                scale_y=scale_y,
                                shear_x=shear_xy,
                                shear_y=shear_xy)
    augmented = tf.concat([affine_, sample["ink"][:, :, 2:]], axis=-1)
    sample["ink"] = augmented
    return sample
  
  def pp_random_scale(self, sample):
    """Randomly scales x and y coordinates."""
    scale_xy = 1.0 + tf.random.uniform([2],
                                       minval=-self.scale_factor,
                                       maxval=self.scale_factor,
                                       dtype=tf.float32)
    scale_ = tf.concat([scale_xy, tf.ones_like(scale_xy)], axis=0)
    sample["ink"] *= scale_
    return sample

  def pp_temporal_resampling(self, sample):
    """Uniform re-sampling over time dimension."""
    pen = sample["ink"][:, -1:]

    # factor = tf.cast(
    #     tf.cond(sample["stroke_length"] < 20,
    #             lambda: 1,
    #             lambda: self.resampling_factor),
    #     dtype=tf.int64)
    
    factor = tf.cast(
        tf.cond(pred=tf.reduce_max(sample["stroke_length"]) < 20,
                true_fn=lambda: 1,
                false_fn=lambda: self.resampling_factor // 2),
        dtype=tf.int32)

    factor = tf.cast(
        tf.cond(pred=tf.reduce_max(sample["stroke_length"]) > 100,
                true_fn=lambda: self.resampling_factor,
                false_fn=lambda: factor),
        dtype=tf.int64)
    
    # if self.resample_target:
    #   target_freq = tf.random_uniform([1],
    #                                   minval=1,
    #                                   maxval=factor + 1,
    #                                   dtype=tf.int64)[0]
    #   sample["target_ink"] = sample["ink"][:, ::target_freq, :]
    #   sample["target_ink"].set_shape((None, None, 4))
    #   # We keep the pen event.
    #   sample["target_ink"] = tf.concat([sample["target_ink"][:, :-1], pen],
    #                                    axis=1)
    #   sample["target_stroke_length"] = tf.cast(
    #       tf.math.ceil(sample["stroke_length"] / target_freq), tf.int64)
    #   sample["target_freq"] = 1.0 / tf.cast(target_freq, dtype=tf.float32)
    
    if self.resample_target:
      sample["target_ink"] = sample["ink"]
      sample["target_stroke_length"] = sample["stroke_length"]
      sample["target_freq"] = 1.0

    freq = factor
    # freq = tf.random_uniform([1], minval=1, maxval=factor + 1,
    #                          dtype=tf.int64)[0]
    
    sample["ink"] = sample["ink"][:, ::freq, :]
    sample["ink"].set_shape((None, None, 4))
    # We keep the pen event.
    sample["ink"] = tf.concat([sample["ink"][:, :-1], pen], axis=1)
    sample["stroke_length"] = tf.cast(
        tf.math.ceil(sample["stroke_length"] / freq), tf.int64)
    return sample

  def pp_extract_from_padded(self, sample):
    sample["ink"] = sample["ink"][:sample["stroke_length"], :]
    return sample

  def __pp_filter(self, sample):
    has_strokes, is_long_enough = True, True
    if self.min_length_threshold > 0:
      is_long_enough = sample["stroke_length"] > self.min_length_threshold
    if self.max_length_threshold > 0:
      is_long_enough = tf.math.logical_and(
          is_long_enough, sample["stroke_length"] < self.max_length_threshold)
    if self.num_strokes_threshold > 0:
      has_strokes = sample["num_strokes"] > self.num_strokes_threshold
    return tf.math.logical_and(has_strokes, is_long_enough)

  def expand_to_batch(self, sample):
    """Inserts a batch dimension."""
    sample["ink"] = tf.expand_dims(sample["ink"], axis=0)
    return sample

  def pp_reverse(self, sample):
    sample["ink"] = tf.cond(pred=self.reverse_prob > tf.random.uniform([1], maxval=1.0)[0],
                            true_fn=lambda: tf.reverse_sequence(input=sample["ink"], seq_lengths=tf.expand_dims(sample["stroke_length"], axis=0), seq_axis=1),
                            false_fn=lambda: sample["ink"])
    return sample
  
  def set_start_end_coord(self, sample):
    """Sets the start and end point coordinates."""
    sample[C.INP_START_COORD] = sample["ink"][:, 0:1, 0:2]
    # Strokes are padded. The end coordinate has the pen-up event where it is 1
    # and the rest has pen-up feature 0.
    tmp_ = sample["ink"][:, :, 0:2] * sample["ink"][:, :, 3:4]
    sample[C.INP_END_COORD] = tf.reduce_sum(input_tensor=tmp_, axis=1, keepdims=True)
    return sample

  def pp_translate_to_origin(self, sample):
    """Translate strokes to origin."""
    # Batch mode.
    if "target_ink" in sample:
      t_pen_event = sample["target_ink"][:, :, -1:]
      t_start_coord = sample["target_ink"][:, 0:1, 0:3]
      sample["target_ink"] = tf.concat(
          [sample["target_ink"][:, :, 0:3] - t_start_coord, t_pen_event],
          axis=-1)

    pen_event = sample["ink"][:, :, -1:]
    start_coord = sample["ink"][:, 0:1, 0:3]
    sample["ink"] = tf.concat(
        [sample["ink"][:, :, 0:3] - start_coord, pen_event], axis=-1)

    # sample["xy_cov"] = self.sequence_cov(sample["ink"][:, :, 0:2], sample["stroke_length"])
    return sample

  def pp_relative_position(self, sample):
    """Calculate offsets."""
    # Batch mode.
    if "target_ink" in sample:
      t_pen_event = sample["target_ink"][:, :, -1:]
      t_diff = sample["target_ink"][:, 1:, 0:3] - sample["target_ink"][:, 0:-1,
                                                                       0:3]
      diff = tf.concat(
          [tf.zeros((tf.shape(input=t_diff)[0], 1, tf.shape(input=t_diff)[2])), t_diff],
          axis=1)
      sample["target_ink"] = tf.concat([diff, t_pen_event], axis=-1)

    pen_event = sample["ink"][:, :, -1:]
    diff = sample["ink"][:, 1:, 0:3] - sample["ink"][:, 0:-1, 0:3]
    # To preserve the sequence length insert 0 as the start point.
    diff = tf.concat(
        [tf.zeros((tf.shape(input=diff)[0], 1, tf.shape(input=diff)[2])), diff], axis=1)
    sample["ink"] = tf.concat([diff, pen_event], axis=-1)
    return sample

  def __to_model_batch(self, tf_sample_dict):
    """Transforms a TFRecord sample into a more general sample representation.

    We use global keys to represent the required fields by the models.
    Args:
      tf_sample_dict:

    Returns:
    """
    # Targets are the inputs shifted by one step.
    # We ignore the timestamp and pen event.
    model_input = dict()
    ink_ = tf.concat(
        [tf_sample_dict["ink"][0, :, 0:2], tf_sample_dict["ink"][0, :, 3:]],
        axis=-1)
    if self.mask_pen:
      model_input[C.INP_SEQ_LEN] = tf_sample_dict["stroke_length"] - 1
      model_input[C.INP_ENC] = ink_[0:-1]
    else:
      model_input[C.INP_SEQ_LEN] = tf_sample_dict["stroke_length"]
      model_input[C.INP_ENC] = ink_

    model_input[C.INP_START_COORD] = tf_sample_dict[C.INP_START_COORD][0]
    model_input[C.INP_END_COORD] = tf_sample_dict[C.INP_END_COORD][0]
    model_input[C.INP_END_COORD] = tf_sample_dict[C.INP_END_COORD][0]
    model_input[C.INP_T] = tf_sample_dict[C.INP_T]
    model_input[C.TARGET_T_INK] = tf_sample_dict[C.TARGET_T_INK]
    # model_input[C.INP_NUM_STROKE] = tf.shape(tf_sample_dict["stroke_length"])

    model_target = dict()
    if "target_ink" in tf_sample_dict:
      ink_t = tf.concat([
          tf_sample_dict["target_ink"][0, :, 0:2],
          tf_sample_dict["target_ink"][0, :, 3:4]
      ],
                        axis=-1)
      model_input[C.INP_DEC] = tf.concat(
          [tf.zeros_like(ink_t[0:1]), ink_t[0:-1]], axis=0)

      model_target["stroke"] = tf_sample_dict["target_ink"][0, :, 0:2]
      model_target["pen"] = tf_sample_dict["target_ink"][0, :, 3:4]
      model_target[C.BATCH_SEQ_LEN] = tf_sample_dict["target_stroke_length"]
      model_target["target_freq"] = tf_sample_dict["target_freq"]
    else:
      model_target["stroke"] = tf_sample_dict["ink"][0, :, 0:2]
      model_target["pen"] = tf_sample_dict["ink"][0, :, 3:4]
      model_target[C.BATCH_SEQ_LEN] = tf_sample_dict["stroke_length"]
      model_input[C.INP_DEC] = tf.concat([tf.zeros_like(ink_[0:1]), ink_[0:-1]],
                                         axis=0)
    model_target[C.INP_START_COORD] = tf_sample_dict[C.INP_START_COORD][0]
    model_target[C.INP_END_COORD] = tf_sample_dict[C.INP_END_COORD][0]
    model_target[C.TARGET_T_INK] = tf_sample_dict[C.TARGET_T_INK]
    model_target[C.TARGET_T_STROKE] = tf_sample_dict[C.TARGET_T_INK][:, 0:2]
    # timestamp already discarded.
    model_target[C.TARGET_T_PEN] = tf_sample_dict[C.TARGET_T_INK][:, 2:3]
    return model_input, model_target

  def np_undo_preprocessing(self, ink_batch, start_point=None, seq_len=None):
    """Reverts the preprocessing steps applied in tf_preprocessing.

    Args:
        ink_batch (numpy): (batch size, padded length, 3) where the
          3-dimensional feature consists of <x, y, pen>.
        start_point: (batch size, 1, 3) denoting the initial point of strokes.
        seq_len (numpy): (batch size) length of each stroke. it must be updated
          if the start_point is inserted at the beginning.

    Returns:
    """
    if self.normalize:
      ink_batch = ink_batch * self.undo_std_channel + self.undo_mean_channel

    # Absolute position.
    if self.pp_relative_pos:
      abs_ink_batch = np.cumsum(ink_batch, axis=1)
      abs_ink_batch[:, :, 2] = ink_batch[:, :, 2]
    else:
      abs_ink_batch = ink_batch
    # Translate.
    if (self.pp_to_origin or self.pp_relative_pos) and start_point is not None:
      # Insert the start point back as we don't predict it.
      abs_ink_batch = np.concatenate(
          [np.zeros_like(abs_ink_batch[:, 0:1]), abs_ink_batch], axis=1)
      abs_ink_batch += np.concatenate(
          [start_point, np.zeros_like(start_point[:, :, 0:1])], axis=-1)
      # Inserting the start point increases length by one.
      if seq_len is not None:
        seq_len = np.copy(seq_len) + 1
    return abs_ink_batch, seq_len
  
  def tf_undo_preprocessing(self, ink_batch, start_point=None, seq_len=None):
    """Reverts the preprocessing steps applied in tf_preprocessing.

    Args:
        ink_batch (numpy): (batch size, padded length, 3) where the
          3-dimensional feature consists of <x, y, pen>.
        start_point: (batch size, 1, 3) denoting the initial point of strokes.
        seq_len (numpy): (batch size) length of each stroke. it must be updated
          if the start_point is inserted at the beginning.

    Returns:
    """
    if self.normalize:
      ink_batch = ink_batch * self.undo_std_channel + self.undo_mean_channel

    # Absolute position.
    if self.pp_relative_pos:
      abs_xy_batch = tf.cumsum(ink_batch[:, :, 0:2], axis=1)
      abs_ink_batch = tf.concat(abs_xy_batch, ink_batch[:, :, 2:3], axis=-2)
    else:
      abs_ink_batch = ink_batch

    abs_ink_batch = tf.cast(abs_ink_batch, tf.float32)
    # Translate.
    if (self.pp_to_origin or self.pp_relative_pos) and start_point is not None:
      # Insert the start point back as we don't predict it.
      abs_ink_batch = tf.concat(
          [tf.zeros_like(abs_ink_batch[:, 0:1]), abs_ink_batch], axis=1)
      abs_ink_batch += tf.concat(
          [start_point, tf.zeros_like(start_point[:, :, 0:1])], axis=-1)
      # Inserting the start point increases length by one.
      if seq_len is not None:
        seq_len += 1
    return abs_ink_batch, seq_len

  def parse_tfexample_fn(self, proto):
    """Parses a single tfrecord proto storing diagram sequence as strokes.

    Args:
      proto:

    Returns:
    """
    feature_to_type = {
        "ink": tf.io.VarLenFeature(dtype=tf.float32),
        "stroke_length": tf.io.VarLenFeature(dtype=tf.int64),
        "num_strokes": tf.io.FixedLenFeature([], dtype=tf.int64),
        # "shape": tf.FixedLenFeature([3], dtype=tf.int64),
        # "ink_hash": tf.FixedLenFeature([], dtype=tf.string),
    }

    parsed_features = tf.io.parse_single_example(serialized=proto, features=feature_to_type)
    parsed_features["ink"] = tf.reshape(
        tf.sparse.to_dense(parsed_features["ink"]),
        (parsed_features["num_strokes"], -1, 4))
    parsed_features["stroke_length"] = tf.sparse.to_dense(
        parsed_features["stroke_length"])
    parsed_features["num_strokes"] = tf.tile(
        tf.expand_dims(parsed_features["num_strokes"], axis=0),
        [parsed_features["num_strokes"]])
    return parsed_features


class TFRecordBatchDiagram(TFRecordStroke):
  """Creates batches of diagrams.

  All strokes of a diagram are available in the batch. An additional
  information of number of strokes per diagram is also provided in order to
  fetch an individual diagram.
  """

  def __init__(self,
               data_path,
               meta_data_path,
               batch_size=1,
               pp_to_origin=True,
               pp_relative_pos=True,
               shuffle=False,
               normalize=False,
               run_mode=C.RUN_ESTIMATOR,
               **kwargs):

    # Sequences shorter or longer than the thresholds are discarded.
    # self.min_length_threshold = kwargs.get("min_length_threshold", 0)
    # self.max_length_threshold = kwargs.get("max_length_threshold", 200)
    # # Sequences with less strokes are discarded.
    # self.num_strokes_threshold = kwargs.get("num_strokes_threshold", 4)
    # self.num_parallel_calls = kwargs.get("num_parallel_calls", 4)
    # self.mask_pen = kwargs.get("mask_pen", False)

    super(TFRecordBatchDiagram, self).__init__(
        data_path,
        meta_data_path,
        batch_size=batch_size,
        pp_to_origin=pp_to_origin,
        pp_relative_pos=pp_relative_pos,
        shuffle=shuffle,
        normalize=normalize,
        run_mode=run_mode,
        **kwargs)
    # min_length_threshold=self.min_length_threshold,
    # max_length_threshold=self.max_length_threshold,
    # num_strokes_threshold=self.num_strokes_threshold,
    # num_parallel_calls=self.num_parallel_calls,
    # mask_pen=self.mask_pen)

  def get_tf_samples(self):
    inputs, targets = self.tf_samples
    return self.batch_diagram_to_stroke(inputs, targets)

  def get_next(self):
    inputs, targets = self.iterator.get_next()
    return self.batch_diagram_to_stroke(inputs, targets)

  def batch_diagram_to_stroke(self, inputs, targets):
    """Converts batch of diagrams into batch of strokes.

    Reshapes [num_diagrams, num_strokes, ...] shaped inputs and targets into
    [num_diagrams x num_strokes, ...]. Note that num_strokes are padded to the
    maximum number of strokes in the batch.
    Args:
      inputs (dict): model inputs, i.e., return of __to_model_batch.
      targets (dict): model targets, i.e., return of __to_model_batch.

    Returns:
      (tuple) of reshaped inputs and targets.
    """
    # To discard padding strokes.
    # stroke_mask = tf.reshape(tf.sequence_mask(inputs["num_strokes"]), [-1])
    n_samples = tf.shape(input=inputs[C.INP_ENC])[0]
    n_strokes = tf.shape(input=inputs[C.INP_ENC])[1]
    batch_dim =  n_samples*n_strokes
    
    inputs[C.INP_ENC] = tf.reshape(inputs[C.INP_ENC], [batch_dim, -1, self.model_inp_size])
    inputs[C.INP_DEC] = tf.reshape(inputs[C.INP_DEC], [batch_dim, -1, self.model_inp_size])
    inputs[C.INP_SEQ_LEN] = tf.reshape(inputs[C.INP_SEQ_LEN], [batch_dim])
    inputs[C.INP_START_COORD] = tf.reshape(inputs[C.INP_START_COORD], [batch_dim, 1, 2])
    inputs[C.INP_END_COORD] = tf.reshape(inputs[C.INP_END_COORD], [batch_dim, 1, 2])
    # inputs["xy_cov"] = tf.reshape(inputs["xy_cov"], [batch_dim, 1, 4])
    
    targets["stroke"] = tf.reshape(targets["stroke"], [batch_dim, -1, 2])
    targets["pen"] = tf.reshape(targets["pen"], [batch_dim, -1, 1])
    targets[C.INP_SEQ_LEN] = tf.reshape(targets[C.INP_SEQ_LEN], [batch_dim])
    targets[C.INP_START_COORD] = tf.reshape(targets[C.INP_START_COORD], [batch_dim, 1, 2])
    targets[C.INP_END_COORD] = tf.reshape(targets[C.INP_END_COORD], [batch_dim, 1, 2])
    # targets["xy_cov"] = tf.reshape(targets["xy_cov"], [batch_dim, 1, 4])

    inputs[C.INP_T] = tf.reshape(inputs[C.INP_T], [batch_dim, -1])
    inputs[C.TARGET_T_INK] = tf.reshape(inputs[C.TARGET_T_INK], [-1, 3])
    targets[C.TARGET_T_INK] = tf.reshape(targets[C.TARGET_T_INK], [-1, 3])
    targets[C.TARGET_T_PEN] = tf.reshape(targets[C.TARGET_T_PEN], [-1, 1])
    targets[C.TARGET_T_STROKE] = tf.reshape(targets[C.TARGET_T_STROKE], [-1, 2])
    targets["stroke_mask"] = tf.reshape(tf.compat.v1.where(inputs[C.INP_T] > 0., tf.ones_like(inputs[C.INP_T]), tf.zeros_like(inputs[C.INP_T])), [-1])

    return inputs, targets

  def tf_data_transformations(self):
    """Loads the raw data and apply preprocessing.

    This method is also used in calculation of the dataset statistics
    (i.e., meta-data file).
    """
    self.tf_data = tf.data.TFRecordDataset.list_files(
        self.data_path, seed=self.seed, shuffle=self.shuffle)
    self.tf_data = self.tf_data.interleave(
        tf.data.TFRecordDataset,
        cycle_length=self.num_parallel_calls,
        block_length=1)
    
    self.tf_data = self.tf_data.map(
        functools.partial(self.parse_tfexample_fn),
        num_parallel_calls=self.num_parallel_calls)
    self.tf_data = self.tf_data.prefetch(self.batch_size * 2)
    self.tf_data = self.tf_data.filter(functools.partial(self.__pp_filter))

  def tf_data_normalization(self):
    # Apply normalization.
    super(TFRecordBatchDiagram, self).tf_data_normalization()

    # After preprocessing and normalization steps, the padded entries
    # may have non-zero values. Here we mask them.
    self.tf_data = self.tf_data.map(
        functools.partial(self.pp_seq_mask),
        num_parallel_calls=self.num_parallel_calls)

  def tf_data_to_model(self):

    if self.fixed_len:
      self.tf_data = self.tf_data.map(
          functools.partial(self.pp_pad_to_max_len),
          num_parallel_calls=self.num_parallel_calls)

    self.tf_data = self.tf_data.map(
        functools.partial(self.pp_get_t_targets),
        num_parallel_calls=self.num_parallel_calls)

    if self.concat_t_inputs:
      self.tf_data = self.tf_data.map(
          functools.partial(self.pp_concat_t_inputs),
          num_parallel_calls=self.num_parallel_calls)
    
    def element_length_func(model_inputs, _):
      return tf.cast(model_inputs[C.INP_NUM_STROKE], tf.int32)

    # Converts the data into the format that a model expects.
    # Creates input, target, sequence_length, etc.
    self.tf_data = self.tf_data.map(functools.partial(self.__to_model_batch))
    # TODO(aksan) configurable bucket_batch_size
    if self.batch_size > 1:
      bucket_batch_size = [
          self.batch_size,
          int(math.ceil(self.batch_size / 2)),
          int(math.ceil(self.batch_size / 3)),
          int(math.ceil(self.batch_size / 4)),
          int(math.ceil(self.batch_size / 5)),
      ]
      self.tf_data = self.tf_data.apply(
          tf.data.experimental.bucket_by_sequence_length(
              element_length_func=element_length_func,
              bucket_batch_sizes=bucket_batch_size,
              bucket_boundaries=[8, 13, 18, 23],
              pad_to_bucket_boundary=False))
    else:
      self.tf_data = self.tf_data.padded_batch(
          batch_size=1, padded_shapes=self.tf_data.output_shapes)

  def pp_seq_mask(self, sample):
    sample["ink"] *= tf.expand_dims(
        tf.sequence_mask(sample["stroke_length"], dtype=tf.float32), axis=2)
    return sample

  def pp_concat_t_inputs(self, sample):
    """
    Concatenate (x,y,pen) tuples with the corresponding t in [0,1].
    
    Args:
      sample:
    Returns:
    """
    n_strokes = tf.shape(input=sample["stroke_length"])[0]
    # max_len = tf.shape(sample["ink"])[1]
    max_len = tf.reduce_max(input_tensor=sample["stroke_length"])
    steps = tf.tile(tf.expand_dims(tf.range(max_len), axis=0), (n_strokes, 1))
    t = tf.cast(steps, tf.float32) / tf.expand_dims(tf.cast(sample["stroke_length"]-1, tf.float32), axis=1)
    sample["ink"] = tf.concat([sample["ink"], tf.expand_dims(t, axis=-1)], axis=2)
    sample["ink"] *= tf.expand_dims(tf.cast(tf.sequence_mask(sample["stroke_length"]), tf.float32), axis=2)
    return sample
  
  def pp_reverse(self, sample):
    sample["ink"] = tf.cond(pred=self.reverse_prob > tf.random.uniform([1], maxval=1.0)[0],
                            true_fn=lambda: tf.reverse_sequence(input=sample["ink"], seq_lengths=sample["stroke_length"], seq_axis=1),
                            false_fn=lambda: sample["ink"])
    return sample
    
  def pp_get_t_targets(self, sample):
    """Draw a random t from [0,1] and get the interpolated point in the sequence.
    
    Handles multiple stroke and multiple t cases.
    Args:
      sample:

    Returns:
    """
    if self.int_t_samples:
      n_strokes = tf.shape(input=sample["ink"])[0]
      t = tf.random.uniform([n_strokes, self.n_t_targets], minval=0, maxval=1,
                            dtype=tf.float32)
      len_t = t*tf.cast(tf.expand_dims(sample["stroke_length"], axis=-1), tf.float32)
      len_t = tf.floor(len_t)
      t = len_t / tf.tile(tf.expand_dims(tf.cast(sample["stroke_length"]-1, tf.float32), axis=1), (1, self.n_t_targets))
      lower_idx = tf.cast(len_t, tf.int32)
      
      batch_indices = tf.ones_like(lower_idx)
      batch_indices *= tf.expand_dims(tf.range(n_strokes), axis=-1)
      
      gather_lower_idx = tf.stack([
          batch_indices,
          lower_idx
          ], axis=-1)

      lower_points = tf.gather_nd(sample["ink"], gather_lower_idx)
      inter_points = tf.concat([lower_points[:, :, :-2], lower_points[:, :, -1:]], axis=-1)
    else:
      n_strokes = tf.shape(input=sample["ink"])[0]
      t = tf.random.uniform([n_strokes, self.n_t_targets], minval=0, maxval=1,
                            dtype=tf.float32)
      len_t = t*tf.cast(tf.expand_dims(sample["stroke_length"], axis=-1) - 1,
                        tf.float32)
      
      # Identify lower and upper points.
      lower_idx = tf.cast(tf.floor(len_t), tf.int32)
      upper_idx = tf.cast(tf.math.ceil(len_t), tf.int32)
  
      batch_indices = tf.ones_like(lower_idx)
      batch_indices *= tf.expand_dims(tf.range(n_strokes), axis=-1)
  
      gather_lower_idx = tf.stack([
          batch_indices,
          lower_idx
          ], axis=-1)
  
      gather_upper_idx = tf.stack([
          batch_indices,
          upper_idx
          ], axis=-1)
  
      lower_points = tf.gather_nd(sample["ink"], gather_lower_idx)
      upper_points = tf.gather_nd(sample["ink"], gather_upper_idx)
  
      factor = tf.expand_dims((len_t - tf.floor(len_t)), axis=-1)
      inter_points = factor*upper_points + (1 - factor)*lower_points
      
      max_pen = tf.maximum(lower_points[:, :, -1:], upper_points[:, :, -1:])
      inter_points = tf.concat([inter_points[:, :, :-2], max_pen], axis=-1)
    sample[C.INP_T] = t
    sample[C.TARGET_T_INK] = inter_points
    return sample

  def __pp_filter(self, sample):
    """Filters diagram samples.

    Works in batch mode. In other words, if an individual stroke of a diagram
    violates the conditions, then the entire diagram is discarded.
    Hence, the conditions should be relaxed.
    Args:
      sample:

    Returns:
    """
    has_strokes, is_long_enough = True, True
    if self.min_length_threshold > 0:
      is_long_enough = tf.math.greater(
          tf.reduce_min(input_tensor=sample["stroke_length"]), self.min_length_threshold)
    if self.max_length_threshold > 0:
      is_long_enough = tf.math.logical_and(
          is_long_enough,
          tf.math.less(
              tf.reduce_max(input_tensor=sample["stroke_length"]),
              self.max_length_threshold))
    if self.num_strokes_threshold > 0:
      has_strokes = (
          tf.shape(input=sample["num_strokes"])[0] > self.num_strokes_threshold)
    return tf.math.logical_and(has_strokes, is_long_enough)

  def __to_model_batch(self, tf_sample_dict):
    """Transforms a TFRecord sample into a more general sample representation.

    We use global keys to represent the required fields by the models.
    Args:
        tf_sample_dict:

    Returns:
    """
    # Target are the inputs shifted by one step.
    # We ignore the timestamp and pen event.
    model_input = dict()
    ink_ = tf.concat(
        [tf_sample_dict["ink"][:, :, 0:2], tf_sample_dict["ink"][:, :, 3:]],
        axis=-1)  # Ignore the timestamp.
    if self.mask_pen:
      mask_ = tf.sequence_mask(
          tf_sample_dict["stroke_length"] - 1,
          dtype=tf.float32,
          maxlen=tf.reduce_max(input_tensor=tf_sample_dict["stroke_length"]))
      model_input[C.INP_SEQ_LEN] = tf_sample_dict["stroke_length"] - 1
      model_input[C.INP_ENC] = (ink_ * tf.expand_dims(mask_, axis=2))[:, 0:-1]
    else:
      model_input[C.INP_SEQ_LEN] = tf_sample_dict["stroke_length"]
      model_input[C.INP_ENC] = ink_
    model_input[C.INP_DEC] = tf.concat(
        [tf.zeros_like(ink_[:, 0:1]), ink_[:, 0:-1]], axis=1)

    model_input[C.INP_START_COORD] = tf_sample_dict[C.INP_START_COORD]
    model_input[C.INP_END_COORD] = tf_sample_dict[C.INP_END_COORD]
    model_input[C.INP_NUM_STROKE] = tf.shape(input=tf_sample_dict["stroke_length"])[0]
    model_input[C.INP_T] = tf_sample_dict[C.INP_T]
    model_input[C.TARGET_T_INK] = tf_sample_dict[C.TARGET_T_INK]
    # model_input["xy_cov"] = tf_sample_dict["xy_cov"]

    model_target = dict()
    model_target["stroke"] = tf_sample_dict["ink"][:, :, 0:2]
    model_target["pen"] = tf_sample_dict["ink"][:, :, 3:4]
    model_target[C.INP_SEQ_LEN] = tf_sample_dict["stroke_length"]
    model_target[C.INP_NUM_STROKE] = tf.shape(
        input=tf_sample_dict["stroke_length"])[0]

    model_target[C.INP_START_COORD] = model_input[C.INP_START_COORD]
    model_target[C.INP_END_COORD] = model_input[C.INP_END_COORD]

    model_target[C.TARGET_T_INK] = tf_sample_dict[C.TARGET_T_INK]
    model_target[C.TARGET_T_STROKE] = tf_sample_dict[C.TARGET_T_INK][:, :, 0:2]
    # timestamp already discarded.
    model_target[C.TARGET_T_PEN] = tf_sample_dict[C.TARGET_T_INK][:, :, 2:3]
    # model_target["xy_cov"] = tf_sample_dict["xy_cov"]
    
    return model_input, model_target


class TFRecordSingleDiagram(TFRecordStroke):
  """A batch consists of all strokes of a diagram in order.

  Useful for testing and visualization.
  """

  def __init__(self,
               data_path,
               meta_data_path,
               batch_size=1,
               pp_to_origin=True,
               pp_relative_pos=True,
               shuffle=False,
               normalize=False,
               run_mode=C.RUN_ESTIMATOR,
               **kwargs):
    super(TFRecordSingleDiagram, self).__init__(
        data_path,
        meta_data_path,
        batch_size=1,
        pp_to_origin=pp_to_origin,
        pp_relative_pos=pp_relative_pos,
        shuffle=False,
        normalize=normalize,
        run_mode=run_mode,
        **kwargs)

  def tf_data_transformations(self):
    """Loads the raw data and apply preprocessing.

    This method is also used in calculation of the dataset statistics
    (i.e., meta-data file).
    """
    self.tf_data = tf.data.TFRecordDataset.list_files(
        self.data_path, seed=self.seed, shuffle=self.shuffle)
    self.tf_data = self.tf_data.interleave(
        tf.data.TFRecordDataset,
        cycle_length=self.num_parallel_calls,
        block_length=1)
    
    self.tf_data = self.tf_data.map(
        functools.partial(self.parse_tfexample_fn),
        num_parallel_calls=self.num_parallel_calls)
    self.tf_data = self.tf_data.prefetch(self.batch_size * 2)

    self.tf_data = self.tf_data.map(
        functools.partial(self.__pp_filter_by_length),
        num_parallel_calls=self.num_parallel_calls)
    self.tf_data = self.tf_data.filter(functools.partial(self.__pp_filter))

  def tf_preprocessing(self):
    super(TFRecordSingleDiagram, self).tf_preprocessing()
    # After preprocessing and normalization steps, the padded entries
    # may have non-zero values: mask them out.
    self.tf_data = self.tf_data.map(
        functools.partial(self.pp_seq_mask),
        num_parallel_calls=self.num_parallel_calls)

  def tf_data_normalization(self):
    # Apply normalization.
    super(TFRecordSingleDiagram, self).tf_data_normalization()

    # After preprocessing and normalization steps, the padded entries
    # may have non-zero values. Here we mask them.
    self.tf_data = self.tf_data.map(
        functools.partial(self.pp_seq_mask),
        num_parallel_calls=self.num_parallel_calls)

  def tf_data_to_model(self):
    if self.fixed_len:
      self.tf_data = self.tf_data.map(
          functools.partial(self.pp_pad_to_max_len),
          num_parallel_calls=self.num_parallel_calls)
      
    if self.concat_t_inputs:
      self.tf_data = self.tf_data.map(
          functools.partial(self.pp_concat_t_inputs),
          num_parallel_calls=self.num_parallel_calls)

    # Converts the data into the format that a model expects.
    # Creates input, target, sequence_length, etc.
    self.tf_data = self.tf_data.map(functools.partial(self.__to_model_batch))

  def pp_reverse(self, sample):
    sample["ink"] = tf.cond(pred=self.reverse_prob > tf.random.uniform([1], maxval=1.0)[0],
                            true_fn=lambda: tf.reverse_sequence(input=sample["ink"], seq_lengths=sample["stroke_length"], seq_axis=1),
                            false_fn=lambda: sample["ink"])
    return sample

  def pp_concat_t_inputs(self, sample):
    """
    Concatenate (x,y,pen) tuples with the corresponding t in [0,1].
    
    Args:
      sample:
    Returns:
    """
    n_strokes = tf.shape(input=sample["stroke_length"])[0]
    max_len = tf.reduce_max(input_tensor=sample["stroke_length"])
    steps = tf.tile(tf.expand_dims(tf.range(max_len), axis=0), (n_strokes, 1))
    t = tf.cast(steps, tf.float32) / tf.expand_dims(tf.cast(sample["stroke_length"]-1, tf.float32), axis=1)
    sample["ink"] = tf.concat([sample["ink"], tf.expand_dims(t, axis=-1)], axis=2)
    sample["ink"] *= tf.expand_dims(tf.cast(tf.sequence_mask(sample["stroke_length"]), tf.float32), axis=2)
    return sample
  
  def pp_seq_mask(self, sample):
    mask_ = tf.sequence_mask(sample["stroke_length"], dtype=tf.float32)
    sample["ink"] *= tf.expand_dims(mask_, axis=2)
    return sample

  def __pp_filter_by_length(self, sample):
    # short_enough = sample["stroke_length"] <= self.max_length_threshold
    long_enough = sample["stroke_length"] > self.min_length_threshold
    # selected_idx = tf.math.logical_and(short_enough, long_enough)
    selected_idx = long_enough
    sample["ink"] = sample["ink"][selected_idx]
    sample["stroke_length"] = sample["stroke_length"][selected_idx]
    sample["num_strokes"] = tf.shape(input=sample["ink"])[0]
    return sample

  def __pp_filter(self, sample):
    is_long_enough, has_strokes = True, True
    length = tf.reduce_max(input_tensor=sample["stroke_length"])
    if self.min_length_threshold > 0:
      is_long_enough = length > self.min_length_threshold

    if self.max_length_threshold > 0:
      is_long_enough = tf.math.logical_and(is_long_enough,
                                           length <= self.max_length_threshold)
    if self.num_strokes_threshold > 0:
      has_strokes = tf.reduce_max(
          input_tensor=sample["num_strokes"]) >= self.num_strokes_threshold
    return tf.math.logical_and(has_strokes, is_long_enough)

  def __to_model_batch(self, tf_sample_dict):
    """Transforms a TFRecord sample into a more general sample representation.

    We use global keys to represent the required fields by the models.
    Args:
        tf_sample_dict:

    Returns:
    """
    # Target are the inputs shifted by one step.
    # We ignore the timestamp and pen event.
    model_input = dict()
    ink_ = tf.concat(
        [tf_sample_dict["ink"][:, :, 0:2], tf_sample_dict["ink"][:, :, 3:]],
        axis=-1)  # Ignore the timestamp.

    if self.mask_pen:
      mask_ = tf.sequence_mask(
          tf_sample_dict["stroke_length"] - 1,
          dtype=tf.float32,
          maxlen=tf.shape(input=ink_)[1])
      model_input[C.INP_SEQ_LEN] = tf_sample_dict["stroke_length"] - 1
      model_input[C.INP_ENC] = (ink_ * tf.expand_dims(mask_, axis=2))[:, 0:-1]
    else:
      model_input[C.INP_SEQ_LEN] = tf_sample_dict["stroke_length"]
      model_input[C.INP_ENC] = ink_
    model_input[C.INP_DEC] = tf.concat(
        [tf.zeros_like(ink_[:, 0:1]), ink_[:, 0:-1]], axis=1)

    model_input[C.INP_START_COORD] = tf_sample_dict[C.INP_START_COORD]
    model_input[C.INP_END_COORD] = tf_sample_dict[C.INP_END_COORD]
    model_input[C.INP_NUM_STROKE] = tf.shape(input=tf_sample_dict["stroke_length"])
    # model_input["xy_cov"] = tf_sample_dict["xy_cov"]

    # TODO(aksan) Use dummy input for now.
    num_strokes = tf.shape(input=tf_sample_dict["ink"])[0]
    model_input[C.INP_T] = tf.zeros((num_strokes, 1))
    model_input[C.TARGET_T_INK] = tf.zeros((num_strokes, 3))

    model_target = dict()
    model_target["stroke"] = tf_sample_dict["ink"][:, :, 0:2]
    model_target["pen"] = tf_sample_dict["ink"][:, :, 3:4]
    model_target[C.INP_SEQ_LEN] = tf_sample_dict["stroke_length"]
    model_target[C.INP_START_COORD] = model_input[C.INP_START_COORD]
    model_target[C.INP_END_COORD] = model_input[C.INP_END_COORD]
    model_target[C.INP_NUM_STROKE] = tf.shape(input=tf_sample_dict["stroke_length"])
    # model_target["xy_cov"] = tf_sample_dict["xy_cov"]
    
    # TODO(aksan) Use dummy target for now.
    model_target[C.TARGET_T_INK] = tf.zeros((num_strokes, 3))
    model_target[C.TARGET_T_STROKE] = tf.zeros((num_strokes, 2))
    model_target[C.TARGET_T_PEN] = tf.zeros((num_strokes, 1))
    return model_input, model_target


if __name__ == "__main__":
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = 0.5
  tf.compat.v1.enable_eager_execution(config=config)

  def log_stats(stats, tag="Online"):
    """Logs statistics.

    Args:
      stats:
      tag:

    Returns:
    """
    print("[{2}] mean: {0}, std: {1}".format(stats["mean_all"],
                                             stats["var_all"], tag))
    print("[{2}] mean channel: {0}, std channel: {1}".format(
        stats["mean_channel"], stats["var_channel"], tag))
    print("[{1}] # samples: {0}".format(stats["n_samples"], tag))
    print("[{2}] min value: {0}, max value: {1}".format(stats["min_all"],
                                                        stats["max_all"], tag))
    print("[{3}] min length: {0}, mean length: {1}, max length: {2}".format(
        stats["min_seq_len"], stats["mean_seq_len"], stats["max_seq_len"], tag))
    print("============")
  
  # DATA_DIR = "/local/home/emre/Projects/google/data/didi/"
  # SPLIT = "test"
  # META_FILE = "didi-stats-origin_abs_pos.npy"
  # tfrecord_pattern = "diagrams_20200131-?????-of-?????"
  # data_path_ = os.path.join(DATA_DIR, SPLIT, tfrecord_pattern)

  DATA_DIR_WO = "/local/home/emre/Projects/google/data/didi_wo_text/"
  tfrecord_pattern_wo = "diagrams_wo_text_20200131-?????-of-?????"
  DATA_DIR = "/local/home/emre/Projects/google/data/didi_all/"
  tfrecord_pattern = "diagrams_20200131-?????-of-?????"
  
  SPLIT = "test"
  META_FILE = "didi_all-stats-origin_abs_pos.npy"
  data_path_ = [os.path.join(DATA_DIR_WO, SPLIT, tfrecord_pattern_wo),
                os.path.join(DATA_DIR, SPLIT, tfrecord_pattern)]

  train_data = TFRecordStroke(
      data_path=data_path_,
      meta_data_path=DATA_DIR + META_FILE,
      batch_size=1,
      shuffle=False,
      normalize=True,
      pp_to_origin=True,
      pp_relative_pos=False,
      run_mode=C.RUN_EAGER,
      max_length_threshold=201,
      fixed_len=False,
      mask_pen=False,
      scale_factor=0,
      resampling_factor=4,
      resample_target=True,
      n_t_targets=1,
      concat_t_inputs=False,
      reverse_prob=0,
      )

  # train_data = TFRecordSingleDiagram(
  #     data_path=data_path_,
  #     meta_data_path=DATA_DIR + META_FILE,
  #     batch_size=16,
  #     shuffle=True,
  #     normalize=False,
  #     pp_to_origin=True,
  #     pp_relative_pos=False,
  #     run_mode=C.RUN_EAGER,
  #     max_length_threshold=201,
  #     fixed_len=False,
  #     mask_pen=False,
  #
  #     affine_prob=1,
  #     reverse_prob=0,
  #     scale_factor=0,
  #     resampling_factor=0,
  #     resample_target=False,
  #     n_t_targets=1,
  #     concat_t_inputs=False
  #     )

  # from visualization.visualization import InkVisualizer
  # from smartink.util.utils import dict_tf_to_numpy
  # import time
  # vis_engine = InkVisualizer(train_data.np_undo_preprocessing,
  #                            DATA_DIR,
  #                            animate=False)
  # seq_lens = []
  # starts = []
  # sample_id = 0
  # ts = str(int(time.time()))
  # lens = []
  # try:
  #   while True:
  #     sample_id += 1
  #     input_batch, target_batch = train_data.get_next()
  #
  #     vis_sample = dict()
  #     vis_sample[sample_id] = dict_tf_to_numpy(target_batch)
  #     vis_engine.vis_strokes(vis_sample, "diagram-" + ts)
  #     if sample_id == 1:
  #       break
  # except tf.errors.OutOfRangeError:
  #   pass

  seq_lens = []
  starts = []
  for input_batch, target_batch in train_data.iterator:
    seq_lens.extend(input_batch[C.INP_SEQ_LEN].numpy().tolist())
    seq_lens.extend(target_batch[C.INP_SEQ_LEN].numpy().tolist())
    starts.extend(input_batch["start_coord"])
    # seq_lens.append(input_batch["seq_len"].numpy().max())
  seq_lens = np.array(seq_lens)
  print("Done")