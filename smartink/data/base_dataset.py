"""Base dataset class for diagram data in tfrecords.

Provides data I/O functionality in tf.data API, preprocessing and normalization
routines.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import numpy as np
import tensorflow as tf
from rdp import rdp

from common.constants import Constants as C
from smartink.util.utils import err_unknown_type


class Dataset(object):
  """A base wrapper class around tf.data.Dataset API.

  Depending on the dataset requirements, it applies data transformations.
  """
  
  def __init__(self,
               data_path,
               meta_data_path,
               batch_size,
               preprocessing=True,
               shuffle=True,
               normalize=True,
               run_mode=C.RUN_ESTIMATOR):
    
    self.tf_data = None
    self.data_path = data_path
    self.batch_size = batch_size
    self.preprocessing = preprocessing
    self.shuffle = shuffle
    self.normalize = normalize
    self.run_mode = run_mode
    self.seed = 1234
    print("Data path: " + str(data_path))
    
    # First apply preprocessing and then normalization.
    self.tf_data_transformations()
    self.tf_preprocessing()
    
    if self.normalize:
      # Load statistics and other data summary stored in the meta-data file.
      self.meta_data = self.load_meta_data(meta_data_path)
      if not self.meta_data:
        print("Calculating statistics...")
        self.meta_data = self.create_meta_data()
        self.save_meta_data(self.meta_data, meta_data_path)
      
      # self.data_summary()
      self.mean_all = self.meta_data[C.MEAN_ALL]
      self.std_all = np.sqrt(self.meta_data[C.VAR_ALL])
      self.mean_channel = self.meta_data[C.MEAN_CHANNEL]
      self.std_channel = np.sqrt(self.meta_data[C.VAR_CHANNEL])

      self.mean_start_pos = self.meta_data.get("mean_start_pos", np.array([0]))
      self.std_start_pos = self.meta_data.get("std_start_pos", np.array([1]))
      
      self.tf_data_normalization()
    
    self.tf_data_to_model()
    
    if self.run_mode == C.RUN_EAGER:
      self.iterator = tf.compat.v1.data.make_one_shot_iterator(self.tf_data)
      self.tf_samples = None
    elif self.run_mode == C.RUN_ESTIMATOR:
      self.tf_data = self.tf_data.repeat()
      self.iterator = tf.compat.v1.data.make_one_shot_iterator(self.tf_data)
      self.tf_samples = self.get_next()
    elif self.run_mode == C.RUN_STATIC:
      self.iterator = tf.compat.v1.data.make_initializable_iterator(self.tf_data)
      self.tf_samples = self.get_next()
  
  @classmethod
  def load_meta_data(cls, meta_data_path):
    """Loads meta-data file given the path.

    It is assumed to be in numpy.

    Args:
        meta_data_path:

    Returns:
        Meta-data dictionary or False if it is not found.
    """
    # if not meta_data_path or not os.path.exists(meta_data_path):
    
    _, ext = os.path.splitext(meta_data_path)
    if ext == ".json":
      meta_fp = tf.io.gfile.GFile(meta_data_path, "r")
      try:
        meta_fp.size()
        print("Loading statistics " + meta_data_path)
        json_stats = json.load(meta_fp)
        stats_np = dict()
        for key_, value_ in json_stats.items():
          stats_np[key_] = np.array(value_) if isinstance(value_, list) else \
            value_
        return stats_np
      except tf.errors.NotFoundError:
        print("Meta-data not found.")
        return False
    
    elif ext == ".npy":
      meta_fp = tf.io.gfile.GFile(meta_data_path, "rb")
      try:
        meta_fp.size()
        print("Loading statistics " + meta_data_path)
        return np.load(meta_fp, allow_pickle=True).item()
      except tf.errors.NotFoundError:
        print("Meta-data not found.")
        return False
    else:
      err_unknown_type(ext)
  
  @classmethod
  def save_meta_data(cls, meta_data_dict, meta_data_path):
    file_path, _ = os.path.splitext(meta_data_path)
    np.save(file_path + ".npy", meta_data_dict)
    # Convert numpy array into python list and store in json format.
    py_meta = dict()
    for key_, value_ in meta_data_dict.items():
      py_meta[key_] = value_.tolist() if isinstance(value_,
                                                    np.ndarray) else value_
    json.dump(py_meta, open(file_path + ".json", "w"), indent=4, sort_keys=True)
  
  def tf_data_transformations(self):
    raise NotImplementedError("Subclass must override sample method")
  
  def tf_preprocessing(self):
    raise NotImplementedError("Subclass must override sample method")
  
  def tf_data_normalization(self):
    raise NotImplementedError("Subclass must override sample method")
  
  def tf_data_to_model(self):
    raise NotImplementedError("Subclass must override sample method")
  
  def create_meta_data(self):
    raise NotImplementedError("Subclass must override sample method")
  
  def data_summary(self):
    print("# of samples: " + str(self.meta_data[C.NUM_SAMPLES]))
    print("Shortest sequence length: " + str(self.meta_data[C.MIN_SEQ_LEN]))
    print("Longest sequence length: " + str(self.meta_data[C.MAX_SEQ_LEN]))
  
  def normalize_zero_mean_unit_variance_all(self, sample_dict, key):
    sample_dict[key] = (sample_dict[key] - self.mean_all)/self.std_all
    return sample_dict
  
  def normalize_zero_mean_unit_variance_channel(self, sample_dict, key):
    sample_dict[key] = (sample_dict[key] - self.mean_channel)/self.std_channel
    return sample_dict
  
  def get_iterator(self):
    return self.iterator
  
  def get_tf_samples(self):
    return self.tf_samples
  
  def get_next(self):
    return self.iterator.get_next()
  
  def make_one_shot_iterator(self):
    self.iterator = tf.compat.v1.data.make_one_shot_iterator(self.tf_data)
  
  @classmethod
  def compute_statistics_online(cls, iterable_data, key=None):
    """Given a da data iterator, gathers data statistics online.

    The whole data isn't required to be loaded. It is eager compatible,
    and hence it is okay to pass numpy array or python list.
    Mean and variance is calculated according to Knuth's method:
    https://www.johndcook.com/blog/standard_deviation/
    Args:
        iterable_data: where each sample is assumed to be a dictionary with
          <key,value> pairs. If the key argument is None, then numpy or
          tf.Tensor samples are also welcome.
        key: data of interest.

    Returns:
        (dict) or data statistics with mean, var, min, max calculated across the
        sequences or all values, and
        seq_len, number of samples.
    """
    n_samples = 0
    n_all, mean_all, var_all, m2_all = 0.0, 0.0, 0.0, 0.0
    n_channel, mean_channel, var_channel, m2_channel = 0.0, 0.0, 0.0, 0.0
    min_all, max_all = np.inf, -np.inf
    min_seq_len, max_seq_len = np.inf, -np.inf
    seq_len_sum = 0
    
    for sample_dict in iterable_data:
      if key is not None:
        sample = sample_dict[key]
      else:
        sample = sample_dict
      
      if isinstance(sample, tf.Tensor):
        sample = sample.numpy()
      
      assert len(sample) == 1
      sample = sample[0]
      sample = sample.astype(np.float64)
      
      n_samples += 1
      seq_len, feature_size = sample.shape
      
      # Global mean&variance
      n_all += seq_len*feature_size
      delta_all = sample - mean_all
      mean_all = mean_all + delta_all.sum()/n_all
      m2_all = m2_all + (delta_all*(sample - mean_all)).sum()
      
      # Channel-wise mean&variance
      n_channel += seq_len
      delta_channel = sample - mean_channel
      mean_channel = mean_channel + delta_channel.sum(axis=0)/n_channel
      m2_channel = m2_channel + (delta_channel*
                                 (sample - mean_channel)).sum(axis=0)
      
      # Global min&max values.
      min_all = np.min(sample) if np.min(sample) < min_all else min_all
      max_all = np.max(sample) if np.max(sample) > max_all else max_all
      
      # Min&max sequence length.
      min_seq_len = seq_len if seq_len < min_seq_len else min_seq_len
      max_seq_len = seq_len if seq_len > max_seq_len else max_seq_len
      
      # Mean sequence length.
      seq_len_sum += seq_len
    
    var_all = m2_all/(n_all - 1)
    var_channel = m2_channel/(n_channel - 1)
    
    stats = {
        C.MEAN_ALL    : mean_all,
        C.MEAN_CHANNEL: mean_channel,
        C.VAR_ALL     : var_all,
        C.VAR_CHANNEL : var_channel,
        C.MIN_ALL     : min_all,
        C.MAX_ALL     : max_all,
        C.MIN_SEQ_LEN : min_seq_len,
        C.MAX_SEQ_LEN : max_seq_len,
        C.MEAN_SEQ_LEN: seq_len_sum/n_samples,
        C.NUM_SAMPLES : n_samples
        }
    return stats
  
  @classmethod
  def sequence_mean(cls, batch_sequence, seq_len):
    mask_ = tf.expand_dims(tf.cast(tf.sequence_mask(seq_len), tf.float32),
                           axis=-1)
    seq_mean = tf.reduce_sum(input_tensor=batch_sequence*mask_, axis=1)/tf.expand_dims(
      tf.cast(tf.maximum(1, tf.cast(seq_len, tf.int32)), tf.float32), axis=-1)
    return tf.expand_dims(seq_mean, axis=1)
  
  @classmethod
  def sequence_cov(cls, batch_sequence, seq_len, mean=None):
    if mean is None:
      mean = cls.sequence_mean(batch_sequence, seq_len)
    
    mask_ = tf.expand_dims(tf.cast(tf.sequence_mask(seq_len), tf.float32),
                           axis=-1)
    centered = (batch_sequence - mean)*mask_
    matmul = tf.matmul(centered, centered, transpose_a=True)
    return matmul/tf.cast(tf.maximum(1, tf.cast(seq_len, tf.int32)),
                          tf.float32)[:, tf.newaxis, tf.newaxis]
  
  @classmethod
  def apply_affine(cls, sample, theta=0.0, scale_x=1.0, scale_y=1.0,
                   shear_x=0.0, shear_y=0.0):
    """
    Affine transformation by applying scaling, rotation and shearing in order.
    The sample is a sequence of 2D points. If size of the transformation factors
    is equal to batch_size, then the operation runs in batch mode.

    The default values correspond to no transformation.
    Args:
      sample: (batch_size, seq_len, 2)
      theta: rotation angle in radians.
      scale_x: scale factor in x-axis.
      scale_y: scale factor in y-axis.
      shear_x: amount of shearing in x direction.
      shear_y: amount of shearing in y direction.

    Returns:
      Transformed sample.
    """
    rot_scale_mat = tf.stack([[scale_x*tf.cos(theta), -scale_y*tf.sin(theta)],
                              [scale_x*tf.sin(theta), scale_y*tf.cos(theta)]])
    rot_scale_mat = tf.transpose(a=tf.reshape(rot_scale_mat, [2, 2, -1]),
                                 perm=[2, 0, 1])
    
    shear_mat = tf.stack(
        [[tf.ones_like(shear_x), shear_x], [shear_y, tf.ones_like(shear_y)]])
    shear_mat = tf.transpose(a=tf.reshape(shear_mat, [2, 2, -1]), perm=[2, 0, 1])
    
    affine_mat = tf.matmul(shear_mat, rot_scale_mat)
    return tf.matmul(sample, affine_mat)

  @tf.function(input_signature=[tf.TensorSpec([None, 3], tf.float32),
                                tf.TensorSpec(None, tf.float32)])
  def tf_rdp_resampling(self, sequence, epsilon):
    """Ramer-Douglas-Peucker re-sampling."""
    resampled =  tf.numpy_function(rdp, [sequence, epsilon], tf.float32)
    resampled.set_shape([None, 3])
    return resampled