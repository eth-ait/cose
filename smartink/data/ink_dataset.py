"""Treats an ink sample as a sequence of points.

In contrast to the classes in stroke_dataset.py, all strokes are concatenated.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import os
import numpy as np
import tensorflow as tf

from smartink.data.stroke_dataset import TFRecordStroke
from common.constants import Constants as C


class TFRecordInkSequence(TFRecordStroke):
  """Creates batches of ink sequences.

  TFRecords store ink samples of shape (n_strokes, padded_stroke_length, 4).
  First, convert it into a sequence of points (1, n_points, 4) and then create
  variable-length batches.
  """

  def __init__(self,
               data_path,
               meta_data_path,
               batch_size=1,
               pp_to_origin=True,
               pp_relative_pos=False,
               shuffle=False,
               normalize=False,
               run_mode=C.RUN_ESTIMATOR,
               **kwargs):

    super(TFRecordInkSequence, self).__init__(
        data_path,
        meta_data_path,
        batch_size=batch_size,
        pp_to_origin=pp_to_origin,
        pp_relative_pos=pp_relative_pos,
        shuffle=shuffle,
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
    self.tf_data = self.tf_data.prefetch(self.batch_size*2)
    
    if self.rdp and self.rdp_didi_pp:
      self.tf_data = self.tf_data.map(
          functools.partial(self.rdp_size_normalization),
          num_parallel_calls=self.num_parallel_calls)
  
    # Convert batch of strokes into a single sequence of points.
    self.tf_data = self.tf_data.map(
        functools.partial(self.to_ink_sequence),
        num_parallel_calls=self.num_parallel_calls)

    self.tf_data = self.tf_data.filter(functools.partial(self.__pp_filter))
  
  def to_ink_sequence(self, sample):
    """Discards the padded entries and concatenates individual strokes."""
    seq_mask = tf.sequence_mask(sample["stroke_length"])
    point_indices = tf.where(seq_mask)
    sample["ink"] = tf.expand_dims(tf.gather_nd(sample["ink"], point_indices), axis=0)
    sample["stroke_length"] = tf.reduce_sum(sample["stroke_length"])
    return sample
  
  def __pp_filter(self, sample):
    has_strokes, is_long_enough = True, True
    if self.min_length_threshold > 0:
      is_long_enough = sample["stroke_length"] > self.min_length_threshold
    if self.max_length_threshold > 0:
      is_long_enough = tf.math.logical_and(
          is_long_enough, sample["stroke_length"] < self.max_length_threshold)
    return is_long_enough


if __name__ == "__main__":
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = 0.5
  tf.compat.v1.enable_eager_execution(config=config)
  
  DATA_DIR = "/local/home/emre/Projects/google/data/didi_wo_text_rdp/"
  tfrecord_pattern = "diagrams_wo_text_20200131-?????-of-?????"
  
  SPLIT = "training"
  # META_FILE = "didi_wo_text_rdp-stats-relative_pos.npy"
  META_FILE = "didi_wo_text-stats-origin_abs_pos.npy"
  data_path_ = [os.path.join(DATA_DIR, SPLIT, tfrecord_pattern)]
  
  train_data = TFRecordInkSequence(
      data_path=data_path_,
      meta_data_path=DATA_DIR + META_FILE,
      batch_size=1,
      shuffle=False,
      normalize=True,
      pp_to_origin=True,
      pp_relative_pos=False,
      run_mode=C.RUN_EAGER,
      max_length_threshold=301,
      fixed_len=False,
      mask_pen=False,
      scale_factor=0,
      resampling_factor=0,
      random_noise_factor=0,
      gt_targets=False,
      n_t_targets=4,
      concat_t_inputs=False,
      reverse_prob=0,
      t_drop_ratio=0,
      affine_prob=0,
      rdp=True,
      rdp_didi_pp=True,
      min_length_threshold=2
      )
  
  from visualization.visualization import InkVisualizer
  from smartink.util.utils import dict_tf_to_numpy
  import time

  vis_engine = InkVisualizer(train_data.np_undo_preprocessing,
                             DATA_DIR,
                             animate=False)
  # seq_lens = []
  # starts = []
  # sample_id = 0
  # ts = str(int(time.time()))
  # lens = []
  #
  # try:
  #   while True:
  #     sample_id += 1
  #     input_batch, target_batch = train_data.get_next()
  #
  #     target_batch = dict_tf_to_numpy(target_batch)
  #     vis_engine.vis_ink_sequence(target_batch, str(sample_id) + "_ink_sequence-" + ts)
  #
  #     if sample_id == 1:
  #       break
  #
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