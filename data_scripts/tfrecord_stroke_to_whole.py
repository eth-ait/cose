import os

import random
import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from smartink.data.stroke_dataset import TFRecordSingleDiagram
from common.constants import Constants as C


OUTPUTFILE="data/diagrams_wo_text/diagrams_wo_text_ramer_0.01.tf_record"
NUM_OUTPUT_SHARDS=25

def create_tfrecord_writers(output_file, num_output_shards):
  writers = []
  for i in range(num_output_shards):
    writers.append(tf.io.TFRecordWriter("%s-%05i-of-%05i" % (output_file, i, num_output_shards)))
  return writers

def close_tfrecord_writers(writers):
  for w in writers:
    w.close()

def _pick_output_shard(min_, max_):
    return random.randint(min_, max_)

def write_tfexample(writers, tf_example, start_, end_):
  writers[_pick_output_shard(start_, end_)].write(tf_example.SerializeToString())


def to_tf_example_concat(stroke_batch):
  ink_batch = stroke_batch[C.INP_ENC].numpy()
  seq_len = stroke_batch[C.INP_SEQ_LEN].numpy()
  strokes = []
  for i in range(ink_batch.shape[0]):
    strokes.append(ink_batch[i][:seq_len[i]])
  ink_ = np.concatenate(strokes, axis=0)
  
  features = dict()
  features['ink'] = tf.train.Feature(
      float_list=tf.train.FloatList(value=ink_.astype(float).flatten()))
  features['shape'] =  tf.train.Feature(
      int64_list=tf.train.Int64List(value=ink_.shape))
  features['num_strokes'] =  tf.train.Feature(
      int64_list=tf.train.Int64List(value=[ink_batch.shape[0]]))
  features['stroke_length'] = tf.train.Feature(
      int64_list=tf.train.Int64List(value=seq_len))
  
  example = tf.train.Example(features=tf.train.Features(feature=features))
  return example

def main():
  DATA_DIR = "data/diagrams_with_strokes_wo_text/"
  SPLIT = "training"
  META_FILE = "strokes_diagrams_wo_text_ramer_0.01_stats-relative_pos.npy"
  tfrecord_pattern = "strokes_diagrams_wo_text_ramer_0.01.tf_record-?????-of-?????"
  data_path_ = os.path.join(DATA_DIR, SPLIT, tfrecord_pattern)

  tfrecord_writers = create_tfrecord_writers(OUTPUTFILE, NUM_OUTPUT_SHARDS)
  train_data = TFRecordSingleDiagram(
      data_path=data_path_,
      meta_data_path=DATA_DIR + META_FILE,
      batch_size=1,
      shuffle=False,
      normalize=False,
      pp_to_origin=False,
      pp_relative_pos=False,
      run_mode=C.RUN_EAGER,
      max_length_threshold=-1,
      fixed_len=False)

  idx = 0
  for input_batch, target_batch in train_data.iterator:
    tf_example = to_tf_example_concat(input_batch)
    write_tfexample(tfrecord_writers, tf_example, 0, 19)
    idx += 1
  print("# training samples: " + str(idx))

  valid_data = TFRecordSingleDiagram(
      data_path=os.path.join(DATA_DIR, "validation", tfrecord_pattern),
      meta_data_path=DATA_DIR + META_FILE,
      batch_size=1,
      shuffle=False,
      normalize=False,
      pp_to_origin=False,
      pp_relative_pos=False,
      run_mode=C.RUN_EAGER,
      max_length_threshold=-1,
      fixed_len=False)

  idx = 0
  for input_batch, target_batch in valid_data.iterator:
    tf_example = to_tf_example_concat(input_batch)
    write_tfexample(tfrecord_writers, tf_example, 20, 21)
    idx += 1
  print("# validation samples: " + str(idx))

  test_data = TFRecordSingleDiagram(
      data_path=os.path.join(DATA_DIR, "test", tfrecord_pattern),
      meta_data_path=DATA_DIR + META_FILE,
      batch_size=1,
      shuffle=False,
      normalize=False,
      pp_to_origin=False,
      pp_relative_pos=False,
      run_mode=C.RUN_EAGER,
      max_length_threshold=-1,
      fixed_len=False)

  idx = 0
  for input_batch, target_batch in test_data.iterator:
    tf_example = to_tf_example_concat(input_batch)
    write_tfexample(tfrecord_writers, tf_example, 22, 24)
    idx += 1
  print("# test samples: " + str(idx))

  close_tfrecord_writers(tfrecord_writers)


if __name__ == "__main__":
  main()