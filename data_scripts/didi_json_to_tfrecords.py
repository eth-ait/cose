"""Data preprocessing script for CoSE model.

Converts .ndjson data into a format CoSE expects and stores in tfrecords.
A ramer resampled variant of the data (aka sketch-rnn format) can be stored as
well.
"""
import collections
import contextlib
import json
import os
import random
import tensorflow as tf

import numpy as np
from rdp import rdp

# Setup and settings.
DATA_DIR = None  # TODO: Set this path.
if DATA_DIR is None and "COSE_DATA_DIR" in os.environ:
  DATA_DIR = os.path.join(os.environ["COSE_DATA_DIR"], "didi_wo_text/")
else:
  raise Exception("Data path must be set")

# JSON_FILES=["full_raw_cat.ndjson"]
# JSON_FILES=["full_raw_elephant.ndjson"]
JSON_FILES=["diagrams_wo_text_20200131.ndjson"]
NUM_TFRECORD_SHARDS = 10


def split_and_pad_strokes(stroke_list):
  max_len = np.array([len(stroke[0]) for stroke in stroke_list]).max()
  
  strokes = []
  stroke_lengths = []
  for stroke in stroke_list:
    stroke_len = len(stroke[0])
    padded_stroke_with_pen = np.zeros([1, max_len, 4], dtype=np.float32)
    padded_stroke_with_pen[0, 0:stroke_len, 0] = stroke[0]
    padded_stroke_with_pen[0, 0:stroke_len, 1] = stroke[1]
    padded_stroke_with_pen[0, 0:stroke_len, 2] = stroke[2]
    padded_stroke_with_pen[0, stroke_len - 1, 3] = 1
    strokes.append(padded_stroke_with_pen)
    stroke_lengths.append(stroke_len)
  
  all_strokes = np.concatenate(strokes, axis=0).astype(float)  # (num_strokes, max_len, 4)
  all_stroke_lengths = np.array(stroke_lengths).astype(int)
  return all_strokes, all_stroke_lengths


def ink_to_tfexample(ink, dot=None):
  """Takes a LabeledInk and outputs a TF.Example with stroke information.

  Args:
    ink: A JSON array containing the drawing information.
    dot: (Optional) textual content of the GrahViz dotfile that was used to
      generate the prompt image.

  Returns:
    a Tensorflow Example proto with the drawing data.
  """
  features = {}
  features["key"] = tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[ink["key"].encode("utf-8")]))
  features["label_id"] = tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[ink["label_id"].encode("utf-8")]))
  if dot:
    features["label_dot"] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[dot.encode("utf-8")]))

  all_strokes, all_stroke_lengths = split_and_pad_strokes(ink["drawing"])
  features["ink"] = tf.train.Feature(
      float_list=tf.train.FloatList(value=all_strokes.flatten()))
  features["stroke_length"] = tf.train.Feature(
      int64_list=tf.train.Int64List(value=all_stroke_lengths))
  features["shape"] = tf.train.Feature(
      int64_list=tf.train.Int64List(value=all_strokes.shape))
  features["num_strokes"] = tf.train.Feature(
      int64_list=tf.train.Int64List(value=[len(ink["drawing"])]))
  
  if "rdp_ink" in ink:
    rdp_all_strokes, rdp_all_stroke_lengths = split_and_pad_strokes(ink["rdp_ink"])
    features["rdp_ink"] = tf.train.Feature(
        float_list=tf.train.FloatList(value=rdp_all_strokes.flatten()))
    features["rdp_stroke_length"] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=rdp_all_stroke_lengths))
    features["rdp_shape"] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=rdp_all_strokes.shape))
    features["rdp_num_strokes"] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[len(ink["rdp_ink"])]))
  
  example = tf.train.Example(features=tf.train.Features(feature=features))
  return example

@contextlib.contextmanager
def create_tfrecord_writers(output_dir, output_file, num_output_shards):
  split_file_path_map = dict(train="training", valid="validation", test="test")
  writers = collections.defaultdict(list)
  for split in ["train", "valid", "test"]:
    output_dir_path = os.path.join(output_dir, split_file_path_map[split])
    if not os.path.exists(output_dir_path):
      os.mkdir(output_dir_path)
    for i in range(num_output_shards):
      writers[split].append(
          tf.io.TFRecordWriter("%s/%s-%05i-of-%05i" %
                                      (output_dir_path, output_file, i, num_output_shards)))
  try:
    yield writers
  finally:
    for split in ["train", "valid", "test"]:
      for w in writers[split]:
        w.close()


def pick_output_shard(num_shards):
  return random.randint(0, num_shards - 1)


def size_normalization(drawing):
  def get_bounding_box(drawing):
    minx = 99999
    miny = 99999
    maxx = 0
    maxy = 0

    for s in drawing:
      minx = min(minx, min(s[0]))
      maxx = max(maxx, max(s[0]))
      miny = min(miny, min(s[1]))
      maxy = max(maxy, max(s[1]))
    return (minx, miny, maxx, maxy)

  bb = get_bounding_box(drawing)
  width, height = bb[2] - bb[0], bb[3] - bb[1]
  offset_x, offset_y = bb[0], bb[1]
  if height < 1e-6:
    height = 1

  size_normalized_drawing = [[[(x - offset_x) / height for x in stroke[0]],
                              [(y - offset_y) / height for y in stroke[1]],
                              [t for t in stroke[2]]]
                             for stroke in drawing]

  return size_normalized_drawing

def resample_ink(drawing, timestep):
  def resample_stroke(stroke, timestep):
    def interpolate(t, t_prev, t_next, v0, v1):
      d0 = abs(t-t_prev)
      d1 = abs(t-t_next)
      dist_sum = d0 + d1
      d0 /= dist_sum
      d1 /= dist_sum
      return d1 * v0 + d0 * v1

    x,y,t = stroke
    if len(t) < 3:
      return stroke
    r_x, r_y, r_t = [x[0]], [y[0]], [t[0]]
    final_time = t[-1]
    stroke_time = final_time - t[0]
    necessary_steps = int(stroke_time / timestep)

    i = 1
    current_time = t[i]
    while current_time < final_time:
      current_time += timestep
      while i < len(t) - 1 and current_time > t[i]:
        i += 1
      r_x.append(interpolate(current_time, t[i-1], t[i], x[i-1], x[i]))
      r_y.append(interpolate(current_time, t[i-1], t[i], y[i-1], y[i]))
      r_t.append(interpolate(current_time, t[i-1], t[i], t[i-1], t[i]))
    return [r_x, r_y, r_t]

  resampled = [resample_stroke(s, timestep) for s in drawing]
  return resampled


def sketch_rnn_preprocess(raw_ink, rdp_epsilon=2.0):
  # ignoring the timestamp.
  processed_ink = []
  for stroke in raw_ink:
    xy = np.transpose(np.stack(stroke[0:2]), [1, 0])
    rdp_mask = rdp(xy, epsilon=rdp_epsilon, return_mask=True)
    rdp_x = np.array(stroke[0])[rdp_mask]
    rdp_y = np.array(stroke[1])[rdp_mask]
    rdp_t = np.array(stroke[2])[rdp_mask]
    # pen = np.zeros_like(rdp_x)
    # pen[-1] = 1
    # rdp_stroke = np.transpose(np.vstack([rdp_x, rdp_y, rdp_t, pen]), [1,0])
    # processed_ink.append(rdp_stroke)
    processed_ink.append([rdp_x.tolist(), rdp_y.tolist(), rdp_t.tolist()])
  return processed_ink


def didi_preprocess(raw_ink, timestep=20):
  raw_ink = size_normalization(raw_ink)
  raw_ink = resample_ink(raw_ink, timestep)
  return raw_ink


for json_file in JSON_FILES:
  i = 0
  counts = collections.defaultdict(int)
  with create_tfrecord_writers(os.path.join(DATA_DIR), json_file.split(".")[0], NUM_TFRECORD_SHARDS) as writers:
    with open(os.path.join(DATA_DIR, json_file)) as f:
      for line in f:
        ink = json.loads(line)
        
        # Randomly (but in reproducible way) define training,validation and test
        # splits if the dataset doesn't do it.
        if "split" not in ink:
          rng = np.random.RandomState(i)
          prob = rng.uniform()
          if prob < 0.75:
            ink["split"] = "train"  # 75% training.
          elif prob > 0.9:
            ink["split"] = "valid"  # 10% validation.
          else:
            ink["split"] = "test"  # 15% test.
        
        if "key" not in ink:
          ink["key"] = str(hash(str(ink["drawing"])))
          ink["label_id"] = ink["key"]
        
        # Ramer resampling.
        ink["rdp_ink"] = sketch_rnn_preprocess(ink["drawing"], rdp_epsilon=2.0)
        ink["drawing"] = didi_preprocess(ink["drawing"], timestep=20)
        
        # dot = get_label_file_contents("dot", ink["label_id"])
        dot = None
        example = ink_to_tfexample(ink, dot)
        
        counts[ink["split"]] += 1
        writers[ink["split"]][pick_output_shard(NUM_TFRECORD_SHARDS)].write(example.SerializeToString())
        
        i += 1
        if i %100 == 0:
          print("# samples ", i)

  print ("Finished writing: %s train: %i valid: %i test: %i" %(json_file, counts["train"], counts["valid"], counts["test"]))