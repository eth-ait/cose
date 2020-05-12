from __future__ import division

import collections
import contextlib
import io
import json
import os
import random
import statistics
import tensorflow as tf

from googleapiclient.discovery import build
# from google.colab import auth
# from google.colab import files
from googleapiclient.http import MediaIoBaseDownload
# from apiclient import errors

import numpy as np

# Setup and settings.

# Settings
# JSON_FILES=["diagrams_wo_text_20200131.ndjson", "diagrams_20200131.ndjson"]
JSON_FILES=["diagrams_20200131.ndjson"]
PROJECT_ID = "digital-ink-diagram-data"
BUCKET_NAME = "digital_ink_diagram_data"
LOCAL_DATA_DIR = "./"
NUM_TFRECORD_SHARDS = 10

# auth.authenticate_user()

# Creating the service client.
gcs_service = build("storage", "v1")

# Download the data
def download_file_from_gcs(filename):
  directory_name = os.path.join(LOCAL_DATA_DIR, os.path.dirname(filename))
  if not os.path.exists(directory_name):
    os.mkdir(directory_name)
  with open(os.path.join(LOCAL_DATA_DIR, filename), "wb") as f:
    request = gcs_service.objects().get_media(bucket=BUCKET_NAME, object=filename)
    media = MediaIoBaseDownload(f, request)

    done = False
    while not done:
      status, done = media.next_chunk()
      if not done:
        print("Downloading '%s': %-3.0f%%" % (filename, status.progress() * 100))

def get_label_file(type, labelid):
  file_id = os.path.join(type, "%s.%s" % (labelid, type))
  fname = os.path.join(LOCAL_DATA_DIR, file_id)
  if os.path.exists(fname):
    return fname
  download_file_from_gcs(file_id)
  return fname

for json_file in JSON_FILES:
  download_file_from_gcs(json_file)


# This cell converts the file to tf.Record of tf.Example.
# This cell takes long time to run.
def get_label_file_contents(type, labelid):
  get_label_file(type, labelid)
  with open(os.path.join(LOCAL_DATA_DIR, type, "%s.%s" %(labelid, type))) as f:
    return f.read()

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

  max_len = np.array([len(stroke[0]) for stroke in ink["drawing"]]).max()

  strokes = []
  stroke_lengths = []
  for stroke in ink["drawing"]:
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

  features["ink"] = tf.train.Feature(
      float_list=tf.train.FloatList(value=all_strokes.flatten()))
  features["stroke_length"] = tf.train.Feature(
      int64_list=tf.train.Int64List(value=all_stroke_lengths))
  features["shape"] = tf.train.Feature(
      int64_list=tf.train.Int64List(value=all_strokes.shape))
  features["num_strokes"] = tf.train.Feature(
      int64_list=tf.train.Int64List(value=[len(ink["drawing"])]))
  example = tf.train.Example(features=tf.train.Features(feature=features))
  return example

@contextlib.contextmanager
def create_tfrecord_writers(output_file, num_output_shards):
  writers = collections.defaultdict(list)
  for split in ["train", "valid", "test"]:
    for i in range(num_output_shards):
      writers[split].append(
          tf.io.TFRecordWriter("%s-%s-%05i-of-%05i" %
                                      (output_file, split, i, num_output_shards)))
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

for json_file in JSON_FILES:
  counts = collections.defaultdict(int)
  with create_tfrecord_writers(os.path.join(LOCAL_DATA_DIR, json_file + ".tfrecord"), NUM_TFRECORD_SHARDS) as writers:
    with open(os.path.join(LOCAL_DATA_DIR, json_file)) as f:
      for line in f:
        ink = json.loads(line)
        dot = get_label_file_contents("dot", ink["label_id"])
        ink["drawing"] = size_normalization(ink["drawing"])
        ink["drawing"] = resample_ink(ink["drawing"], 20)

        example = ink_to_tfexample(ink, dot)
        counts[ink["split"]] += 1
        writers[ink["split"]][pick_output_shard(NUM_TFRECORD_SHARDS)].write(example.SerializeToString())

  print ("Finished writing: %s train: %i valid: %i test: %i" %(json_file, counts["train"], counts["valid"], counts["test"]))