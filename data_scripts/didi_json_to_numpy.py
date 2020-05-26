from __future__ import division

import collections
import json
import os
import functools

import numpy as np
from rdp import rdp

# Setup and settings.

# Settings
# JSON_FILES=["diagrams_wo_text_20200131.ndjson", "diagrams_20200131.ndjson"]
JSON_FILES=["diagrams_wo_text_20200131.ndjson"]
# JSON_FILES=["full_raw_cat.ndjson"]
PROJECT_ID = "digital-ink-diagram-data"
BUCKET_NAME = "digital_ink_diagram_data"
LOCAL_DATA_DIR = "/local/home/emre/Projects/google/data/didi_ndjson/"
DATA_FORMAT = "sketch_rnn"
# DATA_FORMAT = "didi"


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
    xy = np.transpose(np.stack(stroke[0:2]), [1,0])
    rdp_xy = rdp(xy, rdp_epsilon)
    pen = np.zeros_like(rdp_xy[:, 0:1])
    pen[-1, 0] = 1.0
    rdp_stroke = np.concatenate([rdp_xy, pen], axis=1)
    processed_ink.append(rdp_stroke)
  
  all_strokes = np.concatenate(processed_ink, axis=0)
  delta_stroke = np.diff(all_strokes[:, 0:2], axis=0)
  delta_pen = all_strokes[1:, 2:3]
  delta_ink = np.concatenate([delta_stroke, delta_pen], axis=1)
  return delta_ink


def didi_preprocess(raw_ink, timestep=20):
  raw_ink = size_normalization(raw_ink)
  raw_ink = resample_ink(raw_ink, timestep)
  return raw_ink

if DATA_FORMAT == "sketch_rnn":
  preprocessing_fn = functools.partial(sketch_rnn_preprocess, rdp_epsilon=2.0)
elif DATA_FORMAT == "didi":
  preprocessing_fn = functools.partial(didi_preprocess, timestep=20)
else:
  raise Exception()

for json_file in JSON_FILES:
  counts = collections.defaultdict(int)
  splits = dict()
  splits["train"] = []
  splits["valid"] = []
  splits["test"] = []
  i = 0
  
  with open(os.path.join(LOCAL_DATA_DIR, json_file)) as f:
    for line in f:
      ink = json.loads(line)
      ink["drawing"] = preprocessing_fn(ink["drawing"])

      split_ = ink.get("split")
      counts[split_] += 1
      splits[split_].append(ink["drawing"])
      i += 1
      if i % 100 == 0:
        print("# samples ", i)

  print ("Finished writing: %s train: %i valid: %i test: %i" %(json_file, counts["train"], counts["valid"], counts["test"]))
  filename = os.path.join(LOCAL_DATA_DIR, json_file.split(".")[0] + ".npz")
  np.savez_compressed(filename, train=splits["train"], valid=splits["valid"], test=splits["test"])