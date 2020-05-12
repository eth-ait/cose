"""Utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def ink_batch_to_strokes(undo_fn, ink_batch, ink_start, seq_len):
  """Converts a batch of strokes (numpy tensor) to list of strokes.

  First reverts the preprocessing steps applied.
  The return value can be passed to visualization functions directly.
  Args:
    undo_fn: a function gets a batch of strokes and reverts preprocessing
      steps such as normalization, first order derivative, etc.
    ink_batch:
    ink_start:
    seq_len:

  Returns:
  """

  ink_batch, seq_len = undo_fn(ink_batch, ink_start, seq_len)
  # y dimension is mirrored.
  ink_batch[:, :, 1] = -1 * ink_batch[:, :, 1]
  # Discard paddings.
  strokes = []
  for i in range(ink_batch.shape[0]):
    strokes.append(ink_batch[i][:seq_len[i]])
  return strokes
  

def padded_to_stroke_list(padded_strokes, undo_preprocessing_fn=None):
  """Converts padded predictions into a list of strokes.

  First extract the strokes by ignoring the padding and then undo the
  pre-processing.
  Args:
    padded_strokes: a dictionary containing the ink data like stroke and pen
      with shape (n_strokes, max_seq_len, ...).
    undo_preprocessing_fn: function to undo preprocessing steps.
  Returns:
  """
  start_pos = padded_strokes.get("start_coord", None)
  seq_len = padded_strokes.get("seq_len", None)
  
  if isinstance(padded_strokes["stroke"], dict):
    xy_stroke = padded_strokes["stroke"]["mu"]
  else:
    xy_stroke = padded_strokes["stroke"]
  
  if isinstance(xy_stroke, tf.Tensor):
    stroke_tensor = tf.concat([xy_stroke, padded_strokes["pen"]], axis=-1)
    ink_batch, seq_len = undo_preprocessing_fn(stroke_tensor, start_pos, seq_len)
    # y dimension is mirrored.
    ink_batch = tf.concat([ink_batch[:, :, 0:1], -1*ink_batch[:, :, 1:2], ink_batch[:, :, 2:3]], axis=-1)
  else:
    stroke_tensor = np.concatenate([xy_stroke, padded_strokes["pen"]], axis=-1)
    ink_batch, seq_len = undo_preprocessing_fn(stroke_tensor, start_pos, seq_len)
    # y dimension is mirrored.
    ink_batch[:, :, 1] = -1*ink_batch[:, :, 1]
    
  # Discard paddings.
  strokes = []
  for i in range(ink_batch.shape[0]):
    strokes.append(ink_batch[i][:seq_len[i]])
  return strokes