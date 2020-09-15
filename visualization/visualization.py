"""Visualization functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import traceback

import matplotlib
from matplotlib import animation
matplotlib.use("agg")
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
from matplotlib import cm
import matplotlib.collections as mcoll
import matplotlib as mpl
import numpy as np

import tensorflow as tf


def get_min_max(values, offset_ratio=0.0):
  min_ = values.min()
  max_ = values.max()
  offset_ = max(abs(min_), abs(max_))
  min_ -= offset_ * offset_ratio
  max_ += offset_ * offset_ratio
  return (min_, max_)


def animate_strokes(strokes, x_borders=None, y_borders=None, colors=None, interval=40):
  """Animates strokes.

  Args:
    strokes: list of strokes.
    x_borders: a tuple of min, max x coordinates.
    y_borders: a tuple of min, max y coordinates.
    colors:
    interval: Delay between frames in milliseconds.

  Returns:
  """
  # First set up the figure, the axes, and the plot element
  all_strokes = np.concatenate(strokes, axis=0)

  # Set plot limits dynamically.
  x_min, x_max = get_min_max(all_strokes[:, 0], 0.1)
  if x_borders:
    x_min = min(y_borders[0], x_min)
    x_max = max(y_borders[1], x_max)
  x_borders = (x_min, x_max)

  y_min, y_max = get_min_max(all_strokes[:, 1], 0.1)
  if y_borders:
    y_min = min(y_borders[0], y_min)
    y_max = max(y_borders[1], y_max)
  y_borders = (y_min, y_max)

  # Set figure size dynamically. Max resolution is 2000 pixels.
  y_range = abs(y_borders[0]) + abs(y_borders[1])
  x_range = abs(x_borders[0]) + abs(x_borders[1])
  base_size = 2
  max_size = 20

  if y_range / x_range > 4 or x_range / y_range > 4:
    x_size = base_size if x_range < y_range else min(
        max_size, base_size * x_range / y_range)
    y_size = base_size if y_range < x_range else min(
        max_size, base_size * y_range / x_range)
  else:
    x_size = max((x_range / (y_range + x_range)) * max_size, base_size)
    y_size = max((y_range / (y_range + x_range)) * max_size, base_size)

  fig, ax = plt.subplots(figsize=(x_size, y_size), tight_layout=True)
  plt.axis("tight")
  plt.axis('off')
  plt.close()
  ax.set_xlim(x_borders)
  ax.set_ylim(y_borders)

  lines = []
  line_borders = []  # Start and end index of a stroke in the entire drawing.
  current_len = 0
  for i, stroke in enumerate(strokes):
    color = colors[i] if colors is not None else mpl.cm.tab20.colors[i%20]
    line_borders.append((current_len, current_len + stroke.shape[0]))
    lines.append(ax.plot([], [], lw=3, color=color)[0])
    current_len += stroke.shape[0] + 1
  num_frames = current_len

  def init():
    for line in lines:
      line.set_data([], [])
    return lines

  def animate(i):
    """Animation function required by FuncAnimation.

    Args:
      i:

    Returns:
    """
    for idx, (line, (line_start,
                     line_end)) in enumerate(zip(lines, line_borders)):
      if i > line_end:
        # All strokes that have been visualized.
        # line.set_data(strokes[idx][:-1, 0], strokes[idx][:-1, 1])
        line.set_data(strokes[idx][:, 0], strokes[idx][:, 1])
      else:
        line.set_data(strokes[idx][0:(i - line_start), 0],
                      strokes[idx][0:(i - line_start), 1])
        break
    return lines

  anim = animation.FuncAnimation(
      fig, animate, init_func=init, frames=num_frames, interval=interval, blit=True)
  # rc('animation', html='jshtml', embed_limit=128)
  # from IPython.display import HTML
  # HTML(anim.to_html5_video(embed_limit=128))
  return anim, fig


def render_strokes(strokes, x_borders=None, y_borders=None, colors=None, marker_size=0, fig=None, ax=None, alpha=1.0, highlight_start=False):
  """Render all strokes at once.

  Args:
    strokes: list of strokes.
    x_borders: a tuple of min, max x coordinates.
    y_borders: a tuple of min, max y coordinates.
    colors:
    marker_size:

  Returns:
  """
  if len(strokes) > 1:
    all_strokes = np.concatenate(strokes, axis=0)
  else:
    all_strokes = strokes[0]

  if all_strokes.shape[0] == 0:
    return None

  # Set plot limits dynamically.
  x_min, x_max = get_min_max(all_strokes[:, 0], 0.1)
  if x_borders:
    # x_min = min(y_borders[0], x_min)
    # x_max = max(y_borders[1], x_max)
    x_min, x_max = x_borders
  x_borders = (x_min, x_max)

  y_min, y_max = get_min_max(all_strokes[:, 1], 0.1)
  if y_borders:
    # y_min = min(y_borders[0], y_min)
    # y_max = max(y_borders[1], y_max)
    y_min, y_max = y_borders
  y_borders = (y_min, y_max)

  # Set figure size dynamically. Max resolution is 2000 pixels.
  y_range = abs(y_borders[0]) + abs(y_borders[1])
  x_range = abs(x_borders[0]) + abs(x_borders[1])
  base_size = 2
  max_size = 20

  if y_range / x_range > 4 or x_range / y_range > 4:
    x_size = base_size if x_range < y_range else min(
        max_size, base_size * x_range / y_range)
    y_size = base_size if y_range < x_range else min(
        max_size, base_size * y_range / x_range)
  else:
    x_size = max((x_range / (y_range + x_range)) * max_size, base_size)
    y_size = max((y_range / (y_range + x_range)) * max_size, base_size)
  
  if fig is None:
    fig, ax = plt.subplots(figsize=(x_size, y_size))
    
  # plt.close()
  plt.axis("tight")
  plt.axis('off')
  ax.set_xlim(x_borders)
  ax.set_ylim(y_borders)

  # Points with pen-up event (i.e., stroke[:, 2] == 1) indicates the end of
  # the stroke.
  for i, stroke in enumerate(strokes):
    color = colors[i] if colors is not None else mpl.cm.tab20.colors[i%20]
    if marker_size > 0:
      # ax.plot(stroke[:-1, 0], stroke[:-1, 1], lw=2, color=color, marker='o', markersize=marker_size)
      ax.plot(stroke[:, 0], stroke[:, 1], lw=3, color=color, marker='o', markersize=marker_size)
    else:
      plt_stroke=ax.plot(stroke[:-1, 0], stroke[:-1, 1], lw=3, color=color, alpha=alpha)
      
      if highlight_start:
        plt.plot(stroke[0, 0], stroke[0, 1], 'ro', lw=3, markersize=12, color=color)
        mean_pos = stroke.mean(0)
        text_x = mean_pos[0]
        text_y = mean_pos[1]
        on_stroke = np.any(np.linalg.norm(stroke[:, 0:2] - mean_pos[np.newaxis, :2], axis=1) < 0.05)
        if on_stroke:
          mean_pos = stroke[:stroke.shape[0] // 3].mean(0)
          text_x = mean_pos[0]
          text_y = mean_pos[1]
          text_x -= (text_y - stroke[0, 1]) / 2.0
          text_y -= (text_x - stroke[0, 0]) / 2.0
        ax.text(text_x, text_y, str(i + 1), fontsize=25, ha='center', va='center', color=plt_stroke[0].get_color())

  # def make_segments(x, y):
  #   """
  #   Create list of line segments from x and y coordinates, in the correct format
  #   for LineCollection: an array of the form numlines x (points per line) x 2 (x
  #   and y) array
  #   """
  #
  #   points = np.array([x, y]).T.reshape(-1, 1, 2)
  #   segments = np.concatenate([points[:-1], points[1:]], axis=1)
  #   return segments
  #
  # cm_subsection = np.linspace(0, 1.0, all_strokes.shape[0])
  # segments = np.concatenate([make_segments(stroke[:, 0], stroke[:, 1]) for stroke in strokes], axis=0)
  # lc = mcoll.LineCollection(segments, array=cm_subsection, cmap=plt.get_cmap('cool'), norm=plt.Normalize(0.0, 1.0),
  #                           linewidth=2, alpha=1)
  # ax = plt.gca()
  # ax.add_collection(lc)
  
  return fig, ax


class InkVisualizer(object):
  """Renders and animates ink samples."""

  def __init__(self, undo_preprocessing, log_dir, animate=False):
    self.undo_preprocessing = undo_preprocessing
    self.log_dir = log_dir
    self.animate = animate
    self.n_rendered_strokes = 100  # All.
    self.marker_size = 0
        
  def vis_ink_sequence(self, sample,
                       save_name,
                       x_borders=None,
                       y_borders=None,
                       colors=None):
    """Visualizes an in sample stored as a sequence of points.

    The strokes are already concatenated.
    Args:
      sample: dict of ink data with stroke and pen with shape (batch,
        max_seq_len, ...).
      save_name:
      x_borders: a tuple of min, max x coordinates.
      y_borders: a tuple of min, max y coordinates.
      colors: list of colors per stroke.

    Returns:
    """
    stroke_sample = np.concatenate([sample["stroke"], sample["pen"]], axis=-1)
    stroke_sample = self.undo_preprocessing(stroke_sample,
                                            sample.get("start_coord", None),
                                            sample["seq_len"])[0]
    stroke_sample[:, :, 1] = -1 * stroke_sample[:, :, 1]
    save_path = os.path.join(self.log_dir, "")

    stroke_list = np.split(stroke_sample[0], np.where(stroke_sample[0, :, 2] == 1)[0] + 1)
    if len(stroke_list) > 1:
      stroke_list = stroke_list[:-1]
    self.render_sample(
        strokes=stroke_list,
        save_path=save_path + save_name,
        n_strokes=self.n_rendered_strokes,
        animate=self.animate,
        x_borders=x_borders,
        y_borders=y_borders,
        colors=colors,
        marker_size=self.marker_size)

  def vis_stroke(self,
                 sample,
                 save_name,
                 num_strokes=None,
                 x_borders=None,
                 y_borders=None,
                 colors=None):
    """Visualizes ink samples stored as a batch of padded strokes.

    First extract the strokes by ignoring the padding and then undo the
    pre-processing.
    Args:
      sample (dict): a diagram sample containing the ink data like stroke and
        pen with shape (batch, max_seq_len, ...).
      save_name:
      num_strokes: # of strokes to be rendered (stroke_list[:num_strokes]).
      x_borders: a tuple of min, max x coordinates.
      y_borders: a tuple of min, max y coordinates.
      colors: list of colors per stroke.

    Returns:
    """
    save_path = os.path.join(self.log_dir, "")
    if isinstance(sample["stroke"], dict):
      xy_stroke = sample["stroke"]["mu"]
    else:
      xy_stroke = sample["stroke"]
  
    stroke_tensor = np.concatenate([xy_stroke, sample["pen"]], axis=-1)
    stroke_list = self.ink_batch_to_strokes(self.undo_preprocessing,
                                            stroke_tensor,
                                            sample.get("start_coord", None),
                                            sample.get("seq_len", None))
    if num_strokes:
      stroke_list = stroke_list[:min(num_strokes, xy_stroke.shape[0])]
  
    self.render_sample(
        strokes=stroke_list,
        save_path=save_path + save_name,
        n_strokes=self.n_rendered_strokes,
        animate=self.animate,
        x_borders=x_borders,
        y_borders=y_borders,
        colors=colors,
        marker_size=self.marker_size)

  def vis_strokes_dict(self,
                       samples,
                       save_name,
                       num_strokes=None,
                       x_borders=None,
                       y_borders=None,
                       colors=None):
    """Visualizes a number of ink samples stored as a batch of padded strokes.

    First extract the strokes by ignoring the padding and then undo the
    pre-processing.
    Args:
      samples: dict of diagram samples where each sample is also a dictionary
        containing the ink data like stroke and pen with shape (batch,
        max_seq_len, ...).
      save_name:
      num_strokes: # of strokes to be rendered (stroke_list[:num_strokes]).
      x_borders: a tuple of min, max x coordinates.
      y_borders: a tuple of min, max y coordinates.
      colors: list of colors per stroke.

    Returns:
    """
    save_path = os.path.join(self.log_dir, "")
    for idx, sample in samples.items():
      if isinstance(sample["stroke"], dict):
        xy_stroke = sample["stroke"]["mu"]
      else:
        xy_stroke = sample["stroke"]
    
      stroke_tensor = np.concatenate([xy_stroke, sample["pen"]], axis=-1)
      stroke_list = self.ink_batch_to_strokes(self.undo_preprocessing,
                                              stroke_tensor,
                                              sample.get("start_coord", None),
                                              sample.get("seq_len", None))
      if num_strokes:
        stroke_list = stroke_list[:min(num_strokes, xy_stroke.shape[0])]
    
      self.render_sample(
          strokes=stroke_list,
          save_path=save_path + str(idx) + "_" + save_name,
          n_strokes=self.n_rendered_strokes,
          animate=self.animate,
          x_borders=x_borders,
          y_borders=y_borders,
          colors=colors,
          marker_size=self.marker_size)

  @classmethod
  def render_sample(cls,
                    strokes,
                    save_path,
                    n_strokes=-1,
                    animate=False,
                    x_borders=None,
                    y_borders=None,
                    colors=None,
                    marker_size=0):
    """Renders an ink sample.

    Args:
      strokes: list of strokes where features are (x,y,pen).
      save_path: output file path without extension.
      n_strokes: # of strokes to be rendered.
      animate: whether to animate it or not.
      x_borders: (x_min, x_max) coordinates of the plot.
      y_borders: (y_min, y_max) coordinates of the plot.
      colors: list of colors per stroke.
      marker_size: determines the point size. If 0, points will not be visible.

    Returns:
    """
    if animate:
      anim, fig = animate_strokes(strokes[:n_strokes], x_borders, y_borders, colors)
      try:
        anim.save(save_path + ".mp4")
        # with tf.gfile.GFile(save_path + ",mp4", "w") as tf_save_path:
        #   anim.save(tf_save_path)
      except Exception:  # pylint: disable=broad-except
        traceback.print_exc()
    else:
      fig, _ = render_strokes(strokes[:n_strokes], x_borders, y_borders, colors, marker_size)

    if fig is not None:
      # fig.savefig(save_path + ".png", format="png")
      with tf.io.gfile.GFile(save_path + ".png", "w") as tf_save_path:
        fig.savefig(tf_save_path, format="png", bbox_inches='tight', dpi=200)
        plt.close()

      # with open(save_path + ".svg", "w") as tf_save_path:
      #   fig.savefig(tf_save_path, format='svg', dpi=300)
      #   plt.close()

  @classmethod
  def tf_ink_batch_to_strokes(cls, undo_fn, ink_batch, ink_start, seq_len):
    """Converts a batch of strokes (tf tensor) to list of strokes.

    First reverts the preprocessing steps.
    The return value can be passed to visualization functions directly.
    Args:
      undo_fn: a function gets a batch of strokes and reverts preprocessing
        steps such as normalization, first order derivative, etc.
      ink_batch (tf.Tensor):
      ink_start (tf.Tensor):
      seq_len:

    Returns:
    """
    ink_batch = undo_fn(ink_batch.numpy(), ink_start.numpy())
    seq_len = seq_len  # .numpy()
    # y dimension is mirrored.
    ink_batch[:, :, 1] = -1 * ink_batch[:, :, 1]
    # Discard paddings.
    strokes = []
    for i in range(ink_batch.shape[0]):
      strokes.append(ink_batch[i][:seq_len[i]])
    return strokes

  @classmethod
  def ink_batch_to_strokes(cls, undo_fn, ink_batch, ink_start, seq_len):
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
