import numpy as np
import tensorflow as tf

from common.constants import Constants as C
from smartink.loss.chamfer import chamfer_distance_tf
from smartink.util.utils import dict_tf_to_numpy
from smartink.util.ink import padded_to_stroke_list
from visualization.visualization import InkVisualizer


class MetricEngine(object):
  def __init__(self, undo_preprocessing_fn, metrics=(C.METRIC_L2, C.METRIC_CHAMFER), to_origin=True, ignore_pen=True, ignore_pen_step=True):
    """
    Args:
      undo_preprocessing_fn:
      metrics: list of metrics to be evaluated.
      to_origin: translate strokes to the origin first.
      ignore_pen: whether to ignore the pen dimension or not.
      ignore_pen_step: whether the ignore the last step or not.
    """
    self.undo_preprocessing_fn = undo_preprocessing_fn
    self.metrics = metrics
    self.to_origin = to_origin
    self.ignore_pen = ignore_pen
    self.ignore_pen_step = ignore_pen_step
    self.vis_engine = InkVisualizer(undo_preprocessing_fn, "./", animate=False)
  
  def eval(self, targets, predictions, return_all=True):
    targets_ = list()
    predictions_ = list()
    for t_, p_ in zip(targets, predictions):
      if self.to_origin:
        t_ = t_ - t_[0, :]
        p_ = p_ - p_[0, :]

      if self.ignore_pen and self.ignore_pen_step:
        t_ = t_[:-1, 0:2]
        p_ = p_[:-1, 0:2]
      elif self.ignore_pen:
        t_ = t_[:, 0:2]
        p_ = p_[:, 0:2]
      elif self.ignore_pen_step:
        t_ = t_[:-1]
        p_ = p_[:-1]
        
      targets_.append(t_)
      predictions_.append(p_)
    
    results = dict()
    for metric in self.metrics:
      if metric == C.METRIC_L2:
        res = self.euclidean(targets_, predictions_)
      elif metric == C.METRIC_CHAMFER:
        res = self.chamfer(targets_, predictions_)

      if not return_all:
        res = np.array(res).mean()
      results[metric] = res
    return results

  @classmethod
  def euclidean(cls, targets, predictions):
    return [tf.reduce_sum(input_tensor=tf.sqrt(tf.reduce_sum(input_tensor=tf.square(gt - pred), axis=1))).numpy() for gt, pred in zip(targets, predictions)]
    
  @classmethod
  def chamfer(cls, targets, predictions):
    return [chamfer_distance_tf((gt, pred)).numpy() for gt, pred in zip(targets, predictions)]
  
  def chamfer_distance(self, targets, predictions, return_all=True):
    """Calculates Chamfer distance between targets and predictions.
    
    Keeping for backward compatibility.
    Args:
      targets: List of ground-truth strokes where each stroke has a
        shape of (seq_len, <x,y,pen>).
      predictions: List of predicted strokes where each stroke has a
        shape of (seq_len, <x,y,pen>).
      return_all: return average stroke loss or all values.
    Returns:
    """
    if self.to_origin:
      targets = [t_ - t_[0, :] for t_ in targets]
      predictions = [p_ - p_[0, :] for p_ in predictions]

    if self.ignore_pen and self.ignore_pen_step:
      cd = [chamfer_distance_tf((gt[:-1, 0:2], pred[:-1, 0:2])).numpy() for gt, pred in zip(targets, predictions)]
    elif self.ignore_pen:
      cd = [chamfer_distance_tf((gt[:, 0:2], pred[:, 0:2])).numpy() for gt, pred in zip(targets, predictions)]
    elif self.ignore_pen_step:
      cd = [chamfer_distance_tf((gt[:-1], pred[:-1])).numpy() for gt, pred in zip(targets, predictions)]
    else:
      cd = [chamfer_distance_tf((gt, pred)).numpy() for gt, pred in zip(targets, predictions)]
      
    if return_all:
      return cd
    else:
      return np.array(cd).mean()

  def chamfer_eval_raw(self, target_batch, pred_batch, on_diagram=False,
                       return_all=True, target_i=None, pred_i=None):
    """Chamfer distance with customized for our model outputs.
    
    Gets model outputs directly and handles all the pre- and post-processing.
    It is used for custom stuff.
    
    Args:
      target_batch:
      pred_batch:
      on_diagram:
      return_all:
      target_i:
      pred_i:

    Returns:

    """
  
    target_batch = dict_tf_to_numpy(target_batch)
    gt_strokes = padded_to_stroke_list(target_batch,
                                       self.undo_preprocessing_fn)
    if target_i is not None:
      gt_strokes = gt_strokes[target_i]
      pred_batch[C.INP_START_COORD] = target_batch["start_coord"][target_i]
      pred_batch[C.INP_SEQ_LEN] = target_batch["seq_len"][target_i]
    else:
      pred_batch[C.INP_START_COORD] = target_batch["start_coord"]
      pred_batch[C.INP_SEQ_LEN] = target_batch["seq_len"]
  
    pred_strokes = padded_to_stroke_list(dict_tf_to_numpy(pred_batch),
                                         self.undo_preprocessing_fn)
    if pred_i is not None:
      pred_strokes = pred_strokes[pred_i]
  
    # Chamfer distance on individual strokes.
    cd_stroke = self.chamfer_distance(gt_strokes, pred_strokes,
                                      return_all=return_all)
  
    # Chamfer distance on the entire diagram.
    cd_diagram = None
    if on_diagram:
      gt_diagram = np.vstack(gt_strokes)
      pred_diagram = np.vstack(pred_strokes)
      cd_diagram = self.chamfer_distance([gt_diagram], [pred_diagram],
                                         return_all=return_all)
    return cd_stroke, cd_diagram