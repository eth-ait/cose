"""Evaluation engine."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import os
import numpy as np
import tensorflow as tf

from common.constants import Constants as C
from common.logger import GoogleSheetLogger
from smartink.source.eval_metrics import MetricEngine
from smartink.util.utils import AggregateAvg
from smartink.util.utils import dict_tf_to_numpy
from smartink.util.ink import padded_to_stroke_list
from visualization.visualization import InkVisualizer

from smartink.loss.nll import log_likelihood
import matplotlib.pyplot as plt
from visualization.visualization import render_strokes
from visualization.visualization import get_min_max
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
from scipy.stats import multivariate_normal


np.set_printoptions(precision=4, suppress=True)


class EvalEngine(object):
  """Evaluates the given stroke model and visualizes predictions."""

  def __init__(self,
               config,
               eval_dataset,
               embedding_model,
               predictive_model=None,
               glog=True):
    """Restores model parameters.

    Args:
      config:
      eval_dataset: shuffling should be disabled.
      embedding_model:
      predictive_model:
      glog: google sheet entry.

    Raises:
      Exception: if model checkpoint is not found.
    """
    self.config = config
    self.dataset = eval_dataset
    self.embedding_model = embedding_model
    self.predictive_model = predictive_model
    self.model = self.embedding_model if predictive_model is None else predictive_model
    self.gt_len_decoding = True  # Whether to use GT sequence length.
    self.save_video = True  # Whether to use GT sequence length.
    
    self.metrics = MetricEngine(eval_dataset.np_undo_preprocessing,
                                metrics=[C.METRIC_CHAMFER],
                                to_origin=True, ignore_pen=True,
                                ignore_pen_step=True)
    
    self.vis_engine = InkVisualizer(eval_dataset.np_undo_preprocessing,
                                    config.experiment.eval_dir,
                                    animate=False)
    self.log_dir = config.experiment.eval_dir

    if glog and config.get("gdrive", False):
      self.glogger = GoogleSheetLogger(
          tf.io.gfile.GFile(config.gdrive.credential, "r"),
          config.gdrive.workbook, [config.gdrive.sheet + "/test"],
          config.experiment.id,
          static_values={
              "Model ID": config.experiment.id,
              "Model Name": config.experiment.tag,
              "Comment": config.experiment.comment
          })
    else:
      self.glogger = None

    self.model_restored = False
    checkpoint = tf.train.Checkpoint(model=self.model)
    checkpoint_path = tf.train.latest_checkpoint(config.experiment.model_dir)
    if checkpoint_path is None:
      raise Exception("Checkpoint not found.")
    
    print("Loading model " + checkpoint_path)
    checkpoint.restore(checkpoint_path).expect_partial().assert_existing_objects_matched()

  def quantitative_eval(self, num_eval_samples):
    """Evaluates model predictions and loss in eager mode.

    Args:
      num_eval_samples:
    Returns:
    """
    print("======================================")
    print("Running evaluation for {} samples...".format(num_eval_samples))
    print("======================================")
    # Storing evaluation results for visualization.
    eval_loss_summary = AggregateAvg()
    losses = dict()

    start_time = time.perf_counter()
    step = 0
    # Keep track of some statistics
    sample_lengths = list()
    sample_losses = list()
    sample_embeddings = list()
    selected_emb_comps = list()
    try:
      while True:
        input_batch, target_batch = self.dataset.get_next()
        
        if step == num_eval_samples:
          break
        step += 1
        
        if step % 100 == 0:
          print("{} samples evaluated...".format(step))
        
        # tf.keras restores weights only after the first call :(
        if not self.model_restored:
          _ = self.model(inputs=input_batch, training=False)
          self.model_restored = True

        # Get stroke embeddings.
        predictions = self.embedding_model(inputs=input_batch, training=False)
        embeddings = predictions["embedding_sample"]
        target_batch = dict_tf_to_numpy(target_batch)

        ### Decode with original stroke length.
        seq_len = target_batch["seq_len"]
        recon_batch = self.embedding_model.decode_sequence(embeddings,
                                                           seq_len=seq_len)
        
        # Before calculating metrics, convert to 2D positions that's also used
        # for visualization.
        gt_strokes = padded_to_stroke_list(target_batch,
                                           self.dataset.np_undo_preprocessing)
        recon_batch[C.INP_START_COORD] = target_batch["start_coord"]
        recon_batch[C.INP_SEQ_LEN] = target_batch["seq_len"]
        recon_strokes = padded_to_stroke_list(dict_tf_to_numpy(recon_batch),
                                              self.dataset.np_undo_preprocessing)
        
        ### (1) Euclidean and Chamfer distances on the reconstructed strokes.
        res_stroke = self.metrics.eval(gt_strokes, recon_strokes, return_all=True)
        losses["rc_chamfer_stroke"] = res_stroke[C.METRIC_CHAMFER]

        ### (2) Euclidean and Chamfer distances on the reconstructed diagram.
        gt_diagram = np.vstack(gt_strokes)
        recon_diagram = np.vstack(recon_strokes)
        res_diag = self.metrics.eval([gt_diagram], [recon_diagram], return_all=True)
        losses["rc_chamfer_diagram"] = res_diag[C.METRIC_CHAMFER]

        if C.METRIC_L2 in res_stroke:
          losses["rc_l2_stroke"] = res_stroke[C.METRIC_L2]
          losses["rc_l2_diagram"] = res_diag[C.METRIC_L2]
        
        # Evaluating the prediction model.
        if self.predictive_model is not None:
          n_given_strokes = 2  # minimum # of given strokes.
          
          ### (3) Predicted embedding log-likelihood.
          pred_emb, target_emb = self.__predict_embedding_ordered_batch(input_batch, target_batch, embeddings, step, given_strokes=n_given_strokes)
          logli = log_likelihood(target_emb, pred_emb)
          losses["nll_embedding"] = -1*logli[:, 0].numpy()

          ### (4) Chamfer distance of the predicted strokes.
          # Here we consider all GMM components and report the one with the
          # lowest chamfer distance with the ground-truth stroke.
          all_emb_decodings, all_emb_pi = self.__decode_embedding_all_components(pred_emb, target_batch["seq_len"][n_given_strokes:])
          n_components = all_emb_pi.shape[1]
          # Tile start coordinates for every component.
          all_emb_decodings[C.INP_START_COORD] = tf.tile(target_batch["start_coord"][n_given_strokes:], [n_components, 1, 1])
          all_comp_strokes = padded_to_stroke_list(dict_tf_to_numpy(all_emb_decodings), self.dataset.np_undo_preprocessing)
          all_comp_gt = gt_strokes[n_given_strokes:]*n_components
          pred_stroke = self.metrics.eval(all_comp_gt, all_comp_strokes, return_all=True)
          # (n_strokes, n_components)
          all_comp_chamfer = np.transpose(np.reshape(np.array(pred_stroke[C.METRIC_CHAMFER]), [n_components, -1]), [1, 0])
          min_chamfer = np.min(all_comp_chamfer, axis=1)
          min_comp_id = np.argmin(all_comp_chamfer, axis=1)
          
          # Which components are causing the lowest error.
          min_comp_id = np.tile(min_comp_id[:, np.newaxis], [1, n_components])
          sorted_comp_id = np.argsort(all_emb_pi)
          # It is in ascending order.
          ordered_min_comp_id = (n_components - np.argwhere(min_comp_id == sorted_comp_id))[:, 1]

          losses["pred_chamfer_stroke"] = min_chamfer
          selected_emb_comps.extend(ordered_min_comp_id)
        
        eval_loss_summary.add(losses)
        sample_embeddings.extend(embeddings.numpy())
        sample_lengths.extend(seq_len)
        sample_losses.extend(res_stroke[C.METRIC_CHAMFER])
        
    except tf.errors.OutOfRangeError:
      print("Model evaluated on {} samples.".format(step))

    loss_dict, eval_step = eval_loss_summary.summary_and_reset()
    
    print("[RC] Avg stroke CF dist: ", loss_dict["rc_chamfer_stroke"])
    print("[RC] Avg diagram CF dist: ", loss_dict["rc_chamfer_diagram"])

    if "rc_l2_stroke" in loss_dict:
      print("[RC] Avg stroke L2 dist: ", loss_dict["rc_l2_stroke"])
      print("[RC] Avg diagram L2 dist: ", loss_dict["rc_l2_diagram"])
    
    # Plot loss statistics.
    sample_losses = np.array(sample_losses)
    normalized_loss = sample_losses / np.array(sample_lengths)
    plt.scatter(sample_lengths, normalized_loss, s=4)
    plt.title("Stroke Chamfer Loss (normalized by length)")
    plt.xlabel("Stroke Length")
    plt.ylabel("Chamfer Loss")
    plt.savefig(os.path.join(self.log_dir, "chamfer_loss_norm_length_plot.png"), bbox_inches='tight', dpi=200)
    plt.close()

    plt.scatter(sample_lengths, sample_losses, s=4)
    plt.title("Stroke Chamfer Loss")
    plt.xlabel("Stroke Length")
    plt.ylabel("Chamfer Loss")
    plt.savefig(os.path.join(self.log_dir, "chamfer_loss_length_plot.png"), bbox_inches='tight', dpi=200)
    plt.close()
    
    _ = plt.hist(sample_losses, 50, density=True, cumulative=False, log=False)
    plt.savefig(os.path.join(self.log_dir, "chamfer_loss_hist.png"), bbox_inches='tight', dpi=200)
    plt.close()

    if selected_emb_comps:
      left_of_first_bin = 0.5
      right_of_last_bin = n_components + 0.5
      plt.hist(selected_emb_comps, np.arange(left_of_first_bin, right_of_last_bin + 1, 1), density=True, cumulative=False, log=False)
      
      # _ = plt.hist(selected_emb_comps, n_components, density=True, cumulative=False, log=False)
      plt.savefig(os.path.join(self.log_dir, "selected_emb_components.png"),
                  bbox_inches='tight', dpi=200)
      plt.close()
    
    def entropy_hist_dd(x):
      counts = np.histogramdd(x)[0]
      dist = counts/np.sum(counts)
      logs = np.log2(np.where(dist > 0, dist, 1))
      return -np.sum(dist*logs)
    
    def entropy_mvn(x):
      mean_ = np.mean(x, axis=0)
      cov_ = np.cov(x, rowvar=0)
      mvn = multivariate_normal(mean_, cov_)
      return mvn.entropy()

    all_embeddings = np.vstack(sample_embeddings)
    try:
      loss_dict["entropy_hist_dd"] = entropy_hist_dd(all_embeddings)
    except:
      pass
    try:
      loss_dict["entropy_mvn"] = entropy_mvn(all_embeddings)
    except:
      pass

    elapsed_time = (time.perf_counter() - start_time)/eval_step
    print("Elapsed time per sample: {:.4f}".format(elapsed_time))
    if self.glogger:
      self.glogger.update_or_append_row(loss_dict)

  def __predict_embedding_ordered_batch(self, input_batch, target_batch, embeddings, sample_idx, given_strokes=2):
    """Predict all the next embeddings in an ordered fashion for a given sample.
    
    Operates in batch mode for efficiency.
    Args:
      input_batch:
      target_batch:
      embeddings: (1, n_strokes, embedding_size)
      sample_idx:
      given_strokes: minimum # of strokes to be conditioned on.
    Returns:
      Log-likelihood of shape (n_strokes-2) and predicted embeddings of shape
      (n_strokes-2, embedding_size)
    """
    n_strokes = embeddings.numpy().shape[0]
    if n_strokes <= given_strokes:
      return list()
    
    n_predictions = n_strokes - given_strokes

    embeddings = tf.expand_dims(embeddings, axis=0)
    inp_emb = tf.tile(embeddings, [n_predictions, 1, 1])[:, :-1]
    mask_ = tf.linalg.band_part(tf.ones((n_strokes, n_strokes-1)), -1, 0)
    mask_ = mask_[given_strokes-1:-1]
    inp_emb *= tf.expand_dims(mask_, axis=-1)

    start_pos = tf.tile(tf.transpose(a=input_batch[C.INP_START_COORD], perm=[1,0,2]), [n_predictions,1,1])[:, :-1]
    target_pos = input_batch[C.INP_START_COORD][given_strokes:]
    target_emb = embeddings[0, given_strokes:]

    out_ = self.model.predict_embedding_ar(inp_emb,
                                           inp_pos=start_pos,
                                           target_pos=target_pos,
                                           seq_len=tf.range(given_strokes, n_strokes))
    return out_, target_emb

  def __decode_embedding_all_components(self, pred_embeddings, stroke_len):
    """Decodes embedding samples drawn from all components of a GMM.

    If the embedding output model is not a GMM, then it is not required.
    Args:
      pred_embeddings (dict): containing the predicted embedding model with keys
        mu, sigma and pi corresponding to a GMM model. Each has a shape of
        (batch_size, n_components*embedding_size)
      stroke_len: length of the decoded strokes.
    Returns:
        decoded stroke dict with outputs of shape(n_components*batch_size, 3)
        embedding GMM pi values of shape (batch_size, n_components)
    """
    tmp = self.model.predictive_model.output_layer.draw_sample_every_component(pred_embeddings, greedy=True)
    emb_samples = tmp[0][:, 0, :, :]
    emb_pis = tmp[1][:, 0, :].numpy()
    n_components = emb_pis.shape[1]
    
    # emb_samples = tf.reshape(emb_samples, [-1, tf.shape(emb_samples)[-1]])
    emb_samples = tf.reshape(tf.transpose(a=emb_samples, perm=[1, 0, 2]), [-1, tf.shape(input=emb_samples)[-1]])
    # stroke_lens = tf_repeat0(stroke_len, n_components)
    stroke_lens = np.tile(stroke_len, [n_components])

    decoded_batch = self.embedding_model.decode_sequence(emb_samples,
                                                         seq_len=stroke_lens)
    return decoded_batch, emb_pis
    
  
  def qualitative_eval(self, sample_ids):
    print("======================")
    print("Visualizing results...")
    print("======================")

    losses = dict()
    start_time = time.perf_counter()
    for idx in range(1, max(sample_ids) + 1):
      input_batch, target_batch = self.dataset.get_next()

      # tf.keras restores weights only after the first call :(
      if not self.model_restored:
        _ = self.model(inputs=input_batch, training=False)
        self.model_restored = True

      if idx not in sample_ids:
        continue
      
      print("Visualizing results for sample {}...".format(idx))
      
      # Create a directory for each sample.
      out_path_sample = os.path.join(self.config.experiment.eval_dir, "sample_" + str(idx))
      if not os.path.exists(out_path_sample):
        os.mkdir(out_path_sample)

      self.vis_engine.log_dir = out_path_sample
      
      # (1) Visualize real sample.
      target_batch = dict_tf_to_numpy(target_batch)
      n_strokes = target_batch["num_strokes"][0]
      self.vis_engine.vis_stroke(target_batch, save_name="{}_gt".format(idx))
      
      # Get stroke embeddings.
      forward_pass = self.embedding_model(inputs=input_batch, training=False)
      embeddings = forward_pass["embedding_sample"]
      
      # (2) Decode with original stroke length.
      seq_len = target_batch["seq_len"]
      if not self.gt_len_decoding:
        seq_len = np.array([50]*n_strokes)
      decoded_batch = self.embedding_model.decode_sequence(embeddings,
                                                           seq_len=seq_len)
      
      decoded_batch[C.INP_START_COORD] = target_batch["start_coord"]
      self.vis_engine.vis_stroke(decoded_batch, save_name="{}_decoded_original_len".format(idx))
      
      # Log per sample loss: chamfer distance on the reconstructed strokes.
      # Not applicable if we are not using the ground-truth sequence length.
      if self.gt_len_decoding:
        gt_strokes = padded_to_stroke_list(target_batch,
                                           self.dataset.np_undo_preprocessing)
        recon_strokes = padded_to_stroke_list(dict_tf_to_numpy(decoded_batch),
                                              self.dataset.np_undo_preprocessing)
        
        res_stroke = self.metrics.eval(gt_strokes, recon_strokes, return_all=False)
        losses["{}_stroke".format(idx)] = res_stroke[C.METRIC_CHAMFER]
  
        ### (2) Euclidean and Chamfer distances on the reconstructed diagram.
        gt_diagram = np.vstack(gt_strokes)
        recon_diagram = np.vstack(recon_strokes)
        res_diag = self.metrics.eval([gt_diagram], [recon_diagram],
                                     return_all=False)
        losses["{}_diagram".format(idx)] = res_diag[C.METRIC_CHAMFER]
      
      ### Evaluate the predictive model.
      if self.predictive_model is None:
        continue
      
      # Convert batch of strokes into a diagram sample.
      embeddings = self.model.batch_stroke_to_diagram(embeddings,
                                                      input_batch[C.INP_NUM_STROKE])
      
      # (3) Predict a stroke in leave-one-out (loo) fashion (i.e., given the rest)
      # self.__predict_loo(input_batch, target_batch, embeddings, idx)

      # (4) Predict the next stroke given only the strokes so far.
      # self.__predict_ordered(input_batch, target_batch, embeddings, idx)

      # (5) Auto-regressive prediction with random embeddings.
      self.__predict_ar(input_batch, target_batch, embeddings, idx)

      # (6) Auto-regressive prediction with the best embedding.
      self.__predict_ar_best_embedding(input_batch, target_batch, embeddings, idx)

      # (7) Predict the next stroke given a predefined set of strokes.
      # self.__predict_random(input_batch, target_batch, embeddings, idx)

      # (8) Predict the next stroke given a predefined set of strokes.
      if self.model.position_model is not None:
        self.__predict_position_ar(input_batch, target_batch, embeddings, idx)

    elapsed_time = (time.perf_counter() - start_time) / len(sample_ids)
    print("Elapsed time per sample: {:.4f}".format(elapsed_time))
    
    if self.glogger and losses:
      self.glogger.update_or_append_row(losses)
      
  def __predict_loo(self, input_batch, target_batch, embeddings, sample_idx):
    # Predict the next stroke given all remaining strokes in leave-one-out (loo)
    # fashion.
    n_strokes = target_batch["num_strokes"][0]
    for stroke_i in range(n_strokes):
      out_, target = self.model.predict_embedding(embeddings,
                                                  target_idx=np.array([stroke_i], dtype=np.int32),
                                                  seq_lens=input_batch[C.INP_NUM_STROKE],
                                                  start_positions=input_batch[C.INP_START_COORD],
                                                  input_type="leave_one_out")
      emb_ = np.copy(embeddings[0].numpy())
      emb_[stroke_i] = out_["embedding_sample"]

      seq_len = target_batch["seq_len"]
      if not self.gt_len_decoding:
        seq_len = np.array([50]*n_strokes)
        
      predicted_batch = self.embedding_model.decode_sequence(emb_,
                                                             seq_len=seq_len)
      predicted_batch[C.INP_START_COORD] = target_batch["start_coord"]

      colors = ["k"]*n_strokes
      colors[stroke_i] = "r"
      self.vis_engine.vis_stroke(predicted_batch, save_name="{}_predicted_loo_s{}".format(sample_idx, str(stroke_i)), colors=colors)
    
  def __predict_ordered(self, input_batch, target_batch, embeddings, sample_idx):
    # Predict the next stroke given only the strokes so far.
    n_strokes = target_batch["num_strokes"][0]
    given_strokes = 1
    for stroke_i in range(given_strokes, n_strokes):
      out_, target = self.model.predict_embedding(embeddings,
                                                  target_idx=np.array([stroke_i], dtype=np.int32),
                                                  seq_lens=input_batch[C.INP_NUM_STROKE],
                                                  start_positions=input_batch[C.INP_START_COORD],
                                                  input_type="ordered")
      emb_ = np.copy(embeddings[0].numpy()[:stroke_i+1])
      emb_[stroke_i] = out_["embedding_sample"]

      seq_len = target_batch["seq_len"][:stroke_i + 1]
      if not self.gt_len_decoding:
        seq_len = np.array([50]*(stroke_i + 1))
      
      predicted_batch = self.embedding_model.decode_sequence(emb_,
                                                             seq_len=seq_len)
      predicted_batch[C.INP_START_COORD] = target_batch["start_coord"][:stroke_i + 1]
      
      colors = ["k"]*(stroke_i + 1)
      colors[stroke_i] = "r"
      self.vis_engine.vis_stroke(predicted_batch, save_name="{}_predicted_ordered_s{}".format(sample_idx, str(stroke_i)), colors=colors)
      
  def __predict_ar(self, input_batch, target_batch, embeddings, sample_idx):
    """Predict the next embedding randomly and decode.

      Draw an embedding sample from the predicted embedding output model.
      """
    # Auto-regressive prediction.
    n_strokes = target_batch["num_strokes"][0]
    context_ids = 2
    context_embeddings = embeddings[:, :context_ids]
    start_positions = tf.transpose(a=input_batch[C.INP_START_COORD], perm=[1, 0, 2])
    for stroke_i in range(context_ids, n_strokes):
      out_ = self.model.predict_embedding_ar(context_embeddings,
                                            inp_pos=start_positions[:, :stroke_i],
                                            target_pos=start_positions[:, stroke_i:stroke_i + 1],
                                            greedy=False)
      context_embeddings = tf.concat([context_embeddings, tf.expand_dims(out_["embedding_sample"], axis=0)], axis=1)
      emb_ = context_embeddings[0].numpy()

      seq_len = target_batch["seq_len"][:stroke_i + 1]
      if not self.gt_len_decoding:
        seq_len = np.array([50]*(stroke_i + 1))
        
      predicted_batch = self.embedding_model.decode_sequence(emb_,
                                                             seq_len=seq_len)
      predicted_batch[C.INP_START_COORD] = target_batch["start_coord"][:stroke_i+1]

      colors = ["k"]*(stroke_i + 1)
      colors[stroke_i] = "r"
      self.vis_engine.vis_stroke(predicted_batch, save_name="{}_ar_predicted_ordered_s{}".format(sample_idx, str(stroke_i)), colors=colors)
    
    # Animate AR predictions.
    colors = ["r"]*(stroke_i + 1)
    for i in range(context_ids):
      colors[i] = "k"
      
    if self.save_video:
      self.vis_engine.animate = True
      self.vis_engine.vis_stroke(predicted_batch, save_name="{}_ar_predicted_ordered_animation_s{}".format(sample_idx, str(stroke_i)), colors=colors)
      self.vis_engine.animate = False
      
  def __predict_ar_best_embedding(self, input_batch, target_batch, embeddings, sample_idx):
    """Predict the next embedding and identify the one with lowest error.
    
    It requires decoding embedding samples from all gmm components and comparing
    with the ground-truth. Hence, it is using additional GT information to
    identify the most accurate embedding.
    """
    # Auto-regressive prediction.
    n_strokes = target_batch["num_strokes"][0]
    n_given_strokes = 2
    context_embeddings = embeddings[:, :n_given_strokes]
    start_positions = tf.transpose(a=input_batch[C.INP_START_COORD], perm=[1, 0, 2])
    gt_strokes = padded_to_stroke_list(target_batch,
                                       self.dataset.np_undo_preprocessing)
    for stroke_i in range(n_given_strokes, n_strokes):
      pred_emb = self.model.predict_embedding_ar(context_embeddings,
                                                 inp_pos=start_positions[:, :stroke_i],
                                                 target_pos=start_positions[:, stroke_i:stroke_i + 1])
      # Decode all modes.
      all_emb_decodings, all_emb_pi = self.__decode_embedding_all_components(pred_emb, target_batch["seq_len"][stroke_i:stroke_i + 1])
      n_components = all_emb_pi.shape[1]
      # Tile start coordinates for every component.
      all_emb_decodings[C.INP_START_COORD] = tf.tile(target_batch["start_coord"][stroke_i:stroke_i + 1], [n_components, 1, 1])
      all_comp_strokes = padded_to_stroke_list(dict_tf_to_numpy(all_emb_decodings), self.dataset.np_undo_preprocessing)
      all_comp_gt = gt_strokes[stroke_i:stroke_i + 1]*n_components
      pred_stroke = self.metrics.eval(all_comp_gt, all_comp_strokes, return_all=True)
      # (n_strokes, n_components)
      all_comp_chamfer = np.transpose(np.reshape(np.array(pred_stroke[C.METRIC_CHAMFER]), [n_components, -1]), [1, 0])
      min_chamfer = np.min(all_comp_chamfer, axis=1)
      min_comp_id = np.argmin(all_comp_chamfer, axis=1)

      embedding_sample, _ = self.model.predictive_model.output_layer.draw_sample_from_nth(pred_emb, n=min_comp_id[0], greedy=True)
      context_embeddings = tf.concat([context_embeddings, embedding_sample], axis=1)
      emb_ = context_embeddings[0].numpy()

      seq_len = target_batch["seq_len"][:stroke_i + 1]
      if not self.gt_len_decoding:
        seq_len = np.array([50]*(stroke_i + 1))
        
      predicted_batch = self.embedding_model.decode_sequence(emb_,
                                                        seq_len=seq_len)
      predicted_batch[C.INP_START_COORD] = target_batch["start_coord"][:stroke_i+1]

      colors = ["k"]*(stroke_i + 1)
      colors[stroke_i] = "r"
      self.vis_engine.vis_stroke(predicted_batch, save_name="{}_best_ar_predicted_ordered_s{}".format(sample_idx, str(stroke_i)), colors=colors)
    
    # Animate AR predictions.
    colors = ["r"]*(stroke_i + 1)
    for i in range(n_given_strokes):
      colors[i] = "k"
      
    if self.save_video:
      self.vis_engine.animate = True
      self.vis_engine.vis_stroke(predicted_batch, save_name="{}_best_ar_predicted_ordered_animation_s{}".format(sample_idx, str(stroke_i)), colors=colors)
      self.vis_engine.animate = False

  def __predict_random(self, input_batch, target_batch, embeddings, sample_idx):
    # Predict the next stroke given a predefined set of strokes.
    seeds = list(range(20))
    num_samples = 10  # number of random input/target configurations.
    n_strokes = target_batch["num_strokes"][0]
    indices = list(range(n_strokes))
    for i in range(num_samples):
      rng = np.random.RandomState(seeds[i])
      ti = rng.randint(low=1, high=n_strokes, size=1)[0]
      rand_indices = rng.permutation(indices)
      target_idx = rand_indices[ti]
      input_indices = rand_indices[:ti].tolist()

      inp_lens = target_batch["seq_len"][input_indices]
      target_len = target_batch["seq_len"][target_idx:target_idx + 1]
      if not self.gt_len_decoding:
        inp_lens = np.array([50]*len(input_indices))
        target_len = np.array([50])
        
      input_sidx = input_indices.copy()
      pred_n_strokes = len(input_sidx)
      out_, target = self.model.predict_embedding(embeddings,
                                                 target_idx=np.array([target_idx], dtype=np.int32),
                                                 input_idx=np.array([input_sidx], dtype=np.int32),
                                                 seq_lens=input_batch[C.INP_NUM_STROKE],
                                                 start_positions=input_batch[C.INP_START_COORD],
                                                 input_type="ordered")
      input_sidx.append(target_idx)
      emb_ = np.copy(embeddings[0].numpy()[input_sidx])
      emb_[-1] = out_["embedding_sample"]

      seq_len = np.concatenate([inp_lens, target_len], axis=0)
      predicted_batch = self.embedding_model.decode_sequence(emb_,
                                                             seq_len=seq_len)
      predicted_batch[C.INP_START_COORD] = target_batch["start_coord"][input_sidx]
      
      colors = ["k"]*(pred_n_strokes + 1)
      colors[-1] = "r"
      given = str(sorted(input_indices))[1:-1].replace(", ", "_")
      self.vis_engine.vis_stroke(predicted_batch,
                                 save_name="{}_predicted_{}_given_{}".format(sample_idx, target_idx, given),
                                 colors=colors)
      
  def __predict_position_ar(self, input_batch, target_batch, embeddings, sample_idx):
    # Set plot limits by using the ground-truth canvas.
    gt_strokes = padded_to_stroke_list(dict_tf_to_numpy(target_batch),
                                       self.dataset.np_undo_preprocessing)
    all_strokes = np.concatenate(gt_strokes, axis=0)
    x_min, x_max = get_min_max(all_strokes[:, 0], 0.3)
    y_min, y_max = get_min_max(all_strokes[:, 1], 0.3)
  
    n_bins = 50
    x, y = np.meshgrid(np.linspace(x_min - 0.5, x_max + 0.5, n_bins),
                       np.linspace(-y_max - 0.5, -y_min + 0.5, n_bins))
  
    xy = np.empty(x.shape + (2,))
    xy[:, :, 0] = x
    xy[:, :, 1] = y
    xy = np.reshape(xy, [-1, 2]).astype(np.float32)
    
    # Auto-regressive prediction.
    n_strokes = target_batch["num_strokes"][0]
    if n_strokes < 10:
      n_strokes += 2
    context_ids = 2
    context_embeddings = embeddings[:, :context_ids]
    # start_positions = tf.transpose(input_batch[C.INP_START_COORD], [1, 0, 2])
    start_positions = np.transpose(target_batch[C.INP_START_COORD], [1, 0, 2])
    ar_start_pos = [start_positions[:, 0:1], start_positions[:, 1:2]]
    for stroke_i in range(context_ids, n_strokes):
      input_pos = np.concatenate(ar_start_pos[:stroke_i], axis=1)
      pos_ = self.model.predict_position_ar(context_embeddings,
                                            inp_pos=input_pos)
      
      target_pos = np.expand_dims(pos_["position_sample"].numpy(), axis=0)
      ar_start_pos.append(target_pos)
      
      out_ = self.model.predict_embedding_ar(context_embeddings,
                                            inp_pos=input_pos,
                                            target_pos=target_pos)
      context_embeddings = tf.concat([context_embeddings, tf.expand_dims(out_["embedding_sample"], axis=0)], axis=1)
      emb_ = context_embeddings[0].numpy()

      # seq_len = target_batch["seq_len"][:stroke_i + 1]
      seq_len = np.array([50]*(stroke_i + 1))
      if not self.gt_len_decoding:
        seq_len = np.array([50]*(stroke_i + 1))
        
      predicted_batch = self.embedding_model.decode_sequence(emb_,
                                                        seq_len=seq_len)
      predicted_batch[C.INP_START_COORD] = np.transpose(np.concatenate(ar_start_pos[:stroke_i+1], axis=1), [1,0,2])

      colors = ["k"]*(stroke_i + 1)
      colors[stroke_i] = "b"
      
      ### Plot heatmap
      # Render strokes.
      predicted_strokes = padded_to_stroke_list(dict_tf_to_numpy(predicted_batch),
                                                self.dataset.np_undo_preprocessing)
      fig, ax = render_strokes(predicted_strokes, colors=colors, x_borders=(x_min, x_max), y_borders=(y_min, y_max))
      
      # Calculate position densities.
      logli = log_likelihood(xy, pos_)
      # if C.PI in pos_:
      #   logli = logli_gmm_logsumexp(xy, pos_[C.MU], pos_[C.SIGMA], pos_[C.PI])
      # else:
      #   logli = logli_normal_diagonal(xy, pos_[C.MU], pos_[C.SIGMA])
      probs = np.reshape(logli.numpy(), [n_bins, n_bins])

      # Customize the colormap by adding some alpha.
      cmap = pl.cm.OrRd
      my_cmap = cmap(np.arange(cmap.N))
      my_cmap[:, -1] = np.linspace(0, 0.8, cmap.N)  # Set alpha
      my_cmap = ListedColormap(my_cmap)        # Create new colormap
      
      plt.contourf(x, -y, probs, cmap=my_cmap)
      plt.plot(target_pos[0, 0, 0], -target_pos[0, 0, 1], 'ro', color="k")
      plt.colorbar()
      fig.savefig(os.path.join(self.vis_engine.log_dir, "{}_pos_ar_heatmap_ordered_s{}.png".format(sample_idx, str(stroke_i))), format="png", bbox_inches='tight', dpi=200)
      plt.close()
      
    # Animate AR predictions.
    colors = ["r"]*(stroke_i + 1)
    for i in range(context_ids):
      colors[i] = "k"

    if self.save_video:
      self.vis_engine.animate = True
      self.vis_engine.vis_stroke(predicted_batch, save_name="{}_pos_ar_predicted_ordered_animation_s{}".format(sample_idx, str(stroke_i)), colors=colors)
      self.vis_engine.animate = False