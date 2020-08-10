"""Evaluation engine."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from common.constants import Constants as C
from common.logger import GoogleSheetLogger
from smartink.source.eval_metrics import MetricEngine
from smartink.util.utils import AggregateAvg
from smartink.util.utils import dict_tf_to_numpy
from smartink.util.ink import padded_to_stroke_list
from visualization.visualization import InkVisualizer

from smartink.loss.nll import log_likelihood
from visualization.visualization import render_strokes
from visualization.visualization import get_min_max

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.colors as colors
from sklearn.cluster import SpectralClustering, KMeans
from sklearn import metrics


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
    self.decoded_length = 50  # 40 mean sequence-length
    self.emb_greedy = False
    
    # Rendering options. Mostly cosmetics.
    self.save_video = False
    self.render_initial_point = False
    self.render_position_heatmap = False
    self.render_binary_colors = False  # Black context, red prediction.
    self.prediction_color = "#c3090aff"  # Prediction color (red).
    
    self.embedding_analysis = True
    self.reconstruction_analysis = True
    self.prediction_analysis = True
    self.loss_plots = False
    
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
    else:
      print("Loading model " + checkpoint_path)
    checkpoint.restore(checkpoint_path).expect_partial()

  @tf.function
  def tf_decoding_fn(self, fn, embeddings, seq_len):
    return fn(embeddings, seq_len)
  
  def embedding_eval(self, embedding_samples=None, glog_entry=False, metric="sqeuclidean"):
    """Conducts a number of analysis on the embedding samples.
    
    Args:
      embedding_samples: numpy matrix of shape (n_samples, latent_units)
      glog_entry:
      metric:
    Returns:
    """
    eval_results = dict()

    if embedding_samples is None:
      if os.path.exists(os.path.join(self.log_dir, "test_gt_embeddings.npy")):
        print("Loading embedding samples from " + os.path.join(self.log_dir, "test_gt_embeddings.npy"))
        embedding_samples = np.load(os.path.join(self.log_dir, "test_gt_embeddings.npy"))
      else:
        print("Calculating embeddings from scratch...")
        try:
          embedding_container = list()
          step = 0
          num_eval_samples = np.inf
          while True:
            input_batch, target_batch = self.dataset.get_next()
            
            if step > num_eval_samples:
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
            embedding_container.extend(predictions["embedding_sample"].numpy())
            
        except tf.errors.OutOfRangeError:
          print("Model evaluated on {} samples.".format(step))
          embedding_samples = np.vstack(embedding_container)
          np.save(os.path.join(self.log_dir, "test_gt_embeddings"), embedding_samples)

    n_batches = 10
    n_samples = embedding_samples.shape[0]
    glog_spectral_name_ = "SC_{}"
    glog_kmeans_name_ = "KM_{}"
    
    # Calculate silhouette coefficient.
    # Clustering takes a lot of time. Instead of running on the entire test
    # split, we run in a cross-validation fashion. Randomly split the test set
    # into 10 splits and report mean and std as well.
    n_samples = (n_samples // n_batches)*n_batches
    emb_copy = embedding_samples.copy()
    np.random.seed(1)
    np.random.shuffle(emb_copy)
    emb_copy = emb_copy[:n_samples]
    
    # Normalize embedding vectors so that Euclidean metric in KMeans and
    # Spectral clustering behave like scaled cosine distance.
    # Derivation: https://stats.stackexchange.com/a/299016/41958
    if metric == "cosine":
      norms = np.linalg.norm(emb_copy, axis=1)
      emb_copy = emb_copy / norms[:, np.newaxis]
      glog_spectral_name_ = "SC_cos_{}"
      glog_kmeans_name_ = "KM_cos_{}"
    
    cluster_splits = np.split(emb_copy, n_batches)
    
    print("Calculating Silhouette Coefficient Metric with KMeans on {} samples...".format(n_samples))
    n_clusters = [5, 10, 15, 20, 25]
    for n_cluster in n_clusters:
      try:
        kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(emb_copy)
        si_kmeans = metrics.silhouette_score(emb_copy, kmeans.labels_, metric=metric)
        eval_results[glog_kmeans_name_.format(n_cluster)] = si_kmeans
        print(glog_kmeans_name_.format(n_cluster) + ": " + str(si_kmeans))
      except:
        pass

    n_clusters = [10, 15, 20, 25]  # Spectral clustering sometimes never terminates with 5.
    print("Calculating Silhouette Coefficient Metric with Spectral Clustering on {} samples...".format(n_samples))
    for n_cluster in n_clusters:
      vals_spectral = list()
      for i in range(n_batches):
        try:
          spect = SpectralClustering(n_clusters=n_cluster, assign_labels="discretize", n_jobs=8).fit(cluster_splits[i])
          si_spect = metrics.silhouette_score(cluster_splits[i], spect.labels_, metric=metric)
          vals_spectral.append(si_spect)
          print("Batch {}: {}".format(i, si_spect))
        except:
          pass

      vals_spectral = np.array(vals_spectral)
      eval_results[glog_spectral_name_.format(n_cluster)] = vals_spectral.mean()
      eval_results[glog_spectral_name_.format(n_cluster) + "_std"] = vals_spectral.std()
      print(glog_spectral_name_.format(n_cluster) + ": " + str(vals_spectral.mean()))
      print(glog_spectral_name_.format(n_cluster) + "_std: " + str(vals_spectral.std()))

    # Calculate entropy and KLD or the fitted latent distribution.
    def entropy_hist_dd(x):
      counts = np.histogramdd(x)[0]
      dist = counts/np.sum(counts)
      logs = np.log2(np.where(dist > 0, dist, 1))
      return -np.sum(dist*logs)

    try:
      eval_results["entropy_hist_dd"] = entropy_hist_dd(embedding_samples)
    except:
      pass

    try:
      agg_mean = np.mean(embedding_samples, axis=0)
      agg_cov = np.cov(embedding_samples, rowvar=0)
      agg_q_mvn = tfp.distributions.MultivariateNormalFullCovariance(tf.convert_to_tensor(agg_mean, dtype=tf.float32), tf.convert_to_tensor(agg_cov, dtype=tf.float32))
      prior = tfp.distributions.MultivariateNormalDiag(tf.zeros([embedding_samples.shape[1]]), tf.ones([embedding_samples.shape[1]]))
      eval_results["entropy_mvn"] = agg_q_mvn.entropy().numpy()
      eval_results["kld_q_p"] = agg_q_mvn.kl_divergence(prior).numpy()
      eval_results["kld_p_q"] = prior.kl_divergence(agg_q_mvn).numpy()
    except:
      pass

    if glog_entry and self.glogger:
      self.glogger.update_or_append_row(eval_results)
    return eval_results
  
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
    selected_predicted_emb = list()
    selected_emb_comps = list()
    try:
      while True:
        input_batch, target_batch = self.dataset.get_next()
        
        if step > num_eval_samples:
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
        if self.reconstruction_analysis:
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
        if self.prediction_analysis and self.predictive_model is not None:
          n_given_strokes = 2  # minimum # of given strokes.
          
          ### (3) Predicted embedding log-likelihood.
          pred_emb, target_emb = self.__predict_embedding_ordered_batch(input_batch, target_batch, embeddings, step, given_strokes=n_given_strokes)
          logli = log_likelihood(target_emb, pred_emb)
          losses["nll_embedding"] = -1*logli[:, 0].numpy()

          ### (4) Chamfer distance of the predicted strokes.
          # Here we consider all GMM components and report the one with the
          # lowest chamfer distance with the ground-truth stroke.
          all_emb_decodings, all_emb_pi, all_emb_samples = self.__decode_embedding_all_components(pred_emb, target_batch["seq_len"][n_given_strokes:])
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
          
          best_embedding_idx = tf.stack([tf.range(tf.shape(all_emb_samples)[0]), min_comp_id], axis=-1)
          best_embeddings = tf.gather_nd(all_emb_samples, best_embedding_idx)
          
          # Which components are causing the lowest error.
          min_comp_id = np.tile(min_comp_id[:, np.newaxis], [1, n_components])
          sorted_comp_id = np.argsort(all_emb_pi)
          # It is in ascending order.
          ordered_min_comp_id = (n_components - np.argwhere(min_comp_id == sorted_comp_id))[:, 1]

          losses["pred_chamfer_stroke"] = min_chamfer
          selected_emb_comps.extend(ordered_min_comp_id)
          selected_predicted_emb.extend(best_embeddings.numpy())

        sample_embeddings.extend(embeddings.numpy())
        if self.reconstruction_analysis:
          sample_lengths.extend(seq_len)
          sample_losses.extend(res_stroke[C.METRIC_CHAMFER])
        if losses:
          eval_loss_summary.add(losses)
        
    except tf.errors.OutOfRangeError:
      print("Model evaluated on {} samples.".format(step))

    loss_dict, eval_step = eval_loss_summary.summary_and_reset()
    
    if self.reconstruction_analysis:
      print("[Stroke Reconstruction] Avg stroke reconstruction CF dist: {:.4f}".format(loss_dict["rc_chamfer_stroke"]))

    if self.prediction_analysis and self.predictive_model is not None:
      print("[Stroke Prediction] Avg stroke prediction CF dist: {:.4f}".format(loss_dict["pred_chamfer_stroke"]))
    
    # Plot loss statistics.
    if self.loss_plots and self.reconstruction_analysis:
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
    
    if selected_predicted_emb:
      all_pred_embeddings = np.vstack(selected_predicted_emb)
      np.save(os.path.join(self.log_dir, "test_best_predicted_embeddings"), all_pred_embeddings)

    all_embeddings = np.vstack(sample_embeddings)
    np.save(os.path.join(self.log_dir, "test_gt_embeddings"), all_embeddings)
    
    # if self.embedding_analysis:
    #   emb_results = self.embedding_eval(all_embeddings)
    #   loss_dict.update(emb_results)
      
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

    start_positions = tf.tile(tf.transpose(a=input_batch[C.INP_START_COORD], perm=[1,0,2]), [n_predictions,1,1])[:, :-1]
    if self.predictive_model.end_positions:
      end_positions = tf.tile(tf.transpose(a=input_batch[C.INP_END_COORD], perm=[1,0,2]), [n_predictions,1,1])[:, :-1]
      context_pos = tf.concat([start_positions, end_positions], axis=-1)
    else:
      context_pos = start_positions
    
    target_pos = input_batch[C.INP_START_COORD][given_strokes:]
    target_emb = embeddings[0, given_strokes:]

    out_ = self.model.predict_embedding_ar(inp_emb,
                                           inp_pos=context_pos,
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
    tmp = self.model.predictive_model.output_layer.draw_sample_every_component(pred_embeddings, greedy=self.emb_greedy)
    emb_samples = tmp[0][:, 0, :, :]
    emb_pis = tmp[1][:, 0, :].numpy()
    n_components = emb_pis.shape[1]
    
    # emb_samples = tf.reshape(emb_samples, [-1, tf.shape(emb_samples)[-1]])
    emb_samples_compwise = tf.reshape(tf.transpose(a=emb_samples, perm=[1, 0, 2]), [-1, tf.shape(input=emb_samples)[-1]])
    # stroke_lens = tf_repeat0(stroke_len, n_components)
    stroke_lens = np.tile(stroke_len, [n_components])

    decoded_batch = self.embedding_model.decode_sequence(emb_samples_compwise,
                                                         seq_len=stroke_lens)
    return decoded_batch, emb_pis, emb_samples
    
  
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
        seq_len = np.array([self.decoded_length]*n_strokes)
      decoded_batch = self.embedding_model.decode_sequence(embeddings,
                                                           seq_len=seq_len)
      
      decoded_batch[C.INP_START_COORD] = target_batch["start_coord"]
      self.vis_engine.vis_stroke(decoded_batch, save_name="{}_decoded_original_len".format(idx))

      gt_strokes = padded_to_stroke_list(dict_tf_to_numpy(target_batch),
                                         self.dataset.np_undo_preprocessing)
      # Log per sample loss: chamfer distance on the reconstructed strokes.
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

      # Get min, max values for the plots so that they will be consistent.
      all_strokes = np.concatenate(gt_strokes, axis=0)
      x_min, x_max = get_min_max(all_strokes[:, 0], 0.3)
      y_min, y_max = get_min_max(all_strokes[:, 1], 0.3)

      # (3) Predict a stroke in leave-one-out (loo) fashion (i.e., given the rest)
      # self.__predict_loo(input_batch, target_batch, embeddings, idx)

      # # (4) Predict the next stroke given only the strokes so far.
      # self.__predict_ordered(input_batch, target_batch, embeddings, idx, plot_x=(x_min, x_max), plot_y=(y_min, y_max))

      # # (5) Auto-regressive prediction with random embeddings.
      # self.__predict_ar(input_batch, target_batch, embeddings, idx)
      
      # # (6) Auto-regressive prediction with the best embedding.
      # self.__predict_ar_best_embedding(input_batch, target_batch, embeddings, idx)

      # (7) Predict the next stroke given a predefined set of strokes.
      # self.__predict_random(input_batch, target_batch, embeddings, idx)

      # (8) Predict the next stroke given a predefined set of strokes.
      if self.model.position_model is not None:
        self.__predict_position_ar(input_batch, target_batch, embeddings, idx)
        # self.__predict_position_ar_alternatives(input_batch, target_batch, embeddings, idx)

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
        seq_len = np.array([self.decoded_length]*n_strokes)
        
      predicted_batch = self.embedding_model.decode_sequence(emb_,
                                                             seq_len=seq_len)
      predicted_batch[C.INP_START_COORD] = target_batch["start_coord"]

      stroke_colors = ["k"]*n_strokes
      stroke_colors[stroke_i] = "r"
      self.vis_engine.vis_stroke(predicted_batch, save_name="{}_predicted_loo_s{}".format(sample_idx, str(stroke_i)), colors=stroke_colors)
    
  def __predict_ordered(self, input_batch, target_batch, embeddings, sample_idx, plot_x=None, plot_y=None):
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
        seq_len = np.array([self.decoded_length]*(stroke_i + 1))
      
      predicted_batch = self.embedding_model.decode_sequence(emb_,
                                                             seq_len=seq_len)
      predicted_batch[C.INP_START_COORD] = target_batch["start_coord"][:stroke_i + 1]
      
      stroke_colors = ["k"]*(stroke_i + 1)
      stroke_colors[stroke_i] = "r"
      self.vis_engine.vis_stroke(predicted_batch, save_name="{}_predicted_ordered_s{}".format(sample_idx, str(stroke_i)), colors=stroke_colors, x_borders=plot_x, y_borders=plot_y)
      
  def __predict_ar(self, input_batch, target_batch, embeddings, sample_idx):
    """Predict the next embedding randomly and decode.

      Draw an embedding sample from the predicted embedding output model.
      """
    # Auto-regressive prediction.
    n_strokes = target_batch["num_strokes"][0]
    context_ids = 2
    context_embeddings = embeddings[:, :context_ids]
    start_positions = tf.transpose(a=input_batch[C.INP_START_COORD], perm=[1, 0, 2])
    if self.predictive_model.end_positions:
      end_positions = tf.transpose(a=input_batch[C.INP_END_COORD], perm=[1, 0, 2])
      context_pos = tf.concat([start_positions, end_positions], axis=-1)
    else:
      context_pos = start_positions
    for stroke_i in range(context_ids, n_strokes):
      out_ = self.model.predict_embedding_ar(context_embeddings,
                                            inp_pos=context_pos[:, :stroke_i],
                                            target_pos=start_positions[:, stroke_i:stroke_i + 1],
                                            greedy=self.emb_greedy)
      context_embeddings = tf.concat([context_embeddings, tf.expand_dims(out_["embedding_sample"], axis=0)], axis=1)
      emb_ = context_embeddings[0].numpy()

      seq_len = target_batch["seq_len"][:stroke_i + 1]
      if not self.gt_len_decoding:
        seq_len = np.array([self.decoded_length]*(stroke_i + 1))
        
      predicted_batch = self.embedding_model.decode_sequence(emb_,
                                                        seq_len=seq_len)
      predicted_batch[C.INP_START_COORD] = target_batch["start_coord"][:stroke_i+1]

      stroke_colors = ["k"]*(stroke_i + 1)
      stroke_colors[stroke_i] = "r"
      self.vis_engine.vis_stroke(predicted_batch, save_name="{}_ar_predicted_ordered_s{}".format(sample_idx, str(stroke_i)), colors=stroke_colors)
    
    # Animate AR predictions.
    stroke_colors = ["r"]*(stroke_i + 1)
    for i in range(context_ids):
      stroke_colors[i] = "k"
      
    if self.save_video:
      self.vis_engine.animate = True
      self.vis_engine.vis_stroke(predicted_batch, save_name="{}_ar_predicted_ordered_animation_s{}".format(sample_idx, str(stroke_i)), colors=stroke_colors)
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
    
    if self.predictive_model.end_positions:
      end_positions = tf.transpose(a=input_batch[C.INP_END_COORD], perm=[1, 0, 2])
      context_pos = tf.concat([start_positions, end_positions], axis=-1)
    else:
      context_pos = start_positions
    
    for stroke_i in range(n_given_strokes, n_strokes):
      pred_emb = self.model.predict_embedding_ar(context_embeddings,
                                                 inp_pos=context_pos[:, :stroke_i],
                                                 target_pos=start_positions[:, stroke_i:stroke_i + 1])
      # Decode all modes.
      all_emb_decodings, all_emb_pi, _ = self.__decode_embedding_all_components(pred_emb, target_batch["seq_len"][stroke_i:stroke_i + 1])
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

      embedding_sample, _ = self.model.predictive_model.output_layer.draw_sample_from_nth(pred_emb, n=min_comp_id[0], greedy=self.emb_greedy)
      context_embeddings = tf.concat([context_embeddings, embedding_sample], axis=1)
      emb_ = context_embeddings[0].numpy()

      seq_len = target_batch["seq_len"][:stroke_i + 1]
      if not self.gt_len_decoding:
        seq_len = np.array([self.decoded_length]*(stroke_i + 1))
        
      predicted_batch = self.embedding_model.decode_sequence(emb_,
                                                        seq_len=seq_len)
      predicted_batch[C.INP_START_COORD] = target_batch["start_coord"][:stroke_i+1]

      stroke_colors = ["k"]*(stroke_i + 1)
      stroke_colors[stroke_i] = "r"
      self.vis_engine.vis_stroke(predicted_batch, save_name="{}_best_ar_predicted_ordered_s{}".format(sample_idx, str(stroke_i)), colors=stroke_colors)
    
    # Animate AR predictions.
    stroke_colors = ["r"]*(stroke_i + 1)
    for i in range(n_given_strokes):
      stroke_colors[i] = "k"
      
    if self.save_video:
      self.vis_engine.animate = True
      self.vis_engine.vis_stroke(predicted_batch, save_name="{}_best_ar_predicted_ordered_animation_s{}".format(sample_idx, str(stroke_i)), colors=stroke_colors)
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
        inp_lens = np.array([self.decoded_length]*len(input_indices))
        target_len = np.array([self.decoded_length])
        
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
      
      stroke_colors = ["k"]*(pred_n_strokes + 1)
      stroke_colors[-1] = "r"
      given = str(sorted(input_indices))[1:-1].replace(", ", "_")
      self.vis_engine.vis_stroke(predicted_batch,
                                 save_name="{}_predicted_{}_given_{}".format(sample_idx, target_idx, given),
                                 colors=stroke_colors)
      
  def __predict_position_ar(self, input_batch, target_batch, embeddings, sample_idx):
    # Set plot limits by using the ground-truth canvas.
    gt_strokes = padded_to_stroke_list(dict_tf_to_numpy(target_batch),
                                       self.dataset.np_undo_preprocessing)
    all_strokes = np.concatenate(gt_strokes, axis=0)
    x_min, x_max = get_min_max(all_strokes[:, 0], 0.3)
    y_min, y_max = get_min_max(all_strokes[:, 1], 0.3)
    v_min, v_max = None, None
    
    # Auto-regressive prediction.
    n_strokes = target_batch["num_strokes"][0]
    n_strokes += 5
    context_ids = 2
    context_embeddings = embeddings[:, :context_ids]
    start_positions = np.transpose(target_batch[C.INP_START_COORD], [1, 0, 2])
    end_positions = np.transpose(target_batch[C.INP_END_COORD], [1, 0, 2])
    
    # ar_start_pos = [start_positions[:, 0:1], start_positions[:, 1:2]]
    # ar_end_pos = [end_positions[:, 0:1], end_positions[:, 1:2]]
    ar_start_pos = np.split(start_positions[:, 0:context_ids], context_ids, axis=1)
    ar_end_pos = np.split(end_positions[:, 0:context_ids], context_ids, axis=1)
    for stroke_i in range(context_ids, n_strokes):
      input_pos = np.concatenate(ar_start_pos[:stroke_i], axis=1)

      if self.predictive_model.end_positions:
        end_pos = np.concatenate(ar_end_pos[:stroke_i], axis=1)
        input_pos = np.concatenate([input_pos, end_pos], axis=-1)
      
      pos_ = self.model.predict_position_ar(context_embeddings,
                                            inp_pos=input_pos,
                                            greedy=self.emb_greedy)

      # logli = log_likelihood(xy, pos_).numpy()
      # max_pos = xy[np.where(logli >= logli.max())[0]]
      # target_pos = tf.convert_to_tensor(max_pos[np.newaxis])
      target_pos = np.expand_dims(pos_["position_sample"].numpy(), axis=0)
      ar_start_pos.append(target_pos)

      out_ = self.model.predict_embedding_ar(context_embeddings,
                                             inp_pos=input_pos,
                                             target_pos=target_pos,
                                             greedy=self.emb_greedy)
      
      context_embeddings = tf.concat([context_embeddings, tf.expand_dims(out_["embedding_sample"], axis=0)], axis=1)
      emb_ = context_embeddings[0].numpy()

      # seq_len = target_batch["seq_len"][:stroke_i + 1]
      seq_len = np.array([self.decoded_length]*(stroke_i + 1))
      if not self.gt_len_decoding:
        seq_len = np.array([self.decoded_length]*(stroke_i + 1))
        
      predicted_batch = self.embedding_model.decode_sequence(emb_,
                                                             seq_len=seq_len)
      predicted_batch[C.INP_START_COORD] = np.transpose(np.concatenate(ar_start_pos[:stroke_i+1], axis=1), [1,0,2])

      stoke_colors = None
      if self.render_binary_colors:
        stoke_colors = ["k"]*(stroke_i + 1)
        stoke_colors[stroke_i] = self.prediction_color
      
      ### Plot heatmap
      # Render strokes.
      predicted_strokes = padded_to_stroke_list(dict_tf_to_numpy(predicted_batch),
                                                self.dataset.np_undo_preprocessing)

      all_pred_strokes = np.concatenate(predicted_strokes, axis=0)
      pred_x_min, pred_x_max = get_min_max(all_pred_strokes[:, 0], 0.3)
      pred_y_min, pred_y_max = get_min_max(all_pred_strokes[:, 1], 0.3)
      
      x_min = min(x_min, pred_x_min)
      x_max = max(x_max, pred_x_max)
      y_min = min(y_min, pred_y_min)
      y_max = max(y_max, pred_y_max)
      fig, ax = render_strokes(predicted_strokes, colors=stoke_colors, x_borders=(x_min, x_max), y_borders=(y_min, y_max), highlight_start=self.render_initial_point)
      
      if self.render_position_heatmap:
        n_bins = 100
        x, y = np.meshgrid(np.linspace(x_min - 0.5, x_max + 0.5, n_bins),
                           np.linspace(-y_max - 0.5, -y_min + 0.5, n_bins))
  
        xy = np.empty(x.shape + (2,))
        xy[:, :, 0] = x
        xy[:, :, 1] = y
        xy = np.reshape(xy, [-1, 2]).astype(np.float32)
  
        ar_end_pos.append(predicted_strokes[-1][-1][:2][np.newaxis, np.newaxis])
  
        # Calculate position densities.
        logli = log_likelihood(xy, pos_)
        probs = np.reshape(logli.numpy(), [n_bins, n_bins])
  
        probs_normalized = np.exp(probs)/np.exp(probs).sum()
        probs = probs_normalized
  
        # Customize the colormap by adding some alpha.
        cmap = pl.cm.OrRd
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:, -1] = np.linspace(0, 0.8, cmap.N)  # Set alpha
        my_cmap = colors.ListedColormap(my_cmap)        # Create new colormap
  
        plt.contourf(x, -y, probs, cmap=my_cmap, vmin=v_min, vmax=v_max)
        plt.plot(target_pos[0, 0, 0], -target_pos[0, 0, 1], 'ro', lw=3, markersize=8, color=self.prediction_color)
        # plt.colorbar()
        # pos_str = "{:.2f}, {:.2f}".format(target_pos[0, 0, 0], -target_pos[0, 0, 1])
        # ax.text(target_pos[0, 0, 0], -target_pos[0, 0, 1], pos_str, fontsize=20, ha='center', va='center', color='k') # plt_stroke[0].get_color())
  
      fig.savefig(os.path.join(self.vis_engine.log_dir, "{}_pos_ar_heatmap_ordered_s{}.png".format(sample_idx, str(stroke_i))), format="png", bbox_inches='tight', dpi=200)
      fig.savefig(os.path.join(self.vis_engine.log_dir, "{}_pos_ar_heatmap_ordered_s{}.svg".format(sample_idx, str(stroke_i))), format="svg", bbox_inches='tight', dpi=200)
      plt.close()

    if self.save_video:
      # Animate AR predictions.
      stroke_colors = [self.prediction_color]*(stroke_i + 1)
      for i in range(context_ids):
        stroke_colors[i] = "k"
      
      self.vis_engine.animate = True
      self.vis_engine.vis_stroke(predicted_batch, save_name="{}_pos_ar_predicted_ordered_animation_s{}".format(sample_idx, str(stroke_i)), colors=stroke_colors)
      self.vis_engine.animate = False
      
      
  def __predict_position_ar_alternatives(self, input_batch, target_batch, embeddings, sample_idx):
    # Set plot limits by using the ground-truth canvas.
    gt_strokes = padded_to_stroke_list(dict_tf_to_numpy(target_batch), self.dataset.np_undo_preprocessing)
    all_strokes = np.concatenate(gt_strokes, axis=0)
    x_min, x_max = get_min_max(all_strokes[:, 0], 0.3)
    y_min, y_max = get_min_max(all_strokes[:, 1], 0.3)
    v_min, v_max = None, None
    # x_min, x_max, y_min, y_max = [-0.25, 1.0, -1.25, 0.25]  # sample 64
    x_min, x_max, y_min, y_max = [-0.5, 1.5, -1.5, 0.25]  # sample 2
  
    n_bins = 100
    x, y = np.meshgrid(np.linspace(x_min - 0.5, x_max + 0.5, n_bins),
                       np.linspace(-y_max - 0.5, -y_min + 0.5, n_bins))
  
    xy = np.empty(x.shape + (2,))
    xy[:, :, 0] = x
    xy[:, :, 1] = y
    xy = np.reshape(xy, [-1, 2]).astype(np.float32)
    
    # Auto-regressive prediction.
    n_strokes = target_batch["num_strokes"][0]
    n_strokes += 4
    context_ids = 1
    context_embeddings = embeddings[:, :context_ids]
    start_positions = np.transpose(target_batch[C.INP_START_COORD], [1, 0, 2])
    
    ar_start_pos = np.split(start_positions[:, 0:context_ids], context_ids, axis=1)
    for stroke_i in range(context_ids, n_strokes):
      input_pos = np.concatenate(ar_start_pos[:stroke_i], axis=1)
      
      pos_ = self.model.predict_position_ar(context_embeddings,
                                            inp_pos=input_pos,
                                            greedy=self.emb_greedy)

      all_pos_samples, all_pi = self.model.position_model.output_layer.draw_sample_every_component(pos_, greedy=self.emb_greedy)
      best_pos_id = np.argsort(all_pi.numpy()[0, 0])[-1]
      second_best_pos_id = np.argsort(all_pi.numpy()[0, 0])[-2]
      
      best_pos = all_pos_samples[:, :, best_pos_id].numpy()
      second_best_pos = all_pos_samples[:, :, second_best_pos_id].numpy()
      
      def get_best_two_embedding(context_embeddings, input_pos, target_pos):
        out_ = self.model.predict_embedding_ar(context_embeddings,
                                               inp_pos=input_pos,
                                               target_pos=target_pos,
                                               greedy=self.emb_greedy)
        
        all_emb_samples, all_emb_pi = self.model.predictive_model.output_layer.draw_sample_every_component(out_, greedy=self.emb_greedy)
        sorted_indices = np.argsort(all_emb_pi.numpy()[0, 0])
        best_emb_id = sorted_indices[-1]
        second_best_emb_id = sorted_indices[-2]
        third_best_emb_id = sorted_indices[-3]
        fourth_best_emb_id = sorted_indices[-4]
        fifth_best_emb_id = sorted_indices[-5]
        
        best_emb = all_emb_samples[:, :, best_emb_id]#.numpy()
        second_best_emb = all_emb_samples[:, :, second_best_emb_id]#.numpy()
        third_best_emb = all_emb_samples[:, :, third_best_emb_id]#.numpy()
        fourth_best_emb = all_emb_samples[:, :, fourth_best_emb_id]#.numpy()
        fifth_best_emb = all_emb_samples[:, :, fifth_best_emb_id]#.numpy()
        return best_emb, second_best_emb, third_best_emb, fourth_best_emb, fifth_best_emb
      
      
      def decode_and_plot(current_embeddings, decoding_seq_len_, start_pos_, plot_name, fig=None, ax=None, alpha=1.0, next_stroke_color="#c3090aff", draw_heatmap=True):
        emb_ = current_embeddings[0].numpy()
        predicted_batch = self.embedding_model.decode_sequence(emb_,
                                                               seq_len=decoding_seq_len_)
        predicted_batch[C.INP_START_COORD] = np.transpose(np.concatenate(start_pos_, axis=1), [1,0,2])
        
        ### Plot heatmap
        # Render strokes.
        predicted_strokes = padded_to_stroke_list(dict_tf_to_numpy(predicted_batch),
                                                  self.dataset.np_undo_preprocessing)
        
        n_plot_strokes = len(start_pos_)
        stroke_colors = ["k"]*n_plot_strokes
        # colors[stroke_i] = "b"
        stroke_colors[n_plot_strokes-1] = next_stroke_color
        fig, ax = render_strokes(predicted_strokes, colors=stroke_colors, x_borders=(x_min, x_max), y_borders=(y_min, y_max), fig=fig, ax=ax, alpha=alpha)
        
        # Calculate position densities.
        if draw_heatmap:
          logli = log_likelihood(xy, pos_)
          probs = np.reshape(logli.numpy(), [n_bins, n_bins])
    
          probs_normalized = np.exp(probs)/np.exp(probs).sum()
          probs = probs_normalized
          
          # Customize the colormap by adding some alpha.
          cmap = pl.cm.OrRd
          my_cmap = cmap(np.arange(cmap.N))
          my_cmap[:, -1] = np.linspace(0, 0.8, cmap.N)  # Set alpha
          my_cmap = colors.ListedColormap(my_cmap)        # Create new colormap
    
          plt.contourf(x, -y, probs, cmap=my_cmap, vmin=v_min, vmax=v_max)
          plt.plot(start_pos_[-1][0,0,0], -start_pos_[-1][0,0,1], 'ro', lw=3, markersize=8, color=next_stroke_color)
        
        # pos_str = "{:.2f}, {:.2f}".format(target_pos[0, 0, 0], -target_pos[0, 0, 1])
        # ax.text(target_pos[0, 0, 0], -target_pos[0, 0, 1], pos_str, fontsize=20, ha='center', va='center', color='k') # plt_stroke[0].get_color())
        
        # plt.colorbar()
        fig.savefig(os.path.join(self.vis_engine.log_dir, (plot_name + ".png").format(sample_idx, str(stroke_i))), format="png", bbox_inches='tight', dpi=200)
        fig.savefig(os.path.join(self.vis_engine.log_dir, (plot_name + ".svg").format(sample_idx, str(stroke_i))), format="svg", bbox_inches='tight', dpi=200)
        return fig, ax

      # b_plot_name = "{}_pos_ar_heatmap_ordered_s{}_best_pos"
      # sb_plot_name = "{}_pos_ar_heatmap_ordered_s{}_second_best_pos"
      # decoding_seq_len_ = np.array([self.decoded_length]*(stroke_i + 1))
      # # start_pos_ = ar_start_pos[:stroke_i + 1]
      #
      # best_ar_start_pos = ar_start_pos.copy()
      # best_ar_start_pos.append(best_pos)
      # emb_with_best_pos, second_emb_with_best_pos = get_best_two_embedding(context_embeddings, input_pos, best_pos)
      # b_context_embeddings = tf.concat([context_embeddings, emb_with_best_pos], axis=1)
      # fig_, ax_ = decode_and_plot(b_context_embeddings, decoding_seq_len_, best_ar_start_pos, b_plot_name)
      #
      # # second_best_ar_start_pos = ar_start_pos.copy()
      # # second_best_ar_start_pos.append(second_best_pos)
      # second_best_ar_start_pos = [second_best_pos]
      # emb_with_second_best_pos, _ = get_best_two_embedding(context_embeddings, input_pos, second_best_pos)
      # # sb_context_embeddings = tf.concat([context_embeddings, emb_with_second_best_pos], axis=1)
      # sb_context_embeddings = emb_with_second_best_pos
      # fig_, ax_ = decode_and_plot(sb_context_embeddings, [self.decoded_length], second_best_ar_start_pos, sb_plot_name, fig=fig_, ax=ax_, alpha=0.5, next_stroke_color="dimgray")
      #
      # plt.close()
      
      
      b_plot_name = "{}_pos_ar_s{}_best_emb_"
      # sb_plot_name = "{}_pos_ar_s{}_second_best_em"
      best_emb_ar_start_pos = ar_start_pos.copy()
      best_emb_ar_start_pos.append(best_pos)
      decoding_seq_len_ = np.array([self.decoded_length]*(stroke_i + 1))

      fig_, ax_ = None, None
      draw_heatmap = True
      best_embeddings = get_best_two_embedding(context_embeddings, input_pos, best_pos)
      for idx, embedding in enumerate(best_embeddings):
        # emb_with_best_pos, second_emb_with_best_pos = get_best_two_embedding(context_embeddings, input_pos, best_pos)
        b_emb_context_embeddings = tf.concat([context_embeddings, embedding], axis=1)
        fig_, ax_ = decode_and_plot(b_emb_context_embeddings, decoding_seq_len_, best_emb_ar_start_pos, b_plot_name + str(idx), fig=fig_, ax=ax_, next_stroke_color=mpl.cm.tab20.colors[idx%20], draw_heatmap=draw_heatmap)
        draw_heatmap = False
      plt.close()

      context_embeddings = tf.concat([context_embeddings, best_embeddings[0]], axis=1)
      ar_start_pos = best_emb_ar_start_pos

      # # sb_context_embeddings = tf.concat([context_embeddings, emb_with_second_best_pos], axis=1)
      # sb_context_embeddings = second_emb_with_best_pos
      # fig_, ax_ = decode_and_plot(sb_context_embeddings, [self.decoded_length], [best_pos], sb_plot_name, fig=fig_, ax=ax_, alpha=0.5, next_stroke_color="dimgray")
      #
      # plt.close()
      # context_embeddings = b_context_embeddings
      # ar_start_pos = best_ar_start_pos