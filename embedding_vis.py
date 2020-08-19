"""Evaluation script running in eager mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import argparse

import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
from smartink.config.configuration import Configuration
import smartink.loss.chamfer as chamfer

# plt.style.use('ggplot')
matplotlib.use('Agg')


gpu = tf.config.experimental.list_physical_devices('GPU')[0]
if gpu:
  try:
    tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


def fit_multivariate_normal(data):
  mean_ = np.mean(data, axis=0)
  cov_ = np.cov(data, rowvar=0)
  mvn = tfp.distributions.MultivariateNormalFullCovariance(tf.convert_to_tensor(mean_, dtype=tf.float32),
                                                           tf.convert_to_tensor(cov_, dtype=tf.float32))
  return mvn


def tsne_2d(gt, pred, plot_name=None, c_kmeans=10, norm_kmeans=True):
  perplexity = 50
  n_jobs = 8
  
  if pred is not None:
    all_ = np.concatenate([gt, pred], axis=0)
  else:
    all_ = gt
    
  print("Calculating TSNE projection for {} samples...".format(all_.shape[0]))
  all_TSNE = TSNE(n_components=2, n_jobs=n_jobs, perplexity=perplexity)
  all_2d = all_TSNE.fit_transform(all_)
  
  if pred is not None:
    gt_2d = all_2d[:gt.shape[0]]
    pred_2d = all_2d[-pred.shape[0]:]
    plt.scatter(gt_2d[:, 0], gt_2d[:, 1], s=2, c="xkcd:marine blue", alpha=0.8)
    plt.scatter(pred_2d[:, 0], pred_2d[:, 1], s=2, c="xkcd:yellow orange", alpha=0.2)
    # plt.legend((" embeddings", "Predicted embeddings"), fontsize=6)
    plt.axis('off')
  else:
    plt.scatter(all_2d[:, 0], all_2d[:, 1], s=2, c="xkcd:yellow orange")
    # plt.legend("Ground-truth embeddings", fontsize=6)
    plt.axis('off')
    gt_2d = all_2d
    pred_2d = None
    
  if plot_name is not None:
    plt.savefig("{}_tsne{}.png".format(plot_name, perplexity), bbox_inches='tight', dpi=200)
  plt.close()
  
  # K-means clustering
  kmeans = KMeans(n_clusters=c_kmeans, random_state=0).fit(gt)
  colormap = "tab10"
  plt.scatter(gt_2d[:, 0], gt_2d[:, 1], c=kmeans.labels_, s=2, cmap=colormap)
  plt.axis('off')
  if plot_name is not None:
    plt.savefig("{}_kmeans{}_tsne{}.png".format(plot_name, c_kmeans, perplexity), bbox_inches='tight', dpi=200)
  plt.close()

  # K-means clustering with normalized embeddings.
  if norm_kmeans:
    norms = np.linalg.norm(gt, axis=1)
    normalized_gt = gt/norms[:, np.newaxis]
    kmeans = KMeans(n_clusters=c_kmeans, random_state=0).fit(normalized_gt)
    plt.scatter(gt_2d[:, 0], gt_2d[:, 1], c=kmeans.labels_, s=2, cmap=colormap)
    plt.axis('off')
    if plot_name is not None:
      plt.savefig("{}_norm_kmeans{}_tsne{}.png".format(plot_name, c_kmeans, perplexity), bbox_inches='tight', dpi=200)
    plt.close()
  
  return gt_2d, pred_2d


def pca_2d(gt, pred, plot_name=None, c_kmeans=15, norm_kmeans=True):
  print("Calculating PCA projection for {} samples...".format(gt.shape[0]))
  
  gt_PCA = PCA(n_components=2)
  gt_2d = gt_PCA.fit_transform(gt)
  
  if pred is not None:
    pred_2d = gt_PCA.transform(pred)
    plt.scatter(gt_2d[:, 0], gt_2d[:, 1], s=2, c="xkcd:marine blue", alpha=0.8)
    # plt.scatter(gt_2d[:, 0], gt_2d[:, 1], s=2, cmap="Pastel2", alpha=0.4)
    
    plt.scatter(pred_2d[:, 0], pred_2d[:, 1], s=2, c="xkcd:yellow orange", alpha=0.2)
    # plt.scatter(pred_2d[:, 0], pred_2d[:, 1], s=2, cmap="Pastel2", alpha=0.8)
    # plt.legend(("Ground-truth embeddings", "Predicted embeddings"), fontsize=6)
    plt.axis('off')
  else:
    plt.scatter(gt_2d[:, 0], gt_2d[:, 1], s=2, c="xkcd:yellow orange")
    # plt.legend("Ground-truth embeddings", fontsize=6)
    plt.axis('off')
    pred_2d = None
  
  if plot_name is not None:
    plt.savefig("{}_pca.png".format(plot_name), bbox_inches='tight', dpi=200)
  plt.close()
  
  # K-means clustering
  kmeans = KMeans(n_clusters=c_kmeans, random_state=0).fit(gt)
  # colormap = matplotlib.cm.Dark2.colors
  colormap = "tab10"
  plt.scatter(gt_2d[:, 0], gt_2d[:, 1], c=kmeans.labels_, s=2, cmap=colormap)
  plt.axis('off')
  if plot_name is not None:
    plt.savefig("{}_kmeans{}_pca.png".format(plot_name, c_kmeans), bbox_inches='tight', dpi=200)
  plt.close()

  # K-means clustering with normalized embeddings.
  if norm_kmeans:
    norms = np.linalg.norm(gt, axis=1)
    normalized_gt = gt/norms[:, np.newaxis]
    kmeans = KMeans(n_clusters=c_kmeans, random_state=0).fit(normalized_gt)
    plt.scatter(gt_2d[:, 0], gt_2d[:, 1], c=kmeans.labels_, s=2, cmap=colormap)
    plt.axis('off')
    if plot_name is not None:
      plt.savefig("{}_norm_kmeans{}_pca.png".format(plot_name, c_kmeans), bbox_inches='tight', dpi=200)
    plt.close()
  
  return gt_2d, pred_2d


def calculate_dist_distance(dist1, dist2):
  all_dist = np.vstack([dist1, dist1])
  min_val = all_dist.min()
  max_val = all_dist.max()
  dist1_norm = (dist1 - min_val) / (max_val-min_val)
  dist2_norm = (dist2 - min_val) / (max_val-min_val)
  
  total_dist, dist_2_to_1, dist_1_to_2 = chamfer.chamfer_distance_np_var_len([dist1_norm, dist2_norm])
  print("Total Distance: ", total_dist)
  print("Distance from 1 to 2: ", dist_1_to_2)
  print("Distance from 2 to 1: ", dist_2_to_1)

def main():
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_ids', required=True,
                      help='Experiment ID (experiment timestamp).')
  
  args = parser.parse_args()
  if ',' in args.model_ids:
    model_ids = args.model_ids.split(',')
  else:
    model_ids = [args.model_ids]
    
  try:
    data_root = os.environ["COSE_DATA_DIR"]
    log_dir = os.environ["COSE_LOG_DIR"]
    log_eval_dir = os.environ["COSE_EVAL_DIR"]
    gdrive_key = os.environ["GDRIVE_API_KEY"]
  except KeyError:
    raise Exception("Environment variables are not set.")
  
  dummy = tf.random.normal([10])
  
  for model_id in model_ids:
    print()
    print()
    try:
      print("Model {}".format(model_id))
      # Check if the experiment directory already exists.
      model_dir_query = glob.glob(os.path.join(log_dir, model_id + "*"))
      if not model_dir_query:
        raise Exception("Model not found.")

      model_dir = model_dir_query[0]
      eval_dir = os.path.join(log_eval_dir, os.path.basename(model_dir))
      config = Configuration.from_json(os.path.join(model_dir, "config.json"))
      
      gt_embeddings = np.load(os.path.join(eval_dir, "test_gt_embeddings.npy"))
      
      predicted_emb_path = os.path.join(eval_dir, "test_best_predicted_embeddings.npy")
      predicted_embeddings = None
      if os.path.exists(predicted_emb_path):
        predicted_embeddings = np.load(os.path.join(eval_dir, "test_best_predicted_embeddings.npy"))
      
      # predicted_embeddings = None
      gt_2d_tsne, pred_2d_tsne = tsne_2d(gt_embeddings, predicted_embeddings, os.path.join(eval_dir, "gt_emb"), c_kmeans=10, norm_kmeans=False)
      calculate_dist_distance(gt_2d_tsne, pred_2d_tsne)
      gt_2d_pca, pred_2d_pca = pca_2d(gt_embeddings, predicted_embeddings, os.path.join(eval_dir, "gt_emb"), c_kmeans=10, norm_kmeans=False)

      # Fit multivariate normal and compare.
      if predicted_embeddings is not None:
        mvn_gt = fit_multivariate_normal(gt_embeddings)
        mvn_prediction = fit_multivariate_normal(predicted_embeddings)
        
        kl_gt_to_pred = mvn_gt.kl_divergence(mvn_prediction)
        kl_pred_to_gt = mvn_prediction.kl_divergence(mvn_gt)
        print("kl_gt_to_pred: {}".format(kl_gt_to_pred))
        print("kl_pred_to_gt: {}".format(kl_pred_to_gt))
      
    except Exception as e:
      print("Something went wrong when evaluating model {}".format(model_id))
      raise Exception(e)


if __name__ == "__main__":
  main()
