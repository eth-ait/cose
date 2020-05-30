import tensorflow as tf
import numpy as np
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors


def distance_matrix_batch(array1, array2):
  """
  arguments:
      array1: the array, size: (batch_size, num_point, num_feature)
      array2: the samples, size: (batch_size, num_point, num_feature)
  returns:
      distances: each entry is the distance from a sample to array1
          , it's size: (batch_size, num_point, num_point)
  """
  batch_size, num_point, num_features = array1.shape
  expanded_array1 = tf.tile(array1, (1, num_point, 1))
  expanded_array2 = tf.reshape(
      tf.tile(tf.expand_dims(array2, 2),
              (1, 1, num_point, 1)),
      (batch_size, -1, num_features))
  distances = tf.norm(tensor=expanded_array1 - expanded_array2, axis=-1)
  distances = tf.reshape(distances, (batch_size, num_point, num_point))
  return distances


def distance_matrix(array1, array2):
  """
  arguments:
      array1: the array, size: (num_point, num_feature)
      array2: the samples, size: (num_point, num_feature)
  returns:
      distances: each entry is the distance from a sample to array1
          , it's size: (num_point, num_point)
  """
  num_point, num_features = array1.shape
  expanded_array1 = tf.tile(array1, (num_point, 1))
  expanded_array2 = tf.reshape(
      tf.tile(tf.expand_dims(array2, 1),
              (1, num_point, 1)),
      (-1, num_features))
  distances = tf.norm(tensor=expanded_array1 - expanded_array2, axis=1)
  distances = tf.reshape(distances, (num_point, num_point))
  return distances


def av_dist(array1, array2):
  """
  arguments:
      array1, array2: both size: (num_points, num_feature)
  returns:
      distances: size: (1,)
  """
  distances = distance_matrix(array1, array2)
  distances = tf.reduce_min(input_tensor=distances, axis=1)
  distances = tf.reduce_sum(input_tensor=distances)
  return distances


def chamfer_distance_tf(arrays):
  """
  arguments:
      arrays: array1, array2
  returns:
      sum of av_dist(array1, array2) and av_dist(array2, array1)
  """
  array1, array2 = arrays
  av_dist1 = av_dist(array1, array2)
  av_dist2 = av_dist(array2, array1)
  return av_dist1 + av_dist2


def chamfer_distance_tf_batch(array1, array2):
  dist = tf.reduce_mean(
      input_tensor=tf.map_fn(chamfer_distance_tf, elems=(array1, array2), dtype=tf.float32)
      )
  return dist


def chamfer_distance_np(arrays):
  array1, array2 = arrays
  num_point = array1.shape[0]
  tree1 = KDTree(array1, leafsize=num_point + 1)
  tree2 = KDTree(array2, leafsize=num_point + 1)
  distances1, _ = tree1.query(array2)
  distances2, _ = tree2.query(array1)
  av_dist1 = np.sum(distances1)
  av_dist2 = np.sum(distances2)
  return av_dist1 + av_dist2


def chamfer_distance_np_var_len(arrays):
  """Chamfer distance in numpy supporting arrays with different lengths.
  
  Args:
    arrays:
  Returns:
  """
  x, y = arrays
  x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(x)
  min_y_to_x = x_nn.kneighbors(y)[0]
  y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(y)
  min_x_to_y = y_nn.kneighbors(x)[0]
  chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
  return chamfer_dist