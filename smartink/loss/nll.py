"""Custom loss functions for Tensorflow."""

import numpy as np
import tensorflow as tf


def kld_normal_diagonal(mu1, sigma1, mu2, sigma2, reduce_sum=False):
  """Kullback-Leibler divergence.

  Between two Gaussian with diagonal covariance.
  Args:
      mu1:
      sigma1: log variance.
      mu2:
      sigma2: log variance.
      reduce_sum:

  Returns:
  """
  with tf.compat.v1.name_scope('kld_normal_diagonal'):
    # result = tf.reduce_sum(
    #     tf.log(tf.maximum(1e-6, sigma2)) - tf.log(tf.maximum(1e-6, sigma1)) +
    #     (0.5*tf.square(sigma1) + tf.square(mu1 - mu2)) /
    #     (2 * tf.maximum(1e-6, (tf.square(sigma2)))) - 0.5,
    #     keepdims=True,
    #     axis=-1)

    # Assuming log variance is given.
    result = tf.reduce_sum(
        input_tensor=-0.5 * (sigma1 - sigma2 -
                (tf.exp(sigma1) + tf.square(mu1 - mu2)) / tf.exp(sigma2) + 1.0),
        keepdims=True,
        axis=-1)

    if not reduce_sum:
      return result
    else:
      return tf.reduce_sum(input_tensor=result, axis=reduce_sum)


def kld_normal_diagonal_standard_prior(mu1, sigma1):
  """Kullback-Leibler divergence.

  Between a Gaussian with diagonal covariance and zero-mean unit-variance
  Gaussian.
  Args:
      mu1:
      sigma1: standard deviation vector.

  Returns:
  """
  kld_ = (1 + sigma1 - tf.square(mu1) - tf.exp(sigma1))
  # return -0.5*(tf.reduce_mean(kld_))
  # return -0.5*(tf.reduce_sum(kld_, keepdims=False, axis=-1))
  return tf.reduce_mean(input_tensor=-0.5 * (tf.reduce_sum(input_tensor=kld_, keepdims=False, axis=-1)))


def kld_normal_diagonal_standard_prior_normalized(mu1, sigma1):
  """Kullback-Leibler divergence normalized by the latent size.

  Between a Gaussian with diagonal covariance and zero-mean unit-variance
  Gaussian.
  Args:
      mu1:
      sigma1: standard deviation vector.

  Returns:
  """
  kld_ = (1 + sigma1 - tf.square(mu1) - tf.exp(sigma1))
  return -0.5*(tf.reduce_mean(kld_))


def logli_normal_diagonal(x, mu, sigma):
  """Log-likelihood of a Gaussian with diagonal covariance.

  Args:
      x:
      mu:
      sigma: standard deviation.

  Returns:
  """
  with tf.compat.v1.name_scope('logli_normal_diagonal'):
    var = tf.maximum(1e-6, tf.square(sigma))
    result = -0.5 * (tf.math.log(2 * np.pi * var) + tf.square(x - mu)/var)

    return tf.reduce_sum(input_tensor=result, axis=-1, keepdims=True)


def logli_normal_bivariate(x, mu, sigma, rho, reduce_sum=False):
  """Bivariate Gaussian log-likelihood.

  Rank of arguments is expected to be 3.
  Args:
      x: data samples with shape (batch_size, seq_len, feature_size).
      mu:
      sigma: standard deviation.
      rho:
      reduce_sum: False, None or list of axes.
  Returns:
  """
  last_axis = tf.rank(x) - 1
  x1, x2 = tf.split(x, 2, axis=last_axis)
  mu1, mu2 = tf.split(mu, 2, axis=last_axis)
  sigma1, sigma2 = tf.split(tf.maximum(1e-9, sigma), 2, axis=last_axis)
  s1s2 = tf.maximum(1e-9, tf.multiply(sigma1, sigma2))

  with tf.compat.v1.name_scope('logli_normal_bivariate'):
    x_mu1 = tf.subtract(x1, mu1)
    x_mu2 = tf.subtract(x2, mu2)
    z_denom = tf.square(x_mu1/sigma1) + \
              tf.square(x_mu2/sigma2) - \
              2*(tf.multiply(rho, tf.multiply(x_mu1, x_mu2))/s1s2)

    rho_square_term = tf.maximum(1e-9, 1 - tf.square(rho))
    log_regularize_term = tf.math.log(
        tf.maximum(1e-9,
                   2 * np.pi * tf.multiply(s1s2, tf.sqrt(rho_square_term))))

    log_power_e = z_denom/(2*rho_square_term)
    result = -(log_regularize_term + log_power_e)

    if not reduce_sum:
      return result
    else:
      return tf.reduce_sum(input_tensor=result, axis=reduce_sum)


def logli_gmm_logsumexp(x, mu, sigma, coefficient):
  """Gaussian mixture model log-likelihood.

  Gaussian components with diagonal covariance matrix. More stable
  implementation of GMM log-likelihood.
  Args:
      x: (batch_size, seq_len, units)
      mu: (batch_size, seq_len, units*num_components)
      sigma: std (batch_size, seq_len, units*num_components)
      coefficient: (batch_size, seq_len, num_components)

  Returns:
  """
  with tf.compat.v1.name_scope('logli_gmm_logsumexp'):
    expanded = False
    if len(mu.shape) == 2:
      x = tf.expand_dims(x, axis=1)
      mu = tf.expand_dims(mu, axis=1)
      sigma = tf.expand_dims(sigma, axis=1)
      coefficient = tf.expand_dims(coefficient, axis=1)
      expanded = True

    batch_size = tf.shape(input=mu)[0]
    seq_len = tf.shape(input=mu)[1]
    feature_gmm_components = tf.shape(input=mu)[2]
    num_components = tf.shape(input=coefficient)[-1]
    units = feature_gmm_components // num_components

    mu_ = tf.reshape(mu, (batch_size, seq_len, units, num_components))
    sigma_ = tf.reshape(sigma, (batch_size, seq_len, units, num_components))
    x_ = tf.expand_dims(x, axis=-1)
    log_coeff = tf.math.log(tf.maximum(1e-9, coefficient))

    var = tf.maximum(1e-6, tf.square(sigma_))
    log_normal = -0.5 * tf.reduce_sum(
        input_tensor=(tf.math.log(2 * np.pi * var) + tf.square(x_ - mu_)/var), axis=2)

    nll = tf.reduce_logsumexp(input_tensor=log_coeff + log_normal, axis=-1, keepdims=True)
    if expanded:
      return nll[:, 0]
    else:
      return nll


def logli_gmm(x, mu, sigma, coefficient):
  """Gaussian mixture model log-likelihood.

  Gaussian components with diagonal covariance matrix.
  Args:
      x: (batch_size, seq_len, feature_size)
      mu: (batch_size, seq_len, feature_size*num_gmm_components)
      sigma: std (batch_size, seq_len, feature_size*num_gmm_components)
      coefficient: (batch_size, seq_len, num_gmm_components)

  Returns:
  """
  with tf.compat.v1.name_scope('logli_gmm'):
    batch_size, seq_len, feature_gmm_components = mu.shape.as_list()
    _, _, num_gmm_components = coefficient.shape.as_list()
    feature_size = int(feature_gmm_components / num_gmm_components)
    seq_len = tf.shape(
        input=mu)[1] if seq_len is None else seq_len  # Variable-length sequences.
    batch_size = tf.shape(input=mu)[0] if batch_size is None else batch_size

    mu_ = tf.reshape(mu,
                     (batch_size, seq_len, feature_size, num_gmm_components))
    sigma_ = tf.reshape(sigma,
                        (batch_size, seq_len, feature_size, num_gmm_components))
    x_ = tf.expand_dims(x, axis=-1)
    coefficient_ = tf.expand_dims(coefficient, axis=2)

    var = tf.maximum(1e-6, tf.square(sigma_))
    z_term = 1.0/tf.sqrt(2 * np.pi * var)
    exp_term = tf.exp(-tf.square(x_ - mu_)/ (2*var))
    gaussian_likelihood = z_term * exp_term
    gmm_likelihood = tf.reduce_sum(
        input_tensor=tf.multiply(gaussian_likelihood, coefficient_), axis=-1)
    gmm_loglikelihood = tf.math.log(tf.maximum(1e-6, gmm_likelihood))

    return tf.reduce_sum(input_tensor=gmm_loglikelihood, axis=-1, keepdims=True)
  

def log_likelihood(target, pred):
  """Calculates log likelihood with a given dictionary of predictions.
  
  Args:
    target: target tensor (batch_size, seq_len, feature_size).
    pred (dict): with keys mu, sigma, rho (binormal), pi (gmm) of shape
      (batch_size, seq_len, feature_size).
  Returns:
    (batch_size, seq_len)
  """
  # Log-likelihood Loss
  if "pi" in pred:
    logli = logli_gmm_logsumexp(target, pred["mu"], pred["sigma"], pred["pi"])
  elif "rho" in pred:
    logli = logli_normal_bivariate(target, pred["mu"], pred["sigma"], pred["rho"])
  else:
    logli = logli_normal_diagonal(target, pred["mu"], pred["sigma"])
  return logli
