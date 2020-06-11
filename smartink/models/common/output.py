"""Functions building neural network components."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from common.constants import Constants as C
from smartink.util.utils import dict_tf_to_numpy


class OutputModelDeterministic(tf.keras.Model):
  """Creates output deterministic layers."""

  def __init__(self,
               out_units,
               hidden_units=0,
               hidden_layers=0,
               hidden_activation=tf.keras.activations.relu,
               kernel_regularizer=None,
               bias_regularizer=None,
               prefix=""):

    super(OutputModelDeterministic, self).__init__()
    self.out_units = out_units
    self.prefix = prefix
    if prefix and not prefix.endswith("_"):
      self.prefix += "_"

    self.initializer = tf.compat.v1.random_normal_initializer(stddev=0.001)
    # self.initializer = 'glorot_uniform'

    self.layer_out_mu = tf.keras.Sequential()
    for idx in range(hidden_layers):
      self.layer_out_mu.add(
          tf.keras.layers.Dense(
              units=hidden_units,
              activation=hidden_activation,
              name=prefix + "hidden_mu_" + str(idx)),
              kernel_regularizer=kernel_regularizer,
              bias_regularizer=bias_regularizer)
    self.layer_out_mu.add(
        tf.keras.layers.Dense(
            kernel_initializer=self.initializer,
            units=out_units,
            activation=None,
            name=prefix + "mu",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer))

  def call(self, inputs, training=None, **kwargs):
    out_dict = dict()
    out_dict[self.prefix + C.MU] = self.layer_out_mu(inputs)
    return out_dict

  def draw_sample(self, outputs, greedy=True):
    return outputs[self.prefix + C.MU]

  def draw_sample_np(self, outputs, greedy=True):
    """Draw a sample in numpy.

    Args:
      outputs (dict): the content is retrieved via "mu" key.
      greedy (bool): whether to return mean or not.

    Returns:
    """
    return outputs[self.prefix + C.MU]


class OutputModelNormal(OutputModelDeterministic):
  """Creates output layers following Normal distribution."""

  def __init__(
      self,
      out_units,
      hidden_units=0,
      hidden_layers=0,
      hidden_activation=tf.keras.activations.relu,
      prefix="",
      sigma_activation=tf.keras.activations.softplus,
      logvar=False,
  ):

    super(OutputModelNormal, self).__init__(
        out_units=out_units,
        hidden_units=hidden_units,
        hidden_layers=hidden_layers,
        hidden_activation=hidden_activation,
        prefix=prefix)

    self.logvar = logvar

    self.layer_out_sigma = tf.keras.Sequential()
    for idx in range(hidden_layers):
      self.layer_out_sigma.add(
          tf.keras.layers.Dense(
              units=hidden_units,
              activation=hidden_activation,
              name=prefix + "hidden_sigma_" + str(idx)))
    self.layer_out_sigma.add(
        tf.keras.layers.Dense(
            units=out_units,
            activation=sigma_activation,
            kernel_initializer=self.initializer,
            name=prefix + "sigma"))

  def call(self, inputs, training=None, **kwargs):
    out_dict = super(OutputModelNormal, self).call(inputs, training, **kwargs)
    out_dict[self.prefix + C.SIGMA] = self.layer_out_sigma(inputs)
    return out_dict

  def draw_sample(self, outputs, greedy=False):
    sigma = outputs.get(self.prefix + C.SIGMA, None)
    if greedy or sigma is None:
      return outputs[self.prefix + C.MU]
    if self.logvar:
      # Calculate sigma (i.e., std) from log-variance.
      sigma = tf.exp(sigma / 2.0)
    noise = tf.random.normal(shape=(tf.shape(input=sigma)))
    return noise * sigma + outputs[self.prefix + C.MU]

  def draw_sample_np(self, outputs, greedy=False):
    """Draw a Gaussian sample in numpy.

    Args:
      outputs (dict): container of mean and std statistics of a Normal
        distribution. The content is retrieved via "mu" and "sigma" keys.
      greedy (bool): whether to return mean or not.

    Returns:
    """
    sigma = outputs[self.prefix + C.SIGMA]
    if self.logvar:
      # Calculate sigma (i.e., std) from log-variance.
      sigma = np.exp(sigma / 2.0)

    noise = np.random.normal(size=sigma.shape)
    return noise * sigma + outputs[self.prefix + C.MU]

  def draw_sample_every_component(self, outputs, greedy=False):
    """Just an interface for GMM compatibility.

    Args:
      outputs: a dictionary containing mu and sigma. mu and sigma are of
        shape (batch_size, seq_len, feature_size).
      greedy: whether to return mu directly or sample.

    Returns:
      sample tensor - (batch_size, seq_len, 1, feature_size)
      dummy pi - (batch_size, seq_len, 1)
    """
    mu = outputs[self.prefix + C.MU]
    sigma = outputs[self.prefix + C.SIGMA]
  
    out_shape = tf.shape(input=mu)
    seq_len = 1
    if len(mu._shape_as_list()) == 3:
      seq_len = out_shape[1]
    batch_size = out_shape[0]
    comp_shape = (batch_size, seq_len, self.out_units, 1)
  
    mu = tf.transpose(a=tf.reshape(mu, comp_shape), perm=[0, 1, 3, 2])
    sigma = tf.transpose(a=tf.reshape(sigma, comp_shape), perm=[0, 1, 3, 2])
  
    if greedy:
      sample = mu
    else:
      sample = tf.random.normal(tf.shape(input=mu), mu, sigma/4.0)
    return sample, tf.ones_like(sample[:,:,:,0])

  def draw_sample_from_nth(self, outputs, n, greedy=False):
    """Just an interface for GMM compatibility.

    Args:
      outputs: a dictionary containing mu and sigma. mu and sigma are of
        shape (batch_size, seq_len, feature_size).
      n: component id.
      greedy: whether to return mu directly or sample.

    Returns:
      sample tensor - (batch_size, seq_len, feature_size)
      dummy pi - (batch_size, seq_len, 1)
    """
    sample_, pi_ = self.draw_sample_every_component(outputs, greedy)
    return sample_[:,:,0,:], pi_[:,:,0]


class OutputModelNormal2D(OutputModelNormal):
  """Creates output layers following 2-dimensional Normal distribution."""

  def __init__(
      self,
      out_units,
      hidden_units,
      hidden_layers=0,
      hidden_activation=tf.keras.activations.relu,
      prefix="",
      sigma_activation=tf.keras.activations.softplus,
      logvar=False,
  ):
    assert out_units == 2, "Output dimension must be 2 for 2D Normal model."
    super(OutputModelNormal2D, self).__init__(
        out_units=out_units,
        hidden_units=hidden_units,
        hidden_layers=hidden_layers,
        hidden_activation=hidden_activation,
        prefix=prefix,
        sigma_activation=sigma_activation,
        logvar=logvar)

    self.layer_out_rho = tf.keras.Sequential()
    for idx in range(hidden_layers):
      self.layer_out_rho.add(
          tf.keras.layers.Dense(
              units=hidden_units,
              activation=hidden_activation,
              name=prefix + "hidden_rho_" + str(idx)))
    self.layer_out_rho.add(
        tf.keras.layers.Dense(
            units=out_units,
            activation=tf.keras.activations.tanh,
            kernel_initializer=self.initializer,
            name=prefix + "rho"))

  def call(self, inputs, training=None, **kwargs):
    out_dict = super(OutputModelNormal2D, self).call(inputs, training, **kwargs)
    out_dict[self.prefix + C.RHO] = self.layer_out_rho(inputs)
    return out_dict

  def draw_sample(self, outputs, greedy=False):
    if greedy:
      return outputs[self.prefix + C.MU]

    # TODO Ignoring rho for tf sampling.
    sigma = outputs[self.prefix + C.SIGMA]
    if self.logvar:
      # Calculate sigma (i.e., std) from log-variance.
      sigma = tf.exp(sigma / 2.0)
    noise = tf.random.normal(shape=(tf.shape(input=sigma)))
    return noise * sigma + outputs[self.prefix + C.MU]

  def draw_sample_np(self, outputs, greedy=False):
    """Draw a Gaussian sample in numpy.

    Args:
      outputs (dict): container of mean and std statistics of a Normal
        distribution. The content is retrieved via "mu" and "sigma" keys.
      greedy (bool): whether to return mean or not.

    Returns:
    """
    if greedy:
      return outputs[self.prefix + C.MU]

    # TODO Ignoring rho for tf sampling.
    sigma = outputs[self.prefix + C.SIGMA]
    if self.logvar:
      # Calculate sigma (i.e., std) from log-variance.
      sigma = np.exp(sigma / 2.0)

    noise = np.random.normal(size=sigma.shape)
    return noise * sigma + outputs[self.prefix + C.MU]


class OutputModelGMM(OutputModelNormal):
  """Creates output layers following Gaussian Mixture Model distribution."""

  def __init__(
      self,
      out_units,
      hidden_units,
      hidden_layers=0,
      hidden_activation=tf.keras.activations.relu,
      prefix="",
      sigma_activation=tf.keras.activations.softplus,
      logvar=False,
      num_components=2,
  ):
    assert num_components > 1, "Required more than one GMM components."

    super(OutputModelGMM, self).__init__(
        out_units=out_units * num_components,
        hidden_units=hidden_units,
        hidden_layers=hidden_layers,
        hidden_activation=hidden_activation,
        prefix=prefix,
        sigma_activation=sigma_activation,
        logvar=logvar)

    self.out_units = out_units
    self.num_components = num_components

    self.layer_out_pi = tf.keras.Sequential()
    for idx in range(hidden_layers):
      self.layer_out_pi.add(
          tf.keras.layers.Dense(
              units=hidden_units,
              activation=hidden_activation,
              name=prefix + "hidden_pi_" + str(idx)))
    self.layer_out_pi.add(
        tf.keras.layers.Dense(
            units=num_components,
            activation=tf.keras.activations.softmax,
            kernel_initializer=self.initializer,
            name=prefix + "pi"))

  def call(self, inputs, training=None, **kwargs):
    out_dict = super(OutputModelGMM, self).call(inputs, training, **kwargs)
    out_dict[self.prefix + C.PI] = self.layer_out_pi(inputs)
    return out_dict

  def draw_sample(self, outputs, greedy=False):
    batch_size = tf.shape(input=outputs[self.prefix + C.MU])[0]
    seq_len = tf.shape(input=outputs[self.prefix + C.MU])[1]
    comp_shape = (batch_size, seq_len, self.out_units, self.num_components)

    pi = outputs[self.prefix + C.PI]
    mu = tf.reshape(outputs[self.prefix + C.MU], comp_shape)
    sigma = tf.reshape(outputs[self.prefix + C.SIGMA], comp_shape)
    if self.logvar:
      # Calculate sigma (i.e., std) from log-variance.
      sigma = tf.exp(sigma / 2.0)

    mu = tf.transpose(a=mu, perm=[0, 1, 3, 2])
    sigma = tf.transpose(a=sigma, perm=[0, 1, 3, 2])

    # Select the most likely mixture component.
    probs = tf.reshape(pi, (-1, self.num_components))
    logits = tf.math.log(probs) + 1.0

    if greedy:
      component_indices = tf.reshape(
          tf.argmax(input=logits, axis=1), (batch_size, seq_len))
    else:
      component_indices = tf.reshape(
          tf.random.categorical(logits, 1), (batch_size, seq_len))
    batch_indices = tf.range(batch_size)
    seq_indices = tf.range(seq_len)

    idx_grid = tf.meshgrid(batch_indices, seq_indices)
    gather_idx = tf.stack([
        tf.transpose(a=idx_grid[0]),
        tf.transpose(a=idx_grid[1]),
        tf.cast(component_indices, tf.int32)
    ],
                          axis=-1)
    component_mu = tf.gather_nd(mu, gather_idx)
    component_sigma = tf.gather_nd(sigma, gather_idx)

    if greedy:
      return component_mu
    else:
      return tf.random.normal(
          tf.shape(input=component_mu), component_mu, component_sigma / 2.0)

  def draw_sample_np(self, outputs, greedy=False):
    """Draw a Gaussian sample in numpy.

    Args:
      outputs (dict): container of mean and std statistics of a Normal
        distribution. The content is retrieved via "mu" and "sigma" keys.
      greedy (bool): whether to return mean or not.

    Returns:
    """
    raise Exception("Not implemented.")  # pylint: disable=g-doc-exception


class OutputModelNormal2DDense(tf.keras.Model):
  """Creates output layers following 2-dimensional Normal distribution."""

  def __init__(self,
               prefix="",
               sigma_activation=tf.keras.activations.exponential):
    super(OutputModelNormal2DDense, self).__init__()

    self.prefix = prefix
    self.sigma_activation = sigma_activation
    self.out_units = 2
    self.layer_out = tf.keras.layers.Dense(5, name="binormal_all")

  def call(self, inputs, training=None, **kwargs):
    out_ = self.layer_out(inputs)
    out_dict = dict()
    out_mu, out_sigma, out_rho = tf.split(out_, [2, 2, 1], axis=-1)
    out_dict[self.prefix + C.MU] = out_mu
    out_dict[self.prefix + C.SIGMA] = self.sigma_activation(out_sigma)
    out_dict[self.prefix + C.RHO] = tf.keras.activations.tanh(out_rho)
    return out_dict

  def draw_sample(self, outputs, greedy=False):
    # Greedy.
    return outputs[self.prefix + C.MU]

  def draw_sample_np(self, outputs, greedy=False):
    if greedy:
      return outputs[self.prefix + C.MU]

    outputs = dict_tf_to_numpy(outputs)
    batch_size = len(outputs[self.prefix + C.MU])
    samples = np.zeros((batch_size, 1, 2))
    for i in range(batch_size):
      mu = outputs[self.prefix + C.MU][i, 0]
      sigma = outputs[self.prefix + C.SIGMA][i, 0]
      rho = outputs[self.prefix + C.RHO][i, 0]
      samples[i, 0] = self.sample_gaussian_2d(mu, sigma, rho)
    # return np.reshape(self.sample_gaussian_2d(mu, sigma, rho), (1,1,2))
    return samples

  @classmethod
  def sample_gaussian_2d(cls, mu, sigma, rho, temp=0.5, greedy=False):
    assert mu.shape[0] == 2
    if greedy:
      return mu
    s1 = sigma[0] * temp * temp
    s2 = sigma[1] * temp * temp
    cov = [[s1 * s1, rho[0] * s1 * s2], [rho[0] * s1 * s2, s2 * s2]]
    x = np.random.multivariate_normal(mu, cov, 1)
    return x[0]


class OutputModelGMMDense(tf.keras.Model):
  """Creates output layers following Gaussian Mixture Model distribution.

  All outputs are predicted by the same model first, and then splitted.
  """

  def __init__(self,
               out_units,
               num_components,
               sigma_activation=tf.keras.activations.exponential,
               kernel_regularizer=None,
               bias_regularizer=None,
               prefix=""):
    super(OutputModelGMMDense, self).__init__()

    self.out_units = out_units
    self.num_components = num_components
    self.prefix = prefix
    self.sigma_activation = sigma_activation

    # (mu, sigma) * num_components + num_components
    self.component_size = out_units * num_components
    out_size = 2 * self.component_size + num_components
    self.layer_out = tf.keras.layers.Dense(out_size, name="gmm_all", kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)

  def call(self, inputs, training=None, **kwargs):
    out_ = self.layer_out(inputs)
    out_dict = dict()
    out_mu, out_sigma, out_pi = tf.split(
        out_, [self.component_size, self.component_size, self.num_components],
        axis=-1)
    out_dict[self.prefix + C.MU] = out_mu
    out_dict[self.prefix + C.SIGMA] = self.sigma_activation(out_sigma)
    out_dict[self.prefix + C.PI] = tf.keras.activations.softmax(out_pi)
    return out_dict
  
  def reshape_dist_params(self, outputs):
    is_2d = True
    if len(outputs[self.prefix + C.MU]._shape_as_list()) == 3:
      is_2d = False
    out_shape = tf.shape(input=outputs[self.prefix + C.MU])
    batch_size = out_shape[0]
    seq_len = 1 if is_2d else out_shape[1]
    comp_shape = (batch_size, seq_len, self.out_units, self.num_components)
  
    pi = outputs[self.prefix + C.PI]
    mu = tf.reshape(outputs[self.prefix + C.MU], comp_shape)
    sigma = tf.reshape(outputs[self.prefix + C.SIGMA], comp_shape)
    mu = tf.transpose(a=mu, perm=[0, 1, 3, 2])
    sigma = tf.transpose(a=sigma, perm=[0, 1, 3, 2])  # (batch_size, seq_len, n_components, out_units)
    return dict(mu=mu,
                sigma=sigma,
                pi=pi)
    
  def draw_sample(self, outputs, greedy=False, greedy_mu=True, temp=0.5):
    def adjust_temp(pi_pdf, temp):
      pi_pdf = tf.math.log(pi_pdf)/temp
      pi_pdf -= tf.reduce_max(pi_pdf, axis=-1)
      pi_pdf = tf.math.exp(pi_pdf)
      pi_pdf /= tf.reduce_sum(pi_pdf, axis=-1)
      return pi_pdf
    
    is_2d = True
    if len(outputs[self.prefix + C.MU]._shape_as_list()) == 3:
      is_2d = False
    out_shape = tf.shape(input=outputs[self.prefix + C.MU])
    batch_size = out_shape[0]
    seq_len = 1 if is_2d else out_shape[1]
    comp_shape = (batch_size, seq_len, self.out_units, self.num_components)

    pi = outputs[self.prefix + C.PI]
    mu = tf.reshape(outputs[self.prefix + C.MU], comp_shape)
    sigma = tf.reshape(outputs[self.prefix + C.SIGMA], comp_shape)

    mu = tf.transpose(a=mu, perm=[0, 1, 3, 2])
    sigma = tf.transpose(a=sigma, perm=[0, 1, 3, 2])

    probs = tf.reshape(pi, (-1, self.num_components))
    if greedy:
      logits = tf.math.log(probs) + 1.0
      component_indices = tf.reshape(
          tf.argmax(input=logits, axis=1), (batch_size, seq_len))
    else:
      probs = adjust_temp(probs, temp)
      logits = tf.math.log(probs) + 1.0
      component_indices = tf.reshape(
          tf.random.categorical(logits, 1), (batch_size, seq_len))
    batch_indices = tf.range(batch_size)
    seq_indices = tf.range(seq_len)

    idx_grid = tf.meshgrid(batch_indices, seq_indices)
    gather_idx = tf.stack([
        tf.transpose(a=idx_grid[0]),
        tf.transpose(a=idx_grid[1]),
        tf.cast(component_indices, tf.int32)
    ], axis=-1)
    component_mu = tf.gather_nd(mu, gather_idx)
    component_sigma = tf.gather_nd(sigma, gather_idx)

    if greedy_mu:
      sample=component_mu
    else:
      sample=tf.random.normal(
          tf.shape(input=component_mu), component_mu, component_sigma*temp*temp)
    # sample = component_mu
    if is_2d:
      return sample[:, 0]
    else:
      return sample

  def draw_sample_np(self, outputs, greedy=False):
    return self.draw_sample(outputs, greedy)
  
  def draw_sample_every_component(self, outputs, greedy=False):
    """Draws a sample from every GMM component.
    
    Args:
      outputs: a dictionary containing mu, sigma and pi. mu and sigma are of
        shape (batch_size, seq_len, feature_size*n_components).
        pi is of shape (batch_size, seq_len, n_components).
      greedy: whether to return mu directly or sample.

    Returns:
      sample tensor - (batch_size, seq_len, n_components, feature_size)
      pi values - (batch_size, seq_len, n_components)
    """
    mu = outputs[self.prefix + C.MU]
    sigma = outputs[self.prefix + C.SIGMA]
    pi = outputs[self.prefix + C.PI]
    
    out_shape = tf.shape(input=mu)
    seq_len = 1
    if len(mu._shape_as_list()) == 3:
      seq_len = out_shape[1]
    batch_size = out_shape[0]
    comp_shape = (batch_size, seq_len, self.out_units, self.num_components)
  
    mu = tf.transpose(a=tf.reshape(mu, comp_shape), perm=[0, 1, 3, 2])
    sigma = tf.transpose(a=tf.reshape(sigma, comp_shape), perm=[0, 1, 3, 2])
    pi = tf.reshape(pi, [batch_size, seq_len, self.num_components])
  
    if greedy:
      sample = mu
    else:
      sample = tf.random.normal(tf.shape(input=mu), mu, sigma/4.0)
    return sample, pi

  def draw_sample_from_nth(self, outputs, n, greedy=False):
    """Draws a sample from the nth component.

    Args:
      outputs: a dictionary containing mu, sigma and pi. mu and sigma are of
        shape (batch_size, seq_len, feature_size*n_components).
        pi is of shape (batch_size, seq_len, n_components).
      n: component id.
      greedy: whether to return mu directly or sample.

    Returns:
      sample tensor - (batch_size, seq_len, feature_size)
      pi values - (batch_size, seq_len)
    """
    assert n < self.num_components
    
    mu = outputs[self.prefix + C.MU]
    sigma = outputs[self.prefix + C.SIGMA]
    pi = outputs[self.prefix + C.PI]
  
    out_shape = tf.shape(input=mu)
    seq_len = 1
    if len(mu._shape_as_list()) == 3:
      seq_len = out_shape[1]
    batch_size = out_shape[0]
    comp_shape = (batch_size, seq_len, self.out_units, self.num_components)
  
    mu = tf.reshape(mu, comp_shape)[:, :, :, n]
    sigma = tf.reshape(sigma, comp_shape)[:, :, :, n]
    pi = tf.reshape(pi, [batch_size, seq_len, self.num_components])[:, :, n]
  
    if greedy:
      sample = mu
    else:
      sample = tf.random.normal(tf.shape(input=mu), mu, sigma/4.0)
    return sample, pi
