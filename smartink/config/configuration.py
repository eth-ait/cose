"""Configuration classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os

import tensorflow as tf

from common.constants import Constants as C


class AttrDict(dict):
  """Dictionary like configuration class."""

  # pylint: disable=useless-super-delegation
  def __init__(self, **kwargs):
    super(AttrDict, self).__init__(**kwargs)

  __setattr__ = dict.__setitem__

  def __getattr__(self, key):
    if key in self:
      val = dict.__getitem__(self, key)
      return val
    else:
      raise AttributeError

  def to_json(self):
    return json.dumps(self, indent=4, sort_keys=True)

  def __deepcopy__(self, memo):
    return self.__class__(
        **{k: copy.deepcopy(v, memo) for k, v in self.items()})


class ExperimentConfig(AttrDict):
  """A template configuration for experiment."""

  def __init__(self,
               learning_rate=None,
               max_epochs=None,
               max_steps=None,
               eval_frequency=None,
               **kwargs):
    super(ExperimentConfig, self).__init__(**kwargs)
    self.learning_rate = learning_rate
    self.max_epochs = max_epochs
    self.max_steps = max_steps
    self.eval_frequency = eval_frequency


class DataConfig(AttrDict):
  """A template configuration for data."""

  def __init__(self,
               train_data_path=None,
               valid_data_path=None,
               test_data_path=None,
               meta_data_path=None,
               batch_size=None,
               **kwargs):
    super(DataConfig, self).__init__(**kwargs)
    self.train_data_path = train_data_path
    self.valid_data_path = valid_data_path
    self.test_data_path = test_data_path
    self.meta_data_path = meta_data_path
    self.batch_size = batch_size


class DenseLayerConfig(AttrDict):
  """A template configuration for dense network."""

  def __init__(self,
               layers=None,
               units=None,
               activation=None,
               dropout_rate=0.0,
               **kwargs):
    super(DenseLayerConfig, self).__init__(**kwargs)
    self.type = C.DENSE
    self.layers = layers
    self.units = units
    self.activation = activation
    self.dropout_rate = dropout_rate


class TCNLayerConfig(AttrDict):
  """A template configuration for temporal convolutional network."""

  def __init__(self,
               layers=None,
               units=None,
               activation=None,
               filters=None,
               kernel_width=2,
               strides=1,
               dilation=1,
               **kwargs):
    super(TCNLayerConfig, self).__init__(**kwargs)
    self.type = C.TCN
    self.layers = layers
    self.units = units
    self.activation = activation
    self.filters = filters
    self.kernel_width = kernel_width
    self.strides = strides
    self.dilation = dilation


class RNNLayerConfig(AttrDict):
  """A template configuration for RNN network."""

  def __init__(self, cell_type, layers, units, activation, **kwargs):
    super(RNNLayerConfig, self).__init__(**kwargs)
    self.cell_type = cell_type
    self.layers = layers
    self.units = units
    self.activation = activation


class LossConfig(AttrDict):
  """A template configuration for defining loss terms."""

  def __init__(self, loss_type=None, target_key=None, out_key=None, weight=1.0,
               reduce_type=C.R_MEAN_STEP, eval_only=False, **kwargs):
    super(LossConfig, self).__init__(**kwargs)
    self.loss_type = loss_type  # see constants for the options.
    # key/name of the tensorflow op. looks for <out_key> in model outputs.
    self.out_key = out_key
    # key/name of the target data placeholder.
    self.target_key = target_key
    self.weight = weight
    self.reduce = reduce_type
    self.eval_only = eval_only


class Configuration(AttrDict):
  """Main configuration class for defining models and experiments."""

  # pylint: disable=useless-super-delegation
  def __init__(self, **kwargs):
    super(Configuration, self).__init__(**kwargs)

  def dump(self, path):
    json.dump(
        self,
        tf.io.gfile.GFile(os.path.join(path, 'config.json'), 'w'),
        indent=4,
        sort_keys=True)

  def to_json(self):
    return json.dumps(self, indent=4, sort_keys=True)

  @classmethod
  def from_json(cls, path):
    """Loads json configuration."""
    config_dict = json.load(open(os.path.join(path), 'r'))

    def dict_to_attr(parent_value_):
      if isinstance(parent_value_, dict):
        new_attr_obj = AttrDict()
        for key_, value_ in parent_value_.items():
          new_attr_obj[key_] = dict_to_attr(value_)
        return new_attr_obj
      else:
        return parent_value_

    new_config = Configuration()
    for key_, value_ in config_dict.items():
      new_config[key_] = dict_to_attr(value_)

    return new_config
