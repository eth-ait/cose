"""Main training script by using TF static graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
tf.compat.v1.enable_eager_execution(config=tf_config)

from common.constants import Constants as C
from smartink.source.training_eager import TrainingEngine
from smartink.config.config_predictive_ink import define_flags
from smartink.config.config_predictive_ink import get_config
from smartink.config.config_predictive_ink import build_predictive_model
from smartink.config.config_predictive_ink import build_dataset

FLAGS = define_flags()


def main(argv):
  del argv
  config = get_config(FLAGS)

  # Experiment directory
  if not os.path.exists(config.experiment.model_dir):
    os.mkdir(config.experiment.model_dir)
    config.dump(config.experiment.model_dir)

  # Create Dataset
  train_data = build_dataset(config, C.RUN_EAGER, C.DATA_TRAIN)
  valid_data = build_dataset(config, C.RUN_EAGER, C.DATA_VALID)

  # Create Models
  model = build_predictive_model(config, C.RUN_EAGER)

  # Experiment directory
  if not os.path.exists(config.experiment.model_dir):
    os.mkdir(config.experiment.model_dir)
    config.dump(config.experiment.model_dir)

  # Training Engine
  training_engine = TrainingEngine(
      config=config,
      model=model,
      train_data=train_data,
      valid_data=valid_data,
      test_data=None,
      debug=True)

  # Start Training
  training_engine.run()


if __name__ == "__main__":
  tf.compat.v1.app.run()
