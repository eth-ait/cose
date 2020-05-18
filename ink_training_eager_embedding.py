"""Main training script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from absl import app

from common.constants import Constants as C
from smartink.source.training_eager import TrainingEngine
from smartink.config.config_embedding import define_flags
from smartink.config.config_embedding import get_config
from smartink.config.config_embedding import build_embedding_model
from smartink.config.config_embedding import build_dataset

FLAGS = define_flags()

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
if gpu:
  try:
    tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


def main(argv):
  del argv
  config = get_config(FLAGS)

  # Create Dataset
  train_data = build_dataset(config, C.RUN_EAGER, C.DATA_TRAIN)
  valid_data = build_dataset(config, C.RUN_EAGER, C.DATA_VALID)

  # Create Model
  model = build_embedding_model(config, C.RUN_EAGER)

  # Training Engine
  training_engine = TrainingEngine(
      config=config,
      model=model,
      train_data=train_data,
      valid_data=valid_data,
      test_data=None,
      debug=False)

  # Start Training
  training_engine.run()


if __name__ == "__main__":
  app.run(main)
