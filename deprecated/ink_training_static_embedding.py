"""Main training script by using TF static graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from common.constants import Constants as C
from smartink.source.training_static import TrainingEngine
from smartink.config.config_embedding import define_flags
from smartink.config.config_embedding import get_config
from smartink.config.config_embedding import build_embedding_model
from smartink.config.config_embedding import build_dataset

FLAGS = define_flags()


def main(argv):
  del argv
  config = get_config(FLAGS)

  # Experiment directory
  if not os.path.exists(config.experiment.model_dir):
    os.mkdir(config.experiment.model_dir)
    config.dump(config.experiment.model_dir)

  # Create Dataset
  train_data = build_dataset(config, C.RUN_STATIC, C.DATA_TRAIN)
  valid_data = build_dataset(config, C.RUN_STATIC, C.DATA_VALID)

  # Create Model
  model = build_embedding_model(config, C.RUN_STATIC)

  # Training Engine
  training_engine = TrainingEngine(
      config=config,
      model=model,
      train_data=train_data,
      valid_data=valid_data,
      test_data=None)

  # Start Training
  training_engine.run()

  # # Run evaluation.
  # print("Evaluating model...")
  # if not tf.gfile.Exists(config.experiment.eval_dir):
  #   tf.gfile.MakeDirs(config.experiment.eval_dir)
  #   config.dump(config.experiment.eval_dir)
  #
  # test_data = build_dataset(config, C.RUN_STATIC, C.DATA_TEST)
  #
  # vis_engine = InkVisualizer(
  #     test_data.np_undo_preprocessing,
  #     config.experiment.eval_dir,
  #     animate=False)
  # eval_engine = EvalEngine(config, model, test_data, vis_engine)
  # eval_engine.eval(vis_samples=[1, 2, 3, 4, 5, 6])
  # eval_engine.eval_autoregressive(vis_samples=[1, 2, 3, 4, 5, 6])


if __name__ == "__main__":
  tf.compat.v1.app.run()
