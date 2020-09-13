"""Evaluation script running in eager mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

import random
import numpy as np
import tensorflow as tf

from smartink.config.config_predictive_ink import restore_config as predictive_restore_config
from smartink.config.config_predictive_ink import build_dataset as predictive_build_dataset
from smartink.config.config_predictive_ink import build_predictive_model

from smartink.config.config_embedding import restore_config as embedding_restore_config
from smartink.config.config_embedding import build_dataset as embedding_build_dataset
from smartink.config.config_embedding import build_embedding_model

from common.constants import Constants as C
from smartink.source.eval_engine import EvalEngine
from smartink.util.utils import NotPredictiveModelError
from smartink.util.utils import ModelNotFoundError


gpu = tf.config.experimental.list_physical_devices('GPU')[0]
if gpu:
  try:
    tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


def main(argv):
  del argv

  parser = argparse.ArgumentParser()
  parser.add_argument('--model_ids', required=True, help='Experiment ID (experiment timestamp).')
  parser.add_argument('--quantitative', required=False, action="store_true", help='Quantitative analysis.')
  parser.add_argument('--qualitative', required=False, action="store_true",help='Qualitative analysis.')
  parser.add_argument('--embedding_analysis', required=False, action="store_true",help='Quantitative analysis of the embeddings.')
  parser.add_argument('--random_samples', required=False, type=int, default=0, help='Qualitative analysis with random samples. Uses random_samples many samples.')

  parser.add_argument("--experiment_dir", required=False, type=str, default=None, help="Experiment save directory.")
  parser.add_argument("--eval_dir", required=False, type=str, default=None, help="Where to save evaluation results.")
  parser.add_argument("--data_dir", required=False, type=str, default=None, help="Where to look for data.")

  args = parser.parse_args()
  if ',' in args.model_ids:
    model_ids = args.model_ids.split(',')
  else:
    model_ids = [args.model_ids]

  for model_id in model_ids:
    print()
    print()
    
    try:
      # Try loading as a predictive model.
      config = predictive_restore_config(args, model_id)

      dataset = predictive_build_dataset(config, C.RUN_EAGER, C.DATA_TEST)
      predictive_model = build_predictive_model(config, C.RUN_EAGER)
      embedding_model = predictive_model.embedding_model
      
    except ModelNotFoundError:
      print("Skipping model " + model_id + ": not found.")
      continue
      
    except NotPredictiveModelError:
      try:
        config = embedding_restore_config(args, model_id)
    
        dataset = embedding_build_dataset(config, C.RUN_EAGER, C.DATA_TEST)
        embedding_model = build_embedding_model(config, C.RUN_EAGER)
        predictive_model = None
        
      except ModelNotFoundError:
        print("Skipping model " + model_id + ": not found.")
        continue
      
    if not os.path.exists(config.experiment.eval_dir):
      os.mkdir(config.experiment.eval_dir)
    config.dump(config.experiment.eval_dir)

    try:
      if config.data.data_name == "didi":
        sample_ids = [7, 9, 11, 19]
      elif config.data.data_name == "didi_wo_text":
        # sample_ids = [1, 2, 30] #[10, 17, 29, 30, 1, 64, 73]
        sample_ids = [1, 2, 10, 17, 33, 41, 55, 63, 64]
      elif config.data.data_name == "didi_all":
        sample_ids = [5, 19, 28]
      elif config.data.data_name == "didi_wo_text_rdp":
        sample_ids = [2, 9, 10, 17, 29, 30]
      elif config.data.data_name == "quickdraw_cats":
        sample_ids = [2, 3, 5, 9, 10, 17, 29, 30, 49, 100, 150, 200]
      elif config.data.data_name == "quickdraw_elephant":
        sample_ids = [12, 23, 52, 75, 78, 80, 90, 100, 110]
      elif config.data.data_name == "iamondb":
        sample_ids = [1, 2, 3, 4, 5]
      else:
        raise Exception("Dataset {} not recognized.".format(config.data.data_name))
      
      if args.random_samples > 0:
        r = list(range(500))
        random.shuffle(r)
        sample_ids = r[:args.random_samples]
      
      eval_engine = EvalEngine(config, dataset, embedding_model, predictive_model, glog=True)
      if args.qualitative:
        eval_engine.qualitative_eval(sample_ids)

      if args.quantitative:
        if args.qualitative:
          dataset.make_one_shot_iterator()  # Reset iterator.
        eval_engine.quantitative_eval(np.inf)
      
      if args.embedding_analysis:
        if args.qualitative or args.quantitative:
          dataset.make_one_shot_iterator()  # Reset iterator.
        eval_engine.embedding_eval(glog_entry=True)
        eval_engine.embedding_eval(glog_entry=True, metric="cosine")
      
    except Exception as e:
      print("Something went wrong when evaluating model {}".format(model_id))
      raise Exception(e)

if __name__ == "__main__":
  tf.compat.v1.app.run()
