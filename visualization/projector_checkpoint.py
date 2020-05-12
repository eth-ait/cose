"""Creates checkpoint for visualizing embeddings via Tensorboard Projector.

`projector_embedding_data.py` should be run prior to this script.
Note that this script is useful if you prefer to use Tensorboard locally.

Tensorboard should be started for each model as
> tensorboard --logdir projector_models/<fname>
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from smartink.config.config_embedding import get_config
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


def create_checkpoint(config):
  """Creates checkpoint and Projector config for Tensorboard.

  Args:
    config: experiment config dictionary.

  Returns:
  """
  fname = "{}_{}_{}".format(config.experiment.id,
                            config.experiment.tag.split("_")[0],
                            config.experiment.tag.split("_")[1])
  out_path = os.path.join("projector_models", fname)
  relative_path = ""
  img_size = 64

  model_embeddings = np.load(os.path.join(out_path, "embeddings.npy"))
  print("# of embeddings: " + str(len(model_embeddings)))

  meta_file = os.path.join(relative_path, "meta.tsv")
  sprite_file = os.path.join(relative_path, "sprite_img.png")

  features = tf.Variable(model_embeddings, name=fname)

  # Write summaries for tensorboard
  with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.Saver([features], save_relative_paths=True)

    sess.run(features.initializer)
    saver.save(sess, os.path.join(out_path, "tb_embeddings.ckpt"))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = features.name
    embedding.metadata_path = meta_file

    # This adds the sprite images
    embedding.sprite.image_path = sprite_file
    embedding.sprite.single_image_dim.extend((img_size, img_size))
    projector.visualize_embeddings(tf.compat.v1.summary.FileWriter(out_path), config)


def main(argv):
  del argv
  config = get_config()
  create_checkpoint(config)


if __name__ == "__main__":
  tf.compat.v1.app.run()
