"""Embedding data for visualization with Tensorboard Projector.

Given an experiment id, loads the model and evaluates a number of test samples
to get the embeddings. Saves the embeddings, corresponding images and meta data.

Note that the output .tsv files can also be visualized via
https://projector.tensorflow.org/.

In order to visualize embeddings locally with tensorboard,
`projector_checkpoint.py` script should be run to create checkpoint and
projector configuration.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os

from smartink.config.config_embedding import build_dataset
from smartink.config.config_embedding import build_embedding_model
from smartink.config.config_embedding import get_config
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from common.constants import Constants as C
from smartink.util.utils import dict_tf_to_numpy
import tensorflow as tf

tf.compat.v1.enable_eager_execution()


def create_sprite(data):
  """Tiles images into sprite image by adding any necessary padding.

  Args:
    data: tensor of images.

  Returns:
  """

  # For B&W or greyscale images
  if len(data.shape) == 3:
    data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))

  n = int(np.ceil(np.sqrt(data.shape[0])))
  padding = ((0, n**2 - data.shape[0]), (0, 0), (0, 0), (0, 0))
  data = np.pad(data, padding, mode="constant", constant_values=0)

  # Tile images into sprite
  data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3, 4))
  # print(data.shape) => (n, image_height, n, image_width, 3)

  data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
  # print(data.shape) => (n * image_height, n * image_width, 3)
  return data


def stroke_to_image(stroke, dpi, width_ratio=1, height_ratio=1):
  """Creates an image of a given stroke.

  The image size is determined by the dpi and given w&h ratios:
  height = dpi*height_ratio
  weight = dpi*width_ratio

  Args:
    stroke: numpy array of shape (# of points, 3)
    dpi:
    width_ratio:
    height_ratio:

  Returns:
    Image as numpy array of shape (width, height, 3).
  """
  fig = Figure(figsize=(width_ratio, height_ratio), dpi=dpi)
  canvas = FigureCanvasAgg(fig)
  ax = fig.gca()

  ax.axis("off")
  ax.plot(stroke[:-1, 0], stroke[:-1, 1], lw=2, color="k")
  canvas.draw()

  width, height = width_ratio * dpi, height_ratio * dpi
  img = np.fromstring(canvas.tostring_rgb(), dtype="uint8")
  img = img.reshape(height, width, 3)
  return img


def ink_batch_to_strokes(undo_fn, ink_batch, ink_start=None, seq_len=None):
  """Converts a batch of strokes (numpy tensor) to list of strokes.

  First reverts the preprocessing steps applied.
  The return value can be passed to visualization functions directly.
  Args:
    undo_fn: a function gets a batch of strokes and reverts preprocessing steps
      such as normalization, first order derivative, etc.
    ink_batch:
    ink_start:
    seq_len:

  Returns:
  """

  ink_batch, seq_len = undo_fn(ink_batch, ink_start, seq_len)
  # y dimension is mirrored.
  ink_batch[:, :, 1] = -1 * ink_batch[:, :, 1]
  # Discard paddings.
  strokes = []
  for i in range(ink_batch.shape[0]):
    strokes.append(ink_batch[i][:seq_len[i]])
  return strokes


def apply_noise_perturbations(decode_fn, embeddings, n_noise_samples, seq_lens):
  """Applies Gaussian noise on the embeddings and decodes.

  The amount of noise increases as the number of noise samples increases. For
  every stroke sample in `embeddings`, creates n_noise_samples many noisy
  counterparts. Hence, the result shape is (# of embeddings * n_noise_samples)
  in the first dimension.

  Args:
    decode_fn: model's decoding method expecting embeddings and prediction
      length per stroke.
    embeddings: of shape (# stroke samples, stroke size)
    n_noise_samples:
    seq_lens:

  Returns:
    Decoding results of the original and noisy embeddings.
  """
  if isinstance(embeddings, dict):
    embeddings = embeddings["mu"]

  n_embeddings = embeddings.shape[0]
  emb_size = embeddings.shape[1]

  # Create noise tensor: increase the amount of the noise.
  noise = np.zeros((n_embeddings, n_noise_samples, emb_size))

  for i in range(n_noise_samples):
    noise[:, i, :] = np.random.normal(0, 0.1 * (i + 1),
                                      (n_embeddings, emb_size))
  # Insert "noise" with zeros to get the original stroke.
  zero_noise = np.zeros((n_embeddings, 1, emb_size))
  noise = np.concatenate([zero_noise, noise], axis=1)

  rep_embeddings = np.repeat(embeddings, n_noise_samples + 1, axis=0)
  rep_embeddings += np.reshape(noise, (-1, emb_size))
  inp_embeddings = rep_embeddings.astype(np.float32)
  inp_seq_lens = np.repeat(seq_lens, n_noise_samples + 1, axis=0)

  out_ = decode_fn(inp_embeddings, inp_seq_lens)
  out_[C.INP_SEQ_LEN] = inp_seq_lens
  out_["embeddings"] = inp_embeddings
  
  return out_


def create_embedding_data(config, model, dataset):
  """Loads a model and evaluates data samples to get embeddings.

  Saves stroke data in numpy and tsv, sample images as sprite and meta data.
  Args:
    config: experiment configuration dictionary.
    model: model object.
    dataset: dataset object.
  Returns:
  Raises:
    Exception: if model checkpoint not found.
  """
  fname = "{}_{}_{}".format(config.experiment.id,
                            config.experiment.tag.split("_")[0],
                            config.experiment.tag.split("_")[1])
  out_path = os.path.join("projector_models", fname)
  if not os.path.exists(out_path):
    os.mkdir(out_path)

  sample_ids = list(range(800))
  img_size = 64
  sample_ids_for_noise = [5]
  latent_start_end_id_pairs = [[0, 2], [0, 3], [1, 3]]

  model_restored = False
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint_path = tf.train.latest_checkpoint(config.experiment.model_dir)
  if checkpoint_path is None:
    raise Exception("Checkpoint not found.")
  else:
    print("Loading model " + checkpoint_path)
  checkpoint.restore(checkpoint_path)

  model_embeddings = list()
  stroke_images = list()
  seq_lens = list()
  modification_labels = list()

  for idx in range(1, max(sample_ids) + 1):
    input_batch, target_batch = dataset.get_next()

    # tf.keras restores weights only after the first call :(
    if not model_restored:
      _ = model(inputs=input_batch, training=False)
      model_restored = True

    # Get the embeddings.
    target_batch = dict_tf_to_numpy(target_batch)
    embeddings = model.call_encode(
        input_batch[C.INP_ENC], input_batch[C.INP_SEQ_LEN], training=False)
    embeddings = embeddings["mu"].numpy()

    # Perturb an stroke and label decoded strokes for visualization.
    if idx in sample_ids_for_noise:
      n_noise_samples = 10
      out_ = apply_noise_perturbations(model.decode_sequence, embeddings,
                                       n_noise_samples, target_batch["seq_len"])
      out_ = dict_tf_to_numpy(out_)

      noise_labels = np.array([
          "noise_{}_{}".format(idx, sidx) for sidx in range(embeddings.shape[0])
      ])
      noise_labels = np.repeat(noise_labels, n_noise_samples + 1, axis=0)
      modification_labels.extend(noise_labels)
      model_embeddings.extend(out_["embeddings"])
      seq_lens.extend(out_["seq_len"])

      # Get stroke images to use as thumbnails.
      ink_ = np.concatenate([out_["stroke"], out_["pen"]], axis=-1)
      strokes = ink_batch_to_strokes(dataset.np_undo_preprocessing, ink_, None,
                                     out_["seq_len"])
      for stroke in strokes:
        stroke_images.append(stroke_to_image(stroke, dpi=img_size))
        
      # Latent walk
      n_latent_samples = 10
      for start_id, end_id in latent_start_end_id_pairs:
        latent_seq_len = np.max([target_batch["seq_len"][start_id], target_batch["seq_len"][end_id]])
        out_latent_walk = model.latent_walk(embeddings[start_id:start_id + 1],
                                        embeddings[end_id:end_id + 1],
                                        n_latent_samples,
                                        latent_seq_len)
        out_latent_walk = dict_tf_to_numpy(out_latent_walk)
        noise_labels = np.array(["latent_walk_{}_{}_{}".format(idx, start_id, end_id)])
        noise_labels = np.repeat(noise_labels, n_latent_samples, axis=0)
        modification_labels.extend(noise_labels)
        model_embeddings.extend(out_latent_walk["embeddings"])
        seq_lens.extend(np.array([latent_seq_len]*n_latent_samples))

        # Get stroke images to use as thumbnails.
        ink_ = np.concatenate([out_latent_walk["stroke"], out_latent_walk["pen"]], axis=-1)
        strokes = ink_batch_to_strokes(dataset.np_undo_preprocessing, ink_,
                                       None,
                                       np.array([latent_seq_len]*n_latent_samples))
        for stroke in strokes:
          stroke_images.append(stroke_to_image(stroke, dpi=img_size))

    else:
      modification_labels.extend(np.zeros_like(target_batch["seq_len"]))
      model_embeddings.extend(embeddings)
      seq_lens.extend(target_batch["seq_len"])

      # Get stroke images to use as thumbnails.
      ink_ = np.concatenate([target_batch["stroke"], target_batch["pen"]],
                            axis=-1)
      strokes = ink_batch_to_strokes(dataset.np_undo_preprocessing, ink_, None,
                                     target_batch["seq_len"])
      for stroke in strokes:
        stroke_images.append(stroke_to_image(stroke, dpi=img_size))

  print("# of embeddings: " + str(len(model_embeddings)))

  np.save(os.path.join(out_path, "embeddings"), model_embeddings)

  stroke_images = np.array(stroke_images)
  sprite_img = create_sprite(stroke_images)
  cv2.imwrite(os.path.join(out_path, "sprite_img.png"), sprite_img)

  out_v = io.open(
      os.path.join(out_path, "embeddings.tsv"), "w", encoding="utf-8")
  out_m = io.open(os.path.join(out_path, "meta.tsv"), "w", encoding="utf-8")

  out_m.write("length\tmodifications\n")
  for idx, emb in enumerate(model_embeddings):
    out_m.write(
        str(seq_lens[idx]) + "\t" + str(modification_labels[idx]) + "\n")
    out_v.write("\t".join([str(x) for x in emb]) + "\n")
  out_v.close()
  out_m.close()


def main(argv):
  del argv
  config = get_config()

  if not os.path.exists(config.experiment.eval_dir):
    os.mkdir(config.experiment.eval_dir)
  config.dump(config.experiment.eval_dir)

  dataset = build_dataset(config, C.RUN_EAGER, C.DATA_TEST)
  model = build_embedding_model(config, C.RUN_EAGER)

  create_embedding_data(config, model, dataset)


if __name__ == "__main__":
  tf.compat.v1.app.run()
