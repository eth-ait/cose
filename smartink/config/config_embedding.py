"""Default configuration for stroke experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os

from absl import flags
import builtins as __builtin__

from common.print_function import Print
from common.constants import Constants as C
from smartink.config.configuration import AttrDict
from smartink.config.configuration import Configuration
from smartink.config.configuration import DataConfig
from smartink.config.configuration import ExperimentConfig
from smartink.config.configuration import LossConfig
from smartink.util.utils import err_unknown_type
from smartink.data.stroke_dataset import TFRecordStroke
from smartink.data.stroke_dataset import TFRecordSingleDiagram
from smartink.data.ink_dataset import TFRecordInkSequence
from smartink.models.stroke.t_emb import TEmbedding
from smartink.models.stroke.seq2seq import InkSeq2Seq


def define_flags():
  flags.DEFINE_string("experiment_id", None, "experiment id")
  flags.DEFINE_string("comment", "", "brief description.")
  flags.DEFINE_string("gdrive_api_key", None, "path to google drive API key for glogger.")
  flags.DEFINE_string("experiment_dir", None, "path where to log")
  flags.DEFINE_string("eval_dir", None, "path where to save evaluation results.")
  flags.DEFINE_string("data_dir", None, "path where to look for data.")
  flags.DEFINE_string("data_name", "didi_wo_text", "dataset name.")
  flags.DEFINE_string("data_tfrecord_fname", "diagrams_wo_text_20200131-?????-of-?????", "tfrecord data file name")
  flags.DEFINE_string("data_meta_fname", "didi_wo_text-stats-origin_abs_pos.npy", "meta data file name")
  flags.DEFINE_string("metadata_type", "position", "position or velocity.")
  # Data preprocessing.
  flags.DEFINE_bool("skip_normalization", False, "")
  flags.DEFINE_bool("mask_encoder_pen", False, "whether to mask pen information for encoder or not.")
  flags.DEFINE_integer("resampling_factor", 0, "Temporal resampling rate. Randomly sampled between 1 and the given value.")
  flags.DEFINE_float("scale_factor", 0, "Amount of scaling.")
  flags.DEFINE_float("affine_prob", 0, "Chance of applying affine transf.")
  flags.DEFINE_float("reverse_prob", 0, "Chance of reverting a sequence.")
  flags.DEFINE_integer("n_t_samples", 1, "# of t samples per sequence.")
  flags.DEFINE_boolean("int_t_samples", False, "whether to interpolate t targets or not.")
  flags.DEFINE_boolean("concat_t_inputs", False, "whether to concatenate input points with t.")
  flags.DEFINE_float("t_drop_ratio", 0, "Drop ratio of steps in temporal resampling.")
  flags.DEFINE_boolean("gt_targets", False, "whether to keep the ground-truth targets after pre-processing or not.")
  flags.DEFINE_boolean("ink_dataset", False, "whether to use stroke or full ink representation.")
  flags.DEFINE_boolean("rdp_dataset", False, "whether to use shorter rdp version or the full.")
  flags.DEFINE_boolean("rdp_didi_pp", False, "whether to apply didi preprocessing or not.")
  
  # Experiment details
  flags.DEFINE_integer("batch_size", 100, "batch size for training")
  flags.DEFINE_float("learning_rate", 0.001, "initial learning rate.")
  flags.DEFINE_string("learning_rate_type", "exponential",
                      "LR scheduler: exponential, sketch_rnn or transformer")
  flags.DEFINE_float("grad_clip_norm", 0, "gradient clip by norm threshold.")
  flags.DEFINE_float("grad_clip_value", 0, "gradient clip by value threshold. grad_clip_norm is prioritized.")
  
  flags.DEFINE_string("encoder_model", "rnn", "rnn or transformer.")
  flags.DEFINE_string("stroke_loss", C.NLL_BINORMAL,
                      "nll_normal, nll_binormal, nll_gmm, mse or l1")
  flags.DEFINE_bool("disable_pen_loss", False,
                    "whether to disable pen loss or not.")
  
  flags.DEFINE_string("decoder_model", "t_emb", "stroke decoder: t_emb or rnn.")
  
  # Encoder RNN
  flags.DEFINE_integer("encoder_rnn_layers", 1,
                       "number of layers in encoder rnn model.")
  flags.DEFINE_integer("encoder_rnn_units", 512,
                       "number of units per layer in encoder rnn model.")
  flags.DEFINE_string("encoder_cell_type", C.LSTM,
                      "type of rnn cell instance: " + C.GRU + " or " + C.LSTM)
  flags.DEFINE_bool("bidirectional_encoder", False,
                    "whether to use bidirectional rnn as encoder or not.")
  flags.DEFINE_float("encoder_rdropout", 0.0, "Recurrent ropout rate for the "
                                              "encoder cell.")
  
  # Encoder Transformer
  flags.DEFINE_integer("transformer_dmodel", 64, "representation size.")
  flags.DEFINE_integer("transformer_layers", 6, "number of transformer layers.")
  flags.DEFINE_integer("transformer_heads", 4, "number of attention head.")
  flags.DEFINE_integer("transformer_hidden_units", 256,
                       "size of point_wise_feed_forward_network.")
  flags.DEFINE_float("transformer_dropout", 0, "dropout rate.")
  flags.DEFINE_bool("transformer_pos_encoding", False, "Positional encoding.")
  flags.DEFINE_bool("transformer_scale", False, "Scaling stroke.")
  
  # Embedding
  flags.DEFINE_integer("latent_units", 8,
                       "number of units for stroke embeddings.")
  flags.DEFINE_bool("use_vae", False, "VAE regularizer on the stroke space.")
  flags.DEFINE_float("kld_weight", 1.0, "VAE KLD loss weight.")
  flags.DEFINE_float("kld_start", 0.01, "Initial VAE KLD loss weight.")
  flags.DEFINE_float("kld_increment", 0.0, "VAE KLD loss weight increment per step (0.99995).")
  flags.DEFINE_string("kld_type", "kld_p0", "kld_p0 or kld_p0_norm.")
  # t-decoder
  flags.DEFINE_integer("decoder_layers", 4, "number of dense layers in decoder.")
  flags.DEFINE_list("decoder_hidden_units", "512,512,512,512",
                    "list of hidden units in decoder.")
  flags.DEFINE_string("decoder_activation", C.RELU, "activation function.")
  flags.DEFINE_float("decoder_dropout", 0.0, "Dropout rate after every layer.")
  flags.DEFINE_integer("t_frequency_channels", 0, "# of frequency channels for t frequency encoding.")
  
  # seq2seq decoder
  flags.DEFINE_bool("decoder_dynamic_h0", False, "Parameterized initial rnn state.")
  flags.DEFINE_bool("repeat_vae_sample", False, "Use the same latent sample in all prediction steps.")
  flags.DEFINE_bool("decoder_autoregressive", False, "Feed the decoder with the previous step as well.")
  
  # regularizers
  flags.DEFINE_float("reg_emb_weight", 0.0, "Embedding L2 norm weight")
  flags.DEFINE_float("reg_dec_weight", 0.0, "Decoder parameters L2 norm weight")
  
  return flags.FLAGS


def get_config(FLAGS, experiment_id=None):
  """Defines the default configuration."""
  experiment_id = FLAGS.experiment_id or experiment_id

  config = Configuration()
  config.experiment = ExperimentConfig(
      comment=FLAGS.comment,
      tag="",  # Set automatically.
      model_dir=None,  # Set automatically.
      eval_dir=None,  # Set automatically.
      id=experiment_id,
      max_epochs=None,
      max_steps=200000,
      log_frequency=100,
      eval_steps=500,
      checkpoint_frequency=500,
      grad_clip_norm=FLAGS.grad_clip_norm if FLAGS.grad_clip_value <= 0 else 0,
      grad_clip_value=FLAGS.grad_clip_value)
  
  config.experiment.learning_rate = AttrDict(
      name=FLAGS.learning_rate_type,
      initial_learning_rate=FLAGS.learning_rate,
      )
  if FLAGS.learning_rate_type == "transformer":
    config.experiment.learning_rate.d_model=FLAGS.transformer_dmodel
    config.experiment.learning_rate.warmup_steps=4000
  
  config.data = DataConfig(
      data_dir=FLAGS.data_dir,
      data_name=FLAGS.data_name,
      data_tfrecord_fname=C.DATASET_MAP[FLAGS.data_name]["data_tfrecord_fname"],
      data_meta_fname=C.DATASET_MAP[FLAGS.data_name][FLAGS.metadata_type],
      pp_to_origin="position" in FLAGS.metadata_type,
      pp_relative_pos="velocity" in FLAGS.metadata_type,
      normalize=not FLAGS.skip_normalization,
      batch_size=FLAGS.batch_size,
      max_length_threshold=201,
      mask_pen=FLAGS.mask_encoder_pen,
      resampling_factor=FLAGS.resampling_factor,
      t_drop_ratio=FLAGS.t_drop_ratio,
      gt_targets=FLAGS.gt_targets,
      scale_factor=FLAGS.scale_factor,
      affine_prob=FLAGS.affine_prob,
      reverse_prob=FLAGS.reverse_prob,
      n_t_samples=FLAGS.n_t_samples,
      int_t_samples=FLAGS.int_t_samples,
      concat_t_inputs=FLAGS.concat_t_inputs,
      ink_dataset=FLAGS.ink_dataset,
      rdp_dataset=FLAGS.rdp_dataset,
      rdp_didi_pp=FLAGS.rdp_didi_pp,
  )
  config.gdrive = AttrDict(
      credential=None,  # Set automatically below.
      workbook="1D__n4IEKz_zMmknL5_qL7ILVQgWC5Z6wx-fggKQJDtw",
      sheet=FLAGS.data_name,
  )
  if FLAGS.encoder_model == "rnn":
    config.encoder = AttrDict(
        name="rnn",
        cell_units=FLAGS.encoder_rnn_units,
        cell_layers=FLAGS.encoder_rnn_layers,
        cell_type=FLAGS.encoder_cell_type,
        bidirectional_encoder=FLAGS.bidirectional_encoder,
        rec_dropout_rate=FLAGS.encoder_rdropout)
  elif FLAGS.encoder_model == "transformer":
    config.encoder = AttrDict(
        name="transformer",
        layers=FLAGS.transformer_layers,
        heads=FLAGS.transformer_heads,
        d_model=FLAGS.transformer_dmodel,
        hidden_units=FLAGS.transformer_hidden_units,
        dropout_rate=FLAGS.transformer_dropout,
        pos_encoding=config.data.max_length_threshold
        if FLAGS.transformer_pos_encoding else 0,
        scale=FLAGS.transformer_scale,
        autoregressive=not FLAGS.bidirectional_encoder)
  else:
    err_unknown_type(FLAGS.encoder_model)

  config.embedding = AttrDict(
      latent_units=FLAGS.latent_units,
      use_vae=FLAGS.use_vae,
  )
  
  if FLAGS.decoder_model == "rnn":
    config.decoder = AttrDict(
        name="rnn",
        cell_units=FLAGS.encoder_rnn_units,  # Using the same hyper-param with the encoder.
        cell_layers=FLAGS.encoder_rnn_layers,
        cell_type=FLAGS.encoder_cell_type,
        dropout_rate=FLAGS.decoder_dropout,
        dynamic_h0=FLAGS.decoder_dynamic_h0,
        repeat_vae_sample=FLAGS.repeat_vae_sample,
        autoregressive=FLAGS.decoder_autoregressive,
        )
    target_key_pen = "pen"
    target_key_stroke = "stroke"
  elif FLAGS.decoder_model == "t_emb":
    config.decoder = AttrDict(
        name="t_emb",
        layers=FLAGS.decoder_layers,
        hidden_units=FLAGS.decoder_hidden_units,
        activation=FLAGS.decoder_activation,
        dropout_rate=FLAGS.decoder_dropout,
        t_frequency_channels=FLAGS.t_frequency_channels,
        regularizer_weight=FLAGS.reg_dec_weight
        )
    target_key_pen = C.TARGET_T_PEN
    target_key_stroke = C.TARGET_T_STROKE
  else:
    err_unknown_type(FLAGS.decoder_model)
  
  config.loss = AttrDict()
  config.loss.prefix = "reconstruction"
  config.loss.stroke = LossConfig(
      loss_type=FLAGS.stroke_loss,
      num_components=20,
      target_key=target_key_stroke,
      out_key="stroke_logits",
      reduce_type=C.R_MEAN_STEP)
  config.loss.pen = LossConfig(
      eval_only=FLAGS.disable_pen_loss,
      loss_type=C.NLL_CENT_BINARY,
      target_key=target_key_pen,
      out_key="pen_logits",
      reduce_type=C.R_MEAN_STEP)

  if FLAGS.use_vae:
    config.loss.embedding_kld = LossConfig(
        loss_type=FLAGS.kld_type,  # C.KLD_STANDARD or C.KLD_STANDARD_NORM
        target_key=None,
        out_key="embedding",
        reduce_type=C.R_MEAN_STEP)

    config.loss.embedding_kld.weight = FLAGS.kld_weight
    if FLAGS.kld_increment > 0:
      config.loss.embedding_kld.weight = dict(
          type="linear_decay",
          values=[FLAGS.kld_start, FLAGS.kld_weight, FLAGS.kld_increment])

  if FLAGS.reg_emb_weight > 0:
    config.loss.embedding_l2 = LossConfig(
        loss_type=C.SNORM_L2,
        target_key=None,
        out_key="embedding",
        reduce_type=C.R_MEAN_STEP,
        weight=FLAGS.reg_emb_weight)
    
  try:
    data_root = os.environ["PREDICTIVE_SKETCHING_DATA_DIR"]
    log_dir = os.environ["PREDICTIVE_SKETCHING_LOG_DIR"]
    eval_dir = os.environ["PREDICTIVE_SKETCHING_EVAL_DIR"]
    gdrive_key = os.environ["GDRIVE_API_KEY"]
  except KeyError:
    data_root = FLAGS.data_dir
    log_dir = FLAGS.experiment_dir
    eval_dir = FLAGS.eval_dir
    gdrive_key = FLAGS.gdrive_api_key

  # Check if the experiment directory already exists.
  model_dir_query = glob.glob(os.path.join(log_dir, config.experiment.id + "*"))
  if model_dir_query:
    model_dir = model_dir_query[0]
    __builtin__.print = Print(os.path.join(model_dir, "log.txt"))  # Overload print.
    # Load experiment config.
    config = Configuration.from_json(os.path.join(model_dir, "config.json"))
    config.experiment.model_dir = model_dir
    config.experiment.eval_dir = os.path.join(eval_dir,
                                              os.path.basename(model_dir))
    print("Loading config from " + config.experiment.model_dir)
  else:
    config.experiment.tag = build_experiment_name(config)
    model_dir_name = config.experiment.id + "-" + config.experiment.tag
    config.experiment.model_dir = os.path.join(log_dir, model_dir_name)
    config.experiment.eval_dir = os.path.join(eval_dir, model_dir_name)
    os.mkdir(config.experiment.model_dir) # Create experiment directory
    __builtin__.print = Print(os.path.join(config.experiment.model_dir, "log.txt"))  # Overload print.
    print("Saving config to " + config.experiment.model_dir)

  if not isinstance(config.data.data_tfrecord_fname, list):
    config.data.data_tfrecord_fname = [config.data.data_tfrecord_fname]
    
  data_path = [os.path.join(data_root, config.data.data_name, "{}", dp) for dp in config.data.data_tfrecord_fname]
  config.data.train_data_path = [dp.format("training") for dp in data_path]
  config.data.valid_data_path = [dp.format("validation") for dp in data_path]
  config.data.test_data_path = [dp.format("test") for dp in data_path]
  config.data.meta_data_path = os.path.join(data_root, config.data.data_name,
                                            config.data.data_meta_fname)

  config.gdrive.credential = gdrive_key
  if FLAGS.gdrive_api_key == "nope":
    config.gdrive = None

  config.dump(config.experiment.model_dir)
  return config
    
def restore_config(experiment_id):
  try:
    data_root = os.environ["PREDICTIVE_SKETCHING_DATA_DIR"]
    log_dir = os.environ["PREDICTIVE_SKETCHING_LOG_DIR"]
    eval_dir = os.environ["PREDICTIVE_SKETCHING_EVAL_DIR"]
    gdrive_key = os.environ["GDRIVE_API_KEY"]
  except KeyError:
    raise Exception("Environment variables are not set.")

  # Check if the experiment directory already exists.
  model_dir_query = glob.glob(os.path.join(log_dir, experiment_id + "*"))
  
  if not model_dir_query:
    raise Exception("Model not found: {}".format(model_dir_query))

  # Load experiment config.
  model_dir = model_dir_query[0]
  config = Configuration.from_json(os.path.join(model_dir, "config.json"))
  config.experiment.model_dir = model_dir
  config.experiment.eval_dir = os.path.join(eval_dir, os.path.basename(model_dir))
  
  __builtin__.print = Print(os.path.join(model_dir, "log.txt"))  # Overload print.
  print("Loading from " + config.experiment.model_dir)
  
  # Customize with environment variables.
  if not isinstance(config.data.data_tfrecord_fname, list):
    config.data.data_tfrecord_fname = [config.data.data_tfrecord_fname]
  data_path = [os.path.join(data_root, config.data.data_name, "{}", dp) for dp in config.data.data_tfrecord_fname]
  config.data.train_data_path = [dp.format("training") for dp in data_path]
  config.data.valid_data_path = [dp.format("validation") for dp in data_path]
  config.data.test_data_path = [dp.format("test") for dp in data_path]
  config.data.meta_data_path = os.path.join(data_root, config.data.data_name,
                                            config.data.data_meta_fname)
  config.gdrive.credential = gdrive_key
  return config


def build_experiment_name(config):
  template = "{tag}{model_name}_{encoder}-{latent}-{decoder}-{output}-{experiment}{data}"

  if config.decoder.name == "t_emb":
    model_tags = TEmbedding.get_model_tags(config, config.loss)
  elif config.decoder.name == "rnn":
    model_tags = InkSeq2Seq.get_model_tags(config, config.loss)
  else:
    err_unknown_type(config.decoder.name)
  
  # data = config.data.data_name
  data = ""
  lr = ""
  if config.experiment.learning_rate.name == "transformer":
    lr = "_tr"
  elif config.experiment.learning_rate.name == "exponential":
    lr = "_exp"
  experiment = "B{}_LR{}".format(config.data.batch_size, lr)
  
  return template.format(
      tag=config.experiment.tag + "_" if config.experiment.tag else "",
      model_name=model_tags["model_name"],
      encoder=model_tags["encoder"],
      latent=model_tags["latent"],
      decoder=model_tags["decoder"],
      output=model_tags["output"],
      experiment=experiment,
      data=data,
      )


def build_embedding_model(config_, run_mode=C.RUN_STATIC):
  """Builds model object."""

  if config_.decoder.name == "t_emb":
    model_ = TEmbedding(
        config_encoder=config_.encoder,
        config_embedding=config_.embedding,
        config_decoder=config_.decoder,
        config_loss=config_.loss,
        run_mode=run_mode)
  elif config_.decoder.name == "rnn":
    model_ = InkSeq2Seq(
        config_encoder=config_.encoder,
        config_embedding=config_.embedding,
        config_decoder=config_.decoder,
        config_loss=config_.loss,
        run_mode=run_mode)
  else:
    err_unknown_type(config_.decoder.name)
  
  return model_


def build_dataset(config_, run_mode=C.RUN_STATIC, split=C.DATA_TRAIN):
  """Build dataset object."""

  data_cls = TFRecordStroke
  data_test_cls = TFRecordSingleDiagram
  if config_.data.get("ink_dataset", False):
    data_cls = TFRecordInkSequence
    data_test_cls = TFRecordInkSequence
  
  dataset_ = None
  if split == C.DATA_TRAIN:
    dataset_ = data_cls(
        data_path=config_.data.train_data_path,
        meta_data_path=config_.data.meta_data_path,
        batch_size=config_.data.batch_size,
        pp_to_origin=config_.data.pp_to_origin,
        pp_relative_pos=config_.data.pp_relative_pos,
        normalize=config_.data.normalize,
        shuffle=True,
        run_mode=run_mode,
        max_length_threshold=config_.data.get("max_length_threshold", 201),
        mask_pen=config_.data.get("mask_pen", False),
        fixed_len=config_.data.get("fixed_len", False),
        resampling_factor=config_.data.get("resampling_factor", 0),
        t_drop_ratio=config_.data.get("t_drop_ratio", 0),
        scale_factor=config_.data.get("scale_factor", 0),
        affine_prob=config_.data.get("affine_prob", 0),
        reverse_prob=config_.data.get("reverse_prob", 0),
        gt_targets=config_.data.get("gt_targets", False),
        n_t_targets=config_.data.get("n_t_samples", 1),
        int_t_samples=config_.data.get("int_t_samples", False),
        concat_t_inputs=config_.data.get("concat_t_inputs", False),
        rdp=config_.data.get("rdp_dataset", False),
        rdp_didi_pp=config_.data.get("rdp_didi_pp", False),
        )
  elif split == C.DATA_VALID:
    dataset_ = data_cls(
        data_path=config_.data.valid_data_path,
        meta_data_path=config_.data.meta_data_path,
        batch_size=config_.data.batch_size,
        pp_to_origin=config_.data.pp_to_origin,
        pp_relative_pos=config_.data.pp_relative_pos,
        normalize=config_.data.normalize,
        shuffle=False,
        run_mode=run_mode,
        max_length_threshold=config_.data.get("max_length_threshold", 201),
        mask_pen=config_.data.get("mask_pen", False),
        fixed_len=config_.data.get("fixed_len", False),
        int_t_samples=config_.data.get("int_t_samples", False),
        concat_t_inputs=config_.data.get("concat_t_inputs", False),
        rdp=config_.data.get("rdp_dataset", False),
        rdp_didi_pp=config_.data.get("rdp_didi_pp", False),
        )
  elif split == C.DATA_TEST:
    dataset_ = data_test_cls(
        data_path=config_.data.test_data_path,
        meta_data_path=config_.data.meta_data_path,
        batch_size=1,
        pp_to_origin=config_.data.pp_to_origin,
        pp_relative_pos=config_.data.pp_relative_pos,
        normalize=config_.data.normalize,
        shuffle=False,
        run_mode=run_mode,
        max_length_threshold=config_.data.get("max_length_threshold", 201),
        mask_pen=config_.data.get("mask_pen", False),
        fixed_len=config_.data.get("fixed_len", False),
        int_t_samples=config_.data.get("int_t_samples", False),
        concat_t_inputs=config_.data.get("concat_t_inputs", False),
        rdp=config_.data.get("rdp_dataset", False),
        rdp_didi_pp=config_.data.get("rdp_didi_pp", False),
        )
  else:
    err_unknown_type(split)

  return dataset_
