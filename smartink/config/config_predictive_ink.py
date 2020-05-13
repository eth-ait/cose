"""Default configuration for ink ink experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import getpass
import glob
import os

from absl import flags

from common.constants import Constants as C
from smartink.config.configuration import AttrDict
from smartink.config.configuration import Configuration
from smartink.config.configuration import DataConfig
from smartink.config.configuration import ExperimentConfig
from smartink.config.configuration import LossConfig
from smartink.util.utils import err_unknown_type
from smartink.util.utils import NotPredictiveModelError
from smartink.util.utils import ModelNotFoundError
from smartink.data.stroke_dataset import TFRecordBatchDiagram
from smartink.data.stroke_dataset import TFRecordSingleDiagram
from smartink.models.stroke.t_emb import TEmbedding
from smartink.models.stroke.seq2seq import InkSeq2Seq
from smartink.models.sequence.rnn import RNN
from smartink.models.ink.predictive_models import PredictiveInkModel
from smartink.models.sequence.transformer import TransformerAR
from smartink.models.sequence.transformer import TransformerSeq2seqConditional


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
  flags.DEFINE_bool("mask_encoder_pen", False,
                    "whether to mask pen information for encoder or not.")
  flags.DEFINE_integer(
      "resampling_factor", 0,
      "Temporal resampling rate. Randomly sampled between 1 and the given value.")
  flags.DEFINE_float("scale_factor", 0, "Amount of scaling.")
  flags.DEFINE_float("affine_prob", 0, "Chance of applying affine transf.")
  flags.DEFINE_float("reverse_prob", 0, "Chance of reverting a sequence.")
  flags.DEFINE_integer("n_t_samples", 1, "# of t samples per sequence.")
  flags.DEFINE_boolean("int_t_samples", False, "whether to interpolate t targets or not.")
  flags.DEFINE_boolean("concat_t_inputs", False, "whether to concatenate input points with t.")
  
  # Experiment details
  flags.DEFINE_integer("batch_size", 100, "batch size for training")
  flags.DEFINE_float("learning_rate", 0.001, "initial learning rate.")
  flags.DEFINE_string("learning_rate_type", "exponential",
                      "LR scheduler: exponential, sketch_rnn or transformer")
  flags.DEFINE_float("grad_clip_norm", 0, "gradient clip by norm threshold.")
  flags.DEFINE_float("grad_clip_value", 0, "gradient clip by value threshold. grad_clip_norm is prioritized.")
  flags.DEFINE_string("pretrained_emb_id", None, "ID of a pre-trained stroke model.")
  
  ### Encoder model.
  flags.DEFINE_string("encoder_model", "rnn", "rnn or transformer.")
  # Encoder RNN
  flags.DEFINE_integer("encoder_rnn_layers", 1,
                       "number of layers in encoder rnn model.")
  flags.DEFINE_integer("encoder_rnn_units", 512,
                       "number of units per layer in encoder rnn model.")
  flags.DEFINE_string("encoder_cell_type", C.LSTM,
                      "type of rnn cell instance: " + C.GRU + " or " + C.LSTM)
  flags.DEFINE_bool("bidirectional_encoder", False,
                    "whether to use bidirectional rnn as encoder or not.")
  flags.DEFINE_bool("encoder_cudnn", False,
                    "whether to use cudnn optimized cells or not.")
  flags.DEFINE_float("encoder_rdropout", 0.0, "Recurrent ropout rate for the "
                                              "encoder cell.")
  # Encoder Transformer
  flags.DEFINE_integer("transformer_dmodel", 128, "representation size.")
  flags.DEFINE_integer("transformer_layers", 8, "number of transformer layers.")
  flags.DEFINE_integer("transformer_heads", 8, "number of attention head.")
  flags.DEFINE_integer("transformer_hidden_units", 256,
                       "size of point_wise_feed_forward_network.")
  flags.DEFINE_float("transformer_dropout", 0, "dropout rate.")
  flags.DEFINE_bool("transformer_pos_encoding", False, "Positional encoding.")
  flags.DEFINE_bool("transformer_scale", False, "Scaling stroke.")
  # Embedding
  flags.DEFINE_integer("latent_units", 16,
                       "number of units for stroke embeddings.")
  flags.DEFINE_bool("use_vae", False, "VAE regularizer on the stroke space.")
  flags.DEFINE_float("kld_weight", 1.0, "VAE KLD loss weight.")
  flags.DEFINE_float("kld_start", 0.05, "Initial VAE KLD loss weight.")
  flags.DEFINE_float("kld_increment", 0.99995,
                     "VAE KLD loss weight increment per step.")
  ### Decoder model
  flags.DEFINE_string("decoder_model", "t_emb", "stroke decoder: t_emb or rnn.")
  # t-decoder
  flags.DEFINE_integer("decoder_layers", 3, "number of dense layers in decoder.")
  flags.DEFINE_list("decoder_hidden_units", "512,256,128",
                    "list of hidden units in decoder.")
  flags.DEFINE_string("decoder_activation", C.RELU, "activation function.")
  flags.DEFINE_float("decoder_dropout", 0.0, "Dropout rate after every layer.")
  flags.DEFINE_integer("t_frequency_channels", 0,
                       "# of frequency channels for t frequency encoding.")
  
  # seq2seq decoder
  flags.DEFINE_bool("decoder_dynamic_h0", False, "Parameterized initial rnn state.")
  flags.DEFINE_bool("repeat_vae_sample", False, "Use the same latent sample in all prediction steps.")
  flags.DEFINE_bool("decoder_autoregressive", False, "Feed the decoder with the previous step as well.")
  
  ### Predictive model.
  flags.DEFINE_string("predictive_model", "transformer", "ink models: rnn or transformer.")
  flags.DEFINE_bool("use_start_pos", False, "whether to feed stroke start position or not.")
  flags.DEFINE_bool("use_end_pos", False, "whether to feed stroke end position or not.")
  flags.DEFINE_bool("stop_predictive_grad", False, "whether to stop gradient flow to the embedding model or not.")
  flags.DEFINE_string("pred_input_type", "random", "input/target configuration: single[leave_one_out, last_step], set[random, ordered, hybrid].")
  
  # Predictive: RNN models.
  flags.DEFINE_integer("predictive_rnn_layers", 1,
                       "number of layers in ink rnn.")
  flags.DEFINE_integer("predictive_rnn_units", 512,
                       "number of units per layer in ink rnn.")
  flags.DEFINE_string("predictive_cell_type", C.LSTM,
                      "type of rnn cell instance: " + C.GRU + " or " + C.LSTM)
  # Predictive: Transformer models.
  flags.DEFINE_integer("p_transformer_layers", 8, "number of transformer layers.")
  flags.DEFINE_integer("p_transformer_heads", 8, "number of attention head.")
  flags.DEFINE_integer("p_transformer_dmodel", 128, "transformer representation size.")
  flags.DEFINE_integer("p_transformer_hidden_units", 256,
                       "size of point_wise_feed_forward_network.")
  flags.DEFINE_float("p_transformer_dropout", 0.1, "dropout rate.")
  flags.DEFINE_bool("p_transformer_pos_encoding", False, "Positional encoding.")
  flags.DEFINE_bool("p_transformer_scale", False,
                    "Scaling embeddings in transformer.")

  flags.DEFINE_string("position_model", None, "position models: transformer or None.")
  
  # Loss terms.
  flags.DEFINE_bool("loss_predicted_embedding", False,
                    "Whether to optimize for stroke prediction loss or not")
  flags.DEFINE_bool("loss_predicted_ink", False,
                    "Whether to optimize for ink prediction loss or not")
  flags.DEFINE_bool("loss_reconstructed_ink", False,
                    "Whether to optimize for ink reconstruction loss or not")
  flags.DEFINE_string("stroke_loss", C.NLL_BINORMAL,
                      "nll_normal, nll_binormal, nll_gmm, mse or l1")
  flags.DEFINE_string("embedding_loss", C.MSE, "nll_normal or mse.")
  flags.DEFINE_integer("embedding_gmm_components", 5, "# of gmm components if embedding_loss is nll_gmm.")
  flags.DEFINE_bool("disable_pen_loss", False,
                    "whether to disable pen loss or not.")

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
      grad_clip_value=FLAGS.grad_clip_value,
      pretrained_emb_id=FLAGS.pretrained_emb_id
      )
  config.experiment.learning_rate = AttrDict(
      name=FLAGS.learning_rate_type,
      initial_learning_rate=FLAGS.learning_rate,
      )
  if FLAGS.learning_rate_type == "transformer":
    config.experiment.learning_rate.d_model = FLAGS.transformer_dmodel
    config.experiment.learning_rate.warmup_steps = 4000
    
  config.data = DataConfig(
      data_dir=FLAGS.data_dir,
      data_name=FLAGS.data_name,
      data_tfrecord_fname=C.DATASET_MAP[FLAGS.data_name]["data_tfrecord_fname"],
      data_meta_fname=C.DATASET_MAP[FLAGS.data_name][FLAGS.metadata_type],
      pp_to_origin=FLAGS.metadata_type == "position",
      pp_relative_pos=FLAGS.metadata_type == "velocity",
      normalize=not FLAGS.skip_normalization,
      batch_size=FLAGS.batch_size,
      max_length_threshold=201,
      mask_pen=FLAGS.mask_encoder_pen,
      resampling_factor=FLAGS.resampling_factor,
      scale_factor=FLAGS.scale_factor,
      affine_prob=FLAGS.affine_prob,
      reverse_prob=FLAGS.reverse_prob,
      n_t_samples=FLAGS.n_t_samples,
      int_t_samples=FLAGS.int_t_samples,
      concat_t_inputs=FLAGS.concat_t_inputs
      )
  config.gdrive = AttrDict(
      credential=None,  # Set automatically below.
      workbook="1D__n4IEKz_zMmknL5_qL7ILVQgWC5Z6wx-fggKQJDtw",
      sheet=FLAGS.data_name,
      )
  
  # Embedding model.
  if FLAGS.encoder_model == "rnn":
    config.encoder = AttrDict(
        name="rnn",
        cell_units=FLAGS.encoder_rnn_units,
        cell_layers=FLAGS.encoder_rnn_layers,
        cell_type=FLAGS.encoder_cell_type,
        use_cudnn=FLAGS.encoder_cudnn,
        bidirectional_encoder=FLAGS.bidirectional_encoder,
        rec_dropout_rate = FLAGS.encoder_rdropout)
  elif FLAGS.encoder_model == "transformer":
    config.encoder = AttrDict(
        name="transformer",
        layers=FLAGS.transformer_layers,
        heads=FLAGS.transformer_heads,
        d_model=FLAGS.transformer_dmodel,
        hidden_units=FLAGS.transformer_hidden_units,
        dropout_rate=FLAGS.transformer_dropout,
        pos_encoding=config.data.max_length_threshold if FLAGS.transformer_pos_encoding else 0,
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
        use_cudnn=FLAGS.encoder_cudnn,
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
  
  # Predictive model.
  if FLAGS.predictive_model == "rnn":
    config.predictive_model = AttrDict(
        name="rnn",
        output_size=config.embedding.latent_units,
        cell_units=FLAGS.predictive_rnn_units,
        cell_layers=FLAGS.predictive_rnn_layers,
        cell_type=FLAGS.predictive_cell_type,
        activation=C.RELU,
        use_start_pos=FLAGS.use_start_pos,
        use_end_pos=FLAGS.use_end_pos,
        stop_predictive_grad=FLAGS.stop_predictive_grad,
        pred_input_type=FLAGS.pred_input_type,
        use_cudnn=True,
    )
  elif FLAGS.predictive_model == "transformer":
    config.predictive_model = AttrDict(
        name="transformer",
        output_size=config.embedding.latent_units,
        layers=FLAGS.p_transformer_layers,
        heads=FLAGS.p_transformer_heads,
        d_model=FLAGS.p_transformer_dmodel,
        latent_units=FLAGS.latent_units,
        hidden_units=FLAGS.p_transformer_hidden_units,
        dropout_rate=FLAGS.p_transformer_dropout,
        pos_encoding=FLAGS.p_transformer_pos_encoding,
        scale=FLAGS.p_transformer_scale,
        use_start_pos=FLAGS.use_start_pos,
        use_end_pos=FLAGS.use_end_pos,
        stop_predictive_grad=FLAGS.stop_predictive_grad,
        pred_input_type=FLAGS.pred_input_type,
    )
  else:
    err_unknown_type(FLAGS.predictive_model)
    
  # Sharing flags with the predictive model.
  if FLAGS.position_model == "transformer":
    config.position_model = AttrDict(
        name="transformer",
        output_size=2,
        layers=FLAGS.p_transformer_layers,
        heads=FLAGS.p_transformer_heads,
        d_model=FLAGS.p_transformer_dmodel,
        hidden_units=FLAGS.p_transformer_hidden_units,
        dropout_rate=FLAGS.p_transformer_dropout,
        pos_encoding=FLAGS.p_transformer_pos_encoding,
        scale=FLAGS.p_transformer_scale,
    )
  elif FLAGS.position_model is None:
    config.position_model = None
  else:
    err_unknown_type(FLAGS.predictive_model)

  # Loss
  config.loss = AttrDict()
  
  stroke_ = LossConfig(
      loss_type=FLAGS.stroke_loss,
      num_components=20,
      target_key=target_key_stroke,
      out_key="stroke_logits",
      reduce_type=C.R_MEAN_STEP)
  pen_ = LossConfig(
      eval_only=FLAGS.disable_pen_loss,
      loss_type=C.NLL_CENT_BINARY,
      target_key=target_key_pen,
      out_key="pen_logits",
      reduce_type=C.R_MEAN_STEP)
  
  ink_loss = AttrDict(pen=pen_, stroke=stroke_, prefix = "reconstruction")

  if FLAGS.use_vae:
    ink_loss.embedding_kld = LossConfig(
        loss_type=C.KLD_STANDARD,
        target_key=None,
        out_key="stroke",
        reduce_type=C.R_MEAN_STEP)
    
    # ink_loss.embedding_kld.weight = FLAGS.kld_weight
    ink_loss.embedding_kld.weight = dict(
        type="linear_decay",
        values=[FLAGS.kld_start, FLAGS.kld_weight, FLAGS.kld_increment])
    
  if FLAGS.reg_emb_weight > 0:
    ink_loss.embedding_l2 = LossConfig(
        loss_type=C.SNORM_L2,
        target_key=None,
        out_key="embedding",
        reduce_type=C.R_MEAN_STEP,
        weight=FLAGS.reg_emb_weight)

  embedding_pred = LossConfig(
      loss_type=FLAGS.embedding_loss,
      num_components=FLAGS.embedding_gmm_components,
      target_key="target",
      out_key="prediction",
      reduce_type=C.R_MEAN_STEP)
  
  position_pred = LossConfig(
      loss_type=FLAGS.embedding_loss,
      num_components=FLAGS.embedding_gmm_components,
      target_key="target",
      out_key="prediction",
      reduce_type=C.R_MEAN_STEP)
  
  # embedding_pred.weight = FLAGS.kld_weight
  # embedding_pred.weight = dict(
  #     type="linear_decay",
  #     values=[FLAGS.kld_start, FLAGS.kld_weight, FLAGS.kld_increment])
  
  config.loss = AttrDict(
      ink=ink_loss,
      predicted_embedding=AttrDict(predicted_embedding=embedding_pred),
      predicted_ink=copy.deepcopy(ink_loss))
  
  if config.position_model is not None:
    config.loss.predicted_pos = AttrDict(predicted_pos=position_pred)

  # No kld loss on the predicted ink.
  if "embedding_kld" in config.loss.predicted_ink:
    del config.loss.predicted_ink["embedding_kld"]

  config.loss.apply_predicted_embedding = FLAGS.loss_predicted_embedding
  config.loss.apply_predicted_ink = FLAGS.loss_predicted_ink
  config.loss.apply_reconstructed_ink = FLAGS.loss_reconstructed_ink

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
    # Load experiment config.
    config = config.from_json(os.path.join(model_dir, "config.json"))
    config.experiment.model_dir = model_dir
    config.experiment.eval_dir = os.path.join(eval_dir,
                                              os.path.basename(model_dir))
    if "predictive_model" not in config:
      raise NotPredictiveModelError
    print("Loading from " + config.experiment.model_dir)
  else:
    config.experiment.tag = build_experiment_name(config)
    model_dir_name = config.experiment.id + "-" + config.experiment.tag
    config.experiment.model_dir = os.path.join(log_dir, model_dir_name)
    config.experiment.eval_dir = os.path.join(eval_dir, model_dir_name)
    print("Saving to " + config.experiment.model_dir)
  
  if not isinstance(config.data.data_tfrecord_fname, list):
    config.data.data_tfrecord_fname = [config.data.data_tfrecord_fname]
  data_path = [os.path.join(data_root, config.data.data_name, "{}", dp) for dp in config.data.data_tfrecord_fname]
  config.data.train_data_path = [dp.format("training") for dp in data_path]
  config.data.valid_data_path = [dp.format("validation") for dp in data_path]
  config.data.test_data_path = [dp.format("test") for dp in data_path]
  config.data.meta_data_path = os.path.join(data_root, config.data.data_name,
                                            config.data.data_meta_fname)
  
  config.experiment.pretrained_dir = None
  if config.experiment.get("pretrained_emb_id", None) is not None:
    config.experiment.pretrained_dir = glob.glob(os.path.join(log_dir, config.experiment.pretrained_emb_id + "-*"))[0]
  
  config.gdrive.credential = gdrive_key
  if FLAGS.gdrive_api_key == "nope":
    config.gdrive = None
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
    raise ModelNotFoundError

  # Load experiment config.
  model_dir = model_dir_query[0]
  config = Configuration.from_json(os.path.join(model_dir, "config.json"))
  config.experiment.model_dir = model_dir
  config.experiment.eval_dir = os.path.join(eval_dir,
                                            os.path.basename(model_dir))
  if "predictive_model" not in config:
    raise NotPredictiveModelError
  print("Loading from " + config.experiment.model_dir)
  
  if not isinstance(config.data.data_tfrecord_fname, list):
    config.data.data_tfrecord_fname = [config.data.data_tfrecord_fname]
  data_path = [os.path.join(data_root, config.data.data_name, "{}", dp) for dp in config.data.data_tfrecord_fname]
  config.data.train_data_path = [dp.format("training") for dp in data_path]
  config.data.valid_data_path = [dp.format("validation") for dp in data_path]
  config.data.test_data_path = [dp.format("test") for dp in data_path]
  config.data.meta_data_path = os.path.join(data_root, config.data.data_name,
                                            config.data.data_meta_fname)
  
  config.experiment.pretrained_dir = None
  if config.experiment.get("pretrained_emb_id", None) is not None:
    config.experiment.pretrained_dir = glob.glob(
      os.path.join(log_dir, config.experiment.pretrained_emb_id + "-*"))[0]
  
  config.gdrive.credential = gdrive_key
  return config


def build_experiment_name(config):
  template = "PRED_{tag}{pred_model_name}_{predictive}-{emb_model_name}_{encoder}-{latent}-{decoder}-{output}-{loss}-{experiment}{data}"

  if config.decoder.name == "t_emb":
    emb_model_tags = TEmbedding.get_model_tags(config, config.loss.ink)
  elif config.decoder.name == "rnn":
    emb_model_tags = InkSeq2Seq.get_model_tags(config, config.loss.ink)
  else:
    err_unknown_type(config.decoder.name)
  
  if config.predictive_model.name == "rnn":
    pred_model_tags = RNN.get_model_tags(config.predictive_model)
  elif config.predictive_model.name == "transformer":
    pred_model_tags = TransformerAR.get_model_tags(config.predictive_model)
  else:
    err_unknown_type(config.predictive_model.name)
  
  # data = config.data.data_name
  data = ""
  lr = ""
  if config.experiment.learning_rate.name == "transformer":
    lr = "_tr"
  elif config.experiment.learning_rate.name == "exponential":
    lr = "_exp"
  experiment = "B{}_LR{}".format(config.data.batch_size, lr)
  
  loss = "loss_{}{}{}".format(
      "P" if config.loss.apply_predicted_ink else "",
      "E" if config.loss.apply_predicted_embedding else "",
      "R" if config.loss.apply_reconstructed_ink else "")
  
  return template.format(
      tag=config.experiment.tag + "_" if config.experiment.tag else "",
      pred_model_name=pred_model_tags["model_name"],
      predictive=pred_model_tags["model"],
      emb_model_name=emb_model_tags["model_name"],
      encoder=emb_model_tags["encoder"],
      latent=emb_model_tags["latent"],
      decoder=emb_model_tags["decoder"],
      output=emb_model_tags["output"],
      experiment=experiment,
      loss=loss,
      data=data,
      )


def build_predictive_model(config_, run_mode):
  """Builds model object."""
  
  # Embedding model.
  if config_.decoder.get("name", "t_emb") == "t_emb":
    embedding_model = TEmbedding(
        config_encoder=config_.encoder,
        config_embedding=config_.embedding,
        config_decoder=config_.decoder,
        config_loss=config_.loss.ink,
        run_mode=run_mode)
  elif config_.decoder.get("name", "t_emb") == "rnn":
    embedding_model = InkSeq2Seq(
        config_encoder=config_.encoder,
        config_embedding=config_.embedding,
        config_decoder=config_.decoder,
        config_loss=config_.loss.ink,
        run_mode=run_mode)
  else:
    err_unknown_type(config_.decoder.name)
    
  if config_.predictive_model.get("name", "rnn") == "rnn":
  
    predictive_model = RNN(
        output_size=config_.predictive_model.output_size,
        cell_units=config_.predictive_model.cell_units,
        cell_layers=config_.predictive_model.cell_layers,
        cell_type=config_.predictive_model.cell_type,
        return_sequences=False,
        return_state=False,
        run_mode=run_mode,
        config_loss=config_.loss.predicted_embedding.predicted_embedding
        )
    
  elif config_.predictive_model.name == "transformer":
    predictive_model = TransformerSeq2seqConditional(
        output_size=config_.predictive_model.latent_units,
        num_layers=config_.predictive_model.layers,
        d_model=config_.predictive_model.d_model,
        num_heads=config_.predictive_model.heads,
        dff=config_.predictive_model.hidden_units,
        rate=config_.predictive_model.dropout_rate,
        config_loss=config_.loss.predicted_embedding.predicted_embedding,
        pos_encoding_len=100 if config_.predictive_model.pos_encoding else 0,
        scale=config_.predictive_model.scale,
        run_mode=run_mode,
        autoregressive=False
        )
  else:
    err_unknown_type(config_.predictive_model.name)

  position_model = None
  if config_.get("position_model", None) is not None:
    position_model = TransformerSeq2seqConditional(
        output_size=2,
        num_layers=config_.position_model.layers,
        d_model=config_.position_model.d_model,
        num_heads=config_.position_model.heads,
        dff=config_.position_model.hidden_units,
        rate=config_.position_model.dropout_rate,
        config_loss=config_.loss.predicted_pos.predicted_pos,
        pos_encoding_len=0,
        scale=config_.position_model.scale,
        run_mode=run_mode,
        autoregressive=False
        )
  
  model_ = PredictiveInkModel(
      embedding_model=embedding_model,
      predictive_model=predictive_model,
      position_model=position_model,
      loss_predicted_embedding=config_.loss.apply_predicted_embedding,
      loss_predicted_ink=config_.loss.apply_predicted_ink,
      loss_reconstructed_ink=config_.loss.apply_reconstructed_ink,
      input_type=config_.predictive_model.pred_input_type,
      start_positions=config_.predictive_model.use_start_pos,
      end_positions=config_.predictive_model.get("use_end_pos", False),
      stop_predictive_grad=config_.predictive_model.get("stop_predictive_grad", False),
      config_loss=copy.deepcopy(config_.loss),
      run_mode=run_mode)
  return model_


def build_dataset(config_, run_mode=C.RUN_STATIC, split=C.DATA_TRAIN):
  """Builds dataset object."""

  dataset_ = None
  if split == C.DATA_TRAIN:
    dataset_ = TFRecordBatchDiagram(
        data_path=config_.data.train_data_path,
        meta_data_path=config_.data.meta_data_path,
        batch_size=config_.data.batch_size,
        pp_to_origin=config_.data.pp_to_origin,
        pp_relative_pos=config_.data.pp_relative_pos,
        normalize=config_.data.normalize,
        shuffle=True,
        run_mode=run_mode,
        max_length_threshold=config_.data.max_length_threshold,
        mask_pen=config_.data.mask_pen,
        resampling_factor=config_.data.get("resampling_factor", 0),
        scale_factor=config_.data.get("scale_factor", 0),
        affine_prob=config_.data.get("affine_prob", 0),
        reverse_prob=config_.data.get("reverse_prob", 0),
        resample_target=False,
        n_t_targets=config_.data.get("n_t_samples", 1),
        int_t_samples=config_.data.get("int_t_samples", False),
        concat_t_inputs=config_.data.get("concat_t_inputs", False),
        )
  elif split == C.DATA_VALID:
    dataset_ = TFRecordBatchDiagram(
        data_path=config_.data.valid_data_path,
        meta_data_path=config_.data.meta_data_path,
        batch_size=config_.data.batch_size,
        pp_to_origin=config_.data.pp_to_origin,
        pp_relative_pos=config_.data.pp_relative_pos,
        normalize=config_.data.normalize,
        shuffle=False,
        run_mode=run_mode,
        max_length_threshold=config_.data.max_length_threshold,
        mask_pen=config_.data.mask_pen,
        concat_t_inputs=config_.data.get("concat_t_inputs", False),
        )
  elif split == C.DATA_TEST:
    dataset_ = TFRecordSingleDiagram(
        data_path=config_.data.test_data_path,
        meta_data_path=config_.data.meta_data_path,
        batch_size=config_.data.batch_size,
        pp_to_origin=config_.data.pp_to_origin,
        pp_relative_pos=config_.data.pp_relative_pos,
        normalize=config_.data.normalize,
        shuffle=False,
        max_length_threshold=config_.data.max_length_threshold,
        run_mode=run_mode,
        mask_pen=config_.data.mask_pen,
        concat_t_inputs=config_.data.get("concat_t_inputs", False),
        )
  else:
    err_unknown_type(split)

  return dataset_
