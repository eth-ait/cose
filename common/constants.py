"""Constant strings to be used in the code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Constants(object):
  """Constant strings to be used in the code."""
  SEED = 1234

  # Run Modes
  RUN_EAGER = 'eager'
  RUN_STATIC = 'static'
  RUN_ESTIMATOR = 'estimator'

  # Data splits
  DATA_TRAIN = 'train'
  DATA_VALID = 'valid'
  DATA_TEST = 'test'

  # Data Batch
  BATCH_SEQ_LEN = 'seq_len'
  BATCH_INPUT = 'inputs'
  BATCH_TARGET = 'targets'

  INP_ENC = 'encoder_inputs'
  INP_DEC = 'decoder_inputs'
  INP_SEQ_LEN = BATCH_SEQ_LEN
  INP_NUM_STROKE = 'num_strokes'
  INP_START_COORD = 'start_coord'
  INP_END_COORD = 'end_coord'
  INP_T = 't_input'
  TARGET_T_INK = 't_target_ink'
  TARGET_T_STROKE = 't_target_stroke'
  TARGET_T_PEN = 't_target_pen'

  # Distribution parameters.
  MU = 'mu'
  SIGMA = 'sigma'
  RHO = 'rho'
  PI = 'pi'

  # Statistics
  MEAN_ALL = 'mean_all'
  VAR_ALL = 'var_all'
  MEAN_CHANNEL = 'mean_channel'
  VAR_CHANNEL = 'var_channel'
  MIN_ALL = 'min_all'
  MAX_ALL = 'max_all'
  MIN_SEQ_LEN = 'min_seq_len'
  MAX_SEQ_LEN = 'max_seq_len'
  MEAN_SEQ_LEN = 'mean_seq_len'
  NUM_SAMPLES = 'n_samples'

  # Models.
  MODEL_RNN = 'rnn'
  MODEL_VRNN = 'vrnn'
  MODEL_TCN = 'tcn'

  # Output models
  OUT_DETERMINISTIC = 'deterministic'
  OUT_NORMAL = 'normal'

  # Negative log-likelihood losses.
  NLL_BERNOULLI = 'nll_bernoulli'
  NLL_NORMAL = 'nll_normal'
  NLL_BINORMAL = 'nll_binormal'
  NLL_GMM = 'nll_gmm'
  NLL_BIGMM = 'nll_bigmm'
  NLL_CENT = 'nll_cent'  # Cross-entropy.
  NLL_CENT_BINARY = 'nll_cent_binary'  # Cross-entropy for binary outputs.
  KLD_STANDARD = 'kld_p0'
  KLD = 'kld'
  MSE = 'mse'
  L1 = 'l1'
  SNORM_L2 = 'snorm_l2'  # squared L2 norm
  
  METRIC_L2 = "l2"
  METRIC_CHAMFER = "chamfer"

  # RNN cells and layer types.
  LSTM = 'lstm'
  GRU = 'gru'
  DENSE = 'dense'  # Fully connected layer.
  TCN = 'tcn'  # Temporal convolutional layer, i.e., causal 1D convolution.

  # Activation functions.
  RELU = 'relu'
  ELU = 'elu'
  SIGMOID = 'sigmoid'
  SOFTPLUS = 'softplus'
  TANH = 'tanh'
  SOFTMAX = 'softmax'
  LRELU = 'lrelu'
  CLRELU = 'clrelu'  # Clamped leaky relu.

  # Learning rate scheduler types.
  LR_EXP = 'exponential'
  LR_CONSTANT = 'constant'

  # Loss reduce function types.
  # Take average of average step loss per sample over batch.
  # Uses sequence length.
  R_MEAN_STEP = 'mean_step_loss'
  # Take average of sequence loss (summation of all steps) over batch.
  # Uses sequence length.
  R_MEAN_SEQUENCE = 'mean_sequence_loss'
  R_MEAN = 'mean'  # Calculate average of the whole loss tensor.
  R_SUM = 'sum'  # Sum all entries in the loss tensor.
  # Keep the loss per sample. Uses sequence length.
  B_MEAN_STEP = 'batch_mean_step_loss'
  R_IDENTITY = 'identity'  # No effect.

  # Optimizers
  OPTIMIZER_ADAM = 'adam'
  
  # Stroke decoders
  DECODER_T_EMB = "t_emb"
  DECODER_RNN = "rnn"
  DECODER_TR = "transformer"

  DATASET_MAP = {
      "didi":
          {
            "data_tfrecord_fname": "diagrams_20200131-?????-of-?????",
            "position"           : "didi-stats-origin_abs_pos.npy",
            "velocity"           : "didi-stats-relative_pos.npy"
          },
      
      "didi_wo_text":
          {
            "data_tfrecord_fname": "diagrams_wo_text_20200131-?????-of-?????",
            "position"           : "didi_wo_text-stats-origin_abs_pos.npy",
            "velocity"           : "didi_wo_text-stats-relative_pos.npy"
          },
      
      "didi_all":
        {
            "data_tfrecord_fname": ["diagrams_20200131-?????-of-?????", "diagrams_wo_text_20200131-?????-of-?????"],
            "position"           : "didi_all-stats-origin_abs_pos.npy",
            "velocity"           : "didi_all-stats-relative_pos.npy"
            }
      }