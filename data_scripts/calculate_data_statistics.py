"""Calculate data statistics once and stores."""

import os
from smartink.data.stroke_dataset import TFRecordStroke

DATA_DIR = None  # TODO: Set this path.

if DATA_DIR is None and "COSE_DATA_DIR" in os.environ:
  DATA_DIR = os.path.join(os.environ["COSE_DATA_DIR"], "didi_wo_text/")
else:
  raise Exception("Data path must be set")
  
TFRECORD_PATTERN = "diagrams_wo_text_20200131-?????-of-?????"
META_FILE = "didi_wo_text-stats-origin_abs_pos.npy"

USE_POSITION = True  # Calculate statistics for pixel coordinates (i.e. absolute positions) or relative offsets (i.e., velocity).
MAX_LENGTH = 301  # Longer or shorter strokes will be filtered out.
MIN_LENGTH = 2

train_data = TFRecordStroke(
    data_path=[os.path.join(DATA_DIR, "training", TFRECORD_PATTERN)],
    meta_data_path=DATA_DIR + META_FILE,
    pp_to_origin=USE_POSITION,
    pp_relative_pos=not USE_POSITION,
    max_length_threshold=MAX_LENGTH,
    min_length_threshold=MIN_LENGTH,
    normalize=True,
    batch_size=1,
    shuffle=False,
    run_mode="eager",
    fixed_len=False,
    mask_pen=False,
    scale_factor=0,
    resampling_factor=0,
    random_noise_factor=0,
    gt_targets=False,
    n_t_targets=1,
    concat_t_inputs=False,
    reverse_prob=0,
    t_drop_ratio=0,
    affine_prob=0,
    )