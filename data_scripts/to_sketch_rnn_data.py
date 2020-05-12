import os
import numpy as np

from smartink.data.stroke_dataset import TFRecordStroke
from common.constants import Constants as C


batch_size_ = 5
# DATA_DIR = "../data/quickdraw_cats/"
DATA_DIR = "../data/diagrams_with_strokes_wo_text/"
SPLIT = "training"
# META_FILE = "quickdraw-50m-cats-whole-stats-relative_pos.npy"
META_FILE = "strokes_diagrams_wo_text_ramer_0.01_stats-relative_pos.npy"
# tfrecord_pattern = "quickdraw-50m-cats-whole.tfrecord-?????-of-?????"
tfrecord_pattern = "strokes_diagrams_wo_text_ramer_0.01.tf_record-?????-of-?????"
data_path_ = os.path.join(DATA_DIR, SPLIT, tfrecord_pattern)

# DATA_DIR = "../data/diagrams_with_strokes_wo_text/"
# SPLIT = "training"
# META_FILE = "strokes_diagrams_wo_text_ramer_0.01_stats-origin-relative_pos.npy"
# tfrecord_pattern = SPLIT + "/strokes_diagrams_wo_text_ramer_0.01.tf_record-?????-of-?????"

train_data = TFRecordStroke(
    data_path=data_path_,
    meta_data_path=DATA_DIR + META_FILE,
    batch_size=1,
    shuffle=False,
    normalize=True,
    pp_to_origin=False,
    pp_relative_pos=True,
    run_mode=C.RUN_EAGER,
    max_length_threshold=100,
    fixed_len=False)

valid_data = TFRecordStroke(
    data_path=os.path.join(DATA_DIR, "validation", tfrecord_pattern),
    meta_data_path=DATA_DIR + META_FILE,
    batch_size=1,
    shuffle=False,
    normalize=False,
    pp_to_origin=False,
    pp_relative_pos=True,
    run_mode=C.RUN_EAGER,
    max_length_threshold=100,
    fixed_len=False)

test_data = TFRecordStroke(
    data_path=os.path.join(DATA_DIR, "test", tfrecord_pattern),
    meta_data_path=DATA_DIR + META_FILE,
    batch_size=1,
    shuffle=False,
    normalize=False,
    pp_to_origin=False,
    pp_relative_pos=True,
    run_mode=C.RUN_EAGER,
    max_length_threshold=100,
    fixed_len=False)

def stroke_batch_to_diagram(batch_, undo_fn):
  ink_batch, seq_len = undo_fn(batch_[C.INP_ENC].numpy(), batch_[C.INP_START_COORD].numpy(), batch_[C.INP_SEQ_LEN].numpy())
  # ink_batch = batch_[0][C.INP_ENC].numpy()
  # seq_len = batch_[0][C.INP_SEQ_LEN].numpy()

  strokes = []
  for i in range(ink_batch.shape[0]):
    strokes.append(ink_batch[i][:seq_len[i]])
  ink_ = np.concatenate(strokes, axis=0)
  pen_ = ink_[:, 2:]
  diff_ = ink_[1:, 0:2] - ink_[0:-1, 0:2]
  diff_ = np.concatenate([[[0, 0]], diff_], axis=0)
  ink_ = np.concatenate([diff_, pen_], axis=1)
  return ink_

train_strokes = []
valid_strokes = []
test_strokes = []
lens = []
i = 0
for sample_ in train_data.iterator:
  # sample_ = stroke_batch_to_diagram(sample_[0], train_data.np_undo_preprocessing)
  sample_ = sample_[0][C.INP_ENC][0].numpy()
  if sample_.shape[0] < 300:
    train_strokes.append(sample_)
    lens.append(sample_.shape[0])
    i += 1
lens = np.array(lens)
print("# training samples: {}, min-max lengths {}-{}".format(str(i), lens.min(), lens.max()))

i = 0
lens = []
for sample_ in valid_data.iterator:
  # sample_ = stroke_batch_to_diagram(sample_[0], train_data.np_undo_preprocessing)
  sample_ = sample_[0][C.INP_ENC][0].numpy()
  if sample_.shape[0] < 300:
    valid_strokes.append(sample_)
    lens.append(sample_.shape[0])
    i += 1
lens = np.array(lens)
print("# validation samples: {}, min-max lengths {}-{}".format(str(i), lens.min(), lens.max()))

i = 0
lens = []
for sample_ in test_data.iterator:
  # sample_ = stroke_batch_to_diagram(sample_[0], train_data.np_undo_preprocessing)
  sample_ = sample_[0][C.INP_ENC][0].numpy()
  if sample_.shape[0] < 300:
    test_strokes.append(sample_)
    lens.append(sample_.shape[0])
    i += 1
lens = np.array(lens)
print("# test samples: {}, min-max lengths {}-{}".format(str(i), lens.min(), lens.max()))

# np.savez_compressed("quickdraw_in_sketch_rnn_format.npz",
np.savez_compressed("diagrams_wo_text_in_sketch_rnn_format.npz",
                    train=train_strokes,
                    valid=valid_strokes,
                    test=test_strokes)
print("Done")