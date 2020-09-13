import os
import numpy as np
from visualization import render_strokes
from visualization import animate_strokes


sample_path = "/local/home/emre/Projects/smartink/experiments_eval/1599763052.1-PRED_TR_6x64_256-head_4-drop_0.0-TEMB_TR_64_6x256-head_4-drop_0.0-L8-4x512-gmm-loss_ER-B128_LR_tr/sketchrnn_data_2_pos_ar.npy"
render_up_to = 100

save_name = "sketchrnn_test_run_s1"
save_dir = "/local/home/emre/Projects/smartink/experiments_eval/1599763052.1-PRED_TR_6x64_256-head_4-drop_0.0-TEMB_TR_64_6x256-head_4-drop_0.0-L8-4x512-gmm-loss_ER-B128_LR_tr/"
save_path = os.path.join(save_dir, save_name)


strokes = np.load(sample_path, allow_pickle=True)
stroke_colors = None

fig_im, ax_im = render_strokes(strokes[:render_up_to], colors=stroke_colors)
fig_im.savefig(save_path + ".png", format="png", bbox_inches='tight', dpi=300)

fig_vid, ax_vid = animate_strokes(strokes[:render_up_to], colors=stroke_colors)
fig_vid.save(save_path + ".mp4")