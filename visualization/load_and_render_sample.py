import os
import glob
import numpy as np
from visualization import render_strokes
from visualization import animate_strokes


def visualize_and_save(strokes_, colors_, fps_interval_, save_path_):
  fig_im, ax_im = render_strokes(strokes_, colors=colors_)
  fig_im.savefig(save_path_ + ".png", format="png", bbox_inches='tight', dpi=300)
  
  fig_vid, ax_vid = animate_strokes(strokes_,
                                    colors=colors_,
                                    interval=fps_interval_)
  fig_vid.save(save_path_ + ".mp4")
  


sample_dir = "/local/home/emre/Projects/smartink/cose_sketchrnn_comparison/"

# Cose
sample_names = ["sample64_139/cose/given_1/data_64_pos_ar", "sample64_139/cose/given_full/data_64_pos_ar", "sample63_155/cose/given_half/data_63_pos_ar", "sample55_164/cose/given_2/data_55_pos_ar", "sample55_164/cose/given_full/data_55_pos_ar"]
given_strokes = [1, 5, 5, 2, 5]
n_strokes = [10, 20, 12, 9, 15]  # of strokes to be rendered.
fps_interval=20

# SketchRNN
sample_names = ["sample64_139/sketchrnn/given_1/data_139_pos_ar", "sample64_139/sketchrnn/given_full/data_139_pos_ar", "sample63_155/sketchrnn/given_half/data_155_pos_ar", "sample55_164/sketchrnn/given_2/data_164_pos_ar", "sample55_164/sketchrnn/given_full/data_164_pos_ar"]
given_strokes = [1, 5, 5, 2, 5]
n_strokes = [7, 12, 12, 10, 8]  # of strokes to be rendered.
fps_interval=40


assert  len(sample_names) == len(given_strokes) == len(n_strokes), "Numbers don't match."
save_dir = sample_dir

for sample_name, render_up_to, given_stroke in zip(sample_names, n_strokes, given_strokes):
  print("Visualizing sample " + sample_name + "...")
  save_name = sample_name + "_s" + str(render_up_to)
  
  save_path = os.path.join(save_dir, save_name)
  sample_path = os.path.join(sample_dir, sample_name + ".npy")
  
  strokes = np.load(sample_path, allow_pickle=True)
  stroke_colors = None
  visualize_and_save(strokes[:render_up_to], stroke_colors, fps_interval, save_path)