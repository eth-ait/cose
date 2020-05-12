"""Plot learning rate curves for a number of configurations."""

import tensorflow as tf
from smartink.util.learning_rate import TransformerDecay
import matplotlib.pyplot as plt


warmups = [4000, 4000, 4000, 2000, 10000]
d_models = [64, 128, 1000, 128, 64]
initial_lrs = [1e-3]*len(warmups)

lr_models = []
for i in range(len(warmups)):
  lr_models.append(TransformerDecay(initial_lrs[i],
                                    d_model=d_models[i],
                                    warmup_steps=warmups[i]))

x_range = list(range(1, 50000, 1000))
x_range_tf = tf.Variable(x_range)
for i, model in enumerate(lr_models):
  lr_values = model(x_range_tf)
  plt.plot(x_range, lr_values, label="{}_{}".format(d_models[i], warmups[i]))
  
plt.legend()
plt.savefig("./transformer_lr.png", bbox_inches='tight', dpi=200)
plt.close()

