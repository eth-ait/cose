import tensorflow as tf
import numpy as np

from smartink.models.sequence.transformer import DecoderLayer


d_model = 64
num_heads = 4
dff = 128

batch_size = 19
seq_len = 20

# Data
shuffle_order = np.arange(seq_len)
np.random.shuffle(shuffle_order)
normal_data = np.random.normal(size=(19, 20, d_model))
shuffled_data = normal_data[:, shuffle_order]

normal_data = tf.convert_to_tensor(normal_data, dtype=tf.float32)
shuffled_data = tf.convert_to_tensor(shuffled_data, dtype=tf.float32)

# Model
decoder = DecoderLayer(d_model, num_heads, dff, rate=0)
pooler = lambda x: tf.reduce_mean(x, axis=1)
dropout = tf.keras.layers.Dropout(0.0)

# Run
normal_out = pooler(decoder(normal_data, normal_data, training=True, look_ahead_mask=None, padding_mask=None)[0])
shuffled_out = pooler(decoder(shuffled_data, shuffled_data, training=True, look_ahead_mask=None, padding_mask=None)[0])

print("Diff: ", tf.reduce_sum(tf.square(normal_out - shuffled_out)).numpy())
print(normal_out[0, 0:5])
print(shuffled_out[0, 0:5])
print("Done")