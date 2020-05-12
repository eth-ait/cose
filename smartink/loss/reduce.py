import tensorflow as tf


def reduce_mean_step(sequence, seq_mask):
  """Calculates average per step value.

  A sequence mask is required to get the number of non-padded entries.
  Args:
    sequence: [batch_size, seq_len, feature_size]
    seq_mask: [batch_size, seq_len, 1]

  Returns:
  """
  # return tf.reduce_sum(sequence) / tf.reduce_sum(seq_mask)
  n_elements = tf.maximum(1.0, tf.reduce_sum(input_tensor=seq_mask, axis=[1, 2]))
  per_step_per_sample = tf.reduce_sum(input_tensor=sequence, axis=[1, 2]) / n_elements
  return tf.reduce_mean(input_tensor=per_step_per_sample)


def reduce_mean_sequence(sequence):
  """Calculates average per sequence value.

  First sums over the sequences and then takes average across the batch.
  Args:
    sequence: [batch_size, seq_len, feature_size]

  Returns:
  """
  return tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=sequence, axis=[1, 2]))
