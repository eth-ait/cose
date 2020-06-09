import os
import numpy as np
import tensorflow as tf


class JsonSaverHook(tf.estimator.SessionRunHook):
  """Tensorflow hook to write json into a file.

  It is useful to save model configuration in json with tf.Estimators.
  The file is created before the training starts.
  """

  def __init__(self, save_path, json_str):
    self.save_path = save_path
    self.json_str = json_str
    self.writer_op = None

  def begin(self):
    self.writer_op = tf.io.write_file(self.save_path, self.json_str)

  def after_create_session(self, session, coord):
    session.run(self.writer_op)


class CheckpointEvaluatorHook(tf.estimator.CheckpointSaverListener):
  """Evaluates a checkpoint just after it is saved.

  Evaluation performance is also logged and considered for early stopping.

  The main training script calls only estimator.train. After every checkpoint
  save, estimator.evaluate is called in after_save call. If the evaluation
  performance is better than the previous evaluation, then the checkpoint
  files are renamed.

  Since the saver object is not accessible, the renaming operation is required
  as the saver object keeps track of the older files to be deleted.

  When training ends, the best checkpoint is renamed into the original name
  format and the extra checkpoints are deleted.

  TODO It only keeps the best checkpoint (max_to_keep = 1)
  """

  def __init__(self, eval_func, glogger, save_path):
    self.eval_func = eval_func
    self.glogger = glogger
    self.best_eval_loss = np.inf
    self.save_path = save_path
    self.checkpoint_path = os.path.join(save_path, "model.ckpt-")
    self.best_checkpoint = 0

    self.checkpoint_pl = None
    self.checkpoint_update = None

    self.early_stopping_tolerance = 20
    self.num_steps_wo_improvement = 0
    self.improvement_ratio = 0.0001

  def begin(self):
    self.checkpoint_pl = tf.compat.v1.placeholder(tf.string)
    txt = 'model_checkpoint_path: "model.ckpt-' + self.checkpoint_pl
    txt += '"\nall_model_checkpoint_paths: "model.ckpt-'
    txt += self.checkpoint_pl + '"'

    self.checkpoint_update = tf.io.write_file(
        os.path.join(self.save_path, "checkpoint"), txt)

    # Count and log total number of parameters.
    total_parameters = 0
    for variable in tf.compat.v1.trainable_variables():
      shape = variable.get_shape()
      variable_parameters = 1
      for dim in shape:
        variable_parameters *= dim.value
      total_parameters += variable_parameters
    print("# parameters: " + str(total_parameters))
    tf.compat.v1.logging.info("# parameters: " + str(total_parameters))

  def before_save(self, session, global_step_value):
    pass

  def after_save(self, session, global_step_value):
    # Evaluate the checkpoint. Log results, keep it if it is better, apply
    # early stopping, etc.
    if global_step_value > 0:
      eval_results = self.eval_func()
      new_loss = eval_results["loss"]

      # Reset the counter if there is a significant improvement.
      if (self.best_eval_loss - new_loss) > np.abs(
          self.best_eval_loss * self.improvement_ratio):
        self.num_steps_wo_improvement = 0
      else:
        self.num_steps_wo_improvement += 1

      if new_loss <= self.best_eval_loss:
        # Log the best evaluation loss to google sheet.
        if self.glogger:
          glog_data = dict()
          for key, val in eval_results.items():
            tmp = key.split("validation/")
            if len(tmp) == 2:
              glog_data[tmp[1]] = val
          glog_data["loss"] = new_loss
          glog_data["step"] = global_step_value
          self.glogger.update_or_append_row(glog_data)

        # Delete the previously best checkpoint.
        checkpoint_files = tf.io.gfile.glob(self.checkpoint_path +
                                         str(self.best_checkpoint) + "*")
        for f in checkpoint_files:
          tf.io.gfile.remove(f)

        # Rename the best checkpoint so that it is not removed by the saver.
        checkpoint_files = tf.io.gfile.glob(self.checkpoint_path +
                                         str(global_step_value) + "*")
        for f in checkpoint_files:
          tf.io.gfile.rename(f, f + ".best", True)

        self.best_eval_loss = new_loss
        self.best_checkpoint = global_step_value
        tf.compat.v1.logging.info("Checkpoint update to " + str(self.best_checkpoint))

      # Evaluation performance no longer improves. Send stop signal.
      if self.num_steps_wo_improvement == self.early_stopping_tolerance:
        return True

  def end(self, session, global_step_value):
    # Delete the extra non-best checkpoint files.
    if global_step_value != self.best_checkpoint:
      checkpoint_files = tf.io.gfile.glob(self.checkpoint_path +
                                       str(global_step_value) + "*")
      for f in checkpoint_files:
        tf.io.gfile.remove(f)

    # Rename the "best" checkpoint into original naming format.
    checkpoint_files = tf.io.gfile.glob(self.checkpoint_path +
                                     str(self.best_checkpoint) + "*")
    for f in checkpoint_files:
      tf.io.gfile.rename(f, f[:-5], True)

    # Update checkpoint file.
    feed_dict = {self.checkpoint_pl: str(self.best_checkpoint)}
    session.run(self.checkpoint_update, feed_dict)
