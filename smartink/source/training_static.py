"""Training engine in static graph mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

from smartink.util.utils import TFSummary
from smartink.util.utils import TFSummaryAvg
from smartink.util.utils import AggregateAvg
from smartink.util.utils import LearningRateScheduler
from common.logger import GoogleSheetLogger


class TrainingEngine(object):
  """Provides training and evaluation loop with early stopping."""

  def __init__(self, config, model, train_data, valid_data, test_data=None):
    self.config = config
    self.model = model
    self.train_data = train_data
    self.valid_data = valid_data
    self.test_data = test_data

    self.model_dir = config.experiment.model_dir
    self.max_steps = config.experiment.max_steps
    self.checkpoint_frequency = config.experiment.checkpoint_frequency
    self.log_frequency = config.experiment.log_frequency

    # Create Tensorflow Routines.
    gpu_options = tf.compat.v1.GPUOptions(
        per_process_gpu_memory_fraction=0.95, allow_growth=True)
    self.session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    self.global_step = tf.compat.v1.train.get_or_create_global_step()

    lr_scheduler = LearningRateScheduler(
        initial_lr=config.experiment.learning_rate.initial_learning_rate,
        lr_type=config.experiment.learning_rate.name)
    self.learning_rate = lr_scheduler(self.global_step)

    self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

    self.summary_writer = None
    self.train_summary = None
    self.valid_summary = None

    self.saver = None
    self.checkpoint = None

    if config.get("gdrive", False):
      self.glogger = GoogleSheetLogger(
          tf.io.gfile.GFile(config.gdrive.credential, "r"),
          config.gdrive.workbook, [config.gdrive.sheet + "/valid"],
          config.experiment.id,
          static_values={
              "Model ID": config.experiment.id,
              "Model Name": config.experiment.tag,
              "Comment": config.experiment.comment
          })
    else:
      self.glogger = None

  def run(self):
    """Starts training loop."""
    # Training:
    batch_inputs, batch_targets = self.train_data.get_next()
    op_predictions = self.model(inputs=batch_inputs, training=True)
    op_loss_dict = self.model.loss(op_predictions, batch_targets, training=True)

    print("Model created.")

    # # TODO(aksan) Pre-trained stroke: not optimizing the stroke model.
    # # Renaming variables for restoring from a trained model.
    # var_list = dict()
    # var_name_to = "t_embedding"
    # var_name_from = "predictive_ink_model"
    # for var_ in self.model.embedding_model.variables:
    #   target_name = var_.name.replace(var_name_from, var_name_to)
    #   target_name = target_name.replace(":0", "")
    #   var_list[target_name] = var_

    loss = op_loss_dict["loss"]
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      parameters = tf.compat.v1.trainable_variables()
      
      # # TODO(aksan) Pre-trained stroke: not optimizing the stroke model.
      # parameters = []
      # for param in tf.trainable_variables():
      #   if param not in self.model.embedding_model.variables:
      #     parameters.append(param)
      
      # Gradient clipping.
      if self.config.experiment.grad_clip_value > 0:
        grads_vars = self.optimizer.compute_gradients(loss, parameters)
        g = self.config.experiment.grad_clip_value
        # grads_vars = [(tf.clip_by_value(grad, -g, g), var) for grad, var in
        #               grads_vars]
        clipped = []
        for grad, var in grads_vars:
          if grad is not None:
            clipped.append((tf.clip_by_value(grad, -g, g), var))
        grads_vars = clipped
        
      elif self.config.experiment.grad_clip_norm > 0:
        gradients = tf.gradients(ys=loss, xs=parameters)
        gradients, _ = tf.clip_by_global_norm(
            gradients, self.config.experiment.grad_clip_norm)
        grads_vars = zip(gradients, parameters)
      else:
        grads_vars = self.optimizer.compute_gradients(loss, parameters)

      parameter_update = self.optimizer.apply_gradients(
          grads_and_vars=grads_vars,
          global_step=tf.compat.v1.train.get_global_step())

      num_param = 0
      for var in tf.compat.v1.trainable_variables():
        num_param += np.prod(var.shape.as_list())
        print("{}: {}".format(var.name, np.prod(var.shape.as_list())))
      print("# of parameters: " + str(num_param))
      if self.glogger:
        self.glogger.set_static_cells({"parameters":num_param})

    self.summary_writer = tf.compat.v1.summary.FileWriter(self.model_dir,
                                                self.session.graph)
    self.train_summary = TFSummary(
        session=self.session, writer=self.summary_writer, collection="training")
    self.train_summary.create_summaries(tag="training/", ops=op_loss_dict)
    self.train_summary.create_summaries(
        tag="training/", ops=dict(learning_rate=self.learning_rate))

    # Validation:
    batch_inputs_valid, targets_valid_batch = self.valid_data.get_next()
    op_predictions_valid = self.model(inputs=batch_inputs_valid, training=False)
    op_loss_dict_valid = self.model.loss(op_predictions_valid,
                                         targets_valid_batch, training=False)

    self.valid_summary = TFSummaryAvg(
        session=self.session,
        writer=self.summary_writer,
        collection="validation")
    self.valid_summary.create_summaries(
        tag="validation/", ops=op_loss_dict_valid)

    eval_loss_summary = AggregateAvg()

    # # tf Checkpoint API: there is a bug with StackedRNNCell.
    # self.checkpoint = tf.train.Checkpoint(
    #     optimizer=self.optimizer,
    #     model=self.model,
    #     global_step=self.global_step)
    # self.saver = tf.contrib.checkpoint.CheckpointManager(
    #     self.checkpoint,
    #     directory=self.config.experiment.model_dir,
    #     max_to_keep=1)
    # self.checkpoint.restore(self.saver.latest_checkpoint).initialize_or_restore(
    #     self.session)

    # tf Saver API
    self.session.run(tf.compat.v1.global_variables_initializer())
    self.saver = tf.compat.v1.train.Saver(max_to_keep=1, save_relative_paths=True)
    latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
    if latest_checkpoint is not None:
      self.saver.restore(self.session, latest_checkpoint)
    else:
      self.saver.export_meta_graph(os.path.join(self.model_dir, "meta_graph"))
      
    # # TODO(aksan) Pre-trained stroke:
    # # Restore the stroke model.
    # embedding_model_dir = '/home/eaksan/Warehouse/Projects/google/experiments/1578672834.1-TEMB_biTR_64_6x256-head_4-drop_0.0-L16-3x512-binormal-B100_LR0.001-diagrams_with_strokes_wo_text_resampled20'
    #
    # pretrained = tf.train.list_variables(tf.train.latest_checkpoint(embedding_model_dir))
    # pretrained_names = dict()
    # for var_ in pretrained:
    #   pretrained_names[var_[0]] = var_[1]
    #
    # for var_name, var_ in var_list.items():
    #   if var_name not in pretrained_names:
    #     print(var_name + " not found.")
    #   else:
    #     if pretrained_names[var_name] != var_.shape.as_list():
    #       print(var_name + " shape not matching. ", str(var_.shape.as_list()), str(pretrained_names[var_name]))
    #
    #     del pretrained_names[var_name]
    #
    # if len(pretrained_names) > 0:
    #   print("Remaining vars:" )
    #   for var_ in pretrained_names.keys():
    #     if "Adam" not in var_:
    #       print(var_)
    #
    # embedding_saver = tf.train.Saver(var_list=var_list)
    # embedding_checkpoint = tf.train.latest_checkpoint(embedding_model_dir)
    # embedding_saver.restore(self.session, embedding_checkpoint)

    # Early stopping configuration.
    improvement_ratio = 0.001
    best_valid_loss = np.inf
    num_steps_wo_improvement = 0
    early_stopping_tolerance = 40

    stop_signal = False
    step = self.session.run(self.global_step)
    epoch = 0
    train_iter = self.train_data.get_iterator()
    valid_iter = self.valid_data.get_iterator()

    print("Running Training Loop.")
    print("Experiment directory: " + self.model_dir)
    train_summary_op = self.train_summary.summary_op
    self.session.run(train_iter.initializer)
    self.session.run(valid_iter.initializer)
    while not stop_signal:
      if step >= self.max_steps:
        print("End of Training.")
        break

      # Training.
      for _ in range(self.checkpoint_frequency):
        try:
          start_time = time.perf_counter()
          step += 1
          loss_dict, train_summary, _, _, lr = self.session.run([
              op_loss_dict, train_summary_op, op_predictions, parameter_update,
              self.learning_rate
          ])

          if step % self.log_frequency == 0:
            time_elapsed = (time.perf_counter() - start_time)

            self.train_summary.add_summary(train_summary, step)
            self.model.log_loss(
                loss_dict,
                prefix="Train [{:04d}] \t".format(step),
                suffix="lr: {:.4f} time/batch = {:.3f}".format(
                    lr, time_elapsed))
        except tf.errors.OutOfRangeError:
          self.session.run(train_iter.initializer)
          epoch += 1

      # Evaluation: make a full pass on the evaluation data.
      start_time = time.perf_counter()
      try:
        while True:
          loss_dict_valid, _ = self.session.run(
              [op_loss_dict_valid, op_predictions_valid])
          eval_loss_summary.add(loss_dict_valid)

      except tf.errors.OutOfRangeError:
        self.session.run(valid_iter.initializer)

      eval_loss_dict, eval_step = eval_loss_summary.summary_and_reset()
      self.valid_summary.add_summary(eval_loss_dict, step)
      self.model.log_loss(
          eval_loss_dict,
          prefix="Valid [{:04d}] \t".format(step),
          suffix="time/batch = {:.3f}".format(
              (time.perf_counter() - start_time) / eval_step))

      # Early stopping check.
      valid_loss = eval_loss_dict["loss"]
      if (best_valid_loss - valid_loss) > np.abs(
          best_valid_loss * improvement_ratio):
        num_steps_wo_improvement = 0
      else:
        num_steps_wo_improvement += 1
      if num_steps_wo_improvement == early_stopping_tolerance:
        stop_signal = True

      if valid_loss <= best_valid_loss:
        if self.glogger:
          eval_loss_dict["step"] = step
          self.glogger.update_or_append_row(eval_loss_dict)

        best_valid_loss = valid_loss
        print("Saving model to {}".format(self.model_dir))

        # # tf Checkpoint API
        # with self.session.as_default():
        #   self.saver.save(checkpoint_number=self.global_step)

        # tf Saver API
        self.saver.save(
            self.session,
            os.path.normpath(os.path.join(self.model_dir, "model.ckpt")),
            global_step=step)
