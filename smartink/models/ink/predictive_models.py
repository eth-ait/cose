from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from common.constants import Constants as C
from smartink.util.utils import err_unknown_type
from smartink.models.base_model import BaseModel
from smartink.models.stroke.t_emb import TEmbedding


class PredictiveInkModel(BaseModel):
  """A sequence model that predicts the next stroke.

  It consists of a seq2seq stroke stroke model (InkRNNSeq2Seq) and
  a sequence prediction model (vanilla RNN).
  """

  def __init__(self,
               embedding_model,
               predictive_model,
               position_model,
               loss_predicted_embedding,
               loss_predicted_ink,
               loss_reconstructed_ink,
               input_type,
               start_positions,
               end_positions,
               stop_predictive_grad,
               config_loss,
               num_predictive_inputs=8,
               run_mode=C.RUN_STATIC,
               **kwargs):
    """Constructor.

    Args:
      embedding_model:
      predictive_model:
      loss_predicted_embedding:
      loss_predicted_ink:
      loss_reconstructed_ink:
      config_loss:
      input_type: ink model input configuration. Options are ordered,
        last_step or leave_out_out.
      start_positions: whether to use stroke start coordinates or not.
      end_positions: whether to use stroke end coordinates or not.
      stop_predictive_grad: disable gradient flow to the embedding model.
      num_predictive_inputs: # of input configurations to the predictive model.
      run_mode: eager, static or estimator.
      **kwargs:
    """
    super(PredictiveInkModel, self).__init__(
        config_loss=config_loss, run_mode=run_mode, **kwargs)
    self.embedding_model = embedding_model
    self.predictive_model = predictive_model
    self.position_model=position_model
    self.loss_predicted_position = self.position_model is not None
    self.loss_predicted_embedding = loss_predicted_embedding
    self.loss_predicted_ink = loss_predicted_ink
    self.loss_reconstructed_ink = loss_reconstructed_ink
    self.run_mode = run_mode
    self.input_type = input_type
    self.start_positions = start_positions
    self.end_positions = end_positions
    self.stop_predictive_grad = stop_predictive_grad
    self.num_predictive_inputs = num_predictive_inputs
    
    if self.input_type == "hybrid":
      self.num_predictive_inputs //= 2
      

    self.embedding_size = embedding_model.n_latent_units
    self.is_sequence_decoder = not isinstance(self.embedding_model, TEmbedding)
    # self.is_sequence_decoder = True

  def build(self, input_shape=None):
    pass

  # @tf.function(experimental_relax_shapes=True)
  def call(self, inputs, training=False, **kwargs):
    """Call method.

    Args:
      inputs (dict): expected to contain inputs for the encoder and decoder,
        sequence length and number of strokes ops.
      training: whether in training mode or not.
      **kwargs:

    Returns:
      [batch_size, seq_len, feature_size]
    """
    input_num_strokes = inputs[C.INP_NUM_STROKE]
    # tf.keras compile, fit, predict, etc. methods cause it to be 2-dimensional.
    if len(inputs[C.INP_NUM_STROKE].shape) == 2:
      input_num_strokes = inputs[C.INP_NUM_STROKE][:, 0]

    out_dict = dict()
    gt_reconstruction = self.embedding_model.call(inputs, training=training)
    out_dict["reconstructed_ink"] = gt_reconstruction
    embedding = gt_reconstruction["embedding"]
    out_dict["embedding_sample"] = out_dict["reconstructed_ink"]["embedding_sample"]
    
    # Before making a prediction for the next stroke, reshape
    # embeddings so that we have a sequence of stroke embeddings,
    # representing the diagram samples.
    diagram_embedding = self.batch_stroke_to_diagram(embedding,
                                                     input_num_strokes)
    
    # Probabilistic inputs if the stroke model is probabilistic.
    # draw inputs for the ink model.
    embedding_sample = self.embedding_model.net_embedding.draw_sample(diagram_embedding, greedy=True)
    
    # Determine ink model inputs and targets.
    n_strokes = tf.shape(input=embedding_sample)[0]
    seq_len = tf.shape(input=embedding_sample)[1]
    batch_idx = tf.range(n_strokes)
    # (1) Predict the last step only.
    if self.input_type == "last_step":
      target_idx = input_num_strokes-1

      input_range = tf.tile(tf.range(seq_len)[tf.newaxis, :], [n_strokes, 1])
      mask_ = tf.not_equal(input_range, tf.tile(target_idx[:, tf.newaxis], [1, seq_len]))
      gather_input_idx = tf.reshape(tf.compat.v1.where(mask_), [n_strokes, seq_len - 1, 2])
      pred_input_seq_len = input_num_strokes - 1

      gather_target_idx = tf.stack([
          batch_idx,
          target_idx
          ], axis=-1)
    
    # (2) Leave-one-out with random targets. Use all strokes except to target
    # as input.
    elif self.input_type == "leave_one_out":
      i = tf.random.uniform([n_strokes], minval=0, maxval=1, dtype=tf.float32)
      target_idx = tf.round(i*tf.cast(input_num_strokes - 1, tf.float32))
      target_idx = tf.cast(target_idx, tf.int32)

      input_range = tf.tile(tf.range(seq_len)[tf.newaxis, :], [n_strokes, 1])
      mask_ = tf.not_equal(input_range, tf.tile(target_idx[:, tf.newaxis], [1, seq_len]))
      gather_input_idx = tf.reshape(tf.compat.v1.where(mask_), [n_strokes, seq_len - 1, 2])
      pred_input_seq_len = input_num_strokes - 1

      gather_target_idx = tf.stack([
          batch_idx,
          target_idx
          ], axis=-1)
      
    elif self.input_type in ["random", "ordered", "hybrid"]:
      min_n_stroke = tf.reduce_min(input_tensor=input_num_strokes)
      max_n_stroke = tf.reduce_max(input_tensor=input_num_strokes)
      input_range_ = tf.tile(tf.range(max_n_stroke)[tf.newaxis, :], [n_strokes, 1])
      
      def get_random_inp_target_pairs():
        """Get a randomly generated input set and a target."""
        # Randomly set the number of inputs.
        n_inputs_ = tf.random.uniform([1], minval=2, maxval=min_n_stroke, dtype=tf.int32)[0]
        # Randomly pick a target.
        i_ = tf.random.uniform([n_strokes], minval=0, maxval=1, dtype=tf.float32)
        target_idx_ = tf.round(i_*tf.cast(input_num_strokes - 1, tf.float32))
        target_idx_ = tf.cast(target_idx_, tf.int32)

        mask_ = tf.not_equal(input_range_, tf.tile(target_idx_[:, tf.newaxis], [1, max_n_stroke]))
        all_input_idx_ = tf.reshape(tf.compat.v1.where(mask_), [n_strokes, max_n_stroke - 1, 2])
        all_input_idx_ = tf.transpose(a=all_input_idx_[:, :min_n_stroke - 1], perm=[1, 0, 2])
        all_input_idx_ = tf.random.shuffle(all_input_idx_)
        gather_input_idx_ = tf.cast(tf.transpose(a=all_input_idx_[:n_inputs_], perm=[1,0,2]), tf.int32)
        return gather_input_idx_, target_idx_, n_inputs_
      
      def get_ordered_inp_target_pairs(random_target=False):
        """Get a slice (i.e., window) randomly."""
        # Randomly set the number of inputs.
        n_inputs_ = tf.random.uniform([1], minval=2, maxval=min_n_stroke, dtype=tf.int32)[0]
        # Select start index of the window.
        start_idx = tf.random.uniform([1], minval=0, maxval=min_n_stroke - n_inputs_, dtype=tf.int32)[0]
        
        if not random_target:
          # Target is the next,
          target_idx_ = tf.tile(tf.expand_dims(start_idx+n_inputs_, axis=0), [n_strokes])
        else:
          # Randomly pick a target.
          i_ = tf.random.uniform([n_strokes], minval=0, maxval=1, dtype=tf.float32)
          target_idx_ = tf.round(i_*tf.cast(input_num_strokes - 1, tf.float32))
          target_idx_ = tf.cast(target_idx_, tf.int32)

        mask_ = tf.not_equal(input_range_, tf.tile(target_idx_[:, tf.newaxis], [1, max_n_stroke]))
        all_input_idx_ = tf.reshape(tf.compat.v1.where(mask_), [n_strokes, max_n_stroke - 1, 2])[:, :min_n_stroke]
        gather_input_idx_ = tf.cast(all_input_idx_[:, start_idx:start_idx+n_inputs_], tf.int32)
        return gather_input_idx_, target_idx_, n_inputs_
      
      all_n_inputs = []
      all_gather_input_idx = []
      all_target_idx = []
      all_seq_lens = []
      
      if self.input_type in ["random", "hybrid"]:
        for i in range(self.num_predictive_inputs):
          gather_input_idx, target_idx, n_inputs = get_random_inp_target_pairs()
          all_gather_input_idx.append(gather_input_idx)
          all_target_idx.append(target_idx)
          all_n_inputs.append(n_inputs)
          all_seq_lens.append(tf.ones([n_strokes], dtype=tf.int32)*n_inputs)

      if self.input_type in ["ordered", "hybrid"]:
        for i in range(self.num_predictive_inputs):
          gather_input_idx, target_idx, n_inputs = get_ordered_inp_target_pairs()
          all_gather_input_idx.append(gather_input_idx)
          all_target_idx.append(target_idx)
          all_n_inputs.append(n_inputs)
          all_seq_lens.append(tf.ones([n_strokes], dtype=tf.int32)*n_inputs)
        
      max_len = tf.reduce_max(input_tensor=all_n_inputs)
      for i in range(len(all_n_inputs)):
        all_gather_input_idx[i] = tf.pad(tensor=all_gather_input_idx[i], paddings=[[0, 0], [0, max_len - all_n_inputs[i]], [0, 0]])
        
      gather_input_idx = tf.concat(all_gather_input_idx, axis=0)
      pred_input_seq_len = tf.concat(all_seq_lens, axis=0)

      gather_target_idx = tf.stack([
          tf.tile(batch_idx, [len(all_n_inputs)]),
          tf.concat(all_target_idx, axis=0)
          ], axis=-1)

    else:
      err_unknown_type(self.input_type)
    
    # if "sigma" in diagram_embedding:
    #   pred_targets = dict()
    #   pred_targets["mu"] = tf.stop_gradient(tf.gather_nd(diagram_embedding["mu"], gather_target_idx), name="embedding_mu_target_stop")
    #   pred_targets["sigma"] = tf.stop_gradient(tf.gather_nd(diagram_embedding["sigma"], gather_target_idx), name="embedding_sigma_target_stop")
    # else:
    pred_targets = tf.gather_nd(embedding_sample, gather_target_idx)
    pred_targets = tf.stop_gradient(pred_targets, name="embedding_target_stop")
    
    if self.stop_predictive_grad:
      # Disable gradient flow from the predictive/position models to the
      # embedding model.
      pred_input = tf.stop_gradient(tf.gather_nd(embedding_sample, gather_input_idx), name="embeddings_pred_input_stop")
    else:
      pred_input = tf.gather_nd(embedding_sample, gather_input_idx)

    start_pos = tf.reshape(inputs["start_coord"], [n_strokes, seq_len, 2])
    start_context_pos = tf.gather_nd(start_pos, gather_input_idx)
    if self.end_positions:
      end_pos = tf.reshape(inputs["end_coord"], [n_strokes, seq_len, 2])
      end_context_pos = tf.gather_nd(end_pos, gather_input_idx)
      context_pos = tf.concat([start_context_pos, end_context_pos], axis=-1)
    else:
      context_pos = start_context_pos
      
    target_pos = tf.expand_dims(tf.gather_nd(start_pos, gather_target_idx), axis=1)
    predicted_embedding = self.predictive_model(inputs=dict(input_seq=pred_input,
                                                            input_cond=context_pos,
                                                            target_cond=target_pos,
                                                            seq_len=pred_input_seq_len),
                                                training=True)

    pred_emb_sample = self.predictive_model.output_layer.draw_sample(predicted_embedding, greedy=True)
    embedding_out_dict = dict(
        prediction=predicted_embedding,
        target=pred_targets,
        input_seq_len=pred_input_seq_len,
        embedding_sample=pred_emb_sample)
    out_dict["embedding"] = embedding_out_dict

    if self.position_model is not None:
      predicted_pos = self.position_model(inputs=dict(input_seq=pred_input,
                                                      input_cond=context_pos,
                                                      target_cond=None,
                                                      seq_len=pred_input_seq_len),
                                          training=True)
      pred_pos_sample = self.position_model.output_layer.draw_sample(predicted_pos, greedy=True)
      pos_out_dict = dict(
          prediction=predicted_pos,
          target=target_pos[:, 0],
          input_seq_len=pred_input_seq_len,
          position_sample=pred_pos_sample)
      out_dict["position"] = pos_out_dict
    
    ### Decode the predicted embedding if not using a sequence decoder as it is
    # too slow.
    # if False:
    #   # Select decoder t inputs corresponding to the predicted stroke.
    #   t_batch_diagram = tf.reshape(decoder_inputs, [n_strokes, seq_len, -1])
    #   t_target = tf.gather_nd(t_batch_diagram, gather_target_idx)
    #   predicted_ink = self.embedding_model.call_decode(embedding=pred_emb_sample,
    #                                                    decoder_inputs=t_target,
    #                                                    training=training)
    #   predicted_ink["embedding"] = pred_emb_sample
    #
    #   # TODO(aksan) Need to pass those to select targets.
    #   predicted_ink["shape_0"] = n_strokes
    #   predicted_ink["shape_1"] = seq_len
    #   predicted_ink["gather_target_idx"] = gather_target_idx
    #   out_dict["predicted_ink"] = predicted_ink
      
    return out_dict
  
  def predict_embedding(self, embeddings, target_idx, seq_lens, input_idx=None, input_type="leave_one_out", start_positions=None):
    if isinstance(embeddings, dict):
      embeddings = self.embedding_model.net_embedding.draw_sample(embeddings)
      
    # Determine ink model inputs and targets.
    n_strokes = tf.shape(input=embeddings)[0]
    # seq_len = tf.shape(embeddings)[1]
    seq_len = seq_lens[0]
    batch_idx = tf.range(n_strokes)
    
    if input_idx is None:
      if input_type == "leave_one_out":
        input_range = tf.tile(tf.range(seq_len)[tf.newaxis, :], [n_strokes, 1])
        mask_ = tf.not_equal(input_range,
                             tf.tile(target_idx[:, tf.newaxis], [1, seq_len]))
        gather_input_idx = tf.reshape(tf.compat.v1.where(mask_), [n_strokes, seq_len - 1, 2])
        pred_input_seq_len = seq_lens - 1
      elif input_type == "last_step" or input_type == "ordered":
        input_range = tf.tile(tf.range(seq_len)[tf.newaxis, :], [n_strokes, 1])
        mask_ = tf.less(input_range,
                        tf.tile(target_idx[:, tf.newaxis], [1, seq_len]))
        gather_input_idx = tf.reshape(tf.compat.v1.where(mask_),
                                      [n_strokes, tf.reduce_max(input_tensor=target_idx), 2])
        pred_input_seq_len = target_idx
      else:
        err_unknown_type(input_type)
    else:
      gather_input_idx = tf.stack([
          tf.zeros_like(input_idx),
          input_idx
          ], axis=-1)
      pred_input_seq_len = tf.Variable([tf.shape(input=input_idx)[1]])

    gather_target_idx = tf.stack([
        batch_idx,
        target_idx
        ], axis=-1)
  
    pred_targets = tf.gather_nd(embeddings, gather_target_idx)
    pred_input = tf.gather_nd(embeddings, gather_input_idx)

    start_pos = tf.reshape(start_positions, [n_strokes, seq_len, 2])
    start_context_pos = tf.gather_nd(start_pos, gather_input_idx)
    if self.end_positions:
      end_pos = tf.reshape(start_positions, [n_strokes, seq_len, 2])
      end_context_pos = tf.gather_nd(end_pos, gather_input_idx)
      context_pos = tf.concat([start_context_pos, end_context_pos], axis=-1)
    else:
      context_pos = start_context_pos
    
    target_pos = tf.expand_dims(tf.gather_nd(start_pos, gather_target_idx), axis=1)
    out_ = self.predictive_model(inputs=dict(input_seq=pred_input,
                                             input_cond=context_pos,
                                             target_cond=target_pos,
                                             seq_len=pred_input_seq_len),
                                 training=False)
    
    out_["gather_target_idx"] = gather_target_idx
    out_["gather_input_idx"] = gather_input_idx
    out_["embedding_sample"] = self.predictive_model.output_layer.draw_sample(out_, greedy=True)
    return out_, pred_targets

  def predict_embedding_ar(self, inp_embeddings, inp_pos=None, target_pos=None, seq_len=None, greedy=True):
    # Determine ink model inputs and targets.
    if seq_len is None:
      n_strokes = tf.shape(input=inp_embeddings)[1]
      seq_len = tf.ones_like(inp_embeddings[:, 0, 0], dtype=tf.int32)*n_strokes
    out_ = self.predictive_model(inputs=dict(input_seq=inp_embeddings,
                                             input_cond=inp_pos,
                                             target_cond=target_pos,
                                             seq_len=seq_len),
                                 training=False)
    
    out_["embedding_sample"] = self.predictive_model.output_layer.draw_sample(out_, greedy=greedy)
    return out_

  def predict_position_ar(self, inp_embeddings, inp_pos=None, target_pos=None, seq_len=None, greedy=False):
    # Determine ink model inputs and targets.
    if seq_len is None:
      n_strokes = tf.shape(input=inp_embeddings)[1]
      seq_len = tf.ones_like(inp_embeddings[:, 0, 0], dtype=tf.int32)*n_strokes
    out_ = self.position_model(inputs=dict(input_seq=inp_embeddings,
                                           input_cond=inp_pos,
                                           target_cond=target_pos,
                                           seq_len=seq_len),
                               training=False)
    out_["position_sample"] = self.position_model.output_layer.draw_sample(out_, greedy=greedy)
    return out_

  @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 8], dtype=tf.float32),
                                tf.TensorSpec(shape=[None, None, 2], dtype=tf.float32),
                                tf.TensorSpec(shape=[None, 1, 2], dtype=tf.float32)])
  def serving_predict_embedding(self, inp_embeddings, inp_pos, target_pos):
    """TF Serving interface to predict the next embedding."""
    n_strokes = tf.shape(input=inp_embeddings)[1]
    seq_len = tf.ones_like(inp_embeddings[:, 0, 0], dtype=tf.int32)*n_strokes

    out_ = self.predictive_model(inputs=dict(input_seq=inp_embeddings,
                                             input_cond=inp_pos,
                                             target_cond=target_pos,
                                             seq_len=seq_len),
                                 training=False)
    sample_ = self.predictive_model.output_layer.draw_sample(out_, greedy=True)
    out_ = self.predictive_model.output_layer.reshape_dist_params(out_)
    out_["embedding_sample"] = sample_
    return out_
  
  @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 8], dtype=tf.float32),
                                tf.TensorSpec(shape=[None, None, 2], dtype=tf.float32)])
  def serving_predict_position(self, inp_embeddings, inp_pos):
    """TF Serving interface to predict the next start position."""
    n_strokes = tf.shape(input=inp_embeddings)[1]
    seq_len = tf.ones_like(inp_embeddings[:, 0, 0], dtype=tf.int32)*n_strokes
    out_ = self.position_model(inputs=dict(input_seq=inp_embeddings,
                                           input_cond=inp_pos,
                                           target_cond=None,
                                           seq_len=seq_len),
                               training=False)
    sample_ = self.position_model.output_layer.draw_sample(out_, greedy=True)
    out_ = self.position_model.output_layer.reshape_dist_params(out_)
    out_["position_sample"] = sample_
    return out_

  def loss(self, predictions, targets, seq_len=None, prefix="", training=False):
    """Calculates loss.

    Args:
      predictions:
      targets:
      seq_len:
      prefix:
      training:

    Returns:
    """
    loss_metric_dict = dict()
    loss_val_dict = dict()
    total_loss_ops = list()

    if self.is_sequence_decoder:
      recon_seq_len = targets["seq_len"]
    else:
      recon_seq_len = targets["stroke_mask"]
    
    # Reconstruction of the input sequence.
    reconstruction = self.embedding_model.loss(
        predictions["reconstructed_ink"],
        targets,
        seq_len=recon_seq_len,
        prefix="reconstruction",
        training=training)
    if self.run_mode == C.RUN_ESTIMATOR:
      loss_metric_dict.update(reconstruction[1])
      reconstruction = reconstruction[0]
    loss_val_dict.update(reconstruction)
    if self.loss_reconstructed_ink:
      total_loss_ops.append(reconstruction["reconstruction_loss"])

    # Prediction of the next stroke step.
    # if self.loss_predicted_embedding:
    embedding = self.loss_fn(
        dict(predicted_embedding=self.predictive_model.config_loss),
        predictions["embedding"],
        predictions["embedding"],
        seq_len=predictions["embedding"]["input_seq_len"],
        prefix="embedding",
        run_mode=self.run_mode,
        training=training)
    if self.run_mode == C.RUN_ESTIMATOR:
      loss_metric_dict.update(embedding[1])
      embedding = embedding[0]
    loss_val_dict.update(embedding)
    if self.loss_predicted_embedding:
      total_loss_ops.append(embedding["embedding_loss"])
      
    if self.position_model is not None:
      position = self.loss_fn(
          dict(predicted_position=self.position_model.config_loss),
          predictions["position"],
          predictions["position"],
          seq_len=predictions["position"]["input_seq_len"],
          prefix="position",
          run_mode=self.run_mode,
          training=training)
      if self.run_mode == C.RUN_ESTIMATOR:
        loss_metric_dict.update(position[1])
        position = position[0]
      loss_val_dict.update(position)
      if self.loss_predicted_position:
        total_loss_ops.append(position["position_loss"])

    ### Skip evaluating the decoded strokes if not using a sequence decoder.
    # if False:
    #   # Prediction of the next sequence by using the predicted stroke.
    #   # TODO(aksan) Manually editing targets for reconstruction from the predicted embeddings.
    #   gather_target_idx = predictions["predicted_ink"]["gather_target_idx"]
    #   n_strokes = predictions["predicted_ink"]["shape_0"]
    #   seq_len = predictions["predicted_ink"]["shape_1"]
    #
    #   stroke_target_key = self.config_loss["predicted_ink"]["stroke"]["target_key"]
    #   pen_target_key = self.config_loss["predicted_ink"]["pen"]["target_key"]
    #
    #   all_stroke_targets = tf.reshape(targets[stroke_target_key], [n_strokes, seq_len, -1, 2])
    #   all_pen_targets = tf.reshape(targets[pen_target_key], [n_strokes, seq_len, -1, 1])
    #
    #   stroke_targets = tf.gather_nd(all_stroke_targets, gather_target_idx)
    #   pen_targets = tf.gather_nd(all_pen_targets, gather_target_idx)
    #
    #   tmp_targets = dict()
    #   tmp_targets[stroke_target_key] = tf.reshape(stroke_targets, [-1, 2])
    #   tmp_targets[pen_target_key] = tf.reshape(pen_targets, [-1, 1])
    #
    #   prediction = self.loss_fn(
    #       self.config_loss["predicted_ink"],
    #       predictions["predicted_ink"],
    #       tmp_targets,
    #       seq_len=tf.ones_like(tmp_targets[stroke_target_key][:,0]),
    #       prefix="prediction",
    #       run_mode=self.run_mode,
    #       training=training)
    #
    #   if self.run_mode == C.RUN_ESTIMATOR:
    #     loss_metric_dict.update(prediction[1])
    #     prediction = prediction[0]
    #   loss_val_dict.update(prediction)
    #   if self.loss_predicted_ink:
    #     total_loss_ops.append(prediction["prediction_loss"])

    loss_val_dict["loss"] = tf.math.add_n(total_loss_ops, "total_loss")
    if self.run_mode == C.RUN_ESTIMATOR:
      loss_metric_dict["loss"] = tf.keras.metrics.Mean(loss_val_dict["loss"])
      return loss_val_dict, loss_metric_dict
    else:
      return loss_val_dict

  def batch_stroke_to_diagram(self, stroke_embedding, num_strokes):
    """Reshapes embeddings from batch of strokes to batch of diagrams.

    Args:
      stroke_embedding: Tensor of [num_diagrams x num_strokes, embedding_size]
        or dictionary of tensors with the same shape.
      num_strokes: [num_diagrams]

    Returns:
      Diagram as sequence of stroke embeddings [num_diagrams, num_strokes,
      embedding_size].
    """
    padded_num_strokes = tf.reduce_max(input_tensor=num_strokes)
    num_diagrams = tf.shape(input=num_strokes)[0]

    if isinstance(stroke_embedding, tf.Tensor):
      return tf.reshape(stroke_embedding,
                        [num_diagrams, padded_num_strokes, self.embedding_size])
    else:
      out_dict = dict()
      for key_, value_ in stroke_embedding.items():
        out_dict[key_] = tf.reshape(
            value_, [num_diagrams, padded_num_strokes, self.embedding_size])
      return out_dict

  def batch_diagram_to_stroke(self, diagram_embedding):
    """Reshapes embeddings from batch of diagrams to batch of strokes.

    Args:
      diagram_embedding: Tensor or dictionary of tensors with shape
        [num_diagrams, num_strokes, embedding_size]

    Returns:
      Batch of embeddings [num_diagrams x num_strokes, 1, embedding_size].
    """
    if isinstance(diagram_embedding, tf.Tensor):
      return tf.reshape(diagram_embedding, [-1, self.embedding_size])
    else:
      out_dict = dict()
      for key_, value_ in diagram_embedding.items():
        out_dict[key_] = tf.reshape(value_, [-1, self.embedding_size])
      return out_dict
    
  @classmethod
  def to_diagram_with_t_sample(cls, stroke_batch, n_samples, n_strokes, n_t):
    """Reshapes a batch of "strokes" to batch of diagrams.

    Args:
      stroke_batch:
      n_samples: number of diagram samples.
      n_strokes: max. number of strokes
      n_t: number of t samples

    Returns:
      Diagram as sequence of stroke embeddings [n_samples*n_t, n_strokes,
      feature_size].
    """
    def handle_dict(inp_):
      if isinstance(inp_, tf.Tensor):
        full_dim = tf.reshape(inp_, [n_samples, n_strokes, n_t, -1])
        full_dim = tf.transpose(a=full_dim, perm=[0, 2, 1, 3])
        full_dim = tf.reshape(full_dim, [n_samples*n_t, n_strokes, -1])
        return full_dim
      else:
        out_dict = dict()
        for key_, value_ in inp_.items():
          out_dict[key_] = handle_dict(value_)
        return out_dict

    return handle_dict(stroke_batch)

  @classmethod
  def get_model_tags(cls, config, config_loss=None):
    """Generates a string summarizing experiment parameters.

    Args:
      config:
      config_loss:

    Returns:
    """
  
    if config.predictive_model.get("name", "rnn") == "rnn":
      pred = "{}_{}x{}".format(config.encoder.cell_type,
                               config.predictive_model.cell_layers,
                               config.predictive_model.cell_units)
    elif config.predictive_model.name == "transformer":
      pred = "{}x{}-head_{}-drop_{}".format(
          config.predictive_model.layers,
          config.predictive_model.hidden_units,
          config.predictive_model.heads,
          config.predictive_model.dropout_rate)
    else:
      err_unknown_type(config.predictive_model.name)
  
    return dict(predictive=pred, model_name="PRED")