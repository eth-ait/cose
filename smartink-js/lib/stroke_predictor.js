/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

//const CLOUD_STORAGE_DIR ='http://127.0.0.1:8080/models/1590706693.5-PRED_TR/';
const CLOUD_STORAGE_DIR ='https://js_models.storage.googleapis.com/1590706693.5-PRED_TR/';
const ENCODER_FILE_URL = 'js_encoder/model.json';
const DECODER_FILE_URL = 'js_decoder/model.json';
const EMBEDDING_FILE_URL = 'js_embedding_predictor/model.json';
const POSITION_FILE_URL = 'js_position_predictor/model.json';

class SmartInk {
  constructor() {
    // Input & output graph node names for encoder.
    this.inode_enc_stroke = "input_stroke";
    this.inode_enc_seq_len = "input_seq_len";
    this.onode_enc_emb = "Identity";

    // Input & output graph node names for decoder.
    this.inode_dec_embedding = "embedding_sample";
    this.inode_dec_seq_len = "target_seq_len";
    this.onode_dec_seq_len = "Identity_1";
    this.onode_dec_stroke = "Identity_2";
    this.onode_dec_pen = "Identity";

    // Input & output graph node names for embedding predictor.
    this.inode_emb_tar_pos = "target_pos";
    this.inode_emb_inp_pos = "inp_pos";
    this.inode_emb_inp_emb = "inp_embeddings";
    this.onode_emb_sample = "Identity";

    // Input & output graph node names for position predictor.
    this.inode_pos_inp_pos = "inp_pos";
    this.inode_pos_inp_emb = "inp_embeddings";
    this.onode_pos_sample = "Identity_2";

    // Data mean and std for normalization.
    this.data_mean_channel = tf.tensor([0.0914, 0.0485]);
    this.data_std_channel = tf.tensor([0.3025, 0.1864]);
    
    // Pre-processing options.
    this.pp_to_origin = true;
    this.pp_normalize = true;
    this.pp_screen_scaling = true;
    this.screen_height;
    this.screen_width;
    this.scale_factor = 1.25;

    // State.
    this.content_embeddings = new Map();  // List of embeddings of the strokes on the screen.
    this.content_starts = new Map();  // List of start positions of the strokes on the screen.
  }

  async load() {
    this.encoder = await tf.loadGraphModel(CLOUD_STORAGE_DIR + ENCODER_FILE_URL);
    this.decoder = await tf.loadGraphModel(CLOUD_STORAGE_DIR + DECODER_FILE_URL);
    this.embedding_model = await tf.loadGraphModel(CLOUD_STORAGE_DIR + EMBEDDING_FILE_URL);
    this.position_model = await tf.loadGraphModel(CLOUD_STORAGE_DIR + POSITION_FILE_URL);
  }

  dispose() {
    if (this.encoder) {
      this.encoder.dispose();
    }
    if (this.decoder) {
      this.decoder.dispose();
    }
    if (this.embedding_model) {
      this.embedding_model.dispose();
    }
    if (this.position_model) {
      this.position_model.dispose();
    }
  }
  
  clear() {
    this.content_embeddings.clear();
    this.content_starts.clear();
  };

  delete_embedding(stroke_id) {
    this.content_embeddings.delete(stroke_id);
    this.content_starts.delete(stroke_id);
  };

  /**
   * Encode a stroke.
   *
   * @param strokes array of x,y points and pen-up event.
   * @return embedding vector.
   */
  encode(strokes, stroke_id) {    
    let out = tf.tidy(() => {
      let processed = this.preprocess(tf.tensor([strokes]));
      let tf_stroke = processed[0];
      let pp_options = processed[1];

      let pen_array = new Float32Array(strokes.length);
      pen_array[strokes.length-1] = 1;
      let tf_pen = tf.tensor([pen_array]).expandDims(2);

      let input_ops = {};
      input_ops[this.inode_enc_stroke] = tf.concat([tf_stroke, tf_pen], 2)  // model expects it to be (batch_size, seq_len, 3)
      input_ops[this.inode_enc_seq_len] = tf.tensor([strokes.length]).asType('int32');  // (batch_size)
      let embedding = this.encoder.execute(input_ops, this.onode_enc_emb);
      return [embedding, pp_options];
    });
    this.content_embeddings.set(stroke_id, out[0].squeeze());  // embedding vector
    this.content_starts.set(stroke_id, this.scale_coordinates(out[1]["start_coord"].squeeze()));
    return [out[0], out[1]];
  };

  /**
   * Decode an embedding into a stroke.
   *
   * @param embedding tf tensor with shape (batch_size, latent_units).
   * @param seq_len decoded stroke length. Default is 50 steps.
   * @return stroke array of shape [seq_len, 2]
   */
  decode(embedding, seq_len=50) {
    return tf.tidy(() => {
      let input_ops = {};
      input_ops[this.inode_dec_embedding] = embedding;  //tf.tensor(embedding);
      input_ops[this.inode_dec_seq_len] = tf.tensor(seq_len).asType('int32');  // (batch_size)
      let decoded_stroke = this.decoder.execute(input_ops, this.onode_dec_stroke);
      decoded_stroke = this.undo_preprocess(decoded_stroke).arraySync();
      return decoded_stroke;
    });
  };

  /**
   * Predict embedding given the existing embeddings.
   *
   * @param target_pos conditional start position [x,y].
   * @return an embedding vector of shape [1, n_latent_units]
   */
  predict_embedding(target_pos, stroke_id) {
    let out = tf.tidy(() => {
      let input_ops = {};
      input_ops[this.inode_emb_inp_emb] = tf.stack(Array.from(this.content_embeddings.values())).expandDims(0);
      input_ops[this.inode_emb_inp_pos] = tf.stack(Array.from(this.content_starts.values())).expandDims(0);
      input_ops[this.inode_emb_tar_pos] = this.scale_coordinates(tf.tensor3d(target_pos, [1,1,2]));
      let predicted_emb = this.embedding_model.execute(input_ops, this.onode_emb_sample);
      return predicted_emb;
    });
    this.content_embeddings.set(stroke_id, out.squeeze());  // save the embedding vector.
    this.content_starts.set(stroke_id, this.scale_coordinates(tf.tensor(target_pos)));
    return out;
  };

  /**
   * Predict the next stroke's starting position.
   *
   * @return coordinates of the next stroke of shape [1, 2].
   */
  predict_position() {
    let out = tf.tidy(() => {
      let input_ops = {};
      input_ops[this.inode_pos_inp_emb] = tf.stack(Array.from(this.content_embeddings.values())).expandDims(0);
      input_ops[this.inode_pos_inp_pos] = tf.stack(Array.from(this.content_starts.values())).expandDims(0);
      let predicted_pos = this.position_model.execute(input_ops, this.onode_pos_sample);
      return predicted_pos.squeeze();
    });
    return this.rescale_coordinates(out).arraySync();
  };

  scale_coordinates(coord) {
    return tf.tidy(() => {
      return tf.div(coord, this.screen_height);
    });
  }

  rescale_coordinates(coord) {
    return tf.tidy(() => {
      return tf.mul(coord, this.screen_height);
    });
  }

  /**
   * Apply pre-processing such as offset removal (i.e., translating to the origin) or zero-mean unit-variance normalization
   *
   * @param stroke_tensor tf.tensor of shape (batch_size, seq_len, 2)
   * @return stroke_tensor after pre-processing and pre-processing side-effects such as start coordinates.
   */
  preprocess(stroke_tensor) {
    return tf.tidy(() => {
      let pp_track = {}
      let processed_stroke = stroke_tensor

      stroke_tensor = tf.mul(stroke_tensor, this.scale_factor);
      if (this.pp_to_origin) {  // Remove initial point offset.
        let start_coord = processed_stroke.slice([0, 0], [1, 1, 2]);
        pp_track["start_coord"] = start_coord;
        processed_stroke = tf.sub(processed_stroke, start_coord);
      };
      if (this.pp_screen_scaling) {  // Scale so that y-coordinate is bounded by 1.
        processed_stroke = tf.div(processed_stroke, this.screen_height);  
      };
      if (this.pp_normalize) {  // Zero-mean unit-variance normalization.
        processed_stroke = this.normalize(processed_stroke);
      };
      return [processed_stroke, pp_track];
    });
  };

  /**
   * Undo pre-processing and normalization.
   *
   * @param stroke_tensor tf.tensor of shape (batch_size, seq_len, 2)
   * @return stroke_tensor after reverting pre-processing.
   */
  undo_preprocess(stroke_tensor) {
    return tf.tidy(() => {
      let pp_track = {}
      let processed_stroke = stroke_tensor
      if (this.pp_normalize) {  // Zero-mean unit-variance normalization.
        processed_stroke = this.undo_normalization(processed_stroke);
      };
      if (this.pp_screen_scaling) {  // Scale so that y-coordinate is bounded by 1.
        processed_stroke = tf.mul(processed_stroke, this.screen_height);  
      };

      processed_stroke = tf.div(processed_stroke, this.scale_factor);
      return processed_stroke;
    });
  };
  
  normalize(stroke_tensor) {
    return tf.tidy(() => {
      return stroke_tensor.sub(this.data_mean_channel).div(this.data_std_channel);
    });
  };

  undo_normalization(stroke_tensor) {
    return tf.tidy(() => {
      return stroke_tensor.mul(this.data_std_channel).add(this.data_mean_channel);
    });
  };

  set_device_settings(screen_height, screen_width){
    this.screen_height = tf.tensor(screen_height/this.scale_factor);
    this.screen_width = tf.tensor(screen_width/this.scale_factor);
  };
}
