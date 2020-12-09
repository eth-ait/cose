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

//const CLOUD_STORAGE_DIR ='http://127.0.0.1:8080/models/1589564115.2-PRED_TR/';
const CLOUD_STORAGE_DIR ='https://js_models.storage.googleapis.com/1589564115.2-PRED_TR/';
const ENCODER_FILE_URL = 'js_encoder/model.json';
const DECODER_FILE_URL = 'js_decoder/model.json';

class StrokeEmbedding {
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

    // Data mean and std for normalization.
    this.data_mean_channel = tf.tensor([0.0914, 0.0485]);
    this.data_std_channel = tf.tensor([0.3025, 0.1864]);
    // Pre-processing options.
    this.pp_to_origin = true;
    this.pp_normalize = true;
    this.pp_screen_scaling = true;

    this.screen_height;
    this.screen_width;
  }

  async load() {
    this.encoder = await tf.loadGraphModel(CLOUD_STORAGE_DIR + ENCODER_FILE_URL);
    this.decoder = await tf.loadGraphModel(CLOUD_STORAGE_DIR + DECODER_FILE_URL);
  }

  dispose() {
    if (this.encoder) {
      this.encoder.dispose();
    }
    if (this.decoder) {
      this.decoder.dispose();
    }
  }
  
  /**
   * Encode a stroke.
   *
   * @param strokes array of x,y points and pen-up event.
   * @return embedding vector.
   */
  encode(strokes) {
    return tf.tidy(() => {
      let processed = this.preprocess(tf.tensor([strokes]));
      let tf_stroke = processed[0];
      let pp_options = processed[1];

      let pen_array = new Float32Array(strokes.length);
      pen_array[strokes.length-1] = 1;
      let tf_pen = tf.tensor([pen_array]).expandDims(2);

      let input_ops = {};
      input_ops[this.inode_enc_stroke] = tf.concat([tf_stroke, tf_pen], 2)  // model expects it to be (batch_size, seq_len, 3)
      input_ops[this.inode_enc_seq_len] = tf.tensor([strokes.length]).asType('int32');  // (batch_size)
      let embedding = this.encoder.execute(input_ops, this.onode_enc_emb).arraySync();
      return [embedding, pp_options];
    });
  };

  /**
   * Decode an embedding into a stroke.
   *
   * @param embedding array with shape [batch_size, latent_units].
   * @param seq_len decoded stroke length. Default is 50 steps.
   * @return stroke array of shape [seq_len, 2]
   */
  decode(embedding, seq_len=50) {
    return tf.tidy(() => {
      let input_ops = {};
      input_ops[this.inode_dec_embedding] = tf.tensor(embedding);
      input_ops[this.inode_dec_seq_len] = tf.tensor(seq_len).asType('int32');  // (batch_size)
      let decoded_stroke = this.decoder.execute(input_ops, this.onode_dec_stroke);
      decoded_stroke = this.undo_preprocess(decoded_stroke).arraySync();
      return decoded_stroke;
    });
  };

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
    this.screen_height = tf.tensor(screen_height);
    this.screen_width = tf.tensor(screen_width);
  };
}
