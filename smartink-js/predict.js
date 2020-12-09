// Copyright 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
/**
 * Author of the original content (https://github.com/hardmaru/sketch-rnn-flowchart): David Ha <hadavid@google.com>
 * Edited for CoSE demo by: Emre Aksan <eaksan@inf.ethz.ch>
 */

const model = new SmartInk();
var custom_p5;
var recent_stroke;
var recent_given_start_position;
var recent_predicted_embedding;
var recent_decoded_stroke;

var status_message_written = false;  // Whether the operation is informed on the screen or not.
var start_coord;   // start coordinate of the latest stroke (required for visualization).
var decoded_stroke;  // storing output of the model.

var n_double_clicks=0;

var sketch = function( p ) {
  "use strict";

  var active_stroke_idx = 1;
  var active_stroke_indices = [];
  var active_strokes = new Map();
  var active_stroke_colors = new Map();
  var strokes_to_be_encoded = new Map();  // Stroke ids that is required to be encoded first.

  var screen_margin_bottom = 120;  // Leave some space for sliders and buttons.
  var decoding_length = 50;  // # of decoded points. Controlled by the slider.
  var num_predictions = 1; // # of predicted strokes by the model. Controlled by the slider.
  var strokes_predicted_so_far = 0;
  var current_t = 1;  // Counter to draw steps.
  
  // Variables to control states.
  var model_active = false;
  var ready_new_stroke = false;  // Triggers encoding of a new stroke.
  var ready_to_predict_position = false; // Triggers prediction of the next stroke's start position.
  var ready_new_start_position = false;  // Triggers prediction of a new embedding.
  var ready_embedding_to_decode = false;  // Triggers decoding of an embedding.
  var ready_stroke_to_draw = false;  // Triggers drawing of a decoded stroke.
  var curr_x, curr_y;
  var prev_x, prev_y;

  // variables for the sketch input interface.
  var pen;
  var prev_pen;
  var x, y; // absolute coordinates on the screen of where the pen is
  var start_x, start_y;
  var has_started; // set to true after user starts writing.
  var just_finished_line;
  var epsilon = 3.0; // to ignore data from user's pen staying in one spot.

  var current_raw_line;
  var line_color, predict_line_color; 
  var text_color = "rgba(31,31,31)";

  // UI
  var screen_width, screen_height, decoding_length_slider, num_predictions_slider;
  var line_width = 5.0;


  // dom
  var reset_button, next_stroke_button, delete_stroke_button;
  var text_title, text_decoding_length, text_encoding, text_num_predictions;

  var title_text = "Start drawing a flowchart. ";

  var set_title_text = function(new_text) {
    title_text = new_text.split('_').join(' ');
    text_title.html(title_text);
    text_title.position(screen_width/2-12*title_text.length/2+10, 0);
  };

  var update_decoding_length_text = function() {
    // var the_color="rgba("+Math.round(255*(decoding_length/100))+",0,"+255+",1)";
    text_decoding_length.style("color", text_color); // ff990a
    text_decoding_length.html("Decoding length: "+Math.round(decoding_length));
  };

  var update_num_predictions_text = function() {
    // var the_color="rgba("+Math.round(255*(num_predictions/100))+",0,"+255+",1)";
    text_num_predictions.style("color", text_color); // ff990a
    text_num_predictions.html("# predictions: "+Math.round(num_predictions));
  };
  
  var write_status_message = function(text) {
    // var the_color="rgba("+Math.round(255*(decoding_length/100))+",0,"+255+",1)";
    text_encoding.style("color", text_color); // ff990a
    text_encoding.html(text);
    status_message_written = true;
  };

  var clean_status_message = function() {
    // var the_color="rgba("+Math.round(255*(decoding_length/100))+",0,"+255+",1)";
    text_encoding.style("color", text_color); // ff990a
    text_encoding.html("");
    status_message_written = false;
  };

  var draw_abs_stroke = function(abs_stroke, start_x, start_y, line_color) {
    var i;
    var curr_x, curr_y;
    var prev_x, prev_y;
    prev_x, prev_y = abs_stroke[0];  // Get the first point.
    prev_x += start_x;
    prev_y += start_y;

    for(i=1;i<abs_stroke.length;i++) {
      // sample the next pen's states from our probability distribution
      [curr_x, curr_y] = abs_stroke[i];
      
      // Translate.
      curr_x += start_x;
      curr_y += start_y;

      p.stroke(line_color);
      p.strokeWeight(line_width);
      p.line(prev_x, prev_y, curr_x, curr_y); // draw line connecting prev point to current point
      prev_x = curr_x;
      prev_y = curr_y;
    }

  };

  var init = function() {
    screen_width = p.windowWidth; //window.innerWidth
    screen_height = p.windowHeight; //window.innerHeight
    console.log("Resolution: " + screen_height + " x " + screen_width);
    model.set_device_settings(screen_height, screen_width);

    // dom
    reset_button = p.createButton('Clear Drawing');
    reset_button.position(10, screen_height-120);
    reset_button.mousePressed(reset_button_event); // attach button listener

    next_stroke_button = p.createButton('Next stroke');
    next_stroke_button.position(150, screen_height-120);
    next_stroke_button.mousePressed(next_stroke_event); // attach button listener

    delete_stroke_button = p.createButton('Delete stroke');
    delete_stroke_button.position(290, screen_height-120);
    delete_stroke_button.mousePressed(delete_stroke_event); // attach button listener

    // temp
    decoding_length_slider = p.createSlider(10, 100, decoding_length, 10);
    decoding_length_slider.position(220, screen_height-40);
    decoding_length_slider.style('width', screen_width/3-25+'px');
    decoding_length_slider.changed(decoding_length_slider_event);

    num_predictions_slider = p.createSlider(1, 5, num_predictions, 1);
    num_predictions_slider.position(220, screen_height-80);
    num_predictions_slider.style('width', screen_width/3-25+'px');
    num_predictions_slider.changed(num_predictions_slider_event);

    // title
    text_title = p.createP(title_text);
    text_title.style("font-family", "Courier New");
    text_title.style("font-size", "20");
    text_title.style("color", "#3393d1"); // ff990a
    set_title_text(title_text);

    // decoding_length text
    text_decoding_length = p.createP();
    text_decoding_length.style("font-family", "Courier New");
    text_decoding_length.style("font-size", "16");
    text_decoding_length.position(10, screen_height-52);
    update_decoding_length_text();

    // num_predictions text
    text_num_predictions = p.createP();
    text_num_predictions.style("font-family", "Courier New");
    text_num_predictions.style("font-size", "16");
    text_num_predictions.position(10, screen_height-92);
    update_num_predictions_text();

    // encoding text.
    text_encoding = p.createP();
    text_encoding.style("font-family", "Courier New");
    text_encoding.style("font-size", "16");
    text_encoding.position(screen_width/2, screen_height-64);
    clean_status_message();
  };

  var draw_existing_strokes = function() {
    let stroke_color;
    var i;
    var curr_x, curr_y;
    var prev_x, prev_y;
    
    for (let [k, stroke] of active_strokes) {
      
      prev_x = stroke[0][0];  // Get the first point.
      prev_y = stroke[0][1];  // Get the first point.
      stroke_color = active_stroke_colors.get(k);

      for(i=1;i<stroke.length;i++) {
        
        [curr_x, curr_y] = stroke[i];
  
        p.stroke(stroke_color);
        p.strokeWeight(line_width);
        p.line(prev_x, prev_y, curr_x, curr_y); // draw line connecting prev point to current point
        prev_x = curr_x;
        prev_y = curr_y;
      }
    }
  };

  var add_start_coord = function(stroke, start_coord) {
    let new_stroke = [];
    for(let i=0; i<stroke.length; i++){
      new_stroke.push([stroke[i][0]+start_coord[0], stroke[i][1]+start_coord[1]]);
    }
    return new_stroke;
  };

  var insert_stroke = function(new_stroke, is_prediction) {
    active_strokes.set(active_stroke_idx, new_stroke);
    if (is_prediction) {
      active_stroke_colors.set(active_stroke_idx, predict_line_color);
    } else {
      active_stroke_colors.set(active_stroke_idx, line_color);
      strokes_to_be_encoded.set(active_stroke_idx, active_stroke_idx);
    }
    active_stroke_indices.push(active_stroke_idx);
    active_stroke_idx += 1;
  };

  var remove_stroke = function() {
    let remove_id = active_stroke_indices.pop();
    active_strokes.delete(remove_id);
    active_stroke_colors.delete(remove_id);
    strokes_to_be_encoded.delete(remove_id);
    model.delete_embedding(remove_id);
  };
  
  var restart = function() {

    // reinitialize variables before calling p5.js setup.
    // line_color = p.color(p.random(64, 224), p.random(64, 224), p.random(64, 224));
    // predict_line_color = p.color(p.random(64, 224), p.random(64, 224), p.random(64, 224));
    line_color = p.color(135, 135, 135);
    predict_line_color = p.color(216, 81, 26);

    // make sure we enforce some minimum size of our demo
    screen_width = Math.max(window.innerWidth, 480);
    screen_height = Math.max(window.innerHeight, 320);

    // variables for the sketch input interface.
    pen = 0;
    prev_pen = 1;
    has_started = false; // set to true after user starts writing.
    just_finished_line = false;
    current_raw_line = [];
    // start drawing from somewhere in middle of the canvas
    x = screen_width/2.0;
    y = screen_height/2.0;
    start_x = x;
    start_y = y;

    model.clear();
    active_stroke_idx = 1;
    active_stroke_indices = [];
    active_strokes.clear();
    active_stroke_colors.clear();
    strokes_to_be_encoded.clear();
  };

  var clear_screen = function() {
    p.background(255, 255, 255, 255);
    p.fill(255, 255, 255, 255);
  };

  p.setup = function() {
    init();
    restart();
    p.createCanvas(screen_width, screen_height);
    p.frameRate(60);
    clear_screen();
    console.log('ready.');
  };

  // tracking mouse touchpad
  var tracking = {
    down: false,
    x: 0,
    y: 0
  };

  p.draw = function() {
    if (model_active) {
      // Encode a new stroke. Model keeps track of them. See model.content_embeddings
      if (ready_new_stroke) {
        if (status_message_written) {
          
          for (let [k, stroke_id] of strokes_to_be_encoded) {
            console.time("Encoding time");
            model.encode(active_strokes.get(stroke_id), stroke_id);
            console.timeEnd("Encoding time");
          }
          strokes_to_be_encoded.clear();

          ready_new_stroke = false;
          clean_status_message();
          model_active = true;
          ready_to_predict_position = true;
        } else {
          write_status_message("Encoding...");
        }
      }
      else if (ready_to_predict_position && model.content_embeddings.size > 0) {
        if (status_message_written) {
          console.time("Position Prediction time");
          recent_given_start_position = model.predict_position();
          console.timeEnd("Position Prediction time");
          start_coord = recent_given_start_position;

          ready_to_predict_position = false;
          ready_new_start_position = true;
          clean_status_message();
        } else {
          write_status_message("Predicting Position...")
        };
      }
      // A new start position is given to predict the next stroke and draw from here.
      else if (ready_new_start_position && model.content_embeddings.size > 0) {
        if (status_message_written) {
          console.time("Embedding Prediction time");
          recent_predicted_embedding = model.predict_embedding(recent_given_start_position, active_stroke_idx);
          console.timeEnd("Embedding Prediction time");
          start_coord = recent_given_start_position;
          strokes_predicted_so_far += 1;

          ready_new_start_position = false;
          ready_embedding_to_decode = true;
          clean_status_message();
        } else {
          write_status_message("Predicting Embedding...")
        };
      }
      // Decode the embedding predicted by the model
      else if (ready_embedding_to_decode) {
        if (status_message_written) {
          console.time("Decoding time");
          recent_decoded_stroke = model.decode(recent_predicted_embedding, decoding_length)[0];
          recent_decoded_stroke = add_start_coord(recent_decoded_stroke, start_coord);
          insert_stroke(recent_decoded_stroke, true);
          console.timeEnd("Decoding time");
          ready_embedding_to_decode = false;
          ready_stroke_to_draw = true;
          clean_status_message();
        } else {
          write_status_message("Decoding Embedding...")
        }
      }
      // Finally draw the decoded stroke.
      else if (ready_stroke_to_draw) {
        clean_status_message();
        if (current_t < decoding_length) {
          [prev_x, prev_y] = recent_decoded_stroke[current_t-1];
          [curr_x, curr_y] = recent_decoded_stroke[current_t];

          p.stroke(predict_line_color);
          p.strokeWeight(line_width);
          // p.line(prev_x+start_coord[0], prev_y+start_coord[1], curr_x+start_coord[0], curr_y+start_coord[1]); // Draw line connecting prev point to current point
          p.line(prev_x, prev_y, curr_x, curr_y); // Draw line connecting prev point to current point
          current_t += 1;
        } else {  // Reset states.
          current_t = 1;
          ready_stroke_to_draw = false;
        }
      } 
      
      else{
        if (model.content_embeddings.size > 2 && num_predictions > strokes_predicted_so_far) {
          model_active = true;
          ready_to_predict_position = true;
        } else {
          model_active = false;
        }
      }
      
    // Check mouse event to draw strokes.
    } else {
      deviceEvent();
      // record pen drawing from user:
      if (tracking.down && (tracking.x > 0) && tracking.y < (screen_height-screen_margin_bottom)) { // pen is touching the paper
        if (has_started == false) { // first time anything is written
          has_started = true;
          x = tracking.x;
          y = tracking.y;
          start_x = x;
          start_y = y;
          pen = 0;
          current_raw_line.push([x, y]);
        }
        var dx0 = tracking.x-x; // candidate for dx
        var dy0 = tracking.y-y; // candidate for dy
        
        if (dx0*dx0+dy0*dy0 > epsilon*epsilon) { // only if pen is not in same area
          var dx = dx0;
          var dy = dy0;
          pen = 0;

          if (prev_pen == 0) {
            p.stroke(line_color);
            p.strokeWeight(line_width); // nice thick line
            p.line(x, y, x+dx, y+dy); // draw line connecting prev point to current point.
          }

          // update the absolute coordinates from the offsets
          x += dx;
          y += dy;

          // update raw_lines
          current_raw_line.push([x, y]);
          just_finished_line = true;
        }
      } else { // pen is above the paper
        pen = 1;
        if (just_finished_line) {
          // model_active = true;
          
          // var current_raw_line_simple = DataTool.simplify_line(current_raw_line);
          var current_raw_line_simple = current_raw_line;
          if (current_raw_line_simple.length > 1) {
            // Encode the stroke.
            // ready_new_stroke = true;
            //recent_stroke = current_raw_line_simple;
            insert_stroke(current_raw_line_simple, false);
          } 
          has_started = false;
          current_raw_line = [];
          just_finished_line = false;
        }
      }
    }
    prev_pen = pen;
  };

  var next_stroke_event = function() {
    console.log("Next stroke")
    model_active = true;
    ready_new_stroke = true;
    strokes_predicted_so_far = 0;
  };

  var delete_stroke_event = function() {
    console.log("Delete stroke")
    remove_stroke();
    clear_screen();
    draw_existing_strokes();
  };
  
  var reset_button_event = function() {
    restart();
    clear_screen();
  };

  var decoding_length_slider_event = function() {
    decoding_length = decoding_length_slider.value();
    // clear_screen();
    //draw_example(strokes, start_x, start_y, line_color);
    update_decoding_length_text();
  };

  var num_predictions_slider_event = function() {
    num_predictions = num_predictions_slider.value();
    //draw_example(strokes, start_x, start_y, line_color);
    update_num_predictions_text();
    model_active = true;
  };

  var deviceReleased = function() {
    "use strict";
    tracking.down = false;
  }

  var devicePressed = function(x, y) {
    if (y < (screen_height-screen_margin_bottom)) {
      tracking.x = x;
      tracking.y = y;
      if (!tracking.down) {
        tracking.down = true;
      }
    }
  };

  var deviceEvent = function() {
    if (p.mouseIsPressed) {
      devicePressed(p.mouseX, p.mouseY);
    } else {
      deviceReleased();
    }
  }

};

console.time('Loading of model');
model.load().then(() => {
  console.timeEnd('Loading of model');
  custom_p5 = new p5(sketch, 'sketch');
});
