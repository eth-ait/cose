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
var latest_stroke;

var sketch = function( p ) {
  "use strict";

  var small_class_list = ['flowchart'];
  var class_list = small_class_list;

  var decoding_length = 50;  // Controlled by the slider.
  var current_t = 1;  // Counter to draw steps.
  var embedding_sample;  // embbedding of the latest stroke.
  var start_coord;   // start coordinate of the latest stroke (required for visualization).
  var decoded_stroke;  // storing output of the model.
  
  // Variables to control states.
  var embedding_ready;
  var drawing_ready;
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
  var raw_lines;
  var current_raw_line;
  var strokes;
  var line_color, predict_line_color;

  // UI
  var screen_width, screen_height, decoding_length_slider;
  var line_width = 5.0;
  var screen_scale_factor = 3.0;

  // dom
  var reset_button, model_sel, random_model_button;
  var text_title, text_decoding_length, text_encoding;

  var title_text = "Draw a diagram";

  var set_title_text = function(new_text) {
    title_text = new_text.split('_').join(' ');
    text_title.html(title_text);
    text_title.position(screen_width/2-12*title_text.length/2+10, 0);
  };

  var update_decoding_length_text = function() {
    var the_color="rgba("+Math.round(255*(decoding_length/100))+",0,"+255+",1)";
    text_decoding_length.style("color", the_color); // ff990a
    text_decoding_length.html("Decoding length: "+Math.round(decoding_length));
  };
  
  var update_encoding_text = function(text) {
    var the_color="rgba("+Math.round(255*(decoding_length/100))+",0,"+255+",1)";
    text_encoding.style("color", the_color); // ff990a
    text_encoding.html(text);
  };

  var draw_example = function(example, start_x, start_y, line_color) {
    var i;
    var x=start_x, y=start_y;
    var dx, dy;
    var pen_down, pen_up, pen_end;
    var prev_pen = [1, 0, 0];

    for(i=0;i<example.length;i++) {
      // sample the next pen's states from our probability distribution
      [dx, dy, pen_down, pen_up, pen_end] = example[i];

      if (prev_pen[2] == 1) { // end of drawing.
        break;
      }

      // only draw on the paper if the pen is touching the paper
      if (prev_pen[0] == 1) {
        p.stroke(line_color);
        p.strokeWeight(line_width);
        p.line(x, y, x+dx, y+dy); // draw line connecting prev point to current point.
      }

      // update the absolute coordinates from the offsets
      x += dx;
      y += dy;

      // update the previous pen's state to the current one we just sampled
      prev_pen = [pen_down, pen_up, pen_end];
    }

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

    // model
    /* TODO
    ModelImporter.set_init_model(model_raw_data);

    model_data = ModelImporter.get_model_data();
    model = new SketchRNN(model_data);
    model.set_pixel_factor(screen_scale_factor);
    */

    screen_width = p.windowWidth; //window.innerWidth
    screen_height = p.windowHeight; //window.innerHeight
    console.log("Resolution: " + screen_height + " x " + screen_width);
    model.set_device_settings(screen_height, screen_width);

    // dom

    reset_button = p.createButton('clear drawing');
    reset_button.position(10, screen_height-27-27);
    reset_button.mousePressed(reset_button_event); // attach button listener

    // temp
    decoding_length_slider = p.createSlider(10, 100, decoding_length);
    decoding_length_slider.position(0*screen_width/2-10*0+10, screen_height-27);
    decoding_length_slider.style('width', screen_width/1-25+'px');
    decoding_length_slider.changed(decoding_length_slider_event);

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
    text_decoding_length.position(screen_width-240, screen_height-64);
    update_decoding_length_text();

    // encoding text.
    text_encoding = p.createP();
    text_encoding.style("font-family", "Courier New");
    text_encoding.style("font-size", "16");
    text_encoding.position(screen_width/2, screen_height-64);
    update_encoding_text("");
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
    raw_lines = [];
    current_raw_line = [];
    strokes = [];
    // start drawing from somewhere in middle of the canvas
    x = screen_width/2.0;
    y = screen_height/2.0;
    start_x = x;
    start_y = y;
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
    deviceEvent();
    // record pen drawing from user:
    if (tracking.down && (tracking.x > 0) && tracking.y < (screen_height-60)) { // pen is touching the paper
      if (has_started == false) { // first time anything is written
        has_started = true;
        x = tracking.x;
        y = tracking.y;
        start_x = x;
        start_y = y;
        pen = 0;
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
      update_encoding_text("Encoding...")
    } else { // pen is above the paper
      pen = 1;
      if (just_finished_line) {
        var current_raw_line_simple = DataTool.simplify_line(current_raw_line);
        var idx, last_point, last_x, last_y;

        if (current_raw_line_simple.length > 1) {
          if (raw_lines.length === 0) {
            last_x = start_x;
            last_y = start_y;
          } else {
            idx = raw_lines.length-1;
            last_point = raw_lines[idx][raw_lines[idx].length-1];
            last_x = last_point[0];
            last_y = last_point[1];
          }
          //var stroke = DataTool.line_to_stroke(current_raw_line_simple, [last_x, last_y]);
          //strokes = strokes.concat(stroke);
          
          /*
          // clear_screen();
          console.time("Encoding time")
          let out = model.encode(current_raw_line_simple);
          let emb = out[0];
          let start_coord = out[1]["start_coord"].dataSync();
          console.timeEnd("Encoding time")
          
          console.time("Decoding time")
          let decoded = model.decode(emb, decoding_length);
          console.timeEnd("Decoding time")
          draw_abs_stroke(decoded[0], start_coord[0]+500, start_coord[1], predict_line_color)
          */
          console.time("Encoding time")
          let out = model.encode(current_raw_line_simple);
          embedding_sample = out[0];
          start_coord = out[1]["start_coord"].dataSync();
          console.timeEnd("Encoding time")

          embedding_ready = true;
          drawing_ready = false;

        } else {
          if (raw_lines.length === 0) {
            has_started = false;
          }
        }

        current_raw_line = [];
        just_finished_line = false;
      }

      if (embedding_ready) {  // Decode and draw.
        if (drawing_ready != true) {  // Decode
          console.time("Decoding time")
          decoded_stroke = model.decode(embedding_sample, decoding_length)[0];
          console.timeEnd("Decoding time")
          drawing_ready = true;
          update_encoding_text("")
        };
        
        if (current_t < decoding_length) {  // Draw
          [prev_x, prev_y] = decoded_stroke[current_t-1];
          [curr_x, curr_y] = decoded_stroke[current_t];

          p.stroke(predict_line_color);
          p.strokeWeight(line_width);
          p.line(prev_x+start_coord[0]+500, prev_y+start_coord[1], curr_x+start_coord[0]+500, curr_y+start_coord[1]); // draw line connecting prev point to current point
          current_t += 1;
        } else {  // Reset states.
          current_t = 1;
          embedding_ready = false;
          drawing_ready = false;
        }
      }
    } 
    prev_pen = pen;
  };

  var model_sel_event = function() {
    var c = model_sel.value();
    var model_mode = "gen";
    console.log("user wants to change to model "+c);
    var call_back = function(new_model) {
      model = new_model;
      model.set_pixel_factor(screen_scale_factor);
      encode_strokes(strokes);
      clear_screen();
      draw_example(strokes, start_x, start_y, line_color);
      set_title_text('draw '+model.info.name+'.');
    }
    set_title_text('loading '+c+' model...');
    ModelImporter.change_model(model, c, model_mode, call_back);
  };

  var random_model_button_event = function() {
    var item = class_list[Math.floor(Math.random()*class_list.length)];
    model_sel.value(item);
    model_sel_event();
  };

  var reset_button_event = function() {
    restart();
    clear_screen();
  };

  var decoding_length_slider_event = function() {
    decoding_length = decoding_length_slider.value();
    // clear_screen();
    draw_example(strokes, start_x, start_y, line_color);
    update_decoding_length_text();
  };

  var deviceReleased = function() {
    "use strict";
    tracking.down = false;
  }

  var devicePressed = function(x, y) {
    if (y < (screen_height-60)) {
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
