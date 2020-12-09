# Interactive Demo

A stand alone implementation of our demo. You can open one of the `html` files in your browser.
It loads our model presented in the paper and runs it *locally*.  

`reconstruct.html` encodes and decodes basic shapes such as circles, rectangles, arrows, hexagons, etc. You can also see how our model decodes an embedding by using different number of points.

`predict.html` allows you to interact with the model by drawing flowchart. If the predictions are noisy, you can undo it and ask for a new one. 

Note that the training data was collected on mobile devices and we also haven't fully calibrated this demo for different resolutions, hence our model may not be robust to the inputs on a higher screen resolution or drawings via mouse.   


## How do we build it?
Tensorflow 2 allows you to export models by defining an API. You do not need to implement your model in JS from scratch. 
I admit that it is not very straightforward. You still need to do some manual work. 

### Part I: Python/Tensorflow 
In `smartink/models/ink/predictive_models.py`, you can find how we define the interface for the service model: 
`serving_predict_embedding` and `serving_predict_position` methods. They implements the basic
functionality required in our demo.  

At the end of `smartink/source/training_eager_full_model.py` file, we export our model once the training is over.

You then need to do the following:
```
cd-to-model directory
tensorflowjs_converter ./saved_model_with_signatures ./js_encoder --input_format=tf_saved_model  --saved_model_tags=serve --signature_name encode_stroke
tensorflowjs_converter ./saved_model_with_signatures ./js_decoder --input_format=tf_saved_model  --saved_model_tags=serve --signature_name decode_stroke
tensorflowjs_converter ./saved_model_with_signatures ./js_embedding_predictor --input_format=tf_saved_model  --saved_model_tags=serve --signature_name predict_embedding
tensorflowjs_converter ./saved_model_with_signatures ./js_position_predictor --input_format=tf_saved_model  --saved_model_tags=serve --signature_name predict_position
``` 
which converts your model into a JS compatible format.

### Part II: Javascript
You still need to implement a model class in JS to load the pre-trained models and make calls to the TF models. Check out
`smartink-js/lib/stroke_embedding.js` and `smartink-js/lib/stroke_predictor.js`.

The user interface is implemented in the `reconstruct.js` and `predict.js` by modifying David Ha's [demo](https://github.com/hardmaru/sketch-rnn-flowchart).


Finally, you can store your TF-JS models on storage.googleapis without requiring a dedicated server.     