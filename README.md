# CoSE: Compositional Stroke Embeddings

We haven't yet refactored our code repository for the final public release. The `master` branch is used to train the model
we used in the paper for the tables and figures. The `development` branch implements some ideas improving our model. 

Note that this repository is not in its final state. Two branches will be merged in the final version. We will also include our demo code.


### Environment
Our codebase is in Python3 and using Tensorflow 2.1. We suggest creating a new virtual environment. 

- The required packages can be installed by running `pip install -r requirements.txt`    
- Update `PYTHONPATH` by running `export PYTHONPATH="${PYTHONPATH}:<CODE_PATH>`"
- You can optionally set environment variables or alternatively use FLAGS.
    - `COSE_DATA_DIR` or `--data_dir`: Path to data files. 
    - `COSE_LOG_DIR` or `--experiment_dir`: Path to experiment/model files. 
    - `COSE_EVAL_DIR` or `--eval_dir`: Path to save model evaluation results. This is required only when the evaluation scripts are called.
 

### Dataset
We use [DiDi dataset](https://github.com/google-research/google-research/tree/master/didi_dataset) 
diagram drawings without text. Please skip their preprocessing steps as we provide it. 
Move the .NDJSON file to `COSE_DATA_DIR/didi_wo_text/`.

- Run `data_scripts/didi_json_to_tfrecords.py` to create TFRecord files required for model training and evaluation.
Set `DATA_DIR` variable in this script to `COSE_DATA_DIR/didi_wo_text/`. 


- Run `python data_scripts/calculate_data_statistics.py` to create data statistics file required for model to apply data normalization.
Set `DATA_DIR` variable in this script to `COSE_DATA_DIR/didi_wo_text/`. 

Similarly, [QuickDraw dataset](https://github.com/googlecreativelab/quickdraw-dataset) can also be used. Note that our model require raw files.  

### Training
In `training_commands.json` file, we provide commands for training our main and some of the ablation models.
For example, our model can be trained by running
```
python ink_training_eager_predictive.py --experiment_id <UNIQUE_ID> --gt_targets --use_start_pos --num_pred_inputs 32 --stop_predictive_grad --pred_input_type hybrid --stroke_loss nll_gmm --n_t_samples 4 --batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model transformer --transformer_scale --transformer_pos_encoding --transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 --transformer_hidden_units 256 --transformer_dropout 0.0 --latent_units 8 --decoder_model t_emb --decoder_dropout 0.0 --decoder_layers 4 --decoder_hidden_units 512,512,512,512 --predictive_model transformer --learning_rate_type transformer --p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 --p_transformer_hidden_units 256 --p_transformer_dropout 0.0 --p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model transformer --data_name didi_wo_text --metadata_type position --disable_pen_loss --mask_encoder_pen
``` 
where you are expected to pass a unique identifier (`--experiment_id`). We recommend using timestamp (i.e., output of `date +%s`).

### Evaluation
Qualitative and quantitative evaluation can be done easily by running
```
python eval.py --model_ids <UNIQUE_ID> --qualitative --quantitative --embedding_analysis
```
where `UNIQUE_ID` is the same as above.

### Pre-trained Models
You can download [our main model](https://drive.google.com/drive/folders/1C6m7dbXaL4wn5Z4-K7ZniqoZaNTiQBdP?usp=sharing) and run evaluation script as explained above.  