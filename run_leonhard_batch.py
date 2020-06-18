from subprocess import call
import time

"""
The working directory must be the same with python file's directory.
"""

NUM_CPU = 2
MEMORY = 6000
NUM_GPU = 1
WALL_TIME = 23
# cluster_command_format = 'bsub -G ls_hilli -n {} -W {}:00 -o log_{} -R "rusage[mem={}, ngpus_excl_p={}]" '
# cluster_command_format = 'bsub -G ls_hilli -n {} -W {}:00 -o log_{} -R "rusage[mem={}, ngpus_excl_p={}]" -R "select[gpu_mtotal0>=10240]" '  # Ensures gtx1080Ti or rtx2080Ti
cluster_command_format = 'bsub -G ls_hilli -n {} -W {}:00 -o log_{} -R "rusage[mem={}, ngpus_excl_p={}]" -R "select[gpu_model0==GeForceRTX2080Ti]" ' # Ensures rtx2080Ti


experiment_list_embedding = [
    ]

experiment_list_predictive_transformer = [
    # 'python ink_training_eager_embedding.py '
    # '--comment "tf2-af03-t_s4-t_emb_c_freq2-dec_1e6_emb1e3" --decoder_model t_emb '
    # '--latent_units 8 --grad_clip_norm 1 --batch_size 400 --affine_prob 0.3 '
    # '--encoder_model transformer --transformer_scale --transformer_pos_encoding '
    # '--transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 '
    # '--transformer_hidden_units 256 --transformer_dropout 0.0 '
    # '--resampling_factor 0 --scale_factor 0 --n_t_samples 4 --t_frequency_channels 2 '
    # '--learning_rate_type transformer --stroke_loss nll_gmm '
    # '--reg_emb_weight 1e-3 --reg_dec_weight 1e-6 '
    # '--decoder_layers 4 --decoder_hidden_units 512,512,512,512 '
    # '--data_name didi_wo_text --metadata_type position ',
    #
    # 'python ink_training_eager_embedding.py '
    # '--comment "tf2-af03-t_s4-no_gclip-t_emb_c_freq2-dec_1e6_emb1e3" --decoder_model t_emb '
    # '--latent_units 8 --grad_clip_value 0 --batch_size 400 --affine_prob 0.3 '
    # '--encoder_model transformer --transformer_scale --transformer_pos_encoding '
    # '--transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 '
    # '--transformer_hidden_units 256 --transformer_dropout 0.0 '
    # '--resampling_factor 0 --scale_factor 0 --n_t_samples 4 --t_frequency_channels 2 '
    # '--learning_rate_type transformer --stroke_loss nll_gmm '
    # '--reg_emb_weight 1e-3 --reg_dec_weight 1e-6 '
    # '--decoder_layers 4 --decoder_hidden_units 512,512,512,512 '
    # '--data_name didi_wo_text --metadata_type position ',
    #
    # 'python ink_training_eager_embedding.py '
    # '--comment "tf2-af03-t_s4-no_gclip-t_emb_c_freq4" --decoder_model t_emb '
    # '--latent_units 8 --grad_clip_value 0 --batch_size 400 --affine_prob 0.3 '
    # '--encoder_model transformer --transformer_scale --transformer_pos_encoding '
    # '--transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 '
    # '--transformer_hidden_units 256 --transformer_dropout 0.0 '
    # '--resampling_factor 0 --scale_factor 0 --n_t_samples 4 --t_frequency_channels 4 '
    # '--learning_rate_type transformer --stroke_loss nll_gmm '
    # '--decoder_layers 4 --decoder_hidden_units 512,512,512,512 '
    # '--data_name didi_wo_text --metadata_type position ',
    #
    # 'python ink_training_eager_embedding.py '
    # '--comment "tf2-af03-t_s4-no_gclip-t_emb_c_freq4-dec_1e6_emb1e3" --decoder_model t_emb '
    # '--latent_units 8 --grad_clip_value 0 --batch_size 400 --affine_prob 0.3 '
    # '--encoder_model transformer --transformer_scale --transformer_pos_encoding '
    # '--transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 '
    # '--transformer_hidden_units 256 --transformer_dropout 0.0 '
    # '--resampling_factor 0 --scale_factor 0 --n_t_samples 4 --t_frequency_channels 4 '
    # '--learning_rate_type transformer --stroke_loss nll_gmm '
    # '--reg_emb_weight 1e-3 --reg_dec_weight 1e-6 '
    # '--decoder_layers 4 --decoder_hidden_units 512,512,512,512 '
    # '--data_name didi_wo_text --metadata_type position ',
    
    ##### Predictive model.
    
    'python ink_training_eager_predictive.py '
    '--comment "tf2-af03-t_s4-hyb8-emb_pos_gmm10-gnorm1-stopg-tres2" '
    '--use_start_pos --pred_input_type hybrid --stop_predictive_grad '
    '--stroke_loss nll_gmm --n_t_samples 4 --batch_size 128 --affine_prob 0.3 '
    '--resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 '
    '--encoder_model transformer --transformer_scale --transformer_pos_encoding '
    '--transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 '
    '--transformer_hidden_units 256 --transformer_dropout 0.0 '
    '--latent_units 8 '
    '--decoder_model t_emb --decoder_dropout 0.0 '
    '--decoder_layers 4 --decoder_hidden_units 512,512,512,512 '
    '--predictive_model transformer --learning_rate_type transformer '
    '--p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 '
    '--p_transformer_hidden_units 256 --p_transformer_dropout 0.0 '
    '--p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 '
    '--loss_predicted_embedding --loss_reconstructed_ink '
    '--position_model transformer '
    '--data_name didi_wo_text --metadata_type position '
    '--disable_pen_loss --mask_encoder_pen ',

    'python ink_training_eager_predictive.py '
    '--comment "tf2-af03-t_s4-hyb8-emb_pos_gmm10-gnorm1-stopg-d1e6_e1e3-tres2" '
    '--use_start_pos --pred_input_type hybrid --stop_predictive_grad '
    '--stroke_loss nll_gmm --n_t_samples 4 --batch_size 128 --affine_prob 0.3 '
    '--resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 '
    '--encoder_model transformer --transformer_scale --transformer_pos_encoding '
    '--transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 '
    '--transformer_hidden_units 256 --transformer_dropout 0.0 '
    '--reg_emb_weight 1e-3 --reg_dec_weight 1e-6 '
    '--latent_units 8 '
    '--decoder_model t_emb --decoder_dropout 0.0 '
    '--decoder_layers 4 --decoder_hidden_units 512,512,512,512 '
    '--predictive_model transformer --learning_rate_type transformer '
    '--p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 '
    '--p_transformer_hidden_units 256 --p_transformer_dropout 0.0 '
    '--p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 '
    '--loss_predicted_embedding --loss_reconstructed_ink '
    '--position_model transformer '
    '--data_name didi_wo_text --metadata_type position '
    '--disable_pen_loss --mask_encoder_pen ',

    'python ink_training_eager_predictive.py '
    '--comment "tf2-af03-t_s4-hyb8-emb_pos_gmm10-gnorm1-d1e6_e1e3-tres2" '
    '--use_start_pos --pred_input_type hybrid '
    '--stroke_loss nll_gmm --n_t_samples 4 --batch_size 128 --affine_prob 0.3 '
    '--resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 '
    '--encoder_model transformer --transformer_scale --transformer_pos_encoding '
    '--transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 '
    '--transformer_hidden_units 256 --transformer_dropout 0.0 '
    '--reg_emb_weight 1e-3 --reg_dec_weight 1e-6 '
    '--latent_units 8 '
    '--decoder_model t_emb --decoder_dropout 0.0 '
    '--decoder_layers 4 --decoder_hidden_units 512,512,512,512 '
    '--predictive_model transformer --learning_rate_type transformer '
    '--p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 '
    '--p_transformer_hidden_units 256 --p_transformer_dropout 0.0 '
    '--p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 '
    '--loss_predicted_embedding --loss_reconstructed_ink '
    '--position_model transformer '
    '--data_name didi_wo_text --metadata_type position '
    '--disable_pen_loss --mask_encoder_pen ',
    
    # #### Predictive model didi_all
    # 'python ink_training_eager_predictive.py '
    # '--comment "tf2-af03-t_s4-hyb8-emb_pos_gmm10-gnorm1-stopg" '
    # '--use_start_pos --pred_input_type hybrid --stop_predictive_grad '
    # '--stroke_loss nll_gmm --n_t_samples 4 --batch_size 32 --affine_prob 0.3 '
    # '--resampling_factor 0 --scale_factor 0 --grad_clip_norm 1 '
    # '--encoder_model transformer --transformer_scale --transformer_pos_encoding '
    # '--transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 '
    # '--transformer_hidden_units 256 --transformer_dropout 0.0 '
    # '--latent_units 8 '
    # '--decoder_model t_emb --decoder_dropout 0.0 '
    # '--decoder_layers 4 --decoder_hidden_units 512,512,512,512 '
    # '--predictive_model transformer --learning_rate_type transformer '
    # '--p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 '
    # '--p_transformer_hidden_units 256 --p_transformer_dropout 0.0 '
    # '--p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 '
    # '--loss_predicted_embedding --loss_reconstructed_ink '
    # '--position_model transformer '
    # '--data_name didi_wo_text --metadata_type position '
    # '--disable_pen_loss --mask_encoder_pen ',
    #
    # 'python ink_training_eager_predictive.py '
    # '--comment "tf2-af03-t_s4-hyb8-emb_pos_gmm10-gnorm1-stopg-d1e6_e1e3" '
    # '--use_start_pos --pred_input_type hybrid --stop_predictive_grad '
    # '--stroke_loss nll_gmm --n_t_samples 4 --batch_size 64 --affine_prob 0.3 '
    # '--resampling_factor 0 --scale_factor 0 --grad_clip_norm 1 '
    # '--encoder_model transformer --transformer_scale --transformer_pos_encoding '
    # '--transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 '
    # '--transformer_hidden_units 256 --transformer_dropout 0.0 '
    # '--reg_emb_weight 1e-3 --reg_dec_weight 1e-6 '
    # '--latent_units 8 '
    # '--decoder_model t_emb --decoder_dropout 0.0 '
    # '--decoder_layers 4 --decoder_hidden_units 512,512,512,512 '
    # '--predictive_model transformer --learning_rate_type transformer '
    # '--p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 '
    # '--p_transformer_hidden_units 256 --p_transformer_dropout 0.0 '
    # '--p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 '
    # '--loss_predicted_embedding --loss_reconstructed_ink '
    # '--position_model transformer '
    # '--data_name didi_wo_text --metadata_type position '
    # '--disable_pen_loss --mask_encoder_pen ',
    #
    # 'python ink_training_eager_predictive.py '
    # '--comment "tf2-af03-t_s4-hyb8-emb_pos_gmm10-gnorm1-d1e6_e1e3" '
    # '--use_start_pos --pred_input_type hybrid '
    # '--stroke_loss nll_gmm --n_t_samples 4 --batch_size 64 --affine_prob 0.3 '
    # '--resampling_factor 0 --scale_factor 0 --grad_clip_norm 1 '
    # '--encoder_model transformer --transformer_scale --transformer_pos_encoding '
    # '--transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 '
    # '--transformer_hidden_units 256 --transformer_dropout 0.0 '
    # '--reg_emb_weight 1e-3 --reg_dec_weight 1e-6 '
    # '--latent_units 8 '
    # '--decoder_model t_emb --decoder_dropout 0.0 '
    # '--decoder_layers 4 --decoder_hidden_units 512,512,512,512 '
    # '--predictive_model transformer --learning_rate_type transformer '
    # '--p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 '
    # '--p_transformer_hidden_units 256 --p_transformer_dropout 0.0 '
    # '--p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 '
    # '--loss_predicted_embedding --loss_reconstructed_ink '
    # '--position_model transformer '
    # '--data_name didi_wo_text --metadata_type position '
    # '--disable_pen_loss --mask_encoder_pen ',
    #
    # 'python ink_training_eager_predictive.py '
    # '--comment "tf2-af03-t_s4-hyb8-emb_pos_gmm10-d1e6_e1e3" '
    # '--use_start_pos --pred_input_type hybrid '
    # '--stroke_loss nll_gmm --n_t_samples 4 --batch_size 64 --affine_prob 0.3 '
    # '--resampling_factor 0 --scale_factor 0 --grad_clip_norm 0 '
    # '--encoder_model transformer --transformer_scale --transformer_pos_encoding '
    # '--transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 '
    # '--transformer_hidden_units 256 --transformer_dropout 0.0 '
    # '--reg_emb_weight 1e-3 --reg_dec_weight 1e-6 '
    # '--latent_units 8 '
    # '--decoder_model t_emb --decoder_dropout 0.0 '
    # '--decoder_layers 4 --decoder_hidden_units 512,512,512,512 '
    # '--predictive_model transformer --learning_rate_type transformer '
    # '--p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 '
    # '--p_transformer_hidden_units 256 --p_transformer_dropout 0.0 '
    # '--p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 '
    # '--loss_predicted_embedding --loss_reconstructed_ink '
    # '--position_model transformer '
    # '--data_name didi_wo_text --metadata_type position '
    # '--disable_pen_loss --mask_encoder_pen ',
    
    #### Predictive model didi_all
    'python ink_training_eager_predictive.py '
    '--comment "tf2-af03-t_s4-hyb8-emb_pos_gmm10-gnorm1-stopg" '
    '--use_start_pos --pred_input_type hybrid --stop_predictive_grad '
    '--stroke_loss nll_gmm --n_t_samples 4 --batch_size 32 --affine_prob 0.3 '
    '--resampling_factor 0 --scale_factor 0 --grad_clip_norm 1 '
    '--encoder_model transformer --transformer_scale --transformer_pos_encoding '
    '--transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 '
    '--transformer_hidden_units 256 --transformer_dropout 0.0 '
    '--latent_units 8 '
    '--decoder_model t_emb --decoder_dropout 0.0 '
    '--decoder_layers 4 --decoder_hidden_units 512,512,512,512 '
    '--predictive_model transformer --learning_rate_type transformer '
    '--p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 '
    '--p_transformer_hidden_units 256 --p_transformer_dropout 0.0 '
    '--p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 '
    '--loss_predicted_embedding --loss_reconstructed_ink '
    '--position_model transformer '
    '--data_name didi_all --metadata_type position '
    '--disable_pen_loss --mask_encoder_pen ',
    
    'python ink_training_eager_predictive.py '
    '--comment "tf2-af03-t_s4-hyb8-emb_pos_gmm10-gnorm1-stopg-d1e6_e1e3" '
    '--use_start_pos --pred_input_type hybrid --stop_predictive_grad '
    '--stroke_loss nll_gmm --n_t_samples 4 --batch_size 32 --affine_prob 0.3 '
    '--resampling_factor 0 --scale_factor 0 --grad_clip_norm 1 '
    '--encoder_model transformer --transformer_scale --transformer_pos_encoding '
    '--transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 '
    '--transformer_hidden_units 256 --transformer_dropout 0.0 '
    '--reg_emb_weight 1e-3 --reg_dec_weight 1e-6 '
    '--latent_units 8 '
    '--decoder_model t_emb --decoder_dropout 0.0 '
    '--decoder_layers 4 --decoder_hidden_units 512,512,512,512 '
    '--predictive_model transformer --learning_rate_type transformer '
    '--p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 '
    '--p_transformer_hidden_units 256 --p_transformer_dropout 0.0 '
    '--p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 '
    '--loss_predicted_embedding --loss_reconstructed_ink '
    '--position_model transformer '
    '--data_name didi_all --metadata_type position '
    '--disable_pen_loss --mask_encoder_pen ',
    
    'python ink_training_eager_predictive.py '
    '--comment "tf2-af03-t_s4-ord8-emb_pos_gmm10-gnorm1-stopg" '
    '--use_start_pos --pred_input_type ordered --stop_predictive_grad '
    '--stroke_loss nll_gmm --n_t_samples 4 --batch_size 32 --affine_prob 0.3 '
    '--resampling_factor 0 --scale_factor 0 --grad_clip_norm 1 '
    '--encoder_model transformer --transformer_scale --transformer_pos_encoding '
    '--transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 '
    '--transformer_hidden_units 256 --transformer_dropout 0.0 '
    '--latent_units 8 '
    '--decoder_model t_emb --decoder_dropout 0.0 '
    '--decoder_layers 4 --decoder_hidden_units 512,512,512,512 '
    '--predictive_model transformer --learning_rate_type transformer '
    '--p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 '
    '--p_transformer_hidden_units 256 --p_transformer_dropout 0.0 '
    '--p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 '
    '--loss_predicted_embedding --loss_reconstructed_ink '
    '--position_model transformer '
    '--data_name didi_all --metadata_type position '
    '--disable_pen_loss --mask_encoder_pen ',
    
    'python ink_training_eager_predictive.py '
    '--comment "tf2-af03-t_s4-ord8-emb_pos_gmm10-gnorm1-stopg-d1e6_e1e3" '
    '--use_start_pos --pred_input_type ordered --stop_predictive_grad '
    '--stroke_loss nll_gmm --n_t_samples 4 --batch_size 32 --affine_prob 0.3 '
    '--resampling_factor 0 --scale_factor 0 --grad_clip_norm 1 '
    '--encoder_model transformer --transformer_scale --transformer_pos_encoding '
    '--transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 '
    '--transformer_hidden_units 256 --transformer_dropout 0.0 '
    '--reg_emb_weight 1e-3 --reg_dec_weight 1e-6 '
    '--latent_units 8 '
    '--decoder_model t_emb --decoder_dropout 0.0 '
    '--decoder_layers 4 --decoder_hidden_units 512,512,512,512 '
    '--predictive_model transformer --learning_rate_type transformer '
    '--p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 '
    '--p_transformer_hidden_units 256 --p_transformer_dropout 0.0 '
    '--p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 '
    '--loss_predicted_embedding --loss_reconstructed_ink '
    '--position_model transformer '
    '--data_name didi_all --metadata_type position '
    '--disable_pen_loss --mask_encoder_pen ',
    ]

experiment_list = experiment_list_predictive_transformer
data = ''

experiment_timestamp = str(int(time.time()))
start_id = 1
for work_id, experiment in enumerate(experiment_list):
    experiment_id = "{}.{}".format(experiment_timestamp, start_id+work_id)
    time.sleep(1)
    print(experiment_id)
    experiment_command = experiment + data + ' --experiment_id ' + experiment_id

    cluster_command = cluster_command_format.format(NUM_CPU,
                                                    WALL_TIME,
                                                    experiment_id,
                                                    MEMORY,
                                                    NUM_GPU)
    call([cluster_command + experiment_command], shell=True)
    
"""
python ink_training_eager_predictive.py --experiment_id 1590363000.1 --latent_units 4 --comment "tf2-af03-t_s4-hyb32-emb_pos_gmm10-tres2-stopg" --stop_predictive_grad --use_start_pos --num_pred_inputs 32 --pred_input_type hybrid --stroke_loss nll_gmm --n_t_samples 4 --batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model transformer --transformer_scale --transformer_pos_encoding --transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 --transformer_hidden_units 256 --transformer_dropout 0.0 --decoder_model t_emb --decoder_dropout 0.0 --decoder_layers 4 --decoder_hidden_units 512,512,512,512 --predictive_model transformer --learning_rate_type transformer --p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 --p_transformer_hidden_units 256 --p_transformer_dropout 0.0 --p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model transformer --data_name didi_wo_text --metadata_type position --disable_pen_loss --mask_encoder_pen
python ink_training_eager_predictive.py --experiment_id 1590363000.2 --latent_units 16 --comment "tf2-af03-t_s4-hyb32-emb_pos_gmm10-tres2-stopg" --stop_predictive_grad --use_start_pos --num_pred_inputs 32 --pred_input_type hybrid --stroke_loss nll_gmm --n_t_samples 4 --batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model transformer --transformer_scale --transformer_pos_encoding --transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 --transformer_hidden_units 256 --transformer_dropout 0.0 --decoder_model t_emb --decoder_dropout 0.0 --decoder_layers 4 --decoder_hidden_units 512,512,512,512 --predictive_model transformer --learning_rate_type transformer --p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 --p_transformer_hidden_units 256 --p_transformer_dropout 0.0 --p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model transformer --data_name didi_wo_text --metadata_type position --disable_pen_loss --mask_encoder_pen
python ink_training_eager_predictive.py --experiment_id 1590363000.3 --latent_units 32 --comment "tf2-af03-t_s4-hyb32-emb_pos_gmm10-tres2-stopg" --stop_predictive_grad --use_start_pos --num_pred_inputs 32 --pred_input_type hybrid --stroke_loss nll_gmm --n_t_samples 4 --batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model transformer --transformer_scale --transformer_pos_encoding --transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 --transformer_hidden_units 256 --transformer_dropout 0.0 --decoder_model t_emb --decoder_dropout 0.0 --decoder_layers 4 --decoder_hidden_units 512,512,512,512 --predictive_model transformer --learning_rate_type transformer --p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 --p_transformer_hidden_units 256 --p_transformer_dropout 0.0 --p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model transformer --data_name didi_wo_text --metadata_type position --disable_pen_loss --mask_encoder_pen

python ink_training_eager_predictive.py --experiment_id 1590363000.4 --comment "tf2-af03-t_s4-hyb32-emb_pos_gmm10-tres2-stopg-seq" --use_start_pos --num_pred_inputs 32 --stop_predictive_grad --pred_input_type hybrid --stroke_loss nll_gmm --n_t_samples 4 --batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model transformer --transformer_scale --transformer_pos_encoding --transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 --transformer_hidden_units 256 --transformer_dropout 0.0 --latent_units 8 --decoder_model t_emb --decoder_dropout 0.0 --decoder_layers 4 --decoder_hidden_units 512,512,512,512 --predictive_model transformer --learning_rate_type transformer --p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 --p_transformer_hidden_units 256 --p_transformer_dropout 0.0 --p_transformer_scale --p_transformer_pos_encoding --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model transformer --data_name didi_wo_text --metadata_type position --disable_pen_loss --mask_encoder_pen
python ink_training_eager_predictive.py --experiment_id 1590486000.2 --comment "tf2-af03-t_s4-hyb32-emb_pos_gmm10-tres2-stopg" --use_start_pos --num_pred_inputs 32 --stop_predictive_grad --pred_input_type hybrid --stroke_loss nll_gmm --n_t_samples 1 --batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model rnn --latent_units 8 --decoder_model rnn --decoder_dropout 0.0 --predictive_model transformer --learning_rate_type transformer --p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 --p_transformer_hidden_units 256 --p_transformer_dropout 0.0 --p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model transformer --data_name didi_wo_text --metadata_type position


python ink_training_eager_embedding.py --experiment_id 1590486000.1 --comment "tf2-af03-gnorm1-tres2" --reg_dec_weight 0 --reg_emb_weight 0 --decoder_model rnn --latent_units 8 --grad_clip_norm 1 --batch_size 1000 --affine_prob 0.3 --encoder_model rnn --bidirectional_encoder --resampling_factor 2 --scale_factor 0 --n_t_samples 0 --stroke_loss nll_gmm --learning_rate_type transformer --data_name didi_wo_text --metadata_type position
python ink_training_eager_predictive.py --experiment_id 1590486000.2 --comment "tf2-af03-t_s4-hyb32-emb_pos_gmm10-tres2-stopg" --use_start_pos --num_pred_inputs 32 --stop_predictive_grad --pred_input_type hybrid --stroke_loss nll_gmm --n_t_samples 1 --batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 8 --decoder_model rnn --decoder_dropout 0.0 --predictive_model rnn --learning_rate_type transformer --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model rnn --data_name didi_wo_text --metadata_type position
python ink_training_eager_predictive.py --experiment_id 1590486000.3 --comment "tf2-af03-t_s4-hyb32-emb_pos_gmm10-tres2-stopg" --use_start_pos --num_pred_inputs 32 --stop_predictive_grad --pred_input_type hybrid --stroke_loss nll_gmm --n_t_samples 1 --batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 8 --decoder_model rnn --decoder_dropout 0.0 --predictive_model transformer --learning_rate_type transformer --p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 --p_transformer_hidden_units 256 --p_transformer_dropout 0.0 --p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model transformer --data_name didi_wo_text --metadata_type position
python ink_training_eager_predictive.py --experiment_id 1590486000.4 --comment "tf2-af03-t_s4-hyb32-emb_pos_gmm10-tres2-stopg" --use_start_pos --num_pred_inputs 32 --stop_predictive_grad --pred_input_type hybrid --stroke_loss nll_gmm --n_t_samples 4 --batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model transformer --transformer_scale --transformer_pos_encoding --transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 --transformer_hidden_units 256 --transformer_dropout 0.0 --latent_units 8 --decoder_model t_emb --decoder_dropout 0.0 --decoder_layers 4 --decoder_hidden_units 512,512,512,512 --predictive_model rnn --learning_rate_type transformer --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model rnn --data_name didi_wo_text --metadata_type position --disable_pen_loss --mask_encoder_pen
python ink_training_eager_predictive.py --experiment_id 1590486000.6 --comment "tf2-af03-t_s4-hyb32-emb_pos_gmm10-tres2-fullstopg" --use_start_pos --num_pred_inputs 32 --stop_predictive_grad --pred_input_type hybrid --stroke_loss nll_gmm --n_t_samples 4 --batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model transformer --transformer_scale --transformer_pos_encoding --transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 --transformer_hidden_units 256 --transformer_dropout 0.0 --latent_units 8 --decoder_model t_emb --decoder_dropout 0.0 --decoder_layers 4 --decoder_hidden_units 512,512,512,512 --predictive_model transformer --learning_rate_type transformer --p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 --p_transformer_hidden_units 256 --p_transformer_dropout 0.0 --p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model transformer --data_name didi_wo_text --metadata_type position --disable_pen_loss --mask_encoder_pen
python ink_training_eager_embedding.py --experiment_id 1590486000.7 --comment "tf2-af03-gnorm1-vel" --decoder_model rnn --latent_units 8 --grad_clip_norm 1 --batch_size 1000 --affine_prob 0.3 --encoder_model rnn --bidirectional_encoder --resampling_factor 0 --scale_factor 0 --n_t_samples 0 --stroke_loss nll_gmm --learning_rate_type transformer --data_name didi_wo_text --metadata_type velocity




[StrokeAE][t-emb-vae] python ink_training_eager_embedding.py --experiment_id 1590486000.9 --comment "tf2-af03-t_s4-tres2_gt_targ-kld005" --gt_targets --use_vae --kld_weight 0.05 --decoder_model t_emb --decoder_layers 4 --decoder_hidden_units 512,512,512,512 --latent_units 8 --grad_clip_norm 1 --batch_size 1000 --affine_prob 0.3 --encoder_model transformer --encoder_model transformer --transformer_scale --transformer_pos_encoding --transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 --transformer_hidden_units 256 --transformer_dropout 0.0  --resampling_factor 2 --scale_factor 0 --n_t_samples 4 --stroke_loss nll_gmm --learning_rate_type transformer --disable_pen_loss --mask_encoder_pen --data_name didi_wo_text --metadata_type position
[StrokeAE][t-emb (gt_targets)] python ink_training_eager_embedding.py --experiment_id 1590799043.1 --comment "tf2-af03-t_s4-tres2_gt_targ" --gt_targets --decoder_model t_emb --decoder_layers 4 --decoder_hidden_units 512,512,512,512 --latent_units 8 --grad_clip_norm 1 --batch_size 1000 --affine_prob 0.3 --encoder_model transformer --encoder_model transformer --transformer_scale --transformer_pos_encoding --transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 --transformer_hidden_units 256 --transformer_dropout 0.0  --resampling_factor 2 --scale_factor 0 --n_t_samples 4 --stroke_loss nll_gmm --learning_rate_type transformer --disable_pen_loss --mask_encoder_pen --data_name didi_wo_text --metadata_type position
python ink_training_eager_embedding.py --experiment_id 1590799043.1 --comment "tf2-af03-t_s4-tres2_gt_targ" --gt_targets --decoder_model t_emb --decoder_layers 4 --decoder_hidden_units 512,512,512,512 --latent_units 8 --grad_clip_norm 1 --batch_size 1000 --affine_prob 0.3 --encoder_model transformer --encoder_model transformer --transformer_scale --transformer_pos_encoding --transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 --transformer_hidden_units 256 --transformer_dropout 0.0  --resampling_factor 2 --scale_factor 0 --n_t_samples 4 --stroke_loss nll_gmm --learning_rate_type transformer --disable_pen_loss --mask_encoder_pen --data_name didi_wo_text --metadata_type position; python ink_training_eager_embedding.py --experiment_id 1590799043.2 --comment "tf2-af03-tres2" --n_t_samples 1 --batch_size 1000 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 8 --decoder_model rnn --decoder_dropout 0.0 --learning_rate_type exponential --stroke_loss nll_gmm --data_name didi_wo_text --metadata_type position

[Full][t-emb+TR] python ink_training_eager_predictive.py --experiment_id 1590706693.5 --comment "tf2-af03-t_s4-hyb32-tres2-fullstopg" --use_start_pos --num_pred_inputs 32 --stop_predictive_grad --pred_input_type hybrid --stroke_loss nll_gmm --n_t_samples 4 --batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model transformer --transformer_scale --transformer_pos_encoding --transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 --transformer_hidden_units 256 --transformer_dropout 0.0 --latent_units 8 --decoder_model t_emb --decoder_dropout 0.0 --decoder_layers 4 --decoder_hidden_units 512,512,512,512 --predictive_model transformer --learning_rate_type transformer --p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 --p_transformer_hidden_units 256 --p_transformer_dropout 0.0 --p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model transformer --data_name didi_wo_text --metadata_type position --disable_pen_loss --mask_encoder_pen
[Full][t-emb+TR+ordered] python ink_training_eager_predictive.py --experiment_id 1590706693.4 --comment "tf2-af03-t_s4-ord32-tres2-fullstopg" --use_start_pos --num_pred_inputs 32 --stop_predictive_grad --pred_input_type ordered --stroke_loss nll_gmm --n_t_samples 4 --batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model transformer --transformer_scale --transformer_pos_encoding --transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 --transformer_hidden_units 256 --transformer_dropout 0.0 --latent_units 8 --decoder_model t_emb --decoder_dropout 0.0 --decoder_layers 4 --decoder_hidden_units 512,512,512,512 --predictive_model transformer --learning_rate_type transformer --p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 --p_transformer_hidden_units 256 --p_transformer_dropout 0.0 --p_transformer_scale --p_transformer_pos_encoding --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model transformer --data_name didi_wo_text --metadata_type position --disable_pen_loss --mask_encoder_pen
[Full][t-emb-vae+TR] python ink_training_eager_predictive.py --experiment_id 1590486000.11 --comment "tf2-af03-t_s4-hyb32-tres2_gt_targ-fullstopg-kld001" --gt_targets --use_vae --kld_weight 0.01 --use_start_pos --num_pred_inputs 32 --stop_predictive_grad --pred_input_type hybrid --stroke_loss nll_gmm --n_t_samples 4 --batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model transformer --transformer_scale --transformer_pos_encoding --transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 --transformer_hidden_units 256 --transformer_dropout 0.0 --latent_units 8 --decoder_model t_emb --decoder_dropout 0.0 --decoder_layers 4 --decoder_hidden_units 512,512,512,512 --predictive_model transformer --learning_rate_type transformer --p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 --p_transformer_hidden_units 256 --p_transformer_dropout 0.0 --p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model transformer --data_name didi_wo_text --metadata_type position --disable_pen_loss --mask_encoder_pen
[Full][t-emb+TR (gt_targets)] python ink_training_eager_predictive.py --experiment_id 1590706693.5 --comment "tf2-af03-t_s4-hyb32-tres2-gt_targ-fullstopg" --gt_targets --use_start_pos --num_pred_inputs 32 --stop_predictive_grad --pred_input_type hybrid --stroke_loss nll_gmm --n_t_samples 4 --batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model transformer --transformer_scale --transformer_pos_encoding --transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 --transformer_hidden_units 256 --transformer_dropout 0.0 --latent_units 8 --decoder_model t_emb --decoder_dropout 0.0 --decoder_layers 4 --decoder_hidden_units 512,512,512,512 --predictive_model transformer --learning_rate_type transformer --p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 --p_transformer_hidden_units 256 --p_transformer_dropout 0.0 --p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model transformer --data_name didi_wo_text --metadata_type position --disable_pen_loss --mask_encoder_pen

[Full][t-emb+RNN] python ink_training_eager_predictive.py --experiment_id 1590706693.2 --comment "tf2-af03-t_s4-hyb32-tres2-fullstopg" --use_start_pos --num_pred_inputs 32 --stop_predictive_grad --pred_input_type hybrid --stroke_loss nll_gmm --n_t_samples 4 --batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model transformer --transformer_scale --transformer_pos_encoding --transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 --transformer_hidden_units 256 --transformer_dropout 0.0 --latent_units 8 --decoder_model t_emb --decoder_dropout 0.0 --decoder_layers 4 --decoder_hidden_units 512,512,512,512 --predictive_model rnn --learning_rate_type transformer --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model rnn --data_name didi_wo_text --metadata_type position --disable_pen_loss --mask_encoder_pen
[Full][t-emb+RNN (ord, gt_targ)] python ink_training_eager_predictive.py --experiment_id 1590706693.7 --comment "tf2-af03-t_s4-ord32-tres2_gt_targ-fullstopg" --gt_targets --use_start_pos --num_pred_inputs 32 --stop_predictive_grad --pred_input_type ordered --stroke_loss nll_gmm --n_t_samples 4 --batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model transformer --transformer_scale --transformer_pos_encoding --transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 --transformer_hidden_units 256 --transformer_dropout 0.0 --latent_units 8 --decoder_model t_emb --decoder_dropout 0.0 --decoder_layers 4 --decoder_hidden_units 512,512,512,512 --predictive_model rnn --learning_rate_type transformer --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model rnn --data_name didi_wo_text --metadata_type position --disable_pen_loss --mask_encoder_pen

[Full][seq2seq (exp)] python ink_training_eager_embedding.py --experiment_id 1590863506.1 --comment "tf2-af03-tres2" --n_t_samples 1 --batch_size 1000 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 8 --decoder_model rnn --decoder_dropout 0.0 --learning_rate_type exponential --stroke_loss nll_gmm --data_name didi_wo_text --metadata_type position
[Full][seq2seq-ar (exp)] python ink_training_eager_embedding.py --experiment_id 1590863506.2 --comment "tf2-af03-tres2" --n_t_samples 1 --batch_size 1000 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 8 --decoder_model rnn --decoder_dropout 0.0 --learning_rate_type exponential --stroke_loss nll_gmm --data_name didi_wo_text --metadata_type position

[StrokeAE][seq2seq-vae-ar] python ink_training_eager_embedding.py --experiment_id 1590793466.1 --comment "tf2-af03-tres2" --decoder_autoregressive --use_vae --kld_weight 0.2 --n_t_samples 1 --batch_size 1000 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 8 --decoder_model rnn --decoder_dropout 0.0 --learning_rate_type exponential --stroke_loss nll_gmm --data_name didi_wo_text --metadata_type position
[StrokeAE][seq2seq-vae-ar] python ink_training_eager_embedding.py --experiment_id 1590793466.2 --comment "tf2-af03-tres2" --decoder_autoregressive --use_vae --kld_weight 0.3 --n_t_samples 1 --batch_size 1000 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 8 --decoder_model rnn --decoder_dropout 0.0 --learning_rate_type exponential --stroke_loss nll_gmm --data_name didi_wo_text --metadata_type position
[StrokeAE][seq2seq-vae-ar] python ink_training_eager_embedding.py --experiment_id 1590793466.3 --comment "tf2-af03-tres2" --decoder_autoregressive --use_vae --kld_weight 0.5 --n_t_samples 1 --batch_size 1000 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 8 --decoder_model rnn --decoder_dropout 0.0 --learning_rate_type exponential --stroke_loss nll_gmm --data_name didi_wo_text --metadata_type position
[StrokeAE][seq2seq-vae-ar] python ink_training_eager_embedding.py --experiment_id 1590793466.4 --comment "tf2-af03-tres2-repvae" --repeat_vae_sample --decoder_autoregressive --use_vae --kld_weight 1.0 --n_t_samples 1 --batch_size 1000 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 8 --decoder_model rnn --decoder_dropout 0.0 --learning_rate_type exponential --stroke_loss nll_gmm --data_name didi_wo_text --metadata_type position

[StrokeAE][seq2seq-vel] python ink_training_eager_embedding.py --experiment_id 1590863506.3 --comment "tf2-af03-vel" --n_t_samples 1 --batch_size 1000 --affine_prob 0.3 --resampling_factor 0 --scale_factor 0 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 8 --decoder_model rnn --decoder_dropout 0.0 --learning_rate_type exponential --stroke_loss nll_gmm --data_name didi_wo_text --metadata_type velocity
[StrokeAE][seq2seq-ar-vel] python ink_training_eager_embedding.py --experiment_id 1590863506.4 --comment "tf2-af03-vel" --decoder_autoregressive --n_t_samples 1 --batch_size 1000 --affine_prob 0.3 --resampling_factor 0 --scale_factor 0 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 8 --decoder_model rnn --decoder_dropout 0.0 --learning_rate_type exponential --stroke_loss nll_gmm --data_name didi_wo_text --metadata_type velocity
[StrokeAE][seq2seq-vae-ar-vel] python ink_training_eager_embedding.py --experiment_id 1590793466.5 --comment "tf2-af03-repvae-vel" --repeat_vae_sample --decoder_autoregressive --use_vae --kld_weight 0.5 --n_t_samples 1 --batch_size 1000 --affine_prob 0.3 --resampling_factor 0 --scale_factor 0 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 8 --decoder_model rnn --decoder_dropout 0.0 --learning_rate_type exponential --stroke_loss nll_gmm --data_name didi_wo_text --metadata_type velocity
[StrokeAE][seq2seq-vae-ar-vel] python ink_training_eager_embedding.py --experiment_id 1590793466.6 --comment "tf2-af03-repvae-vel" --repeat_vae_sample --decoder_autoregressive --use_vae --kld_weight 1.0 --n_t_samples 1 --batch_size 1000 --affine_prob 0.3 --resampling_factor 0 --scale_factor 0 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 8 --decoder_model rnn --decoder_dropout 0.0 --learning_rate_type exponential --stroke_loss nll_gmm --data_name didi_wo_text --metadata_type velocity

[Full][seq2seq-vae+TR] python ink_training_eager_predictive.py --experiment_id 1590486000.12 --comment "tf2-af03-t_s4-hyb32-tres2-fullstopg" --use_vae --kld_weight 0.01 --use_start_pos --num_pred_inputs 32 --stop_predictive_grad --pred_input_type hybrid --stroke_loss nll_gmm --n_t_samples 1 --batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 8 --decoder_model rnn --decoder_dropout 0.0 --predictive_model transformer --learning_rate_type transformer --p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 --p_transformer_hidden_units 256 --p_transformer_dropout 0.0 --p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model transformer --data_name didi_wo_text --metadata_type position
[Full][seq2seq+RNN] python ink_training_eager_predictive.py --experiment_id 1590706693.1 --comment "tf2-af03-hyb32-tres2-fullstopg" --use_start_pos --num_pred_inputs 32 --stop_predictive_grad --pred_input_type hybrid --stroke_loss nll_gmm --n_t_samples 1 --batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 8 --decoder_model rnn --decoder_dropout 0.0 --predictive_model rnn --learning_rate_type transformer --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model rnn --data_name didi_wo_text --metadata_type position
[Full][seq2seq-vel+TR] python ink_training_eager_predictive.py --experiment_id 1590706693.3 --comment "tf2-af03-t_s4-hyb32-tres2-fullstopg-vel" --use_start_pos --num_pred_inputs 32 --stop_predictive_grad --pred_input_type hybrid --stroke_loss nll_gmm --n_t_samples 1 --batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 8 --decoder_model rnn --decoder_dropout 0.0 --predictive_model transformer --learning_rate_type transformer --p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 --p_transformer_hidden_units 256 --p_transformer_dropout 0.0 --p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model transformer --data_name didi_wo_text --metadata_type velocity
[Full][seq2seq-vel+RNN] python ink_training_eager_predictive.py --experiment_id 1591013267.1 --comment "tf2-af03-ord32-tres2-fullstopg-vel" --use_start_pos --num_pred_inputs 32 --stop_predictive_grad --pred_input_type hybrid --stroke_loss nll_gmm --n_t_samples 1 --batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 8 --decoder_model rnn --decoder_dropout 0.0 --predictive_model rnn --learning_rate_type exponential --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model rnn --data_name didi_wo_text --metadata_type velocity
[Full][seq2seq+RNN+ordered] python ink_training_eager_predictive.py --experiment_id 1591013267.2 --comment "tf2-af03-ord32-tres2-fullstopg" --use_start_pos --num_pred_inputs 32 --stop_predictive_grad --pred_input_type ordered --stroke_loss nll_gmm --n_t_samples 1 --batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 8 --decoder_model rnn --decoder_dropout 0.0 --predictive_model rnn --learning_rate_type transformer --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model rnn --data_name didi_wo_text --metadata_type position

python ink_training_eager_embedding.py --experiment_id 1590891167.1 --comment 'tf2-skrnn-af03-rdp_vel' --decoder_dynamic_h0 --repeat_vae_sample --decoder_autoregressive --use_vae --kld_type kld_p0_norm --kld_weight 0.5 --batch_size 100 --affine_prob 0.3 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 128 --decoder_model rnn --learning_rate_type exponential --stroke_loss nll_gmm --rdp_dataset --ink_dataset --data_name didi_wo_text_rdp --metadata_type rdp_velocity; python ink_training_eager_embedding.py --experiment_id 1590891167.2 --comment 'tf2-skrnn-af03-rdp_vel' --decoder_dynamic_h0 --repeat_vae_sample --decoder_autoregressive --use_vae --kld_type kld_p0_norm --kld_weight 0.2 --batch_size 100 --affine_prob 0.3 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 128 --decoder_model rnn --learning_rate_type exponential --stroke_loss nll_gmm --rdp_dataset --ink_dataset --data_name didi_wo_text_rdp --metadata_type rdp_velocity; python ink_training_eager_embedding.py --experiment_id 1590891167.3 --comment 'tf2-skrnn-af03-rdp_vel' --decoder_dynamic_h0 --repeat_vae_sample --decoder_autoregressive --use_vae --kld_type kld_p0_norm --kld_weight 1.0 --batch_size 100 --affine_prob 0.3 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 128 --decoder_model rnn --learning_rate_type exponential --stroke_loss nll_gmm --rdp_dataset --ink_dataset --data_name didi_wo_text_rdp --metadata_type rdp_velocity; python ink_training_eager_embedding.py --experiment_id 1590891167.4 --comment 'tf2-skrnn-af03-rdp_vel' --decoder_dynamic_h0 --repeat_vae_sample --decoder_autoregressive --use_vae --kld_type kld_p0_norm --kld_weight 0.5 --batch_size 500 --affine_prob 0.3 --grad_clip_norm 1 --encoder_model rnn --bidirectional_encoder --latent_units 128 --decoder_model rnn --learning_rate_type exponential --stroke_loss nll_gmm --rdp_dataset --ink_dataset --data_name didi_wo_text_rdp --metadata_type rdp_velocity
python ink_training_eager_predictive.py --experiment_id 1590894218.1 --comment "tf2-af03-t_s4-hyb32-rdp_didipp-fullstopg" --use_start_pos --num_pred_inputs 32 --stop_predictive_grad --pred_input_type hybrid --stroke_loss nll_gmm --n_t_samples 4 --batch_size 128 --affine_prob 0.3 --resampling_factor 0 --scale_factor 0 --grad_clip_norm 1 --encoder_model transformer --transformer_scale --transformer_pos_encoding --transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 --transformer_hidden_units 256 --transformer_dropout 0.0 --latent_units 8 --decoder_model t_emb --decoder_dropout 0.0 --decoder_layers 4 --decoder_hidden_units 512,512,512,512 --predictive_model transformer --learning_rate_type transformer --p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 --p_transformer_hidden_units 256 --p_transformer_dropout 0.0 --p_transformer_scale --embedding_loss nll_gmm --embedding_gmm_components 10 --loss_predicted_embedding --loss_reconstructed_ink --position_model transformer --rdp_dataset --rdp_didi_pp --data_name didi_wo_text_rdp --metadata_type rdp_position --disable_pen_loss


python eval.py --model_ids 1590706693.1,1590706693.2,1590486000.5,1590486000.6 --embedding_analysis
python eval.py --model_ids 1590486000.11,1590706693.4,1590706693.3 --embedding_analysis
"""



