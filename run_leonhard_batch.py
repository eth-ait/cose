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
 
    ]

experiment_list_rebuttal = [
    # 'python ink_training_eager_predictive.py '
    # '--comment "leo_test-rebuttal-cose_no_target_pos" --gt_targets '
    # '--num_pred_inputs 32 --stop_predictive_grad --pred_input_type hybrid '
    # '--stroke_loss nll_gmm --n_t_samples 4 --batch_size 128 '
    # '--affine_prob 0.3 --resampling_factor 2 --scale_factor 0 '
    # '--grad_clip_norm 1 '
    # '--encoder_model transformer --transformer_scale --transformer_pos_encoding '
    # '--transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 '
    # '--transformer_hidden_units 256 --transformer_dropout 0.0 --latent_units 8 '
    # '--decoder_model t_emb --decoder_dropout 0.0 --decoder_layers 4 '
    # '--decoder_hidden_units 512,512,512,512 '
    # '--predictive_model transformer --learning_rate_type transformer '
    # '--p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 '
    # '--p_transformer_hidden_units 256 --p_transformer_dropout 0.0 '
    # '--p_transformer_scale '
    # '--embedding_loss nll_gmm --embedding_gmm_components 10 '
    # '--loss_predicted_embedding --loss_reconstructed_ink '
    # '--position_model transformer --data_name didi_wo_text '
    # '--metadata_type position --disable_pen_loss --mask_encoder_pen ',
    
    # # COSE
    # 'python ink_training_eager_predictive.py '
    # '--comment "tf2-af03-t_s4-hyb32-tres2-gt_targ-fullstopg" --gt_targets '
    # '--use_start_pos --num_pred_inputs 32 --stop_predictive_grad '
    # '--pred_input_type hybrid --stroke_loss nll_gmm --n_t_samples 4 '
    # '--batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 '
    # '--grad_clip_norm 1 '
    # '--encoder_model transformer --transformer_scale --transformer_pos_encoding '
    # '--transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 '
    # '--transformer_hidden_units 256 --transformer_dropout 0.0 --latent_units 8 '
    # '--decoder_model t_emb --decoder_dropout 0.0 --decoder_layers 4 '
    # '--decoder_hidden_units 512,512,512,512 --predictive_model transformer '
    # '--learning_rate_type transformer '
    # '--p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 '
    # '--p_transformer_hidden_units 256 --p_transformer_dropout 0.0 '
    # '--p_transformer_scale --position_model transformer '
    # '--embedding_loss nll_gmm --embedding_gmm_components 10 '
    # '--loss_predicted_embedding --loss_reconstructed_ink '
    # '--data_name didi_wo_text --metadata_type position --disable_pen_loss '
    # '--mask_encoder_pen ',
    
  'python ink_training_eager_predictive.py '
    '--comment "dist_cond-af03-t_s4-hyb32-tres2-gt_targ" --gt_targets '
    '--use_start_pos --num_pred_inputs 32 --stop_predictive_grad '
    '--pred_input_type hybrid --stroke_loss nll_gmm --n_t_samples 4 '
    '--batch_size 128 --affine_prob 0.3 --resampling_factor 2 --scale_factor 0 '
    '--grad_clip_norm 1 '
    '--encoder_model transformer --transformer_scale --transformer_pos_encoding '
    '--transformer_layers 6 --transformer_heads 4 --transformer_dmodel 64 '
    '--transformer_hidden_units 256 --transformer_dropout 0.0 --latent_units 8 '
    '--decoder_model t_emb --decoder_dropout 0.0 --decoder_layers 4 '
    '--decoder_hidden_units 512,512,512,512 --predictive_model transformer '
    '--learning_rate_type transformer '
    '--p_transformer_layers 6 --p_transformer_heads 4 --p_transformer_dmodel 64 '
    '--p_transformer_hidden_units 256 --p_transformer_dropout 0.0 '
    '--p_transformer_scale --position_model transformer '
    '--embedding_loss nll_gmm --embedding_gmm_components 10 '
    '--loss_predicted_embedding --loss_reconstructed_ink '
    '--data_name didi_wo_text --metadata_type position --disable_pen_loss '
    '--mask_encoder_pen ',
    ]

experiment_list = experiment_list_rebuttal
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