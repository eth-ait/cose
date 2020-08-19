import time
from subprocess import call


NUM_CPU = 2
MEMORY = 6000
NUM_GPU = 1
WALL_TIME = 4
cluster_command_format = 'bsub -G ls_hilli -n {} -W {}:00 -o log_{} -R "rusage[mem={}, ngpus_excl_p={}]" '
# cluster_command_format = 'bsub -G ls_hilli -n {} -W {}:00 -o log_{} -R "rusage[mem={}, ngpus_excl_p={}]" -R "select[gpu_model0==GeForceGTX1080Ti]" '

model_ids = ["1597224180.2"]
experiments_command = "python eval.py --qualitative --quantitative --embedding_analysis "

# Create a unique experiment timestamp.
for work_id, model_id in enumerate(model_ids):
    time.sleep(1)
    experiment_command = experiments_command + ' --model_id ' + model_id

    cluster_command = cluster_command_format.format(NUM_CPU,
                                                    WALL_TIME,
                                                    model_id + "_eval",
                                                    MEMORY,
                                                    NUM_GPU)
    call([cluster_command + experiment_command], shell=True)
