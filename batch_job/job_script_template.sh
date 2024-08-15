#!/bin/bash -l

#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l gpu_memory=24G    # Specifies GPU memory
#$ -l h_rt=8:00:00      # Specifies the hard time limit for the job

cd /projectnb/aclab/qinziz/trainit
module load python3/3.10.12 cuda/12.2
source env/bin/activate
python check_env.py

# Default batch job. By default without specifying logging.wandb_project, will not log to wandb.
python train_jax.py logging.wandb_project=PROJECT_NAME

# If you don't have enough GPU memory, you can turn off part of the logging stuff.
python train_jax.py logging.wandb_project=PROJECT_NAME \
    logging.store_last_grads=false \
    logging.store_past_grads=false \
    logging.store_last_params=false \
    logging.compute_last_loss=false \
    logging.compute_last_grads=false