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
python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=5000 dataset.total_batch_size=32 \
    optimizer.lr_config.lr=12e-4