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


# 08/16: large batch adam benchmark
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.lr=3e-4
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.lr=1e-3
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.lr=3e-3
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.lr=1e-2
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.lr=3e-2


# 08/16: adam benchmark with random scaling
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.lr=3e-4 train.random_scaling=exponential
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.lr=1e-3 train.random_scaling=exponential
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.lr=3e-3 train.random_scaling=exponential
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.lr=1e-2 train.random_scaling=exponential


# 08/17: restart large batch adam benchmark
# NOTE: I note that the benchmark in 08/16 didn't correctly setup lr schedulers 
# (was still using 50k max-steps with 5k warmups).
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=3e-4
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=1e-3
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=3e-3
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=1e-2
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=3e-2

# B=2 comparison
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=100000 \
#     optimizer.lr_config.warmup=10000 optimizer.lr_config.max_steps=100000

# with random scaling on
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=3e-4 \
#     train.random_scaling=exponential
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=1e-3 \
#     train.random_scaling=exponential
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=3e-3 \
#     train.random_scaling=exponential


# 08/20: adam benchmark: random scaling with smaller lr
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=1e-4 \
#     train.random_scaling=exponential
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=3e-5 \
#     train.random_scaling=exponential
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=1e-5 \
#     train.random_scaling=exponential
