#!/bin/bash -l

#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l gpu_memory=24G    # Specifies GPU memory
#$ -l h_rt=8:00:00      # Specifies the hard time limit for the job

cd /projectnb/aclab/qinziz/trainit
source activate_env.sh

# 09/12-01: B=128 benchmark, different seeds
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=1e-3 \
#     random_seed=21
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=1e-3 \
#     random_seed=157
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=1e-3 \
#     random_seed=2024

# 09/12-02: B=128 benchmark, early stopping lr
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=1000 optimizer.lr_config.lr=1e-3

# 09/12-03: B=1024 (extreme case)
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=250 dataset.total_batch_size=1024 \
#     optimizer.lr_config.warmup=25 optimizer.lr_config.max_steps=250 optimizer.lr_config.lr=1e-3
# Out of curiosity: does the J-shape coincide with warmup (i.e., lr decreasing vs increasing)?
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=250 dataset.total_batch_size=1024 \
#     optimizer.lr_config.warmup=0 optimizer.lr_config.max_steps=250 optimizer.lr_config.lr=1e-3
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=250 dataset.total_batch_size=1024 \
#     optimizer.lr_config.warmup=50 optimizer.lr_config.max_steps=250 optimizer.lr_config.lr=1e-3
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=250 dataset.total_batch_size=1024 \
#     optimizer.lr_config.warmup=100 optimizer.lr_config.max_steps=250 optimizer.lr_config.lr=1e-3

# 09/12-04: B-128, larger shuffle chunk size 10k -> 100k
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=1e-3 \
#     dataset.shuffle_buffer_size=100000
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=250 dataset.total_batch_size=1024 \
#     optimizer.lr_config.warmup=25 optimizer.lr_config.max_steps=250 optimizer.lr_config.lr=1e-3 \
#     dataset.shuffle_buffer_size=100000



# 09/13: The main goal is to re-do previous tests, with larger chunk size (200k)
# 09/13-01: B=128 benchmark: Adam, no RS
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=3e-4
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=1e-3
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=3e-3
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=1e-2

# 09/13-02: RS with exponential scaling
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=3e-4 \
#     train.random_scaling=exponential
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=1e-3 \
#     train.random_scaling=exponential
# python train_jax.py logging.wandb_project=optimistic_o2nc train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=3e-3 \
#     train.random_scaling=exponential

# 09/13-03: RS with interpolate
# python experimental.py experimental.use_interpolate_o2nc=True logging.wandb_project=optimistic_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=3e-4
# python experimental.py experimental.use_interpolate_o2nc=True logging.wandb_project=optimistic_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=1e-3
# python experimental.py experimental.use_interpolate_o2nc=True logging.wandb_project=optimistic_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=3e-3


# 09/14-01: interpolate 1e-3, different random seeds => is it still true that cumulative noise is negative? if so, we need to understand why.
# python experimental.py experimental.use_interpolate_o2nc=True logging.wandb_project=optimistic_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=1e-3 \
#     random_seed=21
# python experimental.py experimental.use_interpolate_o2nc=True logging.wandb_project=optimistic_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=1e-3 \
#     random_seed=99
# python experimental.py experimental.use_interpolate_o2nc=True logging.wandb_project=optimistic_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=1e-3 \
#     random_seed=157


# 09/16-01: warmup random scaling from the middle
# python experimental.py experimental.use_interpolate_o2nc=True logging.wandb_project=optimistic_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=1e-3 \
#     experimental.rs_warmup=500
# python experimental.py experimental.use_interpolate_o2nc=True logging.wandb_project=optimistic_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=1e-3 \
#     experimental.rs_warmup=200
# python experimental.py experimental.use_interpolate_o2nc=True logging.wandb_project=optimistic_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=1e-3 \
#     experimental.rs_warmup=100
# python experimental.py experimental.use_interpolate_o2nc=True logging.wandb_project=optimistic_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.warmup=200 optimizer.lr_config.max_steps=2000 optimizer.lr_config.lr=1e-3 \
#     experimental.rs_warmup=1000


# 09/17-01: NOTE
# - corrected a bug in the implementation of interpolate-RS:
#   where it should be w_n = x_n - (1-s_n) * Delta_n instead of w_n = x_(n-1) + (1-s_n) * Delta_n in the previous code
# - now we need to re-run all experiments with interpolate RS
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=1e-3
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=1e-3 \
#     experimental.use_interpolate_o2nc=True experimental.rs_warmup=0
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=1e-3 \
#     experimental.use_interpolate_o2nc=True experimental.rs_warmup=100
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=1e-3 \
#     experimental.use_interpolate_o2nc=True experimental.rs_warmup=200
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=1e-3 \
#     experimental.use_interpolate_o2nc=True experimental.rs_warmup=500
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=1e-3 \
#     experimental.use_interpolate_o2nc=True experimental.rs_warmup=1000

# 09/17-02: Pseudo-RS
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=1e-3 \
#     experimental.use_interpolate_o2nc=True experimental.use_pseudo_rs=True

# 09/17-03: 01 but with lr=3e-4
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=3e-4
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=3e-4 \
#     experimental.use_interpolate_o2nc=True experimental.rs_warmup=0
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=3e-4 \
#     experimental.use_interpolate_o2nc=True experimental.rs_warmup=100
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=3e-4 \
#     experimental.use_interpolate_o2nc=True experimental.rs_warmup=200
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=3e-4 \
#     experimental.use_interpolate_o2nc=True experimental.rs_warmup=500
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=3e-4 \
#     experimental.use_interpolate_o2nc=True experimental.rs_warmup=1000

# 09/17-04: test s_n = 0 (wn = x(n-1))
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=1e-3 \
#     experimental.grad_at_last_params=True

# 09/17-05: truely random RS per sample
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=1e-3 \
#     experimental.use_per_sample_rs=True


# 09/18-01: save a checkpoint of Adam benchmark at 1000k iter
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=1e-3 \
#     checkpoint.save=True checkpoint.save_path=checkpoint/Adamw_B128_lr1e-3 checkpoint.save_steps=100


# 09/19-01: checkpoint behavior
# python test_checkpoint.py checkpoint.load=True checkpoint.load_path=checkpoint/Adamw_B128_lr1e-3 checkpoint.load_file=iter_1000.ckpt

# 09/19-02: interpolate RS but with fixed RS=0.5
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=1e-3 \
#     experimental.grad_at_middle=True


# 09/23-01: benchmark with additional log: 3rd layer mask update
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=1e-3

# no AMP
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=1e-3 \
#     train.use_amp=False

# NOTE: this is the benchmark after we fix the GPT2 mask layer static field
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=1e-3


# 09/24-01
# Run a new checkpoint of Adamw benchmark, B=128, lr=1e-3, no RS, ckpt iterations at [20,100,500,1000,2000]
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=1e-3 \
#     checkpoint.save=True checkpoint.save_path=checkpoint/new_Adamw_B128_lr1e-3 checkpoint.save_steps="[20,100,500,1000,2000]"

# 02: Adamw benchmark with b1=0 (weighted Adagrad)
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=1e-3 \
#     optimizer.beta1=0 \
#     checkpoint.save=True checkpoint.save_path=checkpoint/Adagrad_B128_lr1e-3 checkpoint.save_steps="[20,100,500,1000,2000]"

# 02-b: some other lr configs
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=3e-4 \
#     optimizer.beta1=0
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=3e-3 \
#     optimizer.beta1=0

# 03: Adamw benchmark with RS
# python experimental.py logging.wandb_project=large_batch_o2nc \
#     train.max_steps=2000 dataset.total_batch_size=128 \
#     optimizer.lr_config.max_steps=2000 optimizer.lr_config.warmup=200 optimizer.lr_config.lr=1e-3 \
#     experimental.use_interpolate_o2nc=True \
#     checkpoint.save=True checkpoint.save_path=checkpoint/Adamw_RS_B128_lr1e-3 checkpoint.save_steps="[20,100,500,1000,2000]"