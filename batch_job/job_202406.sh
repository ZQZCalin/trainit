#!/bin/bash -l

#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l gpu_memory=24G    # Specifies GPU memory
#$ -l h_rt=8:00:00      # Specifies the hard time limit for the job

### Records batch scripts in 2024/06.

cd /projectnb/aclab/qinziz/trainit
module load python3/3.10.12 cuda/12.2
source env/bin/activate
python check_env.py

# Actual training script

# 2024/06/07
# 1. with/out pytorch initialization
# python train_jax.py model.load_pytorch=True train.random_scaling=exponential
# python train_jax.py model.load_pytorch=False train.random_scaling=exponential
# python train_jax.py model.load_pytorch=True train.random_scaling=exponential optimizer=sgdm
# python train_jax.py model.load_pytorch=False train.random_scaling=exponential optimizer=sgdm

# python train_jax.py model.load_pytorch=True
# python train_jax.py model.load_pytorch=False
# python train_jax.py model.load_pytorch=True optimizer=sgdm
# python train_jax.py model.load_pytorch=False optimizer=sgdm


# 2024/06/10
# 1. Use batch_size=2. 
# 1.a. Test on JAX.
# python train_jax.py model.load_pytorch=True dataset.batch_size=2 optimizer=adam
# python train_jax.py model.load_pytorch=False dataset.batch_size=2 optimizer=adam
# python train_jax.py model.load_pytorch=True dataset.batch_size=2 optimizer=sgdm
# python train_jax.py model.load_pytorch=False dataset.batch_size=2 optimizer=sgdm

# 1.b. Test on pytorch.
# python train_torch.py dataset.batch_size=2 optimizer=adam
# python train_torch.py dataset.batch_size=2 optimizer=sgdm


# 2024/06/13
# 1. Test whether it's the datasets that makes the difference.
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=adam model.use_hugging_face=True dataset.use_loadit=True
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=adam model.use_hugging_face=True dataset.use_loadit=False
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=adam model.use_hugging_face=True dataset.use_loadit=True dataset.use_streaming_loadit=True
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=sgdm model.use_hugging_face=True dataset.use_loadit=True
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=sgdm model.use_hugging_face=True dataset.use_loadit=False
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=sgdm model.use_hugging_face=True dataset.use_loadit=True dataset.use_streaming_loadit=True


# 2024/06/14
# 0. Save checkpoint at 900.
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=adam \
#     experimental.use_hugging_face=True experimental.use_loadit=True experimental.use_streaming_loadit=True \
#     experimental.save_checkpoint.use=True logging.wandb_project=null
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=adam \
#     experimental.use_hugging_face=True experimental.use_loadit=True experimental.use_streaming_loadit=False \
#     experimental.save_checkpoint.use=True experimental.save_checkpoint.path=checkpoint/06-14-01/hfgpt_adam_loadit_shuffle_iter_900.pth \
#     logging.wandb_project=null

# 1. Checkpoint vs random initialization; adam vs sgdm
# adam + checkpoint
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=adam \
#     experimental.use_hugging_face=True experimental.use_loadit=True experimental.use_streaming_loadit=True \
#     experimental.load_checkpoint.use=True
# adam + random init
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=adam \
#     experimental.use_hugging_face=True experimental.use_loadit=True experimental.use_streaming_loadit=True \
#     experimental.load_checkpoint.use=False
# sgdm + checkpoint
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=sgdm \
#     experimental.use_hugging_face=True experimental.use_loadit=True experimental.use_streaming_loadit=True \
#     experimental.load_checkpoint.use=True
# sgdm + checkpoint
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=sgdm \
#     experimental.use_hugging_face=True experimental.use_loadit=True experimental.use_streaming_loadit=True \
#     experimental.load_checkpoint.use=False
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=adam \
#     experimental.use_hugging_face=True experimental.use_loadit=True experimental.use_streaming_loadit=False \
#     experimental.load_checkpoint.use=True
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=sgdm \
#     experimental.use_hugging_face=True experimental.use_loadit=True experimental.use_streaming_loadit=False \
#     experimental.load_checkpoint.use=True


# 2024/06/17
# 1. Data conditioning
# Benchmark variant: shuffled loadit with filtering
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=adam \
#     experimental.use_hugging_face=True experimental.use_loadit=True experimental.use_streaming_loadit=False \
#     experimental.data_conditioning.use=True experimental.data_conditioning.threshold=0.25
# Group A: unshuffled loadit with filtering
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=adam \
#     experimental.use_hugging_face=True experimental.use_loadit=True experimental.use_streaming_loadit=True \
#     experimental.data_conditioning.use=True experimental.data_conditioning.threshold=0.25
# Group B: streaming with filtering
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=adam \
#     experimental.use_hugging_face=True experimental.use_loadit=False \
#     experimental.data_conditioning.use=True experimental.data_conditioning.threshold=0.25

# 1.1. Data conditioning with different filter constants
# filter = 0.5
# Benchmark variant: shuffled loadit with filtering
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=adam \
#     experimental.use_hugging_face=True experimental.use_loadit=True experimental.use_streaming_loadit=False \
#     experimental.data_conditioning.use=True experimental.data_conditioning.threshold=0.5
# Group A: unshuffled loadit with filtering
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=adam \
#     experimental.use_hugging_face=True experimental.use_loadit=True experimental.use_streaming_loadit=True \
#     experimental.data_conditioning.use=True experimental.data_conditioning.threshold=0.5
# Group B: streaming with filtering
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=adam \
#     experimental.use_hugging_face=True experimental.use_loadit=False \
#     experimental.data_conditioning.use=True experimental.data_conditioning.threshold=0.5
# filter = 0.1
# Benchmark variant: shuffled loadit with filtering
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=adam \
#     experimental.use_hugging_face=True experimental.use_loadit=True experimental.use_streaming_loadit=False \
#     experimental.data_conditioning.use=True experimental.data_conditioning.threshold=0.1
# Group A: unshuffled loadit with filtering
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=adam \
#     experimental.use_hugging_face=True experimental.use_loadit=True experimental.use_streaming_loadit=True \
#     experimental.data_conditioning.use=True experimental.data_conditioning.threshold=0.1
# Group B: streaming with filtering
# python train_neurips_pipeline.py dataset.batch_size=2 optimizer=adam \
#     experimental.use_hugging_face=True experimental.use_loadit=False \
#     experimental.data_conditioning.use=True experimental.data_conditioning.threshold=0.1


# 2024/06/20-06/21
# Back to JAX
# A. Benchmarking Adam optimizers
# 1. learning rates
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=1e-4
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=3e-4
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=1e-3

# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=1e-4 random_seed=101
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=3e-4 random_seed=101
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=1e-3 random_seed=101

# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=1e-4 random_seed=199
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=3e-4 random_seed=199
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=1e-3 random_seed=199

# 2. schedules
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=3e-4 optimizer.lr_config.schedule=cosine
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=1e-3 optimizer.lr_config.schedule=cosine
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=3e-4 optimizer.lr_config.schedule=constant
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=1e-3 optimizer.lr_config.schedule=constant

# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=3e-4 optimizer.lr_config.schedule=cosine random_seed=101
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=1e-3 optimizer.lr_config.schedule=cosine random_seed=101
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=3e-4 optimizer.lr_config.schedule=constant random_seed=101
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=1e-3 optimizer.lr_config.schedule=constant random_seed=101

# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=3e-4 optimizer.lr_config.schedule=cosine random_seed=199
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=1e-3 optimizer.lr_config.schedule=cosine random_seed=199
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=3e-4 optimizer.lr_config.schedule=constant random_seed=199
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=1e-3 optimizer.lr_config.schedule=constant random_seed=199

# 3. weight decay
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=3e-4 optimizer.weight_decay=0.03
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=3e-4 optimizer.weight_decay=0.3

# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=3e-4 optimizer.weight_decay=0.03 random_seed=101
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=3e-4 optimizer.weight_decay=0.3 random_seed=101

# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=3e-4 optimizer.weight_decay=0.03 random_seed=199
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=3e-4 optimizer.weight_decay=0.3 random_seed=199

# 4. b2
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=3e-4 optimizer.beta2=0.99
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=3e-4 optimizer.beta2=0.95

# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=3e-4 optimizer.beta2=0.99 random_seed=101
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=3e-4 optimizer.beta2=0.95 random_seed=101

# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=3e-4 optimizer.beta2=0.99 random_seed=199
# python train_jax.py logging.wandb_project=jax_benchmark optimizer.lr_config.lr=3e-4 optimizer.beta2=0.95 random_seed=199