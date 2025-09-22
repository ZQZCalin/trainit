#!/bin/bash

# Usage:
#   cd /projectnb/aclab/qinziz/trainit/scripts
#   module load python3/3.10.12 cuda/12.2
#   source /projectnb/aclab/qinziz/trainit/env/bin/activate
#   bash train_local.sh

#-------------------------------------------------------------------------
# Some global variables and the `submit_job()` function

BASE_DIR=/projectnb/aclab/qinziz/trainit
                                # change to your path here; you can just copy from your config.sh
EXP=baseline_step2k_quadratic   # experiment name; you can change if you want
PROJECT=greedy_lr_schedule      # change to your wandb project name
TOTAL_STEPS=2000                # maximum number of training steps
lrs=(1e-1 3.33e-2 1e-2 3.33e-3 1e-3 3.33e-4 1e-4 3.33e-5 1e-5)  # log grid of LRs


#-------------------------------------------------------------------------
# Submit job function. No need to change.

NODES=1
GPU="L40S"
TIME="12:00:00"
DATE=$(date +"%Y-%m-%d")
OUTPUT_PATH=$BASE_DIR/scheduler_outputs/$DATE/$EXP
mkdir -p $OUTPUT_PATH 

#-------------------------------------------------------------------------
# Other training configs.

# mixed precision
USE_AMP=False

# training batch size
BATCH_SIZE=128

# optimizer configs
OPTIMIZER=adamw
BETA1=0.9
BETA2=0.999
WEIGHT_DECAY=0.1
NESTEROV=False

# log additional metrics to wandb
LOG_CALLBACK_DATA=False

# random seed
SEED=42

lr=1e-3
name="lr_${lr}"                                     # ONLY CHANGE args IF NECESSARY
save_path="${OUTPUT_PATH}/${name}/checkpoint"
args=(
    "logging.wandb_project=$PROJECT"
    "logging.wandb_name=$name"
    "logging.wandb_runid=$(uuidgen)"
    "logging.wandb_expname=$name"
    "logging.log_callback_data=$LOG_CALLBACK_DATA"
    "train.max_steps=$TOTAL_STEPS"
    "train.use_amp=$USE_AMP"
    "dataset.total_batch_size=$BATCH_SIZE"
    "random_seed=$SEED"
    "optimizer=$OPTIMIZER"
    "optimizer.beta1=$BETA1"
    "optimizer.beta2=$BETA2"
    "optimizer.weight_decay=$WEIGHT_DECAY"
    "optimizer.use_nesterov=$NESTEROV"
    "checkpoint.save=true"
    "checkpoint.save_path=$save_path"
    "checkpoint.save_steps=$TOTAL_STEPS"
)
# schedule configs
args+=(
    "optimizer/lr_config=quadratic"
    "optimizer.lr_config.lr=$lr"
    "optimizer.lr_config.max_steps=$TOTAL_STEPS"
)
python $BASE_DIR/main.py ${args[@]}