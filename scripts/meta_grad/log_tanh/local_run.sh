#!/bin/bash

# Usage:
#   qrsh -pe omp 8 -l gpus=1 -l gpu_c=7.0 -l gpu_type=L40S
#   cd /projectnb/aclab/qinziz/trainit
#   source activate_env.sh
#   bash /projectnb/aclab/qinziz/trainit/scripts/meta_grad/log_tanh/local_run.sh


#-------------------------------------------------------------------------
# Experiment configs

# basic info
PROJECT=meta_grad_with_reg
RUNID="$(uuidgen)"
NAME="test_${RUNID}"

# mixed precision
USE_AMP=False

# training
TOTAL_STEPS=2000
BATCH_SIZE=128
SEED=42

# optimizer configs
OPTIMIZER=adamw
BETA1=0.9
BETA2=0.999
WEIGHT_DECAY=0.1
NESTEROV=False

# log additional metrics to wandb
LOG_CALLBACK_DATA=True
LOGGER=meta_grad


# base args
args=(
    "logging.wandb_project=$PROJECT"
    "logging.wandb_name=$NAME"
    "logging.wandb_runid=$RUNID"
    "logging.wandb_expname=$NAME"
    "logging.log_callback_data=$LOG_CALLBACK_DATA"
    "logger=$LOGGER"
    "train.max_steps=$TOTAL_STEPS"
    "train.use_amp=$USE_AMP"
    "dataset.total_batch_size=$BATCH_SIZE"
    "random_seed=$SEED"
    "optimizer=$OPTIMIZER"
    "optimizer.beta1=$BETA1"
    "optimizer.beta2=$BETA2"
    "optimizer.weight_decay=$WEIGHT_DECAY"
    "optimizer.use_nesterov=$NESTEROV"
)

# optimizer with meta-gradient
args+=(
    "optimizer/lr_config=constant"  # fix base optimizer schedule to constantly one
    "optimizer.lr_config.lr=1.0"
    "optimizer/wrapper=meta_grad"
    "optimizer.wrapper.base_lr=1e-3"
    "optimizer.wrapper.clip_min=1e-8"
    "optimizer.wrapper.clip_max=null"
    "optimizer.wrapper.momentum=0.99"   # turn off momentum for normal meta-grad
    "optimizer.wrapper.learning_rate.schedule_name=constant"
    "optimizer.wrapper.learning_rate.value=1e-6"
    "optimizer.wrapper.regularizer.name=log_tanh"
    "optimizer.wrapper.regularizer.regularization.schedule_name=constant"
    "optimizer.wrapper.regularizer.regularization.value=1.0"
    # "optimizer.wrapper.regularizer.regularization.warmup=200"
    "optimizer.wrapper.regularizer.regularization.total_steps=2000"
    "optimizer.wrapper.regularizer.decay.schedule_name=constant"
    "optimizer.wrapper.regularizer.decay.value=0.2"
    #NOTE: use +optimizer... to add a non-existing hydra config subfield
    "+optimizer.wrapper.aux_output_lr.schedule_name=warmup_stable_decay"
    "+optimizer.wrapper.aux_output_lr.value=1e-3"
    "+optimizer.wrapper.aux_output_lr.total_steps=2000"
    "+optimizer.wrapper.aux_output_lr.warmup=200"
    "+optimizer.wrapper.aux_output_lr.decay=200"
)

#-------------------------------------------------------------------------
# __main__

python main.py ${args[@]} 2>&1 | tee .log