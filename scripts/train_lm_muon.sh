#!/bin/bash

# project=null
project=precond
run=muon
steps=2000
batch_size=128

optimizer=muon

schedule=linear
muon_lr=0.05
adam_lr=3e-4
warmup=200

momentum=0.95
nesterov=True
ns_steps=6

adam_beta1=0.95
adam_beta2=0.95
adam_eps=1e-8
adam_wd=0.0

# BELOW for saving checkpoint without loggings
# project=null
# log_data=False

# save_checkpoint=True
# save_path=checkpoint/precond/adam_baseline
# save_steps="[2000]"

python main.py \
    logging.wandb_project=$project \
    logging.wandb_name=$run \
    logging.log_callback_data=$log_data \
    train.max_steps=$steps \
    dataset.total_batch_size=$batch_size \
    optimizer=$optimizer \
    optimizer.momentum=$momentum \
    optimizer.nesterov=$nesterov \
    optimizer.ns_steps=$ns_steps \
    optimizer/lr_config=$schedule \
    optimizer.lr_config.lr=$muon_lr \
    optimizer.lr_config.warmup=$warmup \
    optimizer.lr_config.max_steps=$steps \
    optimizer.adam_lr=$adam_lr \
    optimizer.adam_beta1=$adam_beta1 \
    optimizer.adam_beta2=$adam_beta2 \
    optimizer.adam_eps=$adam_eps \
    optimizer.adam_wd=$adam_wd
    # checkpoint.save=$save_checkpoint \
    # checkpoint.save_path=$save_path \
    # checkpoint.save_steps=$save_steps
