#!/bin/bash

# project=null
project=precond
run=adamw
log_data=True
steps=2000
batch_size=128

optimizer=adamw
wd=0.1
nesterov=True

schedule=linear
lr=1e-3
warmup=200

# BELOW for saving checkpoint without loggings
save_checkpoint=False
save_path=checkpoint/precond/adamw
save_steps="[2000]"

python main.py \
    logging.wandb_project=$project \
    logging.wandb_name=$run \
    logging.log_callback_data=$log_data \
    train.max_steps=$steps \
    dataset.total_batch_size=$batch_size \
    optimizer=$optimizer \
    optimizer.weight_decay=$wd \
    optimizer.use_nesterov=$nesterov \
    optimizer/lr_config=$schedule \
    optimizer.lr_config.lr=$lr \
    optimizer.lr_config.warmup=$warmup \
    optimizer.lr_config.max_steps=$steps \
    checkpoint.save=$save_checkpoint \
    checkpoint.save_path=$save_path \
    checkpoint.save_steps=$save_steps
