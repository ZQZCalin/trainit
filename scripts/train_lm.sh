#!/bin/bash

# project=null
project=precond
run=baseline
steps=2000
batch_size=128

optimizer=adam
wd=0.1
use_momentum=True
use_preconditioning=True
decouple_weight_decay=False

schedule=linear
lr=1e-3
warmup=200

# BELOW for saving checkpoint without loggings
project=null
log_data=False

save_checkpoint=True
save_path=checkpoint/precond/adam_baseline
save_steps="[2000]"

python main.py \
    logging.wandb_project=$project \
    logging.wandb_name=$run \
    logging.log_callback_data=$log_data \
    train.max_steps=$steps \
    dataset.total_batch_size=$batch_size \
    optimizer=$optimizer \
    optimizer.weight_decay=$wd \
    optimizer.use_momentum=$use_momentum \
    optimizer.use_preconditioning=$use_preconditioning \
    optimizer.decouple_weight_decay=$decouple_weight_decay \
    optimizer/lr_config=$schedule \
    optimizer.lr_config.lr=$lr \
    optimizer.lr_config.warmup=$warmup \
    optimizer.lr_config.max_steps=$steps \
    checkpoint.save=$save_checkpoint \
    checkpoint.save_path=$save_path \
    checkpoint.save_steps=$save_steps
