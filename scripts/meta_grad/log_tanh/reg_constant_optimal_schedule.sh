#!/bin/bash

# Computes the regularization constant schedule using the optimal schedule with WSD.

# Usage:
#   bash /projectnb/aclab/qinziz/trainit/scripts/meta_grad/log_tanh/reg_constant_optimal_schedule.sh

#-------------------------------------------------------------------------
# Some global variables and the `submit_job()` function

BASE_DIR=/projectnb/aclab/qinziz/trainit
EXP=WSD_meta_grad
PROJECT=meta_grad_with_reg

#-------------------------------------------------------------------------
# Submit job function. No need to change.

NODES=1
GPU="L40S"
TIME="12:00:00"
DATE=$(date +"%Y-%m-%d")
OUTPUT_PATH=$BASE_DIR/scheduler_outputs/$DATE/$EXP
mkdir -p $OUTPUT_PATH

submit_job() {
    local name="$1"
    local runid="$2"
    shift 2
    local args=("$@")

    local job_path="${OUTPUT_PATH}/${name}"
    mkdir -p $job_path

    local job_output=$(qsub <<EOF
#!/bin/bash -l
#$ -pe omp 8
#$ -l h="!scc-506"          # Blacklists bad nodes
#$ -l gpus=${NODES}
#$ -l gpu_type=${GPU}       # Specifies the gpu type
#$ -l h_rt=${TIME}          # Specifies the hard time limit for the job
#$ -N "$name"
#$ -o $job_path/\$JOB_NAME.o\$JOB_ID
#$ -e $job_path/\$JOB_NAME.e\$JOB_ID

cd ${BASE_DIR}
source activate_env.sh
python main.py ${args[@]}
EOF
    )
    # Extract job id and log the submission.
    local job_id=$(echo "$job_output" | awk '{print $3}')
    local wandb_path="optimizedlearning/${PROJECT}/${runid}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') job_id: ${job_id} || ${name} || ${job_path} || ${wandb_path}" >> "${OUTPUT_PATH}/job_list.txt"
    echo "Submitted job: $name"
}


#-------------------------------------------------------------------------
# Experiment configs

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
base_args=(
    "logging.wandb_project=$PROJECT"
    "logging.wandb_expname=$EXP"
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



#-------------------------------------------------------------------------
# Batch submission

aux_schedule=warmup_stable_decay
# aux_warmups=(0 200 400 600 800 1000)
# aux_decays=(0 200 400 600 800 1000)
aux_warmups=(200)
aux_decays=(200)
aux_lrs=(1e-6 1e-5 1e-4 1e-3 1e-2)

lr_schedule=constant
lr_value=1.0    # overwritten by aux anyways
base_lr=1.0     # overwritten by aux anyways

reg_schedule=constant
reg_value=1.0

decay_schedule=constant
decay_value=0.2

for aux_warmup in "${aux_warmups[@]}"; do
    for aux_decay in "${aux_decays[@]}"; do
        for aux_lr in "${aux_lrs[@]}"; do
            name="aux_w${aux_warmup}_d${aux_decay}_lr${aux_lr}"
            runid="$(uuidgen)"
            args="${base_args[@]}"
            args+=(
                "logging.wandb_name=$name"
                "logging.wandb_runid=$runid"
                "optimizer/lr_config=constant"  # fix base optimizer schedule to constantly one
                "optimizer.lr_config.lr=1.0"
                "optimizer/wrapper=meta_grad"
                "optimizer.wrapper.base_lr=$base_lr"
                "optimizer.wrapper.clip_min=1e-8"
                "optimizer.wrapper.clip_max=null"
                "optimizer.wrapper.learning_rate.schedule_name=$lr_schedule"
                "optimizer.wrapper.learning_rate.value=$lr_value"
                "optimizer.wrapper.regularizer.name=log_tanh"
                "optimizer.wrapper.regularizer.regularization.schedule_name=$reg_schedule"
                "optimizer.wrapper.regularizer.regularization.value=$reg_value"
                "optimizer.wrapper.regularizer.decay.schedule_name=$decay_schedule"
                "optimizer.wrapper.regularizer.decay.value=$decay_value"
                "++optimizer.wrapper.aux_output_lr.schedule_name=$aux_schedule"
                "++optimizer.wrapper.aux_output_lr.value=$aux_lr"
                "++optimizer.wrapper.aux_output_lr.total_steps=$TOTAL_STEPS"
                "++optimizer.wrapper.aux_output_lr.warmup=$aux_warmup"
                "++optimizer.wrapper.aux_output_lr.decay=$aux_decay"
            )
            submit_job $name $runid ${args[@]}
            # echo "${args[@]}"
        done
    done
done