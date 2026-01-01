#!/bin/bash

# Baseline tuning for AdamW with WSD schedule.

# Usage:
#   bash /projectnb/aclab/qinziz/trainit/scripts/meta_grad/baselines/adamw_wsd.sh

#-------------------------------------------------------------------------
# Some global variables and the `submit_job()` function

BASE_DIR=/projectnb/aclab/qinziz/trainit
EXP=adamw-wsd-baseline
PROJECT=lr_baselines

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

# add logger
LOG_EMA=0.9
base_args+=(
    "logger.log_update_grad_corr=true"
    "logger.log_ema_update_grad_corr=true"
    "logger.update_ema_constant=$LOG_EMA"
)


#-------------------------------------------------------------------------
# Batch submission

schedule=trapezoid
warmup=200
decays=(0 200 400 600 800 1000)
# base_lrs=(0.1)
base_lrs=(3.33e-2 1e-2 3.33e-3 1e-3 3.33e-4 1e-4 3.33e-5 1e-5)


for decay in "${decays[@]}"; do
    for lr in "${base_lrs[@]}"; do
        name="w-${warmup}_d-${decay}_lr-${lr}"
        runid="$(uuidgen)"
        args="${base_args[@]}"
        args+=(
            "logging.wandb_name=$name"
            "logging.wandb_runid=$runid"
            "optimizer/lr_config=$schedule"
            "optimizer.lr_config.lr=$lr"
            "optimizer.lr_config.warmup=$warmup"
            "optimizer.lr_config.decay=$decay"
            "optimizer.lr_config.max_steps=$TOTAL_STEPS"
        )
        submit_job $name $runid ${args[@]}
    done
done