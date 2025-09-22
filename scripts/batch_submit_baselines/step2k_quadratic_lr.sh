#!/bin/bash

# Usage:
#   cd /projectnb/aclab/qinziz/trainit/
#   module load python3/3.10.12 cuda/12.2
#   source /projectnb/aclab/qinziz/trainit/env/bin/activate
#   bash scripts/batch_submit_baselines/step2k_quadratic_lr.sh

#-------------------------------------------------------------------------
# Some global variables and the `submit_job()` function

BASE_DIR=/projectnb/aclab/qinziz/trainit
                                # change to your path here; you can just copy from your config.sh
EXP=baseline_step2k_quadratic   # experiment name; you can change if you want
PROJECT=greedy_lr_schedule      # change to your wandb project name
TOTAL_STEPS=2000                # maximum number of training steps
lrs=(1.0 3.33e-1 1e-1 3.33e-2 1e-2 3.33e-3 1e-3 3.33e-4 1e-4 3.33e-5 1e-5)  # log grid of LRs


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

for lr in "${lrs[@]}"; do
    name="lr_${lr}"                                     # ONLY CHANGE args IF NECESSARY
    runid="$(uuidgen)"
    save_path="${OUTPUT_PATH}/${name}/checkpoint"
    args=(
        "logging.wandb_project=$PROJECT"
        "logging.wandb_name=$name"
        "logging.wandb_runid=$runid"
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
    submit_job $name $runid ${args[@]}
done