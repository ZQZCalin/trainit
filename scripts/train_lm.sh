#!/bin/bash

# project=null
# project=precond
# run=adamw
# log_data=True
# steps=2000
# batch_size=128

# optimizer=adamw
# wd=0.1
# nesterov=True

# schedule=linear
# lr=1e-3
# warmup=200

# # BELOW for saving checkpoint without loggings
# save_checkpoint=False
# save_path=checkpoint/precond/adamw
# save_steps="[2000]"

# python main.py \
#     logging.wandb_project=$project \
#     logging.wandb_name=$run \
#     logging.log_callback_data=$log_data \
#     train.max_steps=$steps \
#     dataset.total_batch_size=$batch_size \
#     optimizer=$optimizer \
#     optimizer.weight_decay=$wd \
#     optimizer.use_nesterov=$nesterov \
#     optimizer/lr_config=$schedule \
#     optimizer.lr_config.lr=$lr \
#     optimizer.lr_config.warmup=$warmup \
#     optimizer.lr_config.max_steps=$steps \
#     checkpoint.save=$save_checkpoint \
#     checkpoint.save_path=$save_path \
#     checkpoint.save_steps=$save_steps



# project=greedy_lr_schedule
# batch_size=128
# use_amp=False
# log_data=False
# seed=42

# optimizer=adamw
# b1=0.9
# b2=0.999
# wd=0.1
# nesterov=False

# # name=baseline
# lr=1e-3
# steps=2000

# # schedule=linear
# # warmup=200
# # waits=(0 500 1000 1600)
# # wait=${waits[2]}
# # name=trapezoid3

# schedule=cosine
# warmup=200
# name=cosine

# # System variables
# BASE_DIR=/projectnb/aclab/qinziz/trainit
# DATE=$(date +"%Y-%m-%d")
# OUTPUT_PATH=$BASE_DIR/scc_outputs/$DATE/$name

# mkdir -p $OUTPUT_PATH

# qsub <<EOF
# #!/bin/bash -l

# #$ -pe omp 8
# #$ -l gpus=1
# #$ -l gpu_type=L40S
# #$ -l h_rt=8:00:00
# #$ -N $name
# #$ -o $OUTPUT_PATH/\$JOB_NAME.o\$JOB_ID
# #$ -e $OUTPUT_PATH/\$JOB_NAME.e\$JOB_ID

# source activate_env.sh

# python main.py \
#     logging.wandb_project=$project \
#     logging.wandb_name=\$JOB_NAME \
#     logging.wandb_runid=\$JOB_ID \
#     logging.wandb_expname=$name \
#     logging.log_callback_data=$log_data \
#     train.max_steps=$steps \
#     train.use_amp=$use_amp \
#     dataset.total_batch_size=$batch_size \
#     random_seed=$seed \
#     optimizer=$optimizer \
#     optimizer.beta1=$b1 \
#     optimizer.beta2=$b2 \
#     optimizer.weight_decay=$wd \
#     optimizer.use_nesterov=$nesterov \
#     optimizer/lr_config=$schedule \
#     optimizer.lr_config.lr=$lr \
#     optimizer.lr_config.warmup=$warmup \
#     optimizer.lr_config.const=$wait \
#     optimizer.lr_config.max_steps=$steps
# EOF
# echo "Submitted job: $name"



#--------------------------------------------------------------------------------------------------
# Batch submit script
NODES=1
GPU="L40S"
TIME="4:00:00"
BASE_DIR=/projectnb/aclab/qinziz/trainit
DATE=$(date +"%Y-%m-%d")
OUTPUT_PATH=$BASE_DIR/scc_outputs/$DATE/$name

mkdir -p $OUTPUT_PATH

submit_job() {
    local args=("$@")
    job_output=$(qsub <<EOF
#!/bin/bash -l
#$ -pe omp 8
#$ -l h="!scc-506"          # Blacklists bad nodes
#$ -l gpus=${NODES}
#$ -l gpu_type=${GPU}       # Specifies the gpu type
#$ -l h_rt=${TIME}          # Specifies the hard time limit for the job
#$ -N "$name"
#$ -o $OUTPUT_PATH/\$JOB_NAME.o\$JOB_ID
#$ -e $OUTPUT_PATH/\$JOB_NAME.e\$JOB_ID

cd ${BASE_DIR}
source activate_env.sh
python main.py ${args[@]}
EOF
    )
    # Extract job id and log the submission.
    job_id=$(echo "$job_output" | awk '{print $3}')
    echo "$(date '+%Y-%m-%d %H:%M:%S') job_id: ${job_id} || ${name}" >> "${OUTPUT_PATH}/job_list.txt"
    echo "Submitted job: $name"
}

#--------------------------------------------------------------------------------------------------
project=greedy_lr_schedule
steps=2000
batch_size=128
use_amp=False
log_data=False
seed=42

optimizer=adamw
b1=0.9
b2=0.999
wd=0.1
nesterov=False

# name=baseline
# lr=1e-3

# schedule=linear
# warmup=200
# waits=(0 500 1000 1600)
# wait=${waits[2]}
# name=trapezoid3

# schedule=cosine
# warmup=200
# name=cosine

lrs=(1e-3 1.25e-3 0.8e-3 1.5e-3 0.67e-3 2e-3 0.5e-3 3e-3 0.33e-3 5e-3 0.2e-3)


for lr in "${lrs[@]}"; do
    schedule=linear
    warmup=200
    wait=1600
    name="B_trapezoid_lr${lr}"
    args=(
        "logging.wandb_project=$project"
        "logging.wandb_name=\$JOB_NAME"
        "logging.wandb_runid=\$JOB_ID"
        "logging.wandb_expname=$name"
        "logging.log_callback_data=$log_data"
        "train.max_steps=$steps"
        "train.use_amp=$use_amp"
        "dataset.total_batch_size=$batch_size"
        "random_seed=$seed"
        "optimizer=$optimizer"
        "optimizer.beta1=$b1"
        "optimizer.beta2=$b2"
        "optimizer.weight_decay=$wd"
        "optimizer.use_nesterov=$nesterov"
    )
    args+=(
        "optimizer/lr_config=$schedule"
        "optimizer.lr_config.lr=$lr"
        "optimizer.lr_config.warmup=$warmup"
        "optimizer.lr_config.const=$wait"
        "optimizer.lr_config.max_steps=$steps"
    )
    submit_job ${args[@]}
done