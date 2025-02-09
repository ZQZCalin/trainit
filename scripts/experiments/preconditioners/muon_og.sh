#!/bin/bash

project=pile_baseline
log_data=False
steps=2000
batch_size=128

optimizer=muon_og

lr=0.01
momentum=0.95
nesterov=True
ns_steps=6
ns_embedding=False
ns_head=False

schedule=linear
warmup=200
const=null

exp_name=muon_og

# System variables
BASE_DIR=/projectnb/aclab/qinziz/trainit
DATE=$(date +"%Y-%m-%d")
OUTPUT_PATH=$BASE_DIR/scc_outputs/$DATE/$exp_name

mkdir -p $OUTPUT_PATH


# ========================================================================
# ABOVE are default configurations as comparison baseline.
# BELOW are experiment (variable) configs.
# ========================================================================

# ...
# name="muon_og"

# ...
name="muon_og_nshead"
ns_head=True


args=(
    "logging.wandb_project=$project"
    "logging.wandb_name=$name"
    "logging.log_callback_data=$log_data"
    "train.max_steps=$steps"
    "dataset.total_batch_size=$batch_size"
    "optimizer=$optimizer"
    "optimizer.momentum=$momentum"
    "optimizer.nesterov=$nesterov"
    "optimizer.ns_steps=$ns_steps"
    "optimizer.ns_embedding=$ns_embedding"
    "optimizer.ns_head=$ns_head"
    "optimizer/lr_config=$schedule"
    "optimizer.lr_config.lr=$lr"
    "optimizer.lr_config.warmup=$warmup"
    "optimizer.lr_config.const=$const"
    "optimizer.lr_config.max_steps=$steps"
)

qsub <<EOF
#!/bin/bash -l

#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_type=L40S     # Specifies the gpu type.
#$ -l h_rt=8:00:00      # Specifies the hard time limit for the job
#$ -N "$name".sh
#$ -o $OUTPUT_PATH/\$JOB_NAME.o\$JOB_ID     # Escape environment variables with \$
#$ -e $OUTPUT_PATH/\$JOB_NAME.e\$JOB_ID

sleep $(((RANDOM % 1000) / 100))   # Prevents simultaneous reads of loadit dataset

source activate_env.sh
python main.py ${args[@]}
EOF
echo "Submitted job: $name"