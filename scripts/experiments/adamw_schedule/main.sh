#!/bin/bash

project=pile_baseline
name=adamw_trapezoid
log_data=True
steps=2000
batch_size=128

optimizer=adamw
lr=1e-3
wd=0.1
nesterov=False

schedule=trapezoid
warmup=200
decay=200

# System variables
BASE_DIR=/projectnb/aclab/qinziz/trainit
DATE=$(date +"%Y-%m-%d")
OUTPUT_PATH=$BASE_DIR/scc_outputs/$DATE/$name

mkdir -p $OUTPUT_PATH

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

python main.py \
    logging.wandb_project=$project \
    logging.wandb_name="$name"_lr:"$lr" \
    logging.log_callback_data=$log_data \
    train.max_steps=$steps \
    dataset.total_batch_size=$batch_size \
    optimizer=$optimizer \
    optimizer.weight_decay=$wd \
    optimizer.use_nesterov=$nesterov \
    optimizer/lr_config=$schedule \
    optimizer.lr_config.lr=$lr \
    optimizer.lr_config.warmup=$warmup \
    optimizer.lr_config.decay=$decay \
    optimizer.lr_config.max_steps=$steps
EOF
echo "Submitted job: $name"