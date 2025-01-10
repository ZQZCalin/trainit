#!/bin/bash

project=pile_baseline
log_data=True
steps=2000
batch_size=128

optimizer=adamw
wd=0.1
nesterov=False

schedule=linear
# lr=1e-3
lrs=(1e-2 3e-3 1e-3 3e-4 1e-4)
warmup=200

BASE_DIR=/projectnb/aclab/qinziz/trainit
DATE=$(date +"%Y/%m/%d")

for lr in "${lrs[@]}"
do
    qsub <<EOF
#!/bin/bash -l

#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_type=L40S     # Specifies the gpu type.
#$ -l h_rt=8:00:00      # Specifies the hard time limit for the job
#$ -N "$optimizer"_"$lr".sh
#$ -o $BASE_DIR/scc_outputs/$DATE/$optimizer/$JOB_NAME.o$JOB_ID
#$ -e $BASE_DIR/scc_outputs/$DATE/$optimizer/$JOB_NAME.e$JOB_ID

sleep $(((RANDOM % 1000) / 100))   # Prevents simultaneous reads of loadit dataset

source activate_env.sh

python main.py \
    logging.wandb_project=$project \
    logging.wandb_name="$optimizer"_lr:"$lr" \
    logging.log_callback_data=$log_data \
    train.max_steps=$steps \
    dataset.total_batch_size=$batch_size \
    optimizer=$optimizer \
    optimizer.weight_decay=$wd \
    optimizer.use_nesterov=$nesterov \
    optimizer/lr_config=$schedule \
    optimizer.lr_config.lr=$lr \
    optimizer.lr_config.warmup=$warmup \
    optimizer.lr_config.max_steps=$steps
EOF
    echo "Submitted job: $optimizer lr=$lr"
done