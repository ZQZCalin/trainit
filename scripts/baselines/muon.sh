#!/bin/bash

project=pile_baseline
name=muon
log_data=True
steps=2000
batch_size=128

optimizer=muon
# muon_lr=0.03
# muon_lrs=(0.3 0.1 3e-2 1e-2 3e-3)
muon_lrs=(0.1 3e-2 1e-2 3e-3 1e-3)
momentum=0.95
# nesterov=True
nesterov=False
ns_steps=6

adam_lr=3e-4
adam_beta1=0.95
adam_beta2=0.95
adam_eps=1e-8
adam_wd=0.0

schedule=linear
warmup=200
wait=0

# System variables
BASE_DIR=/projectnb/aclab/qinziz/trainit
DATE=$(date +"%Y-%m-%d")
OUTPUT_PATH=$BASE_DIR/scc_outputs/$DATE/$name

mkdir -p $OUTPUT_PATH

for muon_lr in "${muon_lrs[@]}"
do
    qsub <<EOF
#!/bin/bash -l

#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_type=L40S     # Specifies the gpu type.
#$ -l h_rt=8:00:00      # Specifies the hard time limit for the job
#$ -N "$name"_"$muon_lr".sh
#$ -o $OUTPUT_PATH/\$JOB_NAME.o\$JOB_ID     # Escape environment variables with \$
#$ -e $OUTPUT_PATH/\$JOB_NAME.e\$JOB_ID

sleep $(((RANDOM % 1000) / 100))   # Prevents simultaneous reads of loadit dataset

source activate_env.sh

python main.py \
    logging.wandb_project=$project \
    logging.wandb_name="$name"_lr:"$muon_lr" \
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
    optimizer.lr_config.const=$wait \
    optimizer.lr_config.max_steps=$steps \
    optimizer.adam_lr=$adam_lr \
    optimizer.adam_beta1=$adam_beta1 \
    optimizer.adam_beta2=$adam_beta2 \
    optimizer.adam_eps=$adam_eps \
    optimizer.adam_wd=$adam_wd
EOF
    echo "Submitted job: $name lr=$muon_lr"
done