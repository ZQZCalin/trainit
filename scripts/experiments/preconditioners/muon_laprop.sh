#!/bin/bash

project=pile_baseline
log_data=False
steps=2000
batch_size=128

optimizer=muon_laprop
lr=0.01
momentum=0.95
nesterov=True
ns_steps=6
eps=1e-8
lr_1d=3e-4
# lr_1d=$lr
beta2=0.95
# offset_beta=0.99
offset_beta=null

schedule=linear
warmup=200
const=null

name=muon_laprop_no_offset

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
    logging.wandb_name="$name" \
    logging.log_callback_data=$log_data \
    train.max_steps=$steps \
    dataset.total_batch_size=$batch_size \
    optimizer=$optimizer \
    optimizer.momentum=$momentum \
    optimizer.nesterov=$nesterov \
    optimizer.ns_steps=$ns_steps \
    optimizer.eps=$eps \
    optimizer.lr_1d=$lr_1d \
    optimizer.beta2=$beta2 \
    optimizer.offset_beta=$offset_beta \
    optimizer/lr_config=$schedule \
    optimizer.lr_config.lr=$lr \
    optimizer.lr_config.warmup=$warmup \
    optimizer.lr_config.const=$const \
    optimizer.lr_config.max_steps=$steps
EOF
echo "Submitted job: $name"