#!/bin/bash
# Parallel submit script.

# Compute variables
schedule=piecewise_linear

segment_steps=$((end_step-start_step))

# saving
save_path_base="${checkpoint_path}/${start_step}-${end_step}"

# loading
if (( i > 0 )); then
    start_step_prev=${segments[$((i-1))]}
    load=True
    load_path="${checkpoint_path}/${start_step_prev}-${start_step}/lr2:${lr1}"
    load_file="iter_${start_step}.ckpt"
else
    load=False
    load_path=""
    load_file=""
fi


# Parallel submit
OUTPUT_PATH="${BASE_PATH}/scc_outputs/${DATE}/${NAME}/seg$((i+1))"
mkdir -p $OUTPUT_PATH

# Template to test communications
for lr2 in "${lr2_candidates[@]}"; do
    job_name="seg$((i+1))_lr2_${lr2}"
    save_path="${save_path_base}/lr2:${lr2}"
    
    # generate random number in fractions
    rand_int=$(((RANDOM % 10000)))
    sleep_time=$(echo "scale=3; ($rand_int/1000)" | bc)

    # submit training script
    qsub <<EOF
#!/bin/bash -l

#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_type=$GPU_TYPE
#$ -l h_rt=$GPU_HOUR
#$ -N "$job_name"
#$ -o $OUTPUT_PATH/\$JOB_NAME.o\$JOB_ID     # Escape environment variables with \$
#$ -e $OUTPUT_PATH/\$JOB_NAME.e\$JOB_ID

cd $BASE_PATH

echo "\$(date '+%Y-%m-%d %H:%M:%S') - Running job \$JOB_ID..."

# Manual sleep to prevent simultaneous LoadIt reads.
echo "\$(date '+%Y-%m-%d %H:%M:%S') - Sleeping ${sleep_time} seconds..."
sleep "${sleep_time}"

# Main python training script.
source activate_env.sh
python main.py \
    logging.wandb_project=$project \
    logging.wandb_name="$NAME_\$JOB_NAME" \
    logging.wandb_runid=\$JOB_ID \
    logging.log_callback_data=$log_callback_data \
    train.max_steps=$total_steps \
    dataset.total_batch_size=$batch_size \
    random_seed=$random_seed \
    optimizer=$optimizer \
    optimizer.beta1=$beta1 \
    optimizer.beta2=$beta2 \
    optimizer.weight_decay=$weight_decay \
    optimizer.use_nesterov=$nesterov \
    optimizer/lr_config=$schedule \
    optimizer.lr_config.lr1=$lr1 \
    optimizer.lr_config.lr2=$lr2 \
    optimizer.lr_config.start_steps=$start_step \
    optimizer.lr_config.max_steps=$segment_steps \
    checkpoint.save=True \
    checkpoint.save_path=$save_path \
    checkpoint.save_steps=[$end_step] \
    checkpoint.num_steps=$segment_steps \
    checkpoint.load=$load \
    checkpoint.load_path=$load_path \
    checkpoint.load_file=$load_file \
    checkpoint.overwrite_config=True

# Send ACK token to listener.
echo "ACK \$JOB_ID" | nc "$master_host" "$port"

echo "\$(date '+%Y-%m-%d %H:%M:%S') - Job \$JOB_ID completed."
EOF

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Submitter: Submitted job with lr1=${lr1} lr2=${lr2}."
done