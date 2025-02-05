# Contains the submit_job() function.

SCHEDULE="piecewise_linear"

submit_job() {
    local lr1="$1"
    local lr2="$2"
    local seg="$3"                   # index-0 segment number

    local seg_next=$((seg + 1))
    local seg_prev=$((seg - 1))
    local seg_print=$((seg + 1))    # convert to index-1 numbering

    local start_step=${SEGMENTS[$seg]}
    local end_step=${SEGMENTS[$seg_next]}

    local job_name="seg${seg_print}_lr2_$(printf "%.1e" "$lr2")"
    local output_path="${SCC_OUTPUT_PATH}/seg${seg_print}"
    mkdir -p "$output_path"

    # Checkpoint variables
    local save_path="${CHECKPOINT_PATH}/${start_step}-${end_step}/lr2:$(printf "%.1e" "$lr2")"
    if (( seg > 0 )); then
        start_step_prev=${SEGMENTS[$seg_prev]}
        load=True
        load_path="${CHECKPOINT_PATH}/${start_step_prev}-${start_step}/lr2:$(printf "%.1e" "$lr1")"
        load_file="iter_${start_step}.ckpt"
    else
        load=False
        load_path=""
        load_file=""
    fi

    # Generate random sleep time to prevent race conditions
    local rand_int=$(((RANDOM % 6000)))
    local sleep_time=$(echo "scale=3; ($rand_int/100)" | bc)

    # Submit training script
    qsub <<EOF
#!/bin/bash -l

#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_type=$GPU_TYPE
#$ -l h_rt=$GPU_HOUR
#$ -N "$job_name"
#$ -o $output_path/\$JOB_NAME.o\$JOB_ID
#$ -e $output_path/\$JOB_NAME.e\$JOB_ID

cd $BASE_PATH

echo "\$(date '+%Y-%m-%d %H:%M:%S') - Running job \$JOB_ID..."

# Manual sleep to prevent simultaneous LoadIt reads
echo "\$(date '+%Y-%m-%d %H:%M:%S') - Sleeping ${sleep_time} seconds..."
sleep "${sleep_time}"

# Activate environment and run training
echo "\$(date '+%Y-%m-%d %H:%M:%S') - Start training..."
source activate_env.sh
python main.py \
    logging.wandb_project=$PROJECT \
    logging.wandb_name="$NAME_\$JOB_NAME" \
    logging.wandb_runid=\$JOB_ID \
    logging.log_callback_data=$LOG_CALLBACK_DATA \
    train.max_steps=$TOTAL_STEPS \
    dataset.total_batch_size=$BATCH_SIZE \
    random_seed=$RANDOM_SEED \
    optimizer=$OPTIMIZER \
    optimizer.beta1=$BETA1 \
    optimizer.beta2=$BETA2 \
    optimizer.weight_decay=$WEIGHT_DECAY \
    optimizer.use_nesterov=$NESTEROV \
    optimizer/lr_config=$SCHEDULE \
    optimizer.lr_config.lr1=$lr1 \
    optimizer.lr_config.lr2=$lr2 \
    optimizer.lr_config.start_steps=$start_step \
    optimizer.lr_config.max_steps=$((end_step-start_step)) \
    checkpoint.save=True \
    checkpoint.save_path=$save_path \
    checkpoint.save_steps=[$end_step] \
    checkpoint.num_steps=$((end_step-start_step)) \
    checkpoint.load=$load \
    checkpoint.load_path=$load_path \
    checkpoint.load_file=$load_file \
    checkpoint.overwrite_config=True

status=\$?

echo "\$(date '+%Y-%m-%d %H:%M:%S') - Job \$JOB_ID completed with status \$status"

# Create temperary configs for resubmission
if [[ \$status -ne 0 ]]; then
    cat <<JSON > "${SCC_OUTPUT_PATH}/tmp/\$JOB_ID.json"
{
  "lr1": $lr1,
  "lr2": $lr2,
  "seg": $seg
}
JSON
fi

# Send ACK token, together with job ID and exit code
echo "ACK \$JOB_ID \$status" | nc "$MASTER_HOST" "$PORT"
EOF

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Launcher: Submitted job with lr1=${lr1} lr2=${lr2}."
}
