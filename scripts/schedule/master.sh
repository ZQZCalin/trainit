#!/bin/bash -l
# Master script.

# Load environment
module load python3/3.10.12
source env/bin/activate

# Configuration
source scripts/schedule/config.sh
# import log_info()
source scripts/schedule/utils.sh
# import submit_job()
source scripts/schedule/submit_job.sh
# import snapshot()
source scripts/schedule/snapshot.sh


# Create temporary folder for system files.
mkdir -p "${SCC_OUTPUT_PATH}/tmp"

echo "Running experiment ${NAME}." && echo "${DESC}"
echo "master host ip: ${MASTER_HOST}; port number: ${PORT}"

# Take a snapshot of experiment configs.
snapshot_path="${SCC_OUTPUT_PATH}/snapshot.txt"
if [ ! -f $snapshot_path ]; then
  snapshot $snapshot_path
  log_info "[Master]: Save config snapshot to ${snapshot_path}."
fi

# Initialize learning rates.
log_info "[Update]: fetching initial lr1 and lr2_candidates..."
output=$(python3 scripts/schedule/get_next_lr.py \
    --project $PROJECT --job_ids ${received_jobs[@]} --seg 0 --num_segs ${NUM_SEGMENTS})

lr1=$(echo "$output" | jq -r '.lr1')
mapfile -t lr2_candidates < <(echo "$output" | jq -r '.lr2_candidates[]')

log_info "[Update]: Initialize lr1=${lr1}, lr2_candidates=(${lr2_candidates[@]}) for segment 1."

# Master thread
for (( i=0; i < ${#SEGMENTS[@]}-1; i++ )); do
    # Start of segment
    printf '=%.0s' {1..100} && printf "\n"
    log_info "[Master]: Training segment $((i+1)) from iteration ${SEGMENTS[$i]} to ${SEGMENTS[$((i+1))]}..."

    # ====================================================================
    # Submit GPU jobs and track with a listener
    # ====================================================================

    # number of expected ACK tokens, 
    # equal to number of parallel jobs
    expected_acks=${#lr2_candidates[@]}
    # expected_acks=1     # uncomment for test purpose

    log_info "[Listener]: Waiting for ${expected_acks} ACKs on port ${PORT}..."

    received_acks=0
    received_jobs=()
    start_time=$(date +%s)

    # Initialize a dictionary to track number of resubmits
    declare -A job_retries
    for lr2 in "${lr2_candidates[@]}"; do
        job_retries[$lr2]=0
    done

    # a copy of lr2_candidates
    job_queue=( "${lr2_candidates[@]}" )

    while (( received_acks < expected_acks )); do
        # Parallel submit all jobs
        for lr2 in "${job_queue[@]}"; do
            submit_job "$lr1" "$lr2" "$i" false
        done
        job_queue=()    # make sure only submit once

        # Optional timeout mechanism that breaks after a period of time
        current_time=$(date +%s)
        elapsed=$(( current_time-start_time ))
        if (( elapsed >= MAX_LISTEN_TIME )); then
            log_info "[Listener]: Timeout reached after ${elapsed} seconds. Exiting loop."
            break
        fi

        # Listen for one connection and read the incoming message
        # Attempt every X seconds to enable timeout
        msg=$(timeout $LISTENER_BACKOFF nc -l $PORT)
        
        # Parse the message, expecting format: "ACK <JOB_ID>"
        # Split message into an array on whitespace
        read -r ack job_id state <<< "$msg"

        if [[ "$ack" == "" ]]; then
            :   # do nothing if receiving empty message
        elif [[ "$ack" != "ACK" ]]; then
            log_info "[Listener]: Unexpected message: ${msg}"
        elif [[ "$state" -eq 0 ]]; then
            # Increment received_acks
            ((received_acks++))
            # Store job id
            received_jobs+=($job_id)
            log_info "[Listener]: Received ACK ${#received_jobs[@]}/${expected_acks} from job ID ${job_id}"
        else
            log_info "[Listener]: Job ${job_id} failed, resubmitting..."
            # submit_job() creates a temperary config file for resubmission
            # Load these configs and resubmit
            temp_job_config="${SCC_OUTPUT_PATH}/tmp/${job_id}.json"
            lr1=$(jq -r '.lr1' "$temp_job_config")
            lr2=$(jq -r '.lr2' "$temp_job_config")
            seg=$(jq -r '.seg' "$temp_job_config")
            job_retries[$lr2]=$(( job_retries[$lr2] + 1 ))
            # Re-submit failed jobs if within MAX_RETRIES
            if (( job_retries[$lr2] <= MAX_RETRIES )); then
                # Clean up the checkpoint subdir before resubmitting
                submit_job $lr1 $lr2 $seg true
            else
                log_info "(warning) [Listener]: Segment ${seg} lr1=${lr1} lr2=${lr2} failed ${job_retries[$lr2]} times."
                ((received_acks++))
            fi
        fi
    done

    log_info "[Listener]: ${received_acks} / ${expected_acks} ACKs received from jobs (${received_jobs[@]})."

    
    # ====================================================================
    # Initialize / Update learning rates
    # ====================================================================
    log_info "[Update]: computing lr1 and lr2_candidates..."

    # Capture the JSON output from the Python script
    #   received_jobs is undefined in segment 1, thus triggers the default_lr functions
    output=$(python3 scripts/schedule/get_next_lr.py \
        --project $PROJECT --job_ids ${received_jobs[@]} --seg $((i+1)) --num_segs ${NUM_SEGMENTS})

    # If get_next_lr.py returns an error state, exit the main script.
    if [ $? -ne 0 ]; then
        log_info "[Master]: Having a critical error, exiting..."
        exit 1
    fi

    # Extract lr1
    lr1=$(echo "$output" | jq -r '.lr1')

    # Extract the list into a Bash array
    # This uses `jq` to output each element on a new line, then reads them into an array
    mapfile -t lr2_candidates < <(echo "$output" | jq -r '.lr2_candidates[]')

    log_info "[Update]: lr1=${lr1}, lr2_candidates=(${lr2_candidates[@]}) for segment $((i+2))."

    # (Optional) delete checkpoints in other runs
    if [[ $CLEAN_CHECKPOINTS ]]; then
        prev_checkpoint_path="${CHECKPOINT_PATH}/${SEGMENTS[$i]}-${SEGMENTS[$((i+1))]}"
        keep_checkpoint="lr2:$(printf "%.2e" "$lr1")"
        
        log_info "[Master]: cleaning checkpoints other than ${prev_checkpoint_path}/${keep_checkpoint}"
        for sub in "$prev_checkpoint_path"/*; do
            # Check if it's a directory and not the one we want to keep
            if [[ -d "$sub" && "$(basename "$sub")" != "$keep_checkpoint" ]]; then
                rm -rf "$sub"
            fi
        done
    fi

    # ====================================================================
    # End of segment
    log_info "[Master]: Segment $((i+1))/${NUM_SEGMENTS} completed." && echo ""
done

# TODO: process the last segment; fetch optimal runs and locally merge into one plot of loss and lr schedule
log_info "[Master]: Summarizing the experiment..."
python3 scripts/schedule/summarize.py \
    --name "${NAME}" --desc "${DESC}" --ckpt "${CHECKPOINT_PATH}" --proj "${PROJECT}"