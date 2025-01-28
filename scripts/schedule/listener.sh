#!/bin/bash
# Listener script.


lr2_candidates=$1
source scripts/schedule/config.sh
source scripts/schedule/utils.sh
source scripts/schedule/submit_job.sh

# number of expected ACK tokens, 
# equal to number of parallel jobs
expected_acks=${#lr2_candidates[@]}
# expected_acks=1     # uncomment for test purpose

echo "$(date '+%Y-%m-%d %H:%M:%S') - Listener: Waiting for ${expected_acks} ACKs on port ${PORT}..."

received_acks=0
received_jobs=()
start_time=$(date +%s)

while (( received_acks < expected_acks )); do
    # Optional timeout mechanism that breaks after a period of time
    current_time=$(date +%s)
    elapsed=$(( current_time-start_time ))
    if (( elapsed >= MAX_LISTEN_TIME )); then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Listener: Timeout reached after ${elapsed} seconds. Exiting loop."
        break
    fi

    # Listen for one connection and read the incoming message
    # Attempt every 5 seconds to enable timeout
    msg=$(timeout 5 nc -l $port)
    
    # Parse the message, expecting format: "ACK <JOB_ID>"
    # Split message into an array on whitespace
    read -r ack job_id state <<< "$msg"

    if [[ "$ack" == "" ]]; then
        :   # do nothing if receiving empty message
    elif [[ "$ack" != "ACK" ]]; then
        log_info "Listener: Unexpected message: ${msg}"
    elif [[ "$state" -eq 0 ]]; then
        log_info "Listener: Received ACK ${received_acks}/${expected_acks} from job ID ${job_id}"
        # Increment received_acks
        ((received_acks++))
        # Store job id
        received_jobs+=($job_id)
    else
        # Re-submit failed jobs
        log_info "Listener: Job ${job_id} failed, resubmitting..."
        temp_job_config="${SCC_OUTPUT_PATH}/tmp/${job_id}.json"
        lr1=$(jq -r '.lr1' "$temp_job_config")
        lr2=$(jq -r '.lr2' "$temp_job_config")
        seg=$(jq -r '.seg' "$temp_job_config")
        submit_job $lr1 $lr2 $seg
    fi
done

echo "$(date '+%Y-%m-%d %H:%M:%S') - Listener: ${received_acks} / ${expected_acks} ACKs received from jobs (${received_jobs[@]})."