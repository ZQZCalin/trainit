#!/bin/bash
# Listener script.

# number of expected ACK tokens, 
# equal to number of parallel jobs
expected_acks=${#lr2_candidates[@]}
# expected_acks=1     # uncomment for test purpose

echo "$(date '+%Y-%m-%d %H:%M:%S') - Listener: Waiting for ${expected_acks} ACKs on port ${port}..."

received_acks=0
received_jobs=()
start_time=$(date +%s)

while (( received_acks < expected_acks )); do
    # Optional timeout mechanism that breaks after a period of time
    current_time=$(date +%s)
    elapsed=$(( current_time-start_time ))
    if (( elapsed >= max_wait_time )); then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Listener: Timeout reached after ${elapsed} seconds. Exiting loop."
        break
    fi

    # Listen for one connection and read the incoming message
    # Attempt every 5 seconds to enable timeout
    msg=$(timeout 5 nc -l $port)
    
    # Parse the message, expecting format: "ACK <JOB_ID>"
    # Split message into an array on whitespace
    read -r ack job_id <<< "$msg"
    
    if [[ "$ack" == "ACK" ]]; then
        # Increment received_acks
        ((received_acks++))
        # Store job id
        received_jobs+=($job_id)
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Listener: Received ACK ${received_acks}/${expected_acks} from job ID ${job_id}"
    elif [[ "$ack" != "" ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Listener: Unexpected message: ${msg}"
    fi
done

echo "$(date '+%Y-%m-%d %H:%M:%S') - Listener: ${received_acks} / ${expected_acks} ACKs received from jobs (${received_jobs[@]})."