#!/bin/bash -l
# Master script.

# Configuration
source scripts/schedule/config.sh

echo "Running experiment ${NAME}." && echo "${DESC}"

echo "master host ip: ${master_host}; port number: ${port}"

# Master thread
for (( i=0; i < ${#segments[@]}-1; i++ )); do
    start_step=${segments[$i]}
    end_step=${segments[$((i+1))]}

    # Start of segment
    printf '=%.0s' {1..100} && printf "\n"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Master: Training segment $((i+1)) from iteration ${start_step} to ${end_step}..."
    
    # Batch submit jobs in parallel
    source scripts/schedule/launcher.sh

    # Launch a listener for ACK from all jobs
    source scripts/schedule/listener.sh

    # Conclude the segment and update next lr and candidates
    source scripts/schedule/update.sh

    # End of segment
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Master: Segment $((i+1))/${num_segments} completed." && echo ""
done