#!/bin/bash
# Update next segment lrs script.


echo "$(date '+%Y-%m-%d %H:%M:%S') - Update: fetching lr1 and lr2_candidates for the next segment..."

# Capture the JSON output from the Python script
output=$(python3 scripts/schedule/get_next_lr.py --project $project --job_ids ${received_jobs[@]})

# Extract the integer
lr1=$(echo "$output" | jq -r '.lr1')

# Extract the list into a Bash array
# This uses `jq` to output each element on a new line, then reads them into an array
mapfile -t lr2_candidates < <(echo "$output" | jq -r '.lr2_candidates[]')

echo "$(date '+%Y-%m-%d %H:%M:%S') - Update: lr1=${lr1}, lr2_candidates=(${lr2_candidates[@]}) for segment $((i+1))."