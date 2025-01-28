#!/bin/bash

source scripts/schedule/utils.sh

output=$(python3 scripts/schedule/get_next_lr.py --job_ids)

# Extract lr1
lr1=$(echo "$output" | jq -r '.lr1')

# Extract the list into a Bash array
# This uses `jq` to output each element on a new line, then reads them into an array
mapfile -t lr2_candidates < <(echo "$output" | jq -r '.lr2_candidates[]')

echo ${lr1}
echo ${lr2_candidates[@]}
log_info "${lr2_candidates[@]}"

x=(1 2 3 4 5)
log_info "${x[@]} bla bla bla"

# >>
# declare -A dict
# list=(0 1 2 3 4)
# for key in "${list[@]}"; do
#     dict["$key"]=0
# done
# echo ${!dict[@]}
# for key in ${!dict[@]}; do
#     echo "${key}: ${dict[$key]}"
# done
# dict[2]=$((dict[2] + 1))
# for key in ${!dict[@]}; do
#     echo "${key}: ${dict[$key]}"
# done

# >>
# lr1=$(jq -r '.lr1' "tmp/test.json")
# echo "lr1=${lr1}"

# echo "list='${received_jobs[@]}'"
# python scripts/schedule/test.py --list ${var[@]}

# >>
# bash <<EOF
# lr1=0.1
# lr2=0.02
# seg=3

# python "raise KeyboardInterrupt"

# status=\$?

# echo "exit code = \$status"
# if [[ \$status -ne 0 ]]; then
#     echo "updated test.json"
#     cat <<JSON > "tmp/test.json"
# {
# "lr1": \$lr1,
# "lr2": \$lr2,
# "seg": \$seg
# }
# JSON
# fi
# EOF


# >> Testing coproc for background script
# coproc PROC { bash scripts/schedule/listener.sh ; }

# echo "id = $PROC_PID"

# while read -r line <&"${PROC[0]}"; do
#   echo "$line"
# done

# echo "processing other jobs..."

# wait $PROC_ID
# echo "child done."


# >> Testing json file reads and loads
# path="tmp/test.json"
# # touch $path
# mkdir -p tmp
# list=(0 1 2 3 4)
# printf '%s\n' "${list[@]}" | jq -R . | jq -s '.' > $path


# >> Testing unquoted EOF
# echo "$(date '+%Y-%m-%d %H:%M:%S')"
# sleep 3
# bash <<EOF
# log_fn() {
#     echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO: \$1"
# }
# log_fn "something"
# EOF


# >> Testing exit code of python script
# python <<EOF
# # raise KeyboardInterrupt
# print("no problem")
# EOF
# echo "exit code = '$?'."


# >> Testing bash functions
# func() {
#     local var1=$1
#     local var2=$2

#     echo "var1 = '${var1}', var2 = '${var2}'"
# }

# func "boo" "foo"


# >> Testing scientific notation
# val=13.456
# echo $(printf "%.1e" "$val")


# >> Testing next_lr python script
# jobs=(1 2 3 4)
# echo $(python3 scripts/schedule/get_next_lr.py --job_ids ${jobs[@]})


# >> Testing fraction division
# numerator1=3 
# denominator1=4 
# numerator2=2 
# denominator2=5

# rand=$(((RANDOM % 10000)))
# echo $rand

# result=$(echo "scale=3; ($rand/1000)" | bc)

# echo "Result: $result" 


# >> Testing sleep mechanism
# echo $(date '+%Y-%m-%d %H:%M:%S')
# sleep_time=$(((RANDOM % 1000) / 100))
# echo "${sleep_time}"
# sleep "${sleep_time}"
# echo $(date '+%Y-%m-%d %H:%M:%S')


# >> Testing ACK communication
# source scripts/schedule/config.sh
# echo "ACK 0000000" | nc "$master_host" "$port"

# text="
# This is a block text.

# You can write description here.
# "

# echo $text

# L=( $(seq 0 100 2000) )

# # Print the array
# echo "L: ${L[@]}"

# checkpoints=(0 100 200 300)

# # Loop through indices from 0 to length-2
# for (( i=0; i < ${#checkpoints[@]}-1; i++ )); do
#     start=${checkpoints[$i]}
#     end=${checkpoints[$((i+1))]}

#     echo "Segment from $start to $end"
#     # Place additional logic for each segment here

# done



# source scripts/schedule/config.sh
# echo segements: ${segments[@]}
# echo $port


# read -r var1 var2 <<< "ACK hello world"

# echo "word1: $var1; word2: $var2"