#!/bin/bash

# Testing next_lr python script
i=3
path=/SOME/PATH
echo $(python3 scripts/schedule/get_next_lr.py --segment $i --path $path)

# Testing fraction division
# numerator1=3 
# denominator1=4 
# numerator2=2 
# denominator2=5

# rand=$(((RANDOM % 10000)))
# echo $rand

# result=$(echo "scale=3; ($rand/1000)" | bc)

# echo "Result: $result" 

# Testing sleep mechanism
# echo $(date '+%Y-%m-%d %H:%M:%S')
# sleep_time=$(((RANDOM % 1000) / 100))
# echo "${sleep_time}"
# sleep "${sleep_time}"
# echo $(date '+%Y-%m-%d %H:%M:%S')

# Testing ACK communication
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