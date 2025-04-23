#!/bin/bash

source scripts/schedule/config.sh

mkdir -p "$OUTPUT_PATH"

qsub -l h_rt="$CPU_HOUR" \
    -N "${NAME}_master" \
    -o "${OUTPUT_PATH}/master.o\$JOB_ID" \
    -e "${OUTPUT_PATH}/master.e\$JOB_ID" \
    -v CPU_HOUR,NAME,OUTPUT_PATH,uuid6 \
    scripts/schedule/master.sh

echo "Submitted the master script for experiment ${NAME}."