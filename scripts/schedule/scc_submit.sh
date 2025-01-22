#!/bin/bash

source scripts/schedule/config.sh
OUTPUT_PATH="${BASE_PATH}/scc_outputs/${DATE}/${NAME}"

mkdir -p "$OUTPUT_PATH"

qsub -l h_rt="$CPU_HOUR" \
    -N "${NAME}_master" \
    -o "${OUTPUT_PATH}/master.o\$JOB_ID" \
    -e "${OUTPUT_PATH}/master.e\$JOB_ID" \
    -v CPU_HOUR,NAME,OUTPUT_PATH \
    scripts/schedule/master.sh

echo "Submitted the master script for experiment ${NAME}."