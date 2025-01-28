#!/bin/bash

source scripts/schedule/config.sh

mkdir -p "$SCC_OUTPUT_PATH"

qsub -l h_rt="$CPU_HOUR" \
    -N "${NAME}_master" \
    -o "${SCC_OUTPUT_PATH}/master.o\$JOB_ID" \
    -e "${SCC_OUTPUT_PATH}/master.e\$JOB_ID" \
    -v CPU_HOUR,NAME,SCC_OUTPUT_PATH \
    scripts/schedule/master.sh

echo "Submitted the master script for experiment ${NAME}."