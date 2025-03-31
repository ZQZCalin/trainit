#!/bin/bash

source scripts/schedule/config.sh

mkdir -p "$SCC_OUTPUT_PATH"

qsub -l h_rt="$CPU_HOUR" \
    -N "${NAME}_resume" \
    -o "${SCC_OUTPUT_PATH}/resume.o\$JOB_ID" \
    -e "${SCC_OUTPUT_PATH}/resume.e\$JOB_ID" \
    -v CPU_HOUR,NAME,SCC_OUTPUT_PATH \
    scripts/schedule/resume_master.sh

echo "Submitted the resume_master script for experiment ${NAME}."