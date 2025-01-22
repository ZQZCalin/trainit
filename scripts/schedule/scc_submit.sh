#!/bin/bash

source scripts/schedule/config.sh
OUTPUT_PATH="${BASE_PATH}/scc_outputs/${DATE}/${NAME}"

qsub scripts/schedule/master.sh \
    -N "${NAME}_master" \
    -o "${OUTPUT_PATH}/${NAME}_master.o" \
    -e "${OUTPUT_PATH}/${NAME}_master.e"

echo "Submitted the master script for experiment ${NAME}."