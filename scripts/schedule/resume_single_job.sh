#!/bin/bash -l
# Resume (resubmit) one single job


source scripts/schedule/config.sh
source scripts/schedule/submit_job.sh


# this feels dumb, but we need to manually change the DATE
SCC_OUTPUT_PATH=/projectnb/aclab/qinziz/trainit/scc_outputs/2025-04-21/v4_10seg_peak2e-3_eps0.03
job_id=4273194          # fetch the job that fails; you can find it under the `tmp/` folder


temp_job_config="${SCC_OUTPUT_PATH}/tmp/${job_id}.json"
lr1=$(jq -r '.lr1' "$temp_job_config")
lr2=$(jq -r '.lr2' "$temp_job_config")
seg=$(jq -r '.seg' "$temp_job_config")
submit_job $lr1 $lr2 $seg true