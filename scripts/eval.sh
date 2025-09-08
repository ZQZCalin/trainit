#!/bin/bash

# CHANGE THIS
BASE_DIR=/projectnb/aclab/qinziz/trainit
OUTPUT_PATH=$BASE_DIR/test_logs/eval

mkdir -p $OUTPUT_PATH

# CHANGE THIS: a list of string of form "DIR CKPT NAME"
CKPT_LIST=(
  "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-05-04/v5_4seg_peak2e-3_eps0.24const_grid10_0a940a/checkpoint/1400-2000/lr2:7.00e-03 iter_2000_model.ckpt seg4_lr2e-3_eps0.24"
)

submit_eval_job() {
    local dir=$1
    local ckpt=$2
    local name=$3

    job_output=$(qsub <<EOF
#!/bin/bash -l
#$ -pe omp 8
#$ -l h="!scc-506"          # Blacklists bad nodes
#$ -l gpus="1"
#$ -l gpu_type="L40S"       # Specifies the gpu type
#$ -l h_rt="4:00:00"        # Specifies the hard time limit for the job
#$ -N "$name"
#$ -o "$OUTPUT_PATH"/${name}.o\$JOB_ID
#$ -e "$OUTPUT_PATH"/${name}.e\$JOB_ID

cd ${BASE_DIR}
module load python3/3.10.12 cuda/12.2
source ${BASE_DIR}/env/bin/activate
python eval.py --dir "${dir}" --ckpt "${ckpt}"
EOF
    )
    # Extract job id and log the submission.
    job_id=$(echo "$job_output" | awk '{print $3}')
    echo "$(date '+%Y-%m-%d %H:%M:%S') job_id: ${job_id} || ${name}" >> "${OUTPUT_PATH}/job_list.txt"
    echo "Submitted job: $job_id $name"
}


# Iterate through the pairs
for pair in "${CKPT_LIST[@]}"; do
    # Split the pair into dir and ckpt
    set -- $pair
    dir=$1
    ckpt=$2
    name=$3
    submit_eval_job $dir $ckpt $name
done