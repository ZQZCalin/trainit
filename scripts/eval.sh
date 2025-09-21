#!/bin/bash

# CHANGE THIS
BASE_DIR=/projectnb/aclab/qinziz/trainit
DATE=$(date +"%Y-%m-%d")
OUTPUT_PATH=$BASE_DIR/scheduler_outputs/$DATE/eval_2
mkdir -p $OUTPUT_PATH

# CHANGE THIS: a list of string of form "DIR CKPT NAME"
CKPT_LIST=(
    "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-12/step2k_seg4_lr1e-3_grid20_eps0.0_70e394/checkpoint/1500-2000/lr2:1.00e-04 iter_2000_model.ckpt step2k_seg4_lr1e-3_grid20_eps0.0"
    "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-11/step2k_seg4_lr1e-3_grid20_eps0.06_09278e/checkpoint/1500-2000/lr2:0.00e+00 iter_2000_model.ckpt step2k_seg4_lr1e-3_grid20_eps0.06"
    "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-12/step2k_seg4_lr1e-3_grid20_eps0.12_b147e3/checkpoint/1500-2000/lr2:0.00e+00 iter_2000_model.ckpt step2k_seg4_lr1e-3_grid20_eps0.12"
    "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-13/step2k_seg4_lr1e-3_grid20_eps0.24_f01f5d/checkpoint/1500-2000/lr2:0.00e+00 iter_2000_model.ckpt step2k_seg4_lr1e-3_grid20_eps0.24"
    "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-19/step2k_seg10_lr1e-3_grid20_eps0.0_9e290b/checkpoint/1800-2000/lr2:5.40e-05 iter_2000_model.ckpt step2k_seg10_lr1e-3_grid20_eps0.0"
    "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-14/step2k_seg10_lr1e-3_grid20_eps0.06_1df4fd/checkpoint/1800-2000/lr2:0.00e+00 iter_2000_model.ckpt step2k_seg10_lr1e-3_grid20_eps0.06"
    "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-15/step2k_seg10_lr1e-3_grid20_eps0.12_b66742/checkpoint/1800-2000/lr2:0.00e+00 iter_2000_model.ckpt step2k_seg10_lr1e-3_grid20_eps0.12"
    "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-16/step2k_seg10_lr1e-3_grid20_eps0.24_96bb10/checkpoint/1800-2000/lr2:0.00e+00 iter_2000_model.ckpt step2k_seg10_lr1e-3_grid20_eps0.24"
    "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-16/step2k_seg10_lr1e-3_grid20_eps0.24decay_d0f502/checkpoint/1800-2000/lr2:0.00e+00 iter_2000_model.ckpt step2k_seg10_lr1e-3_grid20_eps0.24decay"
    "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-17/step2k_seg10_lr1e-3_grid20_eps0.48decay_15be1a/checkpoint/1800-2000/lr2:0.00e+00 iter_2000_model.ckpt step2k_seg10_lr1e-3_grid20_eps0.48decay"
    "/projectnb/aclab/qinziz/trainit/scheduler_outputs/2025-09-18/step2k_seg10_lr1e-3_grid20_eps0.96decay_71b436/checkpoint/1800-2000/lr2:0.00e+00 iter_2000_model.ckpt step2k_seg10_lr1e-3_grid20_eps0.96decay"
)

# NO NEED TO CHANGE BELOW
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