#!/bin/bash
# Muon baseline.

optimizer=muon_inverse

# ========================================================================
# Global Configs.
# ========================================================================
project=pile_baseline

log_data=False
use_amp=True
steps=2000
batch_size=128

schedule=linear
warmup=200
const=null

# Make scc_outputs dir.
exp_name=muon_inverse
BASE_DIR=/projectnb/aclab/qinziz/trainit
DATE=$(date +"%Y-%m-%d")
OUTPUT_PATH=$BASE_DIR/scc_outputs/$DATE/$exp_name

mkdir -p $OUTPUT_PATH


# ========================================================================
# Optimizer Configs.
# ========================================================================

# inverse_k=0.7
# lrs=(0.03 0.01 3e-3 1e-3 0.1 0.3 1.0)
# lr=${lrs[5]}
# name="muon_inverse-k${inverse_k}_lr${lr}"

# inverse_k=1.4
# lrs=(0.03 0.01 0.1)
# lr=${lrs[2]}
# name="muon_inverse-k${inverse_k}_lr${lr}"

# inverse_k=2.3
# lrs=(0.03 0.01 0.1)
# lr=${lrs[2]}
# name="muon_inverse-k${inverse_k}_lr${lr}"

# inverse_k=4.6
# lrs=(0.03 0.01 0.1)
# lr=${lrs[2]}
# name="muon_inverse-k${inverse_k}_lr${lr}"

# scale_rms=True
# ks=(0.7 1.4 2.3 4.6)
# inverse_k=${ks[3]}
# lrs=(0.03 1e-3 3e-3)     # 0.03/sqrt(768) ~= 0.001
# lr=${lrs[-1]}
# name="muonrms_inverse-k${inverse_k}_lr${lr}"

# inverse_k=0
# # lr=0.03
# # name="muon_inverse-k${inverse_k}_lr${lr}"
# scale_rms=True
# lr=1e-3
# name="muonrms_inverse-k${inverse_k}_lr${lr}"

# scale_rms=True
# inverse_k=2.3
# beta2_list=(0.95 0.99 0.9)
# beta2=${beta2_list[0]}
# lrs=(1e-3 3e-3 3e-4)
# lr=${lrs[2]}
# name="precmuonrms_inverse-k${inverse_k}_beta2${beta2}_lr${lr}"

# ========================================================================
# Submit function.
# ========================================================================
parse() {
    # $1: config string
    # $2: variable name
    if [ -n "${!2+x}" ]; then
        echo "$1=${!2}"
    fi
}

# project=null    # test purpose
args=(
    "logging.wandb_project=$project"
    "logging.wandb_name=$name"
    "logging.log_callback_data=$log_data"
    "train.max_steps=$steps"
    "train.use_amp=$use_amp"
    "dataset.total_batch_size=$batch_size"
    "experimental=null"
    "test=null"
    "optimizer=$optimizer"
    # lr schedule
    "optimizer/lr_config=$schedule"
    "optimizer.lr_config.warmup=$warmup"
    "optimizer.lr_config.const=$const"
    "optimizer.lr_config.max_steps=$steps"
    "$(parse "optimizer.lr_config.lr" "lr")"
    "$(parse "optimizer.adam_lr" "adam_lr")"
)

optimizer_keys=(
    "momentum"
    "nesterov"
    "inverse_k"
    "scale_rms"
    "ns_steps"
    "beta2"
    "adam_beta1"
    "adam_beta2"
    "adam_eps"
    "adam_wd"
)

# Update optimizer configs
for key in "${optimizer_keys[@]}"; do
    args+=( "$(parse "optimizer.config.${key}" "${key}")" )
done

# python main.py ${args[@]}

job_output=$(qsub <<EOF
#!/bin/bash -l

#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_type=L40S     # Specifies the gpu type.
#$ -l h_rt=8:00:00      # Specifies the hard time limit for the job
#$ -N "$name".sh
#$ -o $OUTPUT_PATH/\$JOB_NAME.o\$JOB_ID     # Escape environment variables with \$
#$ -e $OUTPUT_PATH/\$JOB_NAME.e\$JOB_ID

sleep $(((RANDOM % 1000) / 100))   # Prevents simultaneous reads of loadit dataset

source activate_env.sh
python main.py ${args[@]}
EOF
)

# Save job id and associated name to local .txt
# This is extremely helpful to manage a bunch of experiments.
job_id=$(echo "$job_output" | awk '{print $3}')
echo "$(date '+%Y-%m-%d %H:%M:%S') job_id: ${job_id} || ${name}" >> "${OUTPUT_PATH}/job_list.txt"

echo "Submitted job: $name"