#!/bin/bash
# Muon baseline.

optimizer=muon_stable

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
exp_name=muon_stable
BASE_DIR=/projectnb/aclab/qinziz/trainit
DATE=$(date +"%Y-%m-%d")
OUTPUT_PATH=$BASE_DIR/scc_outputs/$DATE/$exp_name

mkdir -p $OUTPUT_PATH


# ========================================================================
# Optimizer Configs.
# ========================================================================

# ns_methods=(
#     "muon"
#     "muon_OG"
#     "k4.6"
#     "k2.3"
#     "k10-b0.01"
#     "k20-b0.01"
# )
# ns_name=${ns_methods[0]}
# lr=1e-3
# scale_rms=True
# # lr=0.03
# # scale_rms=False
# name="muon_stable-${ns_name}_rms-${scale_rms}_lr${lr}"
# # ns_normalize_ord=2
# # name="muon-stable-${ns_name}_ns-spectral_rms-${scale_rms}_lr${lr}"

# ns_name="muon_OG"
# beta2=0.95
# precond_power=0.5
# postcond_power=0
# scale_rms=True
# lr=1e-3
# # scale_rms=False
# # lr=0.03
# name="muon_stable-${ns_name}_rms-${scale_rms}_p${precond_power}-${postcond_power}_lr${lr}"

# ns_name="muon_OG"
# beta2=0.95
# precond_power=0.5
# precond_debias=True
# postcond_power=0
# scale_rms=False
# lr=0.03
# name="muon_stable_recovers_precmuon_baseline"

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
    "ns_name"
    "ns_normalize_ord"
    # Conditioning
    "beta2"
    "precond_power"
    "precond_eps"
    "precond_debias"
    "postcond_power"
    "postcond_eps"
    "postcond_debias"
    # Adamw
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