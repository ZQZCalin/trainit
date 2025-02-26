#!/bin/bash
# Muon baseline.

optimizer=muon

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
exp_name=muon
BASE_DIR=/projectnb/aclab/qinziz/trainit
DATE=$(date +"%Y-%m-%d")
OUTPUT_PATH=$BASE_DIR/scc_outputs/$DATE/$exp_name

mkdir -p $OUTPUT_PATH


# ========================================================================
# Optimizer Configs.
# ========================================================================

# lr=0.01
# adam_lr=0.03
# # name="muon_lr${lr}_adam-lr_${adam_lr}"
# name="muon_baseline"


# lr=0.03
# adam_lr=0.03
# name="muon_lr${lr}_adam-lr${adam_lr}"

# lr=0.03
# adam_lr=0.03
# beta2=0.95
# name="precmuon_lr${lr}_adam-lr${adam_lr}"
# name="precmuon_lr${lr}_adam-lr${adam_lr}_nodebias"

# lr=0.01 #0.03
# adam_lr=0.03
# offset_beta=0.95
# name="muon_lr${lr}_adam-lr${adam_lr}_offset${offset_beta}"

# lr=0.01 #0.03
# adam_lr=0.03
# beta2=0.95
# offset_beta=0.95
# name="precmuon_lr${lr}_adam-lr${adam_lr}_offset${offset_beta}"


# beta2=0.95
# p_pre=0.25
# p_post=0.25
# lrs=(0.03 0.01 3e-3 1e-3 3e-4 1e-4 3e-5)
# lr=${lrs[3]}
# adam_lr=0.03
# # name="muon_p${p_pre}-${p_post}_lr${lr}-${adam_lr}"
# # stabilize="rms"
# # # name="muonstable_p${p_pre}-${p_post}_lr${lr}-${adam_lr}"
# # name="muonstable0.2_p${p_pre}-${p_post}_lr${lr}-${adam_lr}"
# stabilize="mean"
# name="muonstable0.2-mean_p${p_pre}-${p_post}_lr${lr}-${adam_lr}"


# beta2=0.95
# p_pre=0.0
# p_post=0.25
# lrs=(0.03 0.01 3e-3 1e-3 3e-4 1e-4)
# lr=${lrs[3]}
# adam_lr=0.03
# # name="muon_p${p_pre}-${p_post}_lr${lr}-${adam_lr}"
# # name="muonstable_p${p_pre}-${p_post}_lr${lr}-${adam_lr}"
# stabilize="mean"
# name="muonstable0.2-mean_p${p_pre}-${p_post}_lr${lr}-${adam_lr}"


# beta2=0.95
# p_pre=0.5
# p_post=0.5
# lrs=(0.03 0.01 3e-3 1e-3)
# lr=${lrs[3]}
# adam_lr=0.03
# # name="muon_p${p_pre}-${p_post}_lr${lr}-${adam_lr}"
# # name="muonstable_p${p_pre}-${p_post}_lr${lr}-${adam_lr}"
# stabilize="mean"
# name="muonstable0.2-mean_p${p_pre}-${p_post}_lr${lr}-${adam_lr}"


# beta2=0.95
# p_pre=0.0
# p_post=0.5
# lrs=(0.03 0.01 3e-3 1e-3)
# lr=${lrs[3]}
# adam_lr=0.03
# # name="muon_p${p_pre}-${p_post}_lr${lr}-${adam_lr}"
# name="muonstable_p${p_pre}-${p_post}_lr${lr}-${adam_lr}"
# stabilize="mean"
# name="muonstable0.2-mean_p${p_pre}-${p_post}_lr${lr}-${adam_lr}"


# beta2=0.95
# p_pre=0.0
# p_post=0.1
# lrs=(0.03 0.01 3e-3 1e-3)
# lr=${lrs[2]}
# adam_lr=0.03
# # name="muon_p${p_pre}-${p_post}_lr${lr}-${adam_lr}"
# stabilize="rms"
# name="muonstable0.2-rms_p${p_pre}-${p_post}_lr${lr}-${adam_lr}"
# # stabilize="mean"
# # name="muonstable0.2-mean_p${p_pre}-${p_post}_lr${lr}-${adam_lr}"


# beta2=0.95
# p_pre=0.5
# p_post=0.1
# lrs=(0.03 0.01 3e-3 1e-3)
# lr=${lrs[3]}
# adam_lr=0.03
# # name="muon_p${p_pre}-${p_post}_lr${lr}-${adam_lr}"
# stabilize="rms"
# name="muonstable0.2-rms_p${p_pre}-${p_post}_lr${lr}-${adam_lr}"
# # stabilize="mean"
# # name="muonstable0.2-mean_p${p_pre}-${p_post}_lr${lr}-${adam_lr}"


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
)

optimizer_keys=(
    "momentum"
    "nesterov"
    "ns_steps"
    "adam_lr"
    "adam_beta1"
    "adam_beta2"
    "adam_eps"
    "adam_wd"
    "beta2"
    "p_pre"
    "p_post"
    "stabilize"
    "offset_beta"
)

# Update optimizer configs
for key in "${optimizer_keys[@]}"; do
    args+=( "$(parse "optimizer.${key}" "${key}")" )
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