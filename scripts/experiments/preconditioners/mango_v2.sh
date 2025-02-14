# Mango_v2 experiments.

#!/bin/bash

# project=null
project=pile_baseline

log_data=False
steps=2000
batch_size=128

visualize=False

schedule=linear
warmup=200
const=null


exp_name=mango_v2

# System variables
BASE_DIR=/projectnb/aclab/qinziz/trainit
DATE=$(date +"%Y-%m-%d")
OUTPUT_PATH=$BASE_DIR/scc_outputs/$DATE/$exp_name

mkdir -p $OUTPUT_PATH


# ========================================================================
# Mango optimzier configs
# ========================================================================

optimizer=mango_v2

ns_steps=6
eps=1e-8
offset_beta=0.95
scale_clip_low=1.0
scale_clip_high=null

# Normalization defaults
normalize_mat="ns"
normalize_embedding=null
normalize_head="ns"
normalize_attn_w="ns"
normalize_attn_b="l2"
normalize_vec_w=null
normalize_vec_b="l2"

# Scale by weight defaults
# defaults to same.
# you can turn it off by uncommenting the last line
scale_weight_mat="op"
scale_weight_embedding=null
scale_weight_head="op"
scale_weight_attn_w="op"
scale_weight_attn_b="l2"
scale_weight_vec_w=null
scale_weight_vec_b="l2"
# scale_weight=null



# ========================================================================
# Experiments start here
# ========================================================================

# ...
# aka "mango_wnorm_same_p1"
# name="mango_wnorm_default"

# ...
# scale_weight="l2"
# scale_power=1
# lr=3e-4       # scale by 1/sqrt(d)**p, d=768
# # scale_power=0.75
# # lr=8e-4
# # scale_power=0.5
# # lr=2e-3
# # scale_power=0.75
# # lr=4e-3
# name="mango_norm_global_${scale_weight}_p${scale_power}_lr${lr}"

# ...
# # scale_power=1
# # lr=0.01
# # lr=1e-3
# # scale_power=0.75
# # lr=0.01
# # lr=2e-3
# # scale_power=0.5
# # lr=3e-3
# scale_power=0.25
# # lr=0.01
# lr=6e-3
# name="mango_norm_same_p${scale_power}_lr${lr}"

# ...
# use_adamw=True
# scale_weight=null
# # lr=0.01
# # lr=3e-3
# # lr=1e-3
# # lr=3e-4
# # lr=0.03
# lr=0.1
# name="mango_adamw_lr${lr}"

# ...
# use_adamw=True
# scale_weight=null
# nesterov=False
# # lr=0.01
# lr=0.03
# name="mango_adamw_nones_lr${lr}"


# Feb.12: Experiments with corrected implementation of scale_by_weight_norm
# clips weight_norm
beta1=0.95
beta2=0.95
nesterov=True
use_adamw=False

# ...
# scale_power=1
# lr=0.01
# # lr=3e-3
# # lr=1e-3
# # lr=3e-4
# # lr=1e-4
# scale_power=0.75
# scale_power=0.5
# scale_power=0.25
# name="mango_scale_p${scale_power}_lr${lr}"

# ...
# scale_weight_mat=null
# scale_weight_head=null
# scale_weight_attn_w=null

# scale_power=1
# lr=0.01
# # lr=3e-3
# # lr=1e-3
# scale_power=0.75
# scale_power=0.5
# scale_power=0.25
# name="mango_only_biasl2_p${scale_power}_lr${lr}"

# ...
# scale_weight_head=null

# lr=0.01
# lr=3e-3
# lr=1e-3
# # scale_power=1
# scale_power=0.75
# name="mango_head_noscale_p${scale_power}_lr${lr}"

# ...
# normalize_mat="ns"
# normalize_embedding=null
# normalize_head="ns"
# normalize_attn_w="ns"
# normalize_attn_b="l2"
# normalize_vec_w=null
# normalize_vec_b="l2"

# scale_weight_mat="op"
# scale_weight_embedding=null
# scale_weight_head="op"
# scale_weight_attn_w="op"
# scale_weight_attn_b="l2"
# scale_weight_vec_w=null
# scale_weight_vec_b="l2"

# lr_base=0.01
# lr_base=3e-3
# lr_head=1e-3
# lr_head=3e-4

# lr_mat=$lr_base
# lr_embedding=$lr_base
# lr_attn_w=$lr_base
# lr_attn_b=$lr_base
# lr_vec_w=$lr_base
# lr_vec_b=$lr_base

# scale_power=1
# name="mango_same_p${scale_power}_lr${lr_base}_lrhead${lr_head}"

# ...
lr=0.01
beta1=0.95
beta2=0.95
nesterov=True
use_adamw=False

# scale_weight=null
# name="mangov2_baseline"

# ...
# normalize_head="inf_"
# name="mango_head_inf"


# ...
normalize_mat="Spectral"
normalize_attn_w="Spectral"
normalize_embedding="ColNorm"
normalize_head="Sign"
normalize_vec_w="Sign"
normalize_attn_b="Euclidean"
normalize_vec_b="Euclidean"

# # scale_weight=True
# scale_weight=null
# # scale_dim=True
# scale_dim=False

# name="mango_lmo_weight_${scale_weight}_dim_${scale_dim}"

# ...
normalize_embedding=null
normalize_head="Spectral"
normalize_vec_w=null
scale_weight=null
# scale_dim=False
# clip_ns=True
# name="mango_lmo_recover_mango"
# clip_ns=False
# name="mango_lmo_ns_noscale"
scale_dim_mat=True
scale_dim_attn_w=True
scale_dim_head=True
clip_ns=False
name="mango_lmo_ns_noclip"

# ...
# scale_weight=null
# scale_dim=True
# lr=0.03
# name="mango_lmo_lr${lr}"


# ========================================================================
# Below is submit function. Only change the part related to global
# vs param-wise learning rate.
# ========================================================================

parse() {
    # $1: config string
    # $2: variable name
    if [ -n "${!2+x}" ]; then
        echo "$1=${!2}"
    fi
}

# project=null
args=(
    "logging.wandb_project=$project"
    "logging.wandb_name=$name"
    "logging.log_callback_data=$log_data"
    "train.max_steps=$steps"
    "dataset.total_batch_size=$batch_size"
    "experimental=null"
    "test=null"
    "optimizer=$optimizer"
    # global hyperparameters
    "optimizer.ns_steps=$ns_steps"
    "optimizer.eps=$eps"
    "optimizer.offset_beta=$offset_beta"
    "optimizer.scale_clip_low=$scale_clip_low"
    "optimizer.scale_clip_high=$scale_clip_high"
    "$(parse "optimizer.clip_ns" "clip_ns")"
    # lr schedule
    "optimizer/lr_config=$schedule"
    "optimizer.lr_config.warmup=$warmup"
    "optimizer.lr_config.const=$const"
    "optimizer.lr_config.max_steps=$steps"
    # learning rates
    "$(parse "optimizer.lr.mat" "lr_mat")"
    "$(parse "optimizer.lr.embedding" "lr_embedding")"
    "$(parse "optimizer.lr.head" "lr_head")"
    "$(parse "optimizer.lr.attn_w" "lr_attn_w")"
    "$(parse "optimizer.lr.attn_b" "lr_attn_b")"
    "$(parse "optimizer.lr.vec_w" "lr_vec_w")"
    "$(parse "optimizer.lr.vec_b" "lr_vec_b")"
    # beta1
    "$(parse "optimizer.beta1.mat" "beta1_mat")"
    "$(parse "optimizer.beta1.embedding" "beta1_embedding")"
    "$(parse "optimizer.beta1.head" "beta1_head")"
    "$(parse "optimizer.beta1.attn_w" "beta1_attn_w")"
    "$(parse "optimizer.beta1.attn_b" "beta1_attn_b")"
    "$(parse "optimizer.beta1.vec_w" "beta1_vec_w")"
    "$(parse "optimizer.beta1.vec_b" "beta1_vec_b")"
    # beta2
    "$(parse "optimizer.beta2.mat" "beta2_mat")"
    "$(parse "optimizer.beta2.embedding" "beta2_embedding")"
    "$(parse "optimizer.beta2.head" "beta2_head")"
    "$(parse "optimizer.beta2.attn_w" "beta2_attn_w")"
    "$(parse "optimizer.beta2.attn_b" "beta2_attn_b")"
    "$(parse "optimizer.beta2.vec_w" "beta2_vec_w")"
    "$(parse "optimizer.beta2.vec_b" "beta2_vec_b")"
    # nesterov
    "$(parse "optimizer.nesterov.mat" "nesterov_mat")"
    "$(parse "optimizer.nesterov.embedding" "nesterov_embedding")"
    "$(parse "optimizer.nesterov.head" "nesterov_head")"
    "$(parse "optimizer.nesterov.attn_w" "nesterov_attn_w")"
    "$(parse "optimizer.nesterov.attn_b" "nesterov_attn_b")"
    "$(parse "optimizer.nesterov.vec_w" "nesterov_vec_w")"
    "$(parse "optimizer.nesterov.vec_b" "nesterov_vec_b")"
    # use_adamw
    "$(parse "optimizer.use_adamw.mat" "use_adamw_mat")"
    "$(parse "optimizer.use_adamw.embedding" "use_adamw_embedding")"
    "$(parse "optimizer.use_adamw.head" "use_adamw_head")"
    "$(parse "optimizer.use_adamw.attn_w" "use_adamw_attn_w")"
    "$(parse "optimizer.use_adamw.attn_b" "use_adamw_attn_b")"
    "$(parse "optimizer.use_adamw.vec_w" "use_adamw_vec_w")"
    "$(parse "optimizer.use_adamw.vec_b" "use_adamw_vec_b")"
    # normalize
    "$(parse "optimizer.normalize.mat" "normalize_mat")"
    "$(parse "optimizer.normalize.embedding" "normalize_embedding")"
    "$(parse "optimizer.normalize.head" "normalize_head")"
    "$(parse "optimizer.normalize.attn_w" "normalize_attn_w")"
    "$(parse "optimizer.normalize.attn_b" "normalize_attn_b")"
    "$(parse "optimizer.normalize.vec_w" "normalize_vec_w")"
    "$(parse "optimizer.normalize.vec_b" "normalize_vec_b")"
    # scale_weight
    "$(parse "optimizer.scale_weight.mat" "scale_weight_mat")"
    "$(parse "optimizer.scale_weight.embedding" "scale_weight_embedding")"
    "$(parse "optimizer.scale_weight.head" "scale_weight_head")"
    "$(parse "optimizer.scale_weight.attn_w" "scale_weight_attn_w")"
    "$(parse "optimizer.scale_weight.attn_b" "scale_weight_attn_b")"
    "$(parse "optimizer.scale_weight.vec_w" "scale_weight_vec_w")"
    "$(parse "optimizer.scale_weight.vec_b" "scale_weight_vec_b")"
    # scale_power
    "$(parse "optimizer.scale_power.mat" "scale_power_mat")"
    "$(parse "optimizer.scale_power.embedding" "scale_power_embedding")"
    "$(parse "optimizer.scale_power.head" "scale_power_head")"
    "$(parse "optimizer.scale_power.attn_w" "scale_power_attn_w")"
    "$(parse "optimizer.scale_power.attn_b" "scale_power_attn_b")"
    "$(parse "optimizer.scale_power.vec_w" "scale_power_vec_w")"
    "$(parse "optimizer.scale_power.vec_b" "scale_power_vec_b")"
    # scale_dim
    "$(parse "optimizer.scale_dim.mat" "scale_dim_mat")"
    "$(parse "optimizer.scale_dim.embedding" "scale_dim_embedding")"
    "$(parse "optimizer.scale_dim.head" "scale_dim_head")"
    "$(parse "optimizer.scale_dim.attn_w" "scale_dim_attn_w")"
    "$(parse "optimizer.scale_dim.attn_b" "scale_dim_attn_b")"
    "$(parse "optimizer.scale_dim.vec_w" "scale_dim_vec_w")"
    "$(parse "optimizer.scale_dim.vec_b" "scale_dim_vec_b")"
    # Global configs.
    # If any of the following is defined, it will override 
    # the dictionary-specific definition to a float.
    "$(parse "optimizer.lr" "lr")"
    "$(parse "optimizer.beta1" "beta1")"
    "$(parse "optimizer.beta2" "beta2")"
    "$(parse "optimizer.nesterov" "nesterov")"
    "$(parse "optimizer.use_adamw" "use_adamw")"
    "$(parse "optimizer.normalize" "normalize")"
    "$(parse "optimizer.scale_weight" "scale_weight")"
    "$(parse "optimizer.scale_power" "scale_power")"
    "$(parse "optimizer.scale_dim" "scale_dim")"
)

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
echo "job_id: ${job_id} || ${name}" >> "${OUTPUT_PATH}/job_list.txt"

echo "Submitted job: $name"