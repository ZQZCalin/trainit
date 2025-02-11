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


# ...
# aka "mango_wnorm_same_p1"
# name="mango_wnorm_default"

# ...
name="mango_wnorm_global_l2_p1"
scale_weight="l2"

# ...
# name="mango_wnorm_global_l2_p0.75"
# scale_weight="l2|0.75"

# ...
# name="mango_wnorm_global_l2_p0.5"
# scale_weight="l2|0.5"

# ...
# name="mango_wnorm_global_l2_p0.25"
# scale_weight="l2|0.25"

# ...
# p=0.75
# p=0.5
# p=0.25
# name="mango_wnorm_same_p${p}"
# scale_weight_mat="op|${p}"
# scale_weight_embedding="null"
# scale_weight_head="op|${p}"
# scale_weight_attn_w="op|${p}"
# scale_weight_attn_b="l2|${p}"
# scale_weight_vec_w="null"
# scale_weight_vec_b="l2|${p}"


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
    # lr schedule
    "optimizer/lr_config=$schedule"
    "optimizer.lr_config.warmup=$warmup"
    "optimizer.lr_config.const=$const"
    "optimizer.lr_config.max_steps=$steps"
    # learning rates
    "$(parse "optimizer.lr.mat" "lr_mat")"
    "$(parse "optimizer.lr.embedding" "lr_embedding")"
    "$(parse "optimizer.lr.head" "lr_head")"
    "$(parse "optimizer.lr.att_w" "lr_att_w")"
    "$(parse "optimizer.lr.att_b" "lr_att_b")"
    "$(parse "optimizer.lr.vec_w" "lr_vec_w")"
    "$(parse "optimizer.lr.vec_b" "lr_vec_b")"
    # beta1
    "$(parse "optimizer.beta1.mat" "beta1_mat")"
    "$(parse "optimizer.beta1.embedding" "beta1_embedding")"
    "$(parse "optimizer.beta1.head" "beta1_head")"
    "$(parse "optimizer.beta1.att_w" "beta1_att_w")"
    "$(parse "optimizer.beta1.att_b" "beta1_att_b")"
    "$(parse "optimizer.beta1.vec_w" "beta1_vec_w")"
    "$(parse "optimizer.beta1.vec_b" "beta1_vec_b")"
    # beta2
    "$(parse "optimizer.beta2.mat" "beta2_mat")"
    "$(parse "optimizer.beta2.embedding" "beta2_embedding")"
    "$(parse "optimizer.beta2.head" "beta2_head")"
    "$(parse "optimizer.beta2.att_w" "beta2_att_w")"
    "$(parse "optimizer.beta2.att_b" "beta2_att_b")"
    "$(parse "optimizer.beta2.vec_w" "beta2_vec_w")"
    "$(parse "optimizer.beta2.vec_b" "beta2_vec_b")"
    # nesterov
    "$(parse "optimizer.nesterov.mat" "nesterov_mat")"
    "$(parse "optimizer.nesterov.embedding" "nesterov_embedding")"
    "$(parse "optimizer.nesterov.head" "nesterov_head")"
    "$(parse "optimizer.nesterov.att_w" "nesterov_att_w")"
    "$(parse "optimizer.nesterov.att_b" "nesterov_att_b")"
    "$(parse "optimizer.nesterov.vec_w" "nesterov_vec_w")"
    "$(parse "optimizer.nesterov.vec_b" "nesterov_vec_b")"
    # use_adamw
    "$(parse "optimizer.use_adamw.mat" "use_adamw_mat")"
    "$(parse "optimizer.use_adamw.embedding" "use_adamw_embedding")"
    "$(parse "optimizer.use_adamw.head" "use_adamw_head")"
    "$(parse "optimizer.use_adamw.att_w" "use_adamw_att_w")"
    "$(parse "optimizer.use_adamw.att_b" "use_adamw_att_b")"
    "$(parse "optimizer.use_adamw.vec_w" "use_adamw_vec_w")"
    "$(parse "optimizer.use_adamw.vec_b" "use_adamw_vec_b")"
    # normalize
    "$(parse "optimizer.normalize.mat" "normalize_mat")"
    "$(parse "optimizer.normalize.embedding" "normalize_embedding")"
    "$(parse "optimizer.normalize.head" "normalize_head")"
    "$(parse "optimizer.normalize.att_w" "normalize_att_w")"
    "$(parse "optimizer.normalize.att_b" "normalize_att_b")"
    "$(parse "optimizer.normalize.vec_w" "normalize_vec_w")"
    "$(parse "optimizer.normalize.vec_b" "normalize_vec_b")"
    # scale_weight
    "$(parse "optimizer.scale_weight.mat" "scale_weight_mat")"
    "$(parse "optimizer.scale_weight.embedding" "scale_weight_embedding")"
    "$(parse "optimizer.scale_weight.head" "scale_weight_head")"
    "$(parse "optimizer.scale_weight.att_w" "scale_weight_att_w")"
    "$(parse "optimizer.scale_weight.att_b" "scale_weight_att_b")"
    "$(parse "optimizer.scale_weight.vec_w" "scale_weight_vec_w")"
    "$(parse "optimizer.scale_weight.vec_b" "scale_weight_vec_b")"
    # scale_power
    "$(parse "optimizer.scale_power.mat" "scale_power_mat")"
    "$(parse "optimizer.scale_power.embedding" "scale_power_embedding")"
    "$(parse "optimizer.scale_power.head" "scale_power_head")"
    "$(parse "optimizer.scale_power.att_w" "scale_power_att_w")"
    "$(parse "optimizer.scale_power.att_b" "scale_power_att_b")"
    "$(parse "optimizer.scale_power.vec_w" "scale_power_vec_w")"
    "$(parse "optimizer.scale_power.vec_b" "scale_power_vec_b")"
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
)

# python main.py ${args[@]}

qsub <<EOF
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
echo "Submitted job: $name"