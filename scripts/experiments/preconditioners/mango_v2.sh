# Mango_v2 experiments.

#!/bin/bash

project=null
# project=pile_baseline
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
name="mangov2_test"
lr_embedding=0.012


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
    "optimizer=$optimizer"
    "optimizer.ns_steps=$ns_steps"
    "optimizer.eps=$eps"
    "optimizer.offset_beta=$offset_beta"
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
    # global configs.
    # if any of the following is defined, it will override 
    # the dictionary-specific definition.
    "$(parse "optimizer.lr" "lr")"
)

python main.py ${args[@]}

# qsub <<EOF
# #!/bin/bash -l

# #$ -pe omp 8
# #$ -l gpus=1
# #$ -l gpu_type=L40S     # Specifies the gpu type.
# #$ -l h_rt=8:00:00      # Specifies the hard time limit for the job
# #$ -N "$name".sh
# #$ -o $OUTPUT_PATH/\$JOB_NAME.o\$JOB_ID     # Escape environment variables with \$
# #$ -e $OUTPUT_PATH/\$JOB_NAME.e\$JOB_ID

# sleep $(((RANDOM % 1000) / 100))   # Prevents simultaneous reads of loadit dataset

# source activate_env.sh
# python main.py ${args[@]}
# EOF
# echo "Submitted job: $name"