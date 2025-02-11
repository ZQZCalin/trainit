# Extra experiments with mango.

#!/bin/bash

# project=null
project=mango_norms
log_data=False
steps=2000
batch_size=128

optimizer=mango

ns_steps=6
eps=1e-8
visualize=False

schedule=linear
warmup=200
const=null

# we can use constant lrs
lrs=0.01
# or we can use dictionary lrs
lr_mat=$lrs
lr_embedding=$lrs
lr_head=$lrs
lr_attn_w=$lrs
lr_attn_b=$lrs
lr_vec_w=$lrs
lr_vec_b=$lrs


exp_name=mango_test_normalization

# System variables
BASE_DIR=/projectnb/aclab/qinziz/trainit
DATE=$(date +"%Y-%m-%d")
OUTPUT_PATH=$BASE_DIR/scc_outputs/$DATE/$exp_name

mkdir -p $OUTPUT_PATH


# ========================================================================
# Experiment variables
# ========================================================================

lrs=0.01

momentum=0.95
beta2=0.95
offset_beta=0.95

normalization_mat="ns"
normalization_embedding="null"
normalization_head="ns"
normalization_attn_w="ns_split"
normalization_attn_b="l2_split"
normalization_vec_w="null"
normalization_vec_b="l2"

# ...
name="mango_best_visualize"
visualize=True


# ========================================================================
# Below is submit function. Only change the part related to global
# vs param-wise learning rate.
# ========================================================================

args=(
    "logging.wandb_project=$project"
    "logging.wandb_name=$name"
    "logging.log_callback_data=$log_data"
    "train.max_steps=$steps"
    "dataset.total_batch_size=$batch_size"
    "optimizer=$optimizer"
    "optimizer.momentum=$momentum"
    "optimizer.nesterov=$nesterov"
    "optimizer.ns_steps=$ns_steps"
    "optimizer.eps=$eps"
    "optimizer.beta2=$beta2"
    "optimizer.offset_beta=$offset_beta"
    "optimizer/lr_config=$schedule"
    "optimizer.lr_config.warmup=$warmup"
    "optimizer.lr_config.const=$const"
    "optimizer.lr_config.max_steps=$steps"
    # Use this line for global lr.
    "optimizer.lrs=$lrs"
    # Use these lines for different lrs.
    # "optimizer.lrs.mat=$lr_mat"
    # "optimizer.lrs.embedding=$lr_embedding"
    # "optimizer.lrs.head=$lr_head"
    # "optimizer.lrs.attn_w=$lr_attn_w"
    # "optimizer.lrs.attn_b=$lr_attn_b"
    # "optimizer.lrs.vec_w=$lr_vec_w"
    # "optimizer.lrs.vec_b=$lr_vec_b"
    "optimizer.normalizations.mat=$normalization_mat"
    "optimizer.normalizations.embedding=$normalization_embedding"
    "optimizer.normalizations.head=$normalization_head"
    "optimizer.normalizations.attn_w=$normalization_attn_w"
    "optimizer.normalizations.attn_b=$normalization_attn_b"
    "optimizer.normalizations.vec_w=$normalization_vec_w"
    "optimizer.normalizations.vec_b=$normalization_vec_b"
    "optimizer.visualize=$visualize"
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