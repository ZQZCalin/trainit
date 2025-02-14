# Mango_v2 experiments.

#!/bin/bash

# project=null
project=pile_baseline

log_data=False
steps=2000
batch_size=128

schedule=linear
warmup=200
const=null

optimizer=mango_v3

exp_name=mango_v3_fix

# System variables
BASE_DIR=/projectnb/aclab/qinziz/trainit
DATE=$(date +"%Y-%m-%d")
OUTPUT_PATH=$BASE_DIR/scc_outputs/$DATE/$exp_name

mkdir -p $OUTPUT_PATH


# ========================================================================
# Mango optimzier configs
# ========================================================================

# ...
# name="mango_v3_baseline"


# ... Head
# ......
# head_normalize="inf_"
# name="mango_v3_head_inf"

# ......
# head_normalize="l2_row"
# head_scale_dim=True
# head_scale_dim_clip_min=null
# name="mango_v3_head_l2-row"
# # <next>
# head_scale_dim=False
# name="mango_v3_head_l2-row_nodim"

# ......
# head_normalize="l2_col"
# head_scale_dim=True
# head_scale_dim_clip_min=null
# name="mango_v3_head_l2-col"
# # <next>
# head_scale_dim=False
# name="mango_v3_head_l2-col_nodim"

# ......
# head_normalize=null
# name="mango_v3_head_null"


# ... Bias
# ......
# attn_b_scale_dim=True
# vec_b_scale_dim=True
# name="mango_v3_bias_l2_dim"

# ......
# attn_b_normalize=null
# vec_b_normalize=null
# name="mango_v3_bias_null"


# ... Embedding
# ......
# embedding_normalize="l2_col"
# embedding_scale_dim=True
# name="mango_v3_emb_l2col-T_dim"
# # <next>
# embedding_scale_dim=False
# name="mango_v3_emb_l2col-T_nodim"

# ......
# embedding_normalize="l2_col"
# embedding_scale_dim_transpose=False
# name="mango_v3_emb_l2col"
# # <next>
# embedding_scale_dim=True
# name="mango_v3_emb_l2col_dim-scale"


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

layers=("mat" "embedding" "head" "attn_w" "attn_b" "vec_w" "vec_b")
keys=(
  "lr"
  "beta1"
  "beta2"
  "nesterov"
  "eps"
  "normalize"
  "scale_dim"
  "scale_dim_transpose"
  "scale_dim_clip_min"
  "scale_dim_clip_max"
  "ns_steps"
  "num_heads"
  "scale_norm"
  "scale_norm_power"
  "scale_norm_clip_min"
  "scale_norm_clip_max"
  "use_adamw"
  "offset_beta"
  "igt_scale"
)

# Start building the args list.
args=(
  "logging.wandb_project=$project"
  "logging.wandb_name=$name"
  "logging.log_callback_data=$log_data"
  "train.max_steps=$steps"
  "dataset.total_batch_size=$batch_size"
  "experimental=null"
  "test=null"
  "optimizer=$optimizer"
  # lr schedule
  "optimizer/lr_config=$schedule"
  "optimizer.lr_config.warmup=$warmup"
  "optimizer.lr_config.const=$const"
  "optimizer.lr_config.max_steps=$steps"
)

# Add core configs. Override default configs by
# specifying <layer>_<key>=...
#   First argument: optimizer.core.<layer>.<key>
#   Second argument: <layer>_<key>
for layer in "${layers[@]}"; do
  for key in "${keys[@]}"; do
    args+=( "$(parse "optimizer.core.${layer}.${key}" "${layer}_${key}")" )
  done
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
echo "job_id: ${job_id} || ${name}" >> "${OUTPUT_PATH}/job_list.txt"

echo "Submitted job: $name"