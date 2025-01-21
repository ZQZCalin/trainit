# Global configuration variables

# Here's a list of variables that you can specify.
# For the rest of variables, we encourage you to leave
# them as is unless you understand what they do.
# 
# For 
# - 


NAME="test1"
DESC="
Experiment description:

Testing for the automation script. Trial 1.
"


# =========================================================
# >>> GLOBAL VARIABLES
# =========================================================

# root working directory path
BASE_PATH="/projectnb/aclab/qinziz/trainit"

DATE=$(date +"%Y-%m-%d")

GPU_TYPE="L40S"

GPU_HOUR="8:00:00"


# =========================================================
# >>> EXPERIMENT
# =========================================================

# maximum number of training steps
total_steps=2000

# number of segments
num_segments=7

# list of checkpoint iterations
#   i. you can use evenly distributed segments by changing `num_segments`
segments=( $(seq 0 $((total_steps/num_segments)) $total_steps) )
segments[-1]=$total_steps           # set last segment to total_steps

#   ii. alternatively, you can customize unevenly distributed segments
# segments=(0 150 300 500 1000 1500 $total_steps)

# learning rates for the first segment
lr1=0
# lr2_candidates=(1e0 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6)
lr2_candidates=(1 2)


# >>> Other training configs

# global random seed
random_seed=42

# training batch size
batch_size=128

# optimizer configs
optimizer=adamw
beta1=0.9
beta2=0.999
weight_decay=0.1
nesterov=False

# checkpoint subfolder relative path
checkpoint_path="checkpoint/lr_schedule/${NAME}"


# >>> Logging configs

# wandb project name
project="greedy_lr_schedule"

# log additional metrics to wandb
log_callback_data=True


# =========================================================
# >>> LISTENER
# =========================================================

# master host ip address
master_host=$(hostname -I | awk '{print $1}')

# port number for communication
port=51204

# maximum wait time of listener (in seconds)
#   NOTE: a job still sends ACK token if the python job
#   throws an error.
#   currently set to 2 hours. feel free to change
max_wait_mins=120
max_wait_time=$((max_wait_mins*60))
# max_wait_time=10    # uncomment it for test purpose