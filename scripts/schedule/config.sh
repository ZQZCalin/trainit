# Global configuration variables

# PLEASE USE A NEW NAME FOR EVERY NEW EXPERIMENT!
NAME="eps-greedy_10segs(2)"
DESC="
Experiment description:

2000 steps split into 10 segments, with eps-greedy mechanism.
Trial 2 after adding the error catcher.
"

# =========================================================
# >>> GLOBAL VARIABLES
# =========================================================

# root working directory path
BASE_PATH="/projectnb/aclab/qinziz/trainit"

DATE=$(date +"%Y-%m-%d")

# Total cpu hour of the master script.
CPU_HOUR="24:00:00"

GPU_TYPE="L40S"

# GPU hour per parallel job.
GPU_HOUR="4:00:00"


# =========================================================
# >>> EXPERIMENT
# =========================================================

# maximum number of training steps
total_steps=2000

# number of segments
num_segments=10

# list of checkpoint iterations
#   i. you can use evenly distributed segments by changing `num_segments`
segments=( $(seq 0 $((total_steps/num_segments)) $total_steps) )
segments[-1]=$total_steps           # set last segment to total_steps

#   ii. alternatively, you can customize unevenly distributed segments
# segments=(0 150 300 500 1000 1500 $total_steps)

# learning rates for the first segment
lr1=0
lr2_candidates=(1e0 1e-1 1e-2 1e-3 1e-4 1e-5)


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
#   currently set to 4 hours. feel free to change
max_wait_mins=240
max_wait_time=$((max_wait_mins*60))
# max_wait_time=10    # uncomment it for test purpose