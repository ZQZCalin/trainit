# Static configuration variables

# PLEASE USE A NEW NAME FOR EVERY NEW EXPERIMENT!
NAME="linear_grid_test3"
DESC="
Experiment description:

Use the grid i/(i+1) for i in [N] as the grid of next_lrs.
Also use absolute epsilon value eps = 0 (greedy).
Use 10 segments for convenience (so we don't need to force 
first seg to 200 steps and divide the rest into n-1 pieces).
* total_steps = 200 for testing.
"

# =========================================================
# >>> GLOBAL VARIABLES
# =========================================================

# root working directory path
BASE_PATH="/projectnb/aclab/qinziz/trainit"

DATE=$(date +"%Y-%m-%d")

# total cpu hour of the master script
CPU_HOUR="24:00:00"

GPU_TYPE="L40S"

# GPU hour per parallel job
GPU_HOUR="4:00:00"

# path of scc output files
SCC_OUTPUT_PATH="${BASE_PATH}/scc_outputs/${DATE}/${NAME}"

# path of checkpoint files
CHECKPOINT_PATH="${BASE_PATH}/checkpoint/lr_schedule/${NAME}"

# delete checkpoints of suboptimal runs
CLEAN_CHECKPOINTS=True


# =========================================================
# >>> LISTENER
# =========================================================

# master host ip address
MASTER_HOST=$(hostname -I | awk '{print $1}')

# port number for communication
PORT=51205

# backoff time (in seconds) between listener attempts
LISTENER_BACKOFF=5

# maximum wait time of listener (in seconds)
#   NOTE: a job still sends ACK token if the python job
#   throws an error.
MAX_LISTEN_TIME=14400     # 4 hours
# MAX_LISTEN_TIME=10      # uncomment it for test purpose


# =========================================================
# >>> RESUBMIT
# =========================================================

# Maximum number of retry attempts per job
MAX_RETRIES=3

# Optional: Enable or disable resubmission feature
ENABLE_RETRY=true


# =========================================================
# >>> EXPERIMENT
# =========================================================

# maximum number of training steps
# TOTAL_STEPS=2000
TOTAL_STEPS=50      # testing

# number of segments
# NUM_SEGMENTS=10
NUM_SEGMENTS=3      # testing

# list of checkpoint iterations
#   i. you can use evenly distributed segments by changing `num_segments`
SEGMENTS=( $(seq 0 $((TOTAL_STEPS/NUM_SEGMENTS)) $TOTAL_STEPS) )
SEGMENTS[-1]=$TOTAL_STEPS           # set last segment to TOTAL_STEPS

#   ii. alternatively, you can customize unevenly distributed segments
# SEGMENTS=(0 150 300 500 1000 1500 $TOTAL_STEPS)
#       manually adapt NUM_SEGMENTS
# NUM_SEGMENTS=$((${#SEGMENTS[@]} - 1))


# >>> Other training configs

# global random seed
RANDOM_SEED=42

# training batch size
BATCH_SIZE=128

# optimizer configs
OPTIMIZER=adamw
BETA1=0.9
BETA2=0.999
WEIGHT_DECAY=0.1
NESTEROV=False


# >>> Logging configs

# wandb project name
PROJECT="greedy_lr_schedule"

# log additional metrics to wandb
LOG_CALLBACK_DATA=True