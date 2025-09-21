# Static configuration variables

# PLEASE USE A NEW NAME FOR EVERY NEW EXPERIMENT!
NAME="step2k_seg10_lr1e-3_grid20_eps0.48"                                   # CHANGE THIS (every experiment)
DESC="
Part of steps=2k, segs=10 experiment

- initial lr = 1e-3
- 20 grids: (0/1, 1/2, 2/3, ..., 9/10, 1, 10/9, ..., 3/2, 2/1)
- eps = 0.48
"                                                           # CHANGE THIS (every experiment)

# in case of duplicate names, add a 6-digit uuid-v4 to name
# the following definition ensures uuid6 is only defined once (for both scc_submit and master)
: "${uuid6:=$(uuidgen | tr -d '-' | head -c6)}"
export uuid6
NAME+="_${uuid6}"

# =========================================================
# >>> GLOBAL VARIABLES
# =========================================================

# root working directory path
BASE_PATH="/projectnb/aclab/qinziz/trainit"                 # CHANGE THIS upon setup (once)
DATE=$(date +"%Y-%m-%d")

# total cpu hour of the master script
CPU_HOUR="48:00:00"                                         # CHANGE THIS if needed (24 hrs per 20 segs; no more than 72)
GPU_TYPE="L40S"
# GPU hour per parallel job
GPU_HOUR="4:00:00"

# path of all experiment-related outputs
OUTPUT_PATH="${BASE_PATH}/scheduler_outputs/${DATE}/${NAME}"
# path of scc output files
SCC_OUTPUT_PATH="${OUTPUT_PATH}/scc_outputs"
# path of checkpoint files
CHECKPOINT_PATH="${OUTPUT_PATH}/checkpoint"
# path of progress logs
PROGRESS_PATH="${OUTPUT_PATH}/progress.log"
# path of experiment snapshot
SNAPSHOT_PATH="${OUTPUT_PATH}/snapshot.txt"

# delete checkpoints of suboptimal runs
CLEAN_CHECKPOINTS=True


# =========================================================
# >>> LISTENER
# =========================================================

# master host ip address
MASTER_HOST=$(hostname -I | awk '{print $1}')               # DEPRECATED

# port number for communication
PORT=60221                                                  # DEPRECATED

# backoff time (in seconds) between listener attempts
LISTENER_BACKOFF=60                                         # DEPRECATED
# it's too complicated to handle timeout in a fd pipe:
# we need extra step to buffer any message if it's split in two
# due to timeout. For now, I just don't think it's worth adding
# this feature given this LISTENER timeout is rarely used.

# maximum wait time of listener (in seconds)
#   NOTE: a job still sends ACK token if the python job
#   throws an error.
MAX_LISTEN_TIME=14400     # 4 hours                         # DEPRECATED
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
TOTAL_STEPS=2000                                            # CHANGE THIS if needed
# TOTAL_STEPS=50      # testing

# number of segments
NUM_SEGMENTS=10
# NUM_SEGMENTS=3      # testing

# list of checkpoint iterations
#   i. you can use evenly distributed segments by changing `num_segments`
SEGMENTS=( $(seq 0 $((TOTAL_STEPS/NUM_SEGMENTS)) $TOTAL_STEPS) )
SEGMENTS[-1]=$TOTAL_STEPS           # set last segment to TOTAL_STEPS

#   ii. alternatively, you can customize unevenly distributed segments
#       please make sure it always starts with (0 200 ...)
#       below is an example of 3 segments (dividing the rest 1800 steps into 3 segs)
# SEGMENTS=(0 200 800 1400 $TOTAL_STEPS)
# SEGMENTS=(0 200 400 600 800 1000 1200 1400 1600 1800 $TOTAL_STEPS)                      # CHANGE THIS
# 20 segs
# SEGMENTS=(0 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 $TOTAL_STEPS)
# 4 segs
# SEGMENTS=(0 200 800 1400 $TOTAL_STEPS)
#       manually adapt NUM_SEGMENTS
NUM_SEGMENTS=$((${#SEGMENTS[@]} - 1))


# >>> Other training configs

# global random seed
RANDOM_SEED=42

# mixed precision
USE_AMP=False

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
PROJECT="greedy_lr_schedule"                                # CHANGE THIS if needed

# log additional metrics to wandb
LOG_CALLBACK_DATA=False      # we don't need to log other metrics in this task