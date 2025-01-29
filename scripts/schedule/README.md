# Greedy LR Schedule Experiments

## Quick Start

For most of the easy experiments, you just need to change configurations in `config.sh` and `get_next_lr.py`. See [this](#configurations) and [this](#customizing-lr-schedules) for more details.

To run the script, you can either run the master script locally

```bash
bash scripts/schedule/master.sh
```

or submit it to SCC (which is recommended)

```bash
bash scripts/schedule/scc_submit.sh
```

The output files will be automatically stored in `scc_outputs/yyyy-mm-dd/$NAME`, and checkpoints will be stored in `checkpoint/lr_schedule/$NAME`by default. You can track the progress of the experiment in `master.e` and `master.o`.

<details>
<summary>A snippet of progress logs.</summary>

```txt
Running experiment $NAME.

Experiment description:

$DESC

master host ip: 192.168.18.244; port number: 51205
====================================================================================================
2025-01-28 22:28:37 - Master: Training segment 1 from iteration 0 to 200...
2025-01-28 22:28:37 - Update: computing lr1 and lr2_candidates...
2025-01-28 22:28:39 - Update: lr1=0.0, lr2_candidates=(1.0 0.1 0.01 0.001 0.0001 0.00001) for segment 1.
2025-01-28 22:28:39 - Listener: Waiting for 6 ACKs on port 51205...
Your job 2370475 ("seg1_lr2_1.0e+00") has been submitted
2025-01-28 22:28:39 - Launcher: Submitted job with lr1=0.0 lr2=1.0.
Your job 2370476 ("seg1_lr2_1.0e-01") has been submitted
2025-01-28 22:28:39 - Launcher: Submitted job with lr1=0.0 lr2=0.1.
Your job 2370477 ("seg1_lr2_1.0e-02") has been submitted
2025-01-28 22:28:39 - Launcher: Submitted job with lr1=0.0 lr2=0.01.
Your job 2370478 ("seg1_lr2_1.0e-03") has been submitted
2025-01-28 22:28:39 - Launcher: Submitted job with lr1=0.0 lr2=0.001.
Your job 2370479 ("seg1_lr2_1.0e-04") has been submitted
2025-01-28 22:28:39 - Launcher: Submitted job with lr1=0.0 lr2=0.0001.
Your job 2370480 ("seg1_lr2_1.0e-05") has been submitted
2025-01-28 22:28:39 - Launcher: Submitted job with lr1=0.0 lr2=0.00001.
2025-01-28 22:37:17 - Listener: Received ACK 0/6 from job ID 2370475
2025-01-28 22:50:41 - Listener: Received ACK 1/6 from job ID 2370480
2025-01-28 22:50:49 - Listener: Received ACK 2/6 from job ID 2370476
2025-01-28 22:50:52 - Listener: Received ACK 3/6 from job ID 2370477
2025-01-28 22:51:05 - Listener: Received ACK 4/6 from job ID 2370478
2025-01-28 22:52:06 - Listener: Received ACK 5/6 from job ID 2370479
2025-01-28 22:52:07 - Listener: 6 / 6 ACKs received from jobs (2370475 2370480 2370476 2370477 2370478 2370479).
2025-01-28 22:52:07 - Master: Segment 1/10 completed.

...
```
</details>


## Configurations

`config.sh` stores all static variables. Below lists a detailed explanation of each configuration. 

**Important Configs**

- `NAME`: specifies experiment name. 

    *Note:* PLEASE change name for every new experiment. Otherwise, there will be conflicts in `checkpoint/` and `scc_outputs/` and an error will likely be raised (this feature is intended as it prevents overwriting existing checkpoints).

- `BASE_PATH`: change to the path where you clone this repo.

- `PROJECT`: specifies wandb project name. change it as you like.

- `PORT`: specifies the port number for communication between workers and the master thread.

    *Note:* If you submit multiple experiments at the same time, a safe practice is to assign different port numbers to different experiments to avoid port conflicts.

**Experiment Configs**

- `NUM_SEGMENTS`: number of segments

- `SEGMENTS`: a list of intergers indicating the checkpoints. The default is an evenly distributed list. You can also use a customized definition.

- optimizer configs: we are using `AdamW` by default. You can use other optimizers, but you need to make sure the submission script in `submit_job.sh` matches the configs. See example below.
    <details>
    <summary>Replace AdamW with SGDM</summary>

    ```bash
    # config.sh
    # Refer to the main README.md for more details about optimization configs.
    OPTIMIZER=sgdm
    MOMENTUM=0.9
    USE_NESTEROV=False
    WEIGHT_DECAY=0.0
    DECOUPLE_WEIGHT_DECAY=True
    ```

    ```bash
    # submit_job.sh
    qsub <<EOF
    ...
    python main.py \
        ... \
        optimizer=$OPTIMIZER \
        optimizer.momentum=$MOMENTUM \
        optimizer.use_nesterov=$USE_NESTEROV \
        optimizer.weight_decay=$WEIGHT_DECAY \
        optimizer.decouple_weight_decay=$DECOUPLE_WEIGHT_DECAY \
        ...
    ...
    EOF
    ```
    </details>

For other experiment-related configs, please refer to the `Experiment` section in `config.sh`.

**System Configs**

- `DESC`: add a description of your experiment here.

- `CPU_HOUR`: specifies maximum cpu hours for the master script. Defaults to 24 hours. You can increase to up to 720 hours for larger experiments.

- `GPU_TYPE`: specifies a gpu type for training. as of this writing, `L40S` is the most available gpu on SCC.

- `GPU_HOUR`: specifies maximum gpu hours for each single job. a rough guideline is that 200 iterations per segment takes about 20 minutes; so the default 4hr should be sufficient for most experiments.

**Path Configs**

- `SCC_OUTPUT_PATH`: specifies the path for scc stdout and stderr output files.

- `CHECKPOINT_PATH`: specifies the path for checkpoint saving and loading.

**Feature Configs**

- `CLEAN_CHECKPOINTS`: if true, deletes all checkpoints in the last segment besides the one corresponding to the selected learning rate.

- `ENABLE_RETRY`: if true, turns on the resubmit upon failure feature.

- `MAX_RETRIES`: specifies the maximum number of resubmits of each job.

- `MAX_LISTEN_TIME`: specifies the maximum wait time (in seconds) of the listener. The master script moves on to the next segment if it exceeds the max wait time. Defaults to 4 hours, which again should be fine for more experiments. For larger scale experiments, 2x `GPU_HOUR` should be a safe value.


## Customizing LR Schedules

You can customize your own method of choosing learning rates for the next segment in `get_next_lr.py`. The `main()` function already fetches the losses of all submitted runs in the latest segment via WandB API for you.

We provide an example of choosing next lrs. 
1. We first smooth the losses using EMA smoothing with `alpha=0.1`, which corresponds to window size of `10`. 
2. To pick `lr1` for the next segment, we implemented two methods, `greedy` and `eps_greedy`, where the former chooses the learning rate that corresponds to the lowest post-smoothing loss, and the latter chooses the largest lr whose loss is lower than `loss_min + eps`, where `eps` is some tunable parameter. 
3. To generate the list of `lr2_candidates`, we implemented the standard grid search, which creates a list of `[0] + [lr * (c**i) for i in range(-n,n+1)]`. `c` denotes the grid multiplier, and `2n+2` denotes the grid size.

**Further Discussion**

The current automation framework should be easily adapted to 2-step, or even n-step, grid search. Below shows a rough pseudo-code of how to extend it to a 2-step lr search. More work is required to make it work.

<details>
<summary>An example of extending the current automation scripts</summary>

```bash
# Modification of master.sh for 2-step search (log-grid followed by linear-grid).
...
# Master thread
for (( i=0; i < ${#segments[@]}-1; i++ )); do
    start_step=${segments[$i]}
    end_step=${segments[$((i+1))]}

    # Start of segment
    printf '=%.0s' {1..100} && printf "\n"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Master: Training segment $((i+1)) from iteration ${start_step} to ${end_step}..."
    
    # Batch submit jobs in parallel
    source scripts/schedule/launcher.sh

    # Launch a listener for ACK from all jobs
    source scripts/schedule/listener.sh

    # Conclude the segment and update next lr and candidates
    source scripts/schedule/update.sh


    # =====================================================
    # [NEW] add another parallel submission for linear grid search.
    
    # TODO: change configurations for launcher.sh and update.sh

    # Batch submit jobs in parallel
    source scripts/schedule/launcher.sh

    # Launch a listener for ACK from all jobs
    source scripts/schedule/listener.sh

    # Conclude the segment and update next lr and candidates
    source scripts/schedule/update.sh

    # End of new script.
    # =====================================================

    # End of segment
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Master: Segment $((i+1))/${num_segments} completed." && echo ""
done
...
```
</details>


## Updates

- version 0.0.2: 
    - Implements the "resubmit upon failure" feature, which triggers if the exit code of the main python script is non-zero;
    - Implements the checkpoint cleaning feature which deletes all checkpoints other than the optimal one.


## Future Features

- 