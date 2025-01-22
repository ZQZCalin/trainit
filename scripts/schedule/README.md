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
Running experiment eps-greedy_10segs.

Experiment description:

2000 steps split into 10 segments, with eps-greedy mechanism.

master host ip: 192.168.18.244; port number: 51204
====================================================================================================
2025-01-21 23:43:43 - Master: Training segment 1 from iteration 0 to 200...
Your job 2134881 ("seg1_lr2_1.0e+00") has been submitted
2025-01-21 23:43:43 - Submitter: Submitted job with lr1=0 lr2=1e0.
Your job 2134882 ("seg1_lr2_1.0e-01") has been submitted
2025-01-21 23:43:44 - Submitter: Submitted job with lr1=0 lr2=1e-1.
Your job 2134883 ("seg1_lr2_1.0e-02") has been submitted
2025-01-21 23:43:44 - Submitter: Submitted job with lr1=0 lr2=1e-2.
Your job 2134884 ("seg1_lr2_1.0e-03") has been submitted
2025-01-21 23:43:44 - Submitter: Submitted job with lr1=0 lr2=1e-3.
Your job 2134885 ("seg1_lr2_1.0e-04") has been submitted
2025-01-21 23:43:44 - Submitter: Submitted job with lr1=0 lr2=1e-4.
Your job 2134886 ("seg1_lr2_1.0e-05") has been submitted
2025-01-21 23:43:44 - Submitter: Submitted job with lr1=0 lr2=1e-5.
2025-01-21 23:43:44 - Listener: Waiting for 6 ACKs on port 51204...
2025-01-21 23:52:58 - Listener: Received ACK 1/6 from job ID 2134881
2025-01-22 00:04:46 - Listener: Received ACK 2/6 from job ID 2134884
2025-01-22 00:05:50 - Listener: Received ACK 3/6 from job ID 2134882
2025-01-22 00:05:57 - Listener: Received ACK 4/6 from job ID 2134883
2025-01-22 00:14:57 - Listener: Received ACK 5/6 from job ID 2134885
2025-01-22 00:18:10 - Listener: Received ACK 6/6 from job ID 2134886
2025-01-22 00:18:10 - Listener: 6 / 6 ACKs received from jobs (2134881 2134884 2134882 2134883 2134885 2134886).
2025-01-22 00:18:10 - Update: fetching lr1 and lr2_candidates for the next segment...
2025-01-22 00:18:23 - Update: lr1=0.001, lr2_candidates=(0.0 0.00025 0.0005 0.001 0.002 0.004) for segment 1.
2025-01-22 00:18:23 - Master: Segment 1/10 completed.

...
```
</details>


## Configurations

All configurations in the script are located in `config.sh`. Below lists a detailed explanation of each configuration. 

**Important Configs**
- `NAME`: specifies experiment name. 
    *NOTE: please change name for every new experiment. Otherwise, there will be conflicts in `checkpoint/` and `scc_outputs/` and an error will likely be raised (this feature is intended as it prevents overwriting existing checkpoints).*
- `BASE_PATH`: change to the path where you clone this repo.
- `project`: specifies wandb project name. change it as you like.

**Experiment Configs**
- `num_segments`: number of segments
- `segments`: a list of intergers indicating the checkpoints. The default is an evenly distributed list, but you can also customize it. Please make sure `num_segments` is equal to `len(segments)-1`.
- optimizer configs: we are using `AdamW` by default. You can use other optimizers, but you need to make sure the submission script in `launcher.sh` matches the configs. See example below.
    <details>
    <summary>Replace AdamW with SGDM</summary>

    ```bash
    # config.sh
    # Refer to the main README.md for more details about optimization configs.
    optimizer=sgdm
    momentum=0.9
    use_nesterov=False
    weight_decay=0.0
    decouple_weight_decay=True
    ```

    ```bash
    # launcher.sh
    qsub <<EOF
    ...
    python main.py \
        ... \
        optimizer=$optimizer \
        optimizer.momentum=$momentum \
        optimizer.use_nesterov=$use_nesterov \
        optimizer.weight_decay=$weight_decay \
        optimizer.decouple_weight_decay=$decouple_weight_decay \
        ...
    ...
    EOF
    ```
    </details>

For other experiment-related configs, please refer to the `Experiment` section in `config.sh`.

**Optional Configs**
- `DESC`: add a description of your experiment here.
- `CPU_HOUR`: specifies maximum cpu hours for the master script. Defaults to 24 hours. You can increase to up to 720 hours for larger experiments.
- `GPU_TYPE`: specifies a gpu type for training. as of this writing, `L40S` is the most available gpu on SCC.
- `GPU_HOUR`: specifies maximum gpu hours for each single job. a rough guideline is that 200 iterations per segment takes about 20 minutes; so the default 4hr should be sufficient for most experiments.
- `max_wait_mins`: specifies the maximum wait time of the listener. the master script moves on to the next segment if it exceeds the max wait time. Defaults to 4 hours, which again should be fine for more experiments. For larger scale experiments, 2x `GPU_HOUR` should be a safe value.


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


## Future Features

- Right now the script ignores any jobs that fail to run. Might add a function of re-submitting jobs if the failure is caused by invalid cuda environment.