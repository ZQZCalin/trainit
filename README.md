# LLM Training Pipeline

### Table of Content

- [Installation](#installation)
- [Usage](#usage)
- [Advanced](#advanced)

## Installation

### Clone Repo

```bash
git clone https://github.com/ZQZCalin/trainit.git
cd trainit
source scc_setup.sh
```

### Activate Environment

Every time when you need to re-activate the environment:

```bash
cd /YOUR/PATH/trainit
source env/bin/activate
module load python3/3.10.12 cuda pytorch
python check_env.py
```

## Usage

### The Training Pipeline

The main training pipeline is `train_jax.py`. The pipeline consists of several major components:

- **Initialization**
    - `load_lm_data`: initializes the dataloader;
    - `init_tokenizer`: initializes the tokenizer;
    - `init_model`: initializes the LLM model;
    - `init_scheduler`: initializes learning rate scheduler;
    - `init_optimizer`: initializes the optimizer
- **Training Process**
    - `train_step`: one-step training update;
    - `update_aux_state`: optional training update to log training statistics.

### Configurations

We use [hydra](https://hydra.cc/docs/intro/) to manage the training configurations, stored in the `conf/` directory:
```
conf/
|-- dataset/
    |-- pile.yaml
    |-- OTHER DATASETS
|-- model/
    |-- gpt.yaml
    |-- OTHER MODELS
|-- optimizer/
    |-- adam.yaml
    |-- sgdm.yaml
    |-- OTHER OPTIMIZERS
|-- config.yaml
|-- train.yaml
|-- logging.yaml
|-- checkpoint.yaml
|-- experimental.yaml
```
The current configuration is recorded from the optimal Adam benchmark. You can easily change configuration in cmd.

### Submit A Training Batch Job

To reproduce the optimal benchmark:

```bash
python train_jax.py logging.wandb_project=PROJECT_NAME
```

You can also customize your own configurations. For example, if you want to train SGDM:

```bash
python train_jax.py logging.wandb_project=PROJECT_NAME \
    optimizer=sgdm optimizer.lr_config.lr=1.0 ...
```

### Checkpoint

We have implemented checkpointing system for you. All you need is change the checkpoint configuration:

```bash
python train_jax.py logging.wandb_project=PROJECT_NAME \
    checkpoint.save=True checkpoint.save_path=CHECKPOINT_DIR checkpoint.save_steps=10000 \  # enable checkpoint saving
    checkpoint.load=True checkpoint.load_path=CHECKPOINT_DIR/iter_10000.json \              # enable checkpoing loading
    checkpoint.num_steps=null       # specify number of steps in one checkpoint (optional)
```

## Advanced

### Implement Your Own Optimizer

1. Implement your optimizer in `optimizer/NAME.py`.
2. Create a configuration file `conf/optimizer/NAME.yaml`.
3. In the main pipeline `train_jax.py`, add your optimizer in the `init_optimizer` component.