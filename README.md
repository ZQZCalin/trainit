# Trainit -- A ML Training Pipeline

Trainit is a machine learning training pipeline based on [jax](https://jax.readthedocs.io/en/latest/index.html). It is also built on [optax](https://optax.readthedocs.io/en/latest/) for optimizers, [equinox](https://docs.kidger.site/equinox/all-of-equinox/) for models, [wandb](https://wandb.ai/site/) for logging, and [hydra](https://hydra.cc/docs/intro/) for convenient configuration management.

*News:*
This pipeline is now updated to v2.0! In the new update, we made it more available to train other ML tasks beyond language models.

### Table of Content

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Advanced](#advanced)

## Installation

### First-time setup

```bash
git clone https://github.com/ZQZCalin/trainit.git
cd trainit
source scc_setup.sh
```

Please note that you must set up the environment on a device with a supported GPU. For instance, if you are using the SCC, initiate a GPU interactive session and set up on that session.

To setup wandb logging:
```bash
wandb login
```

<!-- We use [minGPT](https://github.com/karpathy/minGPT) to confirm pytorch/jax equivalence.
```bash
git clone https://github.com/karpathy/minGPT.git
``` -->

### Activate Environment

Every time when you need to re-activate the environment:

```bash
cd /YOUR/PATH/trainit
module load python3/3.10.12 cuda/12.2
source env/bin/activate
python check_env.py

# or simply
cd /YOUR/PATH/trainit
source activate_env.sh
```

## Quick Start

### Run the Training Pipeline

We provide an example script of training GPT-2 with AdamW in `scripts/train_lm.sh`. You can modify the configs there and simply run

```bash
bash scripts/train_lm.sh
```

This may fail if your GPU does not have enough memory (24GB should be enough). If you want to use a 12GB GPU like a V100, you can cut down the memory significantly by disabling some logging:

```bash
python train_jax.py \
    # add these lines
    logging.store_last_grads=false \
    logging.store_past_grads=false \
    logging.store_last_params=false \
    logging.compute_last_loss=false \
    logging.compute_last_grads=false
```

If you do not need any logging at all, you can simply change to `log_data=false` in `train_lm.sh` or specify in your command

```bash
python train_jax.py logging.log_callback_data=false
```

### Configuration Management 

You can always customize your own configurations. Trainit uses hydra for config management, so the syntax is the same as hydra. You can find all default configs in the `conf/` subfolder. 

As an example, to change the optimizer from the default AdamW to SGDM, you can simply specify this

```bash
python train_jax.py \
    optimizer=sgdm \
    # you can also change the default hyper-parameters of sgdm here:
    optimizer/lr_config=linear \
    optimizer.lr_config.lr=1.0 \
    optimizer.lr_config.warmup=200 \
    optimizer.lr_config.max_steps=2000
```

For more details, please see the next section.

### Training Checkpoints

We have implemented a checkpointing system for you. All you need is changing the checkpoint configuration:

For saving checkpoint,
```bash
python train_jax.py \
    checkpoint.save=True \
    checkpoint.save_path=CHECKPOINT \
    checkpoint.save_steps=[100, 1000, 10000] \
    checkpoint.num_steps=null
```
- `save_path` specifies the directory to which the checkpoint is saved. It cannot be an existing directory.
- `save_steps` can be either a positive integer or a list of positive integers. If it is an integer, a checkpoint will be saved every `save_steps` iterations; if it is a list, a checkpoint will be saved if the iteration number is in this list. The saved checkpoint will be named `{save_path}/iter_{iteration}.ckpt`.
- `num_steps` specifies the total training steps. It defaults to `train.max_steps` if not specified.

For loading checkpoint,
```bash
python train_jax.py \
    checkpoint.load=True \
    checkpoint.load_path=CHECKPOINT
    checkpoint.load_file=iter_100.ckpt
    checkpoint.overwrite_config: False
    checkpoint.overwrite_optimizer: False
```
- `load_path` specifies the directory from which the checkpoint is loaded.
- `load_file` specifies the checkpoint name, e.g. `"iter_100.ckpt"`.
- `overwrite_config`: Defaults to `False` and loads existing config from `"{load_path}/config.yaml"`. If true, reads the new config instead of the loaded config. USE WITH CAUTION: This should only be turned on if you need to use different config than the loaded checkpoint.
- `overwrite_optimizer`: Defaults to `False` and loads `opt_state` from the saved train_state (and thus continues training the existing optimizer). If true, reinitializes a new optimizer with a fresh `opt_state`.  USE WITH CAUTION: This should only be used if you need to either restart your optimizer or use a different optimizer family (e.g. from Adam to SGDM). If you only need to change optimizer hyper-parameters such as learning rate or momentum, keep `overwrite_optimizer=False` and just turn on `overwrite_config=True`.

## Advanced

*Note:* this part of documentation is still under construction.

### The Training Pipeline

The main training pipeline `train_jax.py` is highly modulized, which consists of the following major components:

- **Initialization**
    - `load_lm_data`: initializes the dataloader;
    - `init_tokenizer`: initializes the tokenizer;
    - `init_model`: initializes the LLM model;
    - `init_scheduler`: initializes learning rate scheduler;
    - `init_optimizer`: initializes the optimizer
- **Training Process**
    - `train_step`: one-step training update;
    - `update_aux_state`: optional training update to log training statistics.

You can modify each component for your own needs.

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
    |-- lr_config/
        |-- constant.yaml
        |-- cosine.yaml
        |-- linear.yaml
        |-- OTHER SCHEDULERS
    |-- adam.yaml
    |-- sgdm.yaml
    |-- OTHER OPTIMIZERS
|-- config.yaml
|-- train.yaml
|-- logging.yaml
|-- checkpoint.yaml
```
The current configuration is recorded from the optimal Adam benchmark. You can easily change configuration in command line as shown earlier. A caveat is overriding the config group `optimizer/lr_config`, in which case you need to use `/` for overriding config group and `.` for specifying config, e.g.
```bash
... optimizer/lr_config=linear optimizer.lr_config.lr=3e-4
```

### Implement Your Own Optimizer

1. Implement your optimizer in `optimizer/NAME.py`.
2. Create a configuration file `conf/optimizer/NAME.yaml`.
3. In the main pipeline `train_jax.py`, add your optimizer in the `init_optimizer` component.


## Credits

- The code of `scalable_shampoo` is derived from [google-research](https://github.com/google-research/google-research/tree/master/scalable_shampoo).