# A simplified script to log non-smooth statistics.
# NOTE: This code is not updated since 2024/06.
import logging
import random
import numpy as np
from itertools import chain
import wandb
import datasets
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import tqdm
from loadit import LoadIt
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import utils
import hydra
from omegaconf import OmegaConf, DictConfig
from argparse import Namespace
from typing import Any, NamedTuple

from datasets.lm_loader import get_lm_loader_next_token

import os, sys
sys.path.append('./minGPT')
from mingpt.model import GPT as torch_GPT


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.34.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)



# ======================================================================
# Util functions for pytorch
# ======================================================================
def tree_subtract(tree1, tree2):
    """Returns tree1-tree2. Both named_parameters."""
    return {name: param.add(-tree2[name]) for name, param in tree1.items()}


def tree_inner(tree1, tree2):
    """Returns inner product of leaves of tree1 and tree2."""
    inner = 0
    for name, param in tree1.items():
        inner += torch.dot(
            torch.flatten(param),
            torch.flatten(tree2[name])
        ).item()
    return inner


def tree_norm_l2(tree):
    """Returns the l2 norm of flattened leaves."""
    norm_sq = 0
    for _, param in tree.items():
        norm_sq += torch.sum(param**2).item()
    return norm_sq**0.5


def get_opt_state(state_name, optimizer, model):
    """Returns a tree of optimizer states."""
    output = {}
    for name, param in model.named_parameters():
        state = optimizer.state[param]
        output.update({
            name: state.get(state_name, None)
        })
    return output


def compute_prev_loss(model, prev_params, batch):
    """Computes loss of model at prev_params. prev_params is a {name: param} dict."""
    current_params = {name: param.data.clone() for name, param in model.named_parameters()}
    try:
        # Replace model parameters with cloned parameters
        for name, param in model.named_parameters():
            param.data.copy_(prev_params[name])
        with torch.no_grad():
            loss = model(**batch).loss
    finally:
        # Restore original parameters
        for name, param in model.named_parameters():
            param.data.copy_(current_params[name])
    return loss


# ======================================================================
# Custom optimizers: adam and sgdm (too lazy to move to another file)
# ======================================================================
class Sgdm(torch.optim.Optimizer):
    """Integrated implemention of O2NC with OGD-MD.

    Updates x_t = x_{t-1} + s_t*Delta_t,
            Delta_{t+1} = (Delta_t - eta_t * g_t) * [beta / (1 + eta_t*mu)]
    """

    def __init__(
        self, 
        params, 
        lr: float, 
        beta: float = 0.0,
        weight_decay: float = 0.0, 
        random_scaling: bool = False,
    ):
        defaults = dict(lr=lr, beta=beta, wd=weight_decay)
        super(Sgdm, self).__init__(params, defaults)
        self.random_scaling = random_scaling
        self.scalar = 1.0
        # Initialize states.
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['momentum'] = torch.zeros_like(p.data)
                state['Delta'] = torch.zeros_like(p.data)

    def step(self, closure = None):
        # IMPORTANT: we need to clone the gradients from O2NC to the online learner.
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Sample a global random scalar first.
        self.scalar = torch.distributions.Exponential(rate=1).sample() if self.random_scaling else 1.0

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            wd = group['wd']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                # Update sgdm.
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Does not support sparse gradients')
                
                state = self.state[p]
                state['step'] += 1
                state['momentum'] = beta*state['momentum'] + (1-beta)*grad
                state['Delta'] = -lr * (state['momentum'] + wd*p.data)

                # Apply random scaling.
                p.data += self.scalar * state['Delta']


class Adam(torch.optim.Optimizer):
    """Integrated implemention of Randomized AdamW."""

    def __init__(
        self, 
        params, 
        lr: float, 
        b1: float = 0.9, 
        b2: float = 0.999, 
        wd: float = 0.0, 
        eps: float = 1e-8, 
        random_scaling: bool = False
    ):
        defaults = dict(lr=lr, b1=b1, b2=b2, wd=wd, eps=eps)
        super(Adam, self).__init__(params, defaults)
        self.random_scaling = random_scaling
        self.scalar = 1.0
        # Initialize states.
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['mu'] = torch.zeros_like(p.data)
                state['nu'] = torch.zeros_like(p.data)
                state['Delta'] = torch.zeros_like(p.data)

    def step(self, closure = None):
        # IMPORTANT: we need to clone the gradients from O2NC to the online learner.
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Sample a global random scalar first.
        self.scalar = torch.distributions.Exponential(rate=1).sample() if self.random_scaling else 1.0

        for group in self.param_groups:
            lr = group['lr']
            b1 = group['b1']
            b2 = group['b2']
            wd = group['wd']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Does not support sparse gradients')
                
                state = self.state[p]
                state['step'] += 1
                state['mu'] = b1*state['mu'] + (1-b1)*grad
                state['nu'] = b2*state['nu'] + (1-b2)*grad**2
                mu_hat = state['mu']/(1-b1**state['step'])
                nu_hat = state['nu']/(1-b2**state['step'])
                state['Delta'] = -lr * (mu_hat / (eps + torch.sqrt(nu_hat)) + wd * p.data)

                p.data += self.scalar * state['Delta']


# ======================================================================
# Main Training Functions
# ======================================================================
def load_lm_data(config: DictConfig, tokenizer: Any, split: str = "train"):
    """Wrapper for Pile dataset. 
    config: global config.

    Returns:
        torch.utils.data.DataLoader.
    """
    context_length = config.model.context_length
    config = config.dataset
    if config.name not in ["c4", "pile"]:
        raise ValueError("dataset name must be c4 or pile.")
    loader = get_lm_loader_next_token(
        tokenizer,
        split=split,
        batch_size=config.batch_size,
        max_length=context_length,
        shuffle_buffer_size=config.shuffle_buffer_size,
        pad_to_multiple_of=context_length,
        num_workers=config.dataloader_workers,
        dataset=config.name,
    )
    return loader


def init_model(vocab_size: int, config: DictConfig):
    """Initializes model. config=config.model"""
    if config.name == "gpt":
        model_config = torch_GPT.get_default_config()
        model_config.model_type = 'gpt2'
        model_config.vocab_size = vocab_size                    # openai's model vocabulary
        model_config.block_size = config.context_length         # openai's model block_size (i.e. input context length)
        model_config.embd_pdrop = config.transformer_dropout
        model_config.resid_pdrop = config.attn_linear_dropout
        model_config.attn_pdrop = config.attn_dropout
        model = torch_GPT(model_config)
    if config.name == "bert":
        raise ValueError("currently doesn't support bert.")
        # model = init_bert(tokenizer, config)
    return model


def init_optimzier(model: torch.nn.Module, config: DictConfig):
    """Initialize optimizer."""
    train_config = config.train
    config = config.optimizer
    lr_config = config.lr_config

    # Need to separate weight decay for different param groups.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # Initialize optimizer.
    use_random_scaling = train_config.random_scaling is not None
    if config.name == "adamw":
        optimizer = Adam(
            optimizer_grouped_parameters, 
            lr=lr_config.lr,
            b1=config.beta1,
            b2=config.beta2,
            wd=config.weight_decay,
            random_scaling=use_random_scaling
        )
    elif config.name == "sgdm":
        optimizer = Sgdm(
            optimizer_grouped_parameters, 
            lr=lr_config.lr, 
            beta=config.beta,
            weight_decay=config.weight_decay,
            random_scaling=use_random_scaling
        )

    # Initialize scheduler.
    lr_scheduler = get_scheduler(
        name=lr_config.schedule,
        optimizer=optimizer,
        num_warmup_steps=lr_config.warmup,
        num_training_steps=lr_config.max_steps
    )
    return optimizer, lr_scheduler


class Logstate(NamedTuple):
    iteration: int
    params_diff: dict
    delta: dict
    random_scalar: torch.Tensor
    logs: dict


def init_logstate(model):
    logstate = Logstate(
        iteration=1,
        params_diff={name: torch.zeros_like(param) for name, param in model.named_parameters()},
        delta={name: torch.zeros_like(param) for name, param in model.named_parameters()},
        random_scalar=torch.ones([]),
        logs = {
            "f(x_t,z_t)": 0.0,
            "f(x_t,z_t)_avg": 0.0,
            "smooth/<g_t, x_t-x_{t-1}>": 0.0,
            "smooth/<g_t, x_t-x_{t-1}>_sum": 0.0,
            "smooth/<g_t, Delta_t>": 0.0,
            "smooth/<g_t, Delta_t>_sum": 0.0,
            "smooth/f(x_t,z_t)-f(x_{t-1},z_t)": 0.0,
            "smooth/f(x_t,z_t)-f(x_{t-1},z_t)_sum": 0.0,
            "norm/s_t": 1.0,
            "norm/|x_t-x_{t-1}|": 0.0,
            "norm/|Delta_t|": 0.0,
            "norm/|g_t|": 0.0,
            "sancheck/|Delta_t|": 0.0,
            "sancheck/<g_t, Delta_t>": 0.0,
        }
    )
    return logstate


def train_step(
    logstate,
    model,
    batch,
    optimizer,
    lr_scheduler,
    accelerator,
    config: DictConfig,
) -> Logstate:
    model.train()
    idx, targets = batch
    with accelerator.accumulate(model):
        # ==========================================================================
        # Auxilliary computation for logging.
        params_diff = logstate.params_diff                                                      # x_t-x_{t-1}
        current_params = {name: param.data.clone() for name, param in model.named_parameters()} # x_t
        prev_params = tree_subtract(current_params, params_diff)                          # x_{t-1}
        # Compute f(x_{t-1},z_t)
        optimizer.zero_grad()
        try:
            # Replace model parameters with cloned parameters
            for name, param in model.named_parameters():
                param.data.copy_(prev_params[name])
            with torch.no_grad():
                logits, _ = model(idx)
                prev_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                prev_loss = prev_loss.detach().float()
        finally:
            # Restore original parameters
            for name, param in model.named_parameters():
                param.data.copy_(current_params[name])

        # ======================================================================
        # Actual training.
        # ======================================================================
        # Forward and backward pass.
        # NOTE: we need to manually call cross-entropy since mingpt uses ignore_index=-1,
        # while our tokenizer uses the default value ignore_index=-100.
        optimizer.zero_grad()
        logits, _ = model(idx)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_val)
        optimizer.step()
        lr_scheduler.step()         # update x_t to x_{t+1}, computes s_{t+1} and Delta_{t+1}

        # Update statistics.
        current_loss = loss.detach().float()                                                    # f(x_t,z_t)
        grads = {name: param.grad.clone() for name, param in model.named_parameters()}          # g(x_t, z_t)
        new_scalar = optimizer.optimizer.scalar                                                 # s_{t+1}
        new_delta = get_opt_state("Delta", optimizer.optimizer, model)                    # Delta_{t+1}=(x_{t+1}-x_t)/s_{t+1}
        next_params = {name: param.data.clone() for name, param in model.named_parameters()}    # x_{t+1}
        new_params_diff = tree_subtract(next_params, current_params)                      # x_{t+1}-x_t

        # ======================================================================
        # Compute logging statistics.
        # ======================================================================
        logs = logstate.logs
        iteration = logstate.iteration                                                          # t
        random_scalar = logstate.random_scalar                                                  # s_t
        params_diff = logstate.params_diff                                                      # x_t-x_{t-1}
        delta = logstate.delta                                                                  # Delta_t

        avg_loss = (logs["f(x_t,z_t)_avg"] * (iteration-1) + current_loss) / iteration
        inner_g_dx = tree_inner(grads, params_diff)                                       # <g_t, x_t-x_{t-1}>
        inner_g_dx_sum = logs["smooth/<g_t, x_t-x_{t-1}>_sum"] + inner_g_dx
        inner_g_delta = tree_inner(grads, delta)                                          # <g_t, Delta_t>
        inner_g_delta_sum = logs["smooth/<g_t, Delta_t>_sum"] + inner_g_delta
        loss_diff = current_loss - prev_loss                                                    # f(x_t,z_t)-f(x_{t-1},z_t)
        loss_diff_sum = logs["smooth/f(x_t,z_t)-f(x_{t-1},z_t)_sum"] + loss_diff
        norm_dx = tree_norm_l2(params_diff)
        norm_delta = tree_norm_l2(delta)
        logs.update({
            "f(x_t,z_t)": current_loss,
            "f(x_t,z_t)_avg": avg_loss,
            "smooth/<g_t, x_t-x_{t-1}>": inner_g_dx,
            "smooth/<g_t, x_t-x_{t-1}>_sum": inner_g_dx_sum,
            "smooth/<g_t, Delta_t>": inner_g_delta,
            "smooth/<g_t, Delta_t>_sum": inner_g_delta_sum,
            "smooth/f(x_t,z_t)-f(x_{t-1},z_t)": loss_diff,
            "smooth/f(x_t,z_t)-f(x_{t-1},z_t)_sum": loss_diff_sum,
            "norm/s_t": random_scalar,
            "norm/|x_t-x_{t-1}|": norm_dx,
            "norm/|Delta_t|": norm_delta,
            "norm/|g_t|": tree_norm_l2(grads),
            "sancheck/|Delta_t|": norm_dx - random_scalar*norm_delta,
            "sancheck/<g_t, Delta_t>": inner_g_dx - random_scalar*inner_g_delta,
        })
    
    return Logstate(
        iteration=iteration+1,
        params_diff=new_params_diff,
        delta=new_delta,
        random_scalar=new_scalar,
        logs=logs,
    )


def train(config: DictConfig) -> None:
    send_example_telemetry("run_clm_no_trainer", Namespace(**config))

    # Initialize pytorch accelerator
    accelerator = Accelerator(gradient_accumulation_steps=1)    # default to no gradient accumulation
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set training seed.
    set_seed(config.random_seed)

    accelerator.wait_for_everyone()

    # ======================================================================
    # Training starts here...
    # ======================================================================
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    train_loader = load_lm_data(config, tokenizer)
    
    model = init_model(len(tokenizer), config.model)
    optimizer, lr_scheduler = init_optimzier(model, config)
    
    # Prepare everything with our `accelerator`.
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Main train loop.
    logstate = init_logstate(model)
    max_steps = config.train.max_steps
    pbar = tqdm.tqdm(enumerate(train_loader), total=max_steps)
    for it, batch in pbar:
        if it > max_steps:
            break
        idx, targets = batch["input_ids"], batch["labels"]
        logstate = train_step(
            logstate, model, (idx, targets), optimizer, lr_scheduler, accelerator, config
        )
        pbar.set_description(f"iteration: {logstate.iteration-1}, avg_train_loss: {logstate.logs['f(x_t,z_t)_avg']:.2f}")
        if config.logging.wandb_project:
            wandb.log(logstate.logs, step=logstate.iteration)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    # TODO: this is temporary
    seed = config.random_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    logging.info(OmegaConf.to_yaml(config))
    if config.logging.wandb_project:
        wandb.init(project=config.logging.wandb_project)
        wandb.config.update(OmegaConf.to_container(config))
    train(config)


if __name__ == "__main__":
    main()