# A simplified script to log non-smooth statistics.
# NOTE: This code is not updated since 2024/06 (and probably won't be updated in the future)
import logging
import random
from itertools import chain
from pathlib import Path
import wandb
import dataloaders
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from dataloaders import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
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
import hydra
from omegaconf import OmegaConf, DictConfig
from argparse import Namespace
from typing import NamedTuple
import sys
sys.path.append('./minGPT')
from mingpt.model import GPT as minGPT
from dataloaders.lm_loader import get_lm_loader_next_token


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


def check_frequency(tensor, threshold, ignores=None):
    """Returns true if there exists some element in tensor that has frequency > threshold.
    
    If ignores (array-like) is not None, will not count the frequency in ignores.
    """
    tensor = tensor.flatten()
    # Mask out ignored elements.
    if ignores is not None:
        mask = torch.ones_like(tensor, dtype=bool)
        for value in ignores:
            mask &= (tensor != value)
        tensor = tensor[mask]
    threshold_count = tensor.numel() * threshold
    _, counts = torch.unique(tensor, return_counts=True)
    return bool((counts > threshold_count).any())


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
def init_tokenizer(config: DictConfig):
    """Initializes tokenizer. config: global config"""
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, trust_remote_code=False)
    if not config.experimental.use_loadit:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer


def init_dataloaders(tokenizer, config: DictConfig):
    """Initializes datasets. config=config.dataset"""
    if config.experimental.use_loadit:
        return LoadIt(root_dir="/projectnb/aclab/tranhp/trainDataloader_pile/", max_workers=1)
    else:
        context_length = config.model.context_length
        config = config.dataset
        if config.name not in ["c4", "pile"]:
            raise ValueError("dataset name must be c4 or pile.")
        return get_lm_loader_next_token(
            tokenizer,
            split="train",
            batch_size=config.batch_size,
            max_length=context_length,
            shuffle_buffer_size=config.shuffle_buffer_size,
            pad_to_multiple_of=context_length,
            num_workers=config.dataloader_workers,
            dataset=config.name,
        )


def init_gpt2(tokenizer):
    """Initializes GPT2 model. config=config.model"""
    model_conf = AutoConfig.from_pretrained("gpt2", trust_remote_code=False)
    ## turn off dropout
    model_conf.attn_pdrop = 0.0
    model_conf.resid_pdrop = 0.0
    model_conf.embd_pdrop = 0.0
    model = AutoModelForCausalLM.from_config(model_conf)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    return model


def init_mingpt(tokenizer):
    model_config = minGPT.get_default_config()
    model_config.model_type = 'gpt2'
    model_config.vocab_size = len(tokenizer)                    # openai's model vocabulary
    model_config.block_size = 1024         # openai's model block_size (i.e. input context length)
    model_config.embd_pdrop = 0
    model_config.resid_pdrop = 0
    model_config.attn_pdrop = 0
    model = minGPT(model_config)
    return model


def init_model(tokenizer, config: DictConfig):
    """Initializes model. config: global config"""
    if config.model.name == "gpt":
        if config.experimental.use_hugging_face:
            return init_gpt2(tokenizer)
        else:
            return init_mingpt(tokenizer)
    # if config.name == "bert-base-uncased":
    else:
        raise ValueError("only support gpt now.")


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


def loss_fn(model, batch, use_hugging_face, use_loadit):
    """Wrapper of loss function: if not using hugging face model (e.g., mingpt), then manually cmopute loss."""
    if use_hugging_face:
        if use_loadit:
            return model(**batch).loss
        else:
            # raise ValueError("currently do not support padding for hugging face.")
            # Manually disable label shifting here.
            logits = model(**batch).logits
            labels = batch["labels"]
            return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    else:
        idx, targets = batch["input_ids"], batch["labels"]
        logits, _ = model(idx)
        if use_loadit:
            # SHIFT the labels by 1 and drop the last label.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
        else:
            # Label already shifted.
            shift_logits = logits
            shift_labels = targets
        return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def train_step(
    logstate,
    model,
    batch,
    optimizer,
    lr_scheduler,
    accelerator,
    config
) -> Logstate:
    clip_norm = config.train.gradient_clip_val
    use_hugging_face = config.experimental.use_hugging_face
    use_loadit = config.experimental.use_loadit

    model.train()
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
                # prev_loss = model(**batch).loss.detach().float()
                prev_loss = loss_fn(model, batch, use_hugging_face, use_loadit)
        finally:
            # Restore original parameters
            for name, param in model.named_parameters():
                param.data.copy_(current_params[name])

        # ==========================================================================
        # Actual training.
        optimizer.zero_grad()
        # loss = model(**batch).loss
        loss = loss_fn(model, batch, use_hugging_face, use_loadit)
        accelerator.backward(loss)
        if clip_norm:               # optional gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        lr_scheduler.step()         # update x_t to x_{t+1}, computes s_{t+1} and Delta_{t+1}

        current_loss = loss.detach().float()                                                    # f(x_t,z_t)
        grads = {name: param.grad.clone() for name, param in model.named_parameters()}          # g(x_t, z_t)
        new_scalar = optimizer.optimizer.scalar                                                 # s_{t+1}
        new_delta = get_opt_state("Delta", optimizer.optimizer, model)                    # Delta_{t+1}=(x_{t+1}-x_t)/s_{t+1}
        next_params = {name: param.data.clone() for name, param in model.named_parameters()}    # x_{t+1}
        new_params_diff = tree_subtract(next_params, current_params)                      # x_{t+1}-x_t

        # ==========================================================================
        # Compute logging statistics.
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
    accelerator = Accelerator(gradient_accumulation_steps=1)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        dataloaders.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        dataloaders.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set training seed.
    set_seed(config.random_seed)

    accelerator.wait_for_everyone()

    # ===============================================================================================
    # Training starts here...
    tokenizer = init_tokenizer(config)
    model = init_model(tokenizer, config)
    # EXPERIMENTAL: load checkpoint state_dict
    ckpt_config = config.experimental.load_checkpoint
    if ckpt_config.use:
        model.load_state_dict(torch.load(ckpt_config.path))
    train_dataloader = init_dataloaders(tokenizer, config)
    eval_dataloader = None
    optimizer, lr_scheduler = init_optimzier(model, config)
    
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Main train loop.
    logstate = init_logstate(model)
    max_steps = config.train.max_steps
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # EXPERIMENTAL: save checkpoint
    ckpt_config = config.experimental.save_checkpoint
    filter_config = config.experimental.data_conditioning

    if config.experimental.use_loadit:
        num_data_points = 1000000
        sample_size = max_steps
        # sampled_indices = random.sample(range(num_data_points), sample_size)
        sampled_indices = random.sample(range(num_data_points), 2*sample_size)

        # EXPERIMENTAL: turn off streaming of loadit
        if config.experimental.use_streaming_loadit:
            # sampled_indices = range(max_steps)
            sampled_indices = range(2*max_steps)

        pbar = tqdm(enumerate(sampled_indices), total=max_steps)
        for it, batch_idx in pbar:
            if logstate.iteration > max_steps:
                break
            # EXPERIMENTAL: save checkpoint model and terminate
            if ckpt_config.use and logstate.iteration > ckpt_config.iter:
                torch.save(
                    accelerator.unwrap_model(model).state_dict(),
                    ckpt_config.path
                )
                break
            batch = train_dataloader[batch_idx]
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

            # EXPERIMENTAL: apply data filtering
            if filter_config.use and check_frequency(batch["input_ids"], filter_config.threshold):
                continue

            logstate = train_step(logstate, model, batch, optimizer, lr_scheduler, accelerator, config)

            pbar.set_description(f"iteration: {it+1}, avg_train_loss: {logstate.logs['f(x_t,z_t)_avg']:.2f}")
            if config.logging.wandb_project:
                logs = logstate.logs
                logs.update({"batches": it})
                wandb.log(logs, step=logstate.iteration)
    else:
        pbar = tqdm(enumerate(train_dataloader), total=max_steps)
        for it, batch in pbar:
            if logstate.iteration > max_steps:
                break
            # EXPERIMENTAL: save checkpoint model and terminate
            if ckpt_config.use and logstate.iteration > ckpt_config.iter:
                torch.save(
                    accelerator.unwrap_model(model).state_dict(),
                    ckpt_config.path
                )
                break

            # EXPERIMENTAL: apply data filtering (NOTE: we ignore the padding token)
            pad_token = tokenizer("<|pad|>")["input_ids"]
            if filter_config.use and check_frequency(batch["input_ids"], filter_config.threshold, ignores=pad_token):
                continue

            logstate = train_step(logstate, model, batch, optimizer, lr_scheduler, accelerator, config)

            pbar.set_description(f"iteration: {it+1}, avg_train_loss: {logstate.logs['f(x_t,z_t)_avg']:.2f}")
            if config.logging.wandb_project:
                logs = logstate.logs
                logs.update({"batches": it})
                wandb.log(logs, step=logstate.iteration)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    logging.info(OmegaConf.to_yaml(config))
    if config.logging.wandb_project:
        wandb.init(project=config.logging.wandb_project)
        wandb.config.update(OmegaConf.to_container(config))
    train(config)


if __name__ == "__main__":
    main()