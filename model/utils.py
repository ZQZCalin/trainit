import jax
from jax import tree_util as jtu
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array

import torch
import equinox as eqx
from equinox import nn


def state_dict_to_pytree():
    """Bruteforce parsing the pytorch state_dict of mingpt model into jax pytree.

    The state_dict should be model.state_dict() where model is a GPT class from
    the [mingpt repo](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py).
    Otherwise, the naming will not be recognized.
    """
    lookup_table = {
        # gpt
        "transformer.wte": "token_embedding",
        "transformer.wpe": "position_embedding",
        "transformer.h": "transformer_blocks",
        "transformer.ln_f": "ln",
        "lm_head": "head",
        # transformer block
        "ln_1": "ln1",
        "attn": "attn",
        "ln_2": "ln2",
        "mlp.c_fc": "expand_fc",
        "mlp.c_proj": "reduce_fc",
        # attention
        "c_attn": "attn_fc",
        "c_proj": "linear",
    }