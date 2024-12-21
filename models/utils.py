import jax
from jax import tree_util as jtu
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array

import re
import torch
import equinox as eqx
from equinox import nn


def parse_state_dict(params: list[str]) -> list[str]:
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
    output = []
    for param in params:
        # Manually drops the mask matrices in multi-head attention layers.
        if ".attn.bias" in param:
            continue
        for key, value in lookup_table.items():
            param = re.sub(re.escape(key), value, param)
        output.append(param)
    return output