import jax
from jax import tree_util as jtu
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, PyTree

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


def summarize_model_parmas(
        model: eqx.Module,
        verbose: bool = True,
) -> PyTree:
    """Summarizes model parameters with aligned output."""
    
    def get_path(path, p):
        # Parse path list into a single string.
        parts = []
        for part in path:
            if isinstance(part, jtu.GetAttrKey):
                parts.append(part.name)
            elif isinstance(part, jtu.SequenceKey):
                parts[-1] += f"[{part.idx}]"
        return ".".join(parts)

    params = eqx.filter(model, eqx.is_array)
    path_tree = jtu.tree_map_with_path(get_path, params)
    path_list, treedef = jtu.tree_flatten(path_tree)
    shape_list, _ = jtu.tree_flatten(
        jtu.tree_map(lambda p: str(p.shape), params)
    )
    
    # Format summarize print lines.
    max_len = max(len(path) for path in path_list)
    lines = [f"{path:<{max_len+3}}| {shape}" for path, shape in zip(path_list, shape_list)]

    if verbose:
        print("\n".join(lines))

    return jtu.tree_unflatten(treedef, lines)