"""
GPT model, based on the papers:
[GPT1] Improving Language Understanding by Generative Pre-Training
[GPT2] Language Models are Unsupervised Multitask Learners
[GPT3] Language Models are Few-Shot Learners
[Att] Attention is All You Need

Equinox implementation of GPT model based on the [min-gpt repo](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py).
"""
import jax
from jax import numpy as jnp

import equinox as eqx
from equinox import nn
from jax import random as jr
from typing import Union, Optional, Sequence

from jax import named_scope
# from jax.random import PRNGKeyArray
from jax import Array as PRNGKeyArray


class CausalSelfAttention(eqx.Module):
    """
    simple attention class.
    """

    attn: eqx.Module

    def __init__(self, config, *, key: PRNGKeyArray):
        dim = config.dim
        num_heads = config.num_heads

        assert dim % num_heads == 0

        self.attn = nn.MultiheadAttention(num_heads, dim, key=key)

    @named_scope("gpt.CausalSelfAttention")
    def __call__(self, data, *, key: Optional[PRNGKeyArray] = None):
        """
        data is [L, D]
        """

        L, D = data.shape
        mask = jnp.tril(jnp.ones((L, L)))
        return self.attn(data, key_=data, value=data, mask=mask, key=key)


# the built-in layer norm in equinox normalizes the entire input.
# we don't want that, so we need to vmap over the dimensions we don't
# want to normalize over.
class AxisLayerNorm(eqx.Module):
    ln: nn.LayerNorm
    axes: Union[int, Sequence[int]] = eqx.field(static=True)

    def __init__(self, *args, axes=0, **kwargs):
        self.ln = nn.LayerNorm(*args, **kwargs)
        self.axes = axes

    @named_scope("gpt.AxisLayerNorm")
    def __call__(self, x, *args, **kwargs):
        def apply_ln(x):
            return self.ln(x, *args, **kwargs)

        vmapped_ln = jax.vmap(apply_ln, in_axes=self.axes, out_axes=self.axes)
        return vmapped_ln(x)


class TransformerBlock(eqx.Module):
    dim: int = eqx.field(static=True)
    fc_dim: int = eqx.field(static=True)
    attn: CausalSelfAttention
    ln1: AxisLayerNorm
    ln2: AxisLayerNorm
    expand_fc: nn.Linear
    reduce_fc: nn.Linear

    def __init__(self, config, *, key=PRNGKeyArray):
        self.dim = config.dim
        self.fc_dim = config.get("fc_dim", 4 * self.dim)
        if self.fc_dim is None:
            self.fc_dim = 4 * self.dim

        attn_key, expand_key, reduce_key = jr.split(key, 3)

        self.attn = CausalSelfAttention(config, key=attn_key)

        self.ln1 = AxisLayerNorm(self.dim)
        self.ln2 = AxisLayerNorm(self.dim)

        self.expand_fc = nn.Linear(
            self.dim, self.fc_dim, use_bias=config.bias, key=expand_key
        )
        self.reduce_fc = nn.Linear(
            self.fc_dim, self.dim, use_bias=config.bias, key=reduce_key
        )

        #
        # [GPT2] says:
        # "
        # A modified initialization which accounts
        # for the accumulation on the residual path with model depth
        # is used. We scale the weights of residual layers at
        # initialization by a factor of 1/\sqrt{N} where N is the number
        # of residual layers.
        # "
        # This is pretty vague: what exactly is a residual layer? I will
        # choose to interpret this as only the last layer of the MLP of the
        # transformer block, but it could just as well also mean the value
        # matrices in the attention layers, or even just all weights in the
        # transformer block.
        #
        # Complicating this story even further, I cannot find any evidence of
        # this rescaling presented in the "official" code here:
        # https://github.com/openai/gpt-2/blob/master/src/model.py
        #
        if config.rescale_residuals:
            self.reduce_fc = eqx.tree_at(
                where=lambda t: t.weight,
                pytree=self.reduce_fc,
                replace=self.reduce_fc.weight / jnp.sqrt(config.num_blocks),
            )

    @named_scope("gpt.TransformerBlock")
    def __call__(self, data, *, key: Optional[PRNGKeyArray] = None):
        """
        data is [L, D] (we will vmap over the batch dimension)
        """
        # Order of these operations described in section 2.3 of [GPT2].
        # That text unfortunately does not actually describe the model,
        # but provides a "diff" from the model described  in Fig 1 of [GPT2]
        # The best reference I can find for the MLP using 4x the input dim as
        # the intermediate layer is [Att]
        #
        out = self.ln1(data)                    # [L, D]
        out = self.attn(out, key=key)           # [L, D]
        post_attn = data + out                  # [L, D]
        out = self.ln2(post_attn)               # [L, D]
        out = jax.vmap(self.expand_fc)(out)     # [L, D] -> [D] with vmap over L -> [L, D]
        out = jax.nn.gelu(out)                  # [L, D]
        out = jax.vmap(self.reduce_fc)(out)     # [L, D] -> [D] with vmap over L -> [L, D]
        out = post_attn + out                   # [L, D]

        return out


class GPT(eqx.Module):
    vocab_size: int = eqx.field(static=True)
    context_length: int = eqx.field(static=True)
    blocks: eqx.Module
    token_embedding: eqx.Module
    position_embedding: eqx.Module
    ln: AxisLayerNorm
    head: nn.Linear

    def __init__(self, vocab_size, config, *, key: PRNGKeyArray):
        # self.config = config

        sequential_key, token_key, position_key, head_key = jr.split(key, 4)
        sequential_key = jr.split(sequential_key, config.num_blocks)

        self.vocab_size = vocab_size
        self.context_length = config.context_length
        self.blocks = nn.Sequential(
            [
                TransformerBlock(config, key=sequential_key[i])
                for i in range(config.num_blocks)
            ]
        )

        self.token_embedding = nn.Embedding(self.vocab_size, config.dim, key=token_key)
        self.position_embedding = nn.Embedding(
            config.context_length, config.dim, key=position_key
        )

        self.ln = AxisLayerNorm(config.dim)

        self.head = nn.Linear(
            config.dim, self.vocab_size, use_bias=config.bias, key=head_key
        )

    @named_scope("gpt.GPT")
    def __call__(self, token_indices, *, key: Optional[PRNGKeyArray] = None):
        """
        token indices will be shape [L] - we will vmap over the batch dimension.
        """

        L = len(token_indices)
        assert L <= self.context_length

        # From Equation (2) of [GPT1], we only do tokens + positions here and
        # then no other position encoding to get GPT-1. The descriptions of
        # GPT-2 and GPT-3 do not say to alter this part.
        # No idea if current GPT-4 type stuff does more fancy things, but even
        # the results in [GPT3] looked amazing.

        tokens = jax.vmap(self.token_embedding, in_axes=0, out_axes=0)(token_indices)  # [L, D]
        positions = self.position_embedding.weight[:L, :]  # [L, D]
        data = tokens + positions
        out = self.blocks(data,  key=key)

        # [GPT2] says to add an extra layer normalization after the transformer
        # blocks. [GPT3] doesn't really say to change anything - it just makes
        # the model bigger.
        out = self.ln(out)

        logits = jax.vmap(self.head)(out)   # [L, D] -> vmap over L -> [L, D]

        return logits