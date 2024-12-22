"""
Equinox implementation of GPT2 model based on the [mingpt repo](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py).

Note: this implementation supports loading parameters from pytorch models.
"""
import jax
from jax import named_scope
from jax import numpy as jnp
from jax import random as jr

import equinox as eqx
from equinox import nn
from . import extend_nn as enn

import torch

from typing import Union, Optional, Sequence, Literal
from jaxtyping import Array, PRNGKeyArray
import re


StateDict = dict[str, torch.Tensor]


class CausalSelfAttention(eqx.Module):
    dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    attn_fc: nn.Linear
    linear: nn.Linear
    attn_dropout: nn.Dropout
    linear_dropout: nn.Dropout
    # mask: Array
    mask: Array = eqx.field(static=True)    # updated 09/23: declare the mask layer to be static to avoid getting updated

    def __init__(self, config, state_dict: Optional[StateDict] = None, *, key: Optional[PRNGKeyArray] = None):
        """Initializes a multi-head self-attention layer for GPT model.

        Args:
            config (_type_): _description_
            key (PRNGKeyArray): _description_
        """
        assert config.dim % config.num_heads == 0, \
            f"Number of heads {config.num_heads} does not divide embedding dimension {config.dim}."
        self.dim = config.dim
        self.num_heads = config.num_heads
        
        if state_dict is None:
            self._init_default(config, key=key)
        else:
            self._init_from_torch(config, state_dict=state_dict)

    def _init_default(self, config, *, key: PRNGKeyArray):
        """Default initialization."""
        attn_key, linear_key = jr.split(key, 2)
        self.attn_fc = nn.Linear(self.dim, 3*self.dim, key=attn_key)
        self.linear = nn.Linear(self.dim, self.dim, key=linear_key)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.linear_dropout = nn.Dropout(config.attn_linear_dropout)
        # Define the mask index matrix: lower triangular one matrix.
        # This masks out upper triangles where j > i in i-th row (which correspond to future tokens).
        self.mask = jnp.tril(jnp.ones((config.context_length, config.context_length)))

    def _init_from_torch(self, config, *, state_dict: StateDict):
        """Initialize from pytorch state_dict."""
        def get_params(name):
            return jnp.array(state_dict[name])
        self.attn_fc = enn.Linear(weight=get_params("c_attn.weight"), bias=get_params("c_attn.bias"))
        self.linear = enn.Linear(weight=get_params("c_proj.weight"), bias=get_params("c_proj.bias"))
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.linear_dropout = nn.Dropout(config.attn_linear_dropout)
        self.mask = jnp.tril(jnp.ones((config.context_length, config.context_length)))

    @named_scope("gpt2.CausalSelfAttention")
    def __call__(self, data, *, key: PRNGKeyArray):
        """data [L,D]"""
        L, D = data.shape
        d = self.dim // self.num_heads
        n = self.num_heads
        assert D == self.dim, \
            f"input embedding size {D} doesn't match model embedding size {self.dim}."
        
        attn_key, linear_key = jr.split(key, 2)
        
        # Apply attn_fc and split into multi-heads of q,k,v.
        qkv = jax.vmap(self.attn_fc)(data)                  # [L,D] -> [L,3D]
        qkv = jnp.reshape(qkv, (L,3,n,d))                   # [L,3D] -> [L,3,n,d]
        qkv = jnp.transpose(qkv, (1,2,0,3))                 # [L,3,n,d] -> [3,n,L,d]
        q, k, v = qkv[0], qkv[1], qkv[2]                    # [n,L,d]

        # Compute attention score for each head.
        def _mat_mul(m1, m2):
            """both [L,d]. returns m1@m1.T [L,L]."""
            return jnp.matmul(m1, jnp.transpose(m2))
        attn = jax.vmap(_mat_mul)(q, k) / jnp.sqrt(d)       # [n,L,d] @ [n,d,L] -> [n,L,L]
        def _causal_mask(m):
            return jnp.where(self.mask[:L, :L], m, float("-inf"))
        attn = jax.vmap(_causal_mask)(attn)                 # [n,L,L]
        attn = jax.nn.softmax(attn, axis=-1)                # [n,L,L]
        attn = self.attn_dropout(attn, key=attn_key)        # [n,L,L]

        # Concatenate all heads and apply final linear layer.
        output = jax.vmap(jnp.matmul)(attn, v)              # [n,L,L] @ [n,L,d] -> [n,L,d]
        output = jnp.transpose(output, (1,0,2))             # [n,L,d] -> [L,n,d]
        output = jnp.reshape(output, (L,D))                 # [L,n,d] -> [L,D]
        output = jax.vmap(self.linear)(output)              # [L,D] -> [L,D]
        output = self.linear_dropout(output, key=linear_key)
        return output


class TransformerBlock(eqx.Module):
    ln1: nn.LayerNorm
    attn: CausalSelfAttention
    ln2: nn.LayerNorm
    expand_fc: nn.Linear
    reduce_fc: nn.Linear
    dropout: nn.Dropout

    def __init__(self, config, state_dict: Optional[StateDict] = None, *, key: Optional[PRNGKeyArray] = None):
        """Initializes a transformer block for GPT model.

        Args:
            config: specifies (dim, transformer_dropout).
            key: random key for initialization.
        """
        if state_dict is None:
            self._init_default(config, key=key)
        else:
            self._init_from_torch(config, state_dict=state_dict)

    def _init_default(self, config, *, key: PRNGKeyArray):
        """Default initialization."""
        attn_key, expand_key, reduce_key = jr.split(key, 3)
        self.ln1 = nn.LayerNorm(config.dim)         # nn.LayerNorm is deterministically initialized to ones, so no key needed.
        self.attn = CausalSelfAttention(config, key=attn_key)
        self.ln2 = nn.LayerNorm(config.dim)
        self.expand_fc = nn.Linear(config.dim, 4*config.dim, key=expand_key)
        self.reduce_fc = nn.Linear(4*config.dim, config.dim, key=reduce_key)
        self.dropout = nn.Dropout(p=config.transformer_dropout)

    def _init_from_torch(self, config, *, state_dict: StateDict):
        """Initialize from pytorch state_dict"""
        # Parse attention params into dictionary.
        attn_params = {}
        pattern = re.compile(r'^attn\.')    # use ^ to indicate word start.
        for k in state_dict.keys():
            match = pattern.match(k)
            if match:
                new_k = pattern.sub('', k)
                attn_params.update({new_k: state_dict[k]})

        # Construct transformer model.
        def get_params(name):
            return jnp.array(state_dict[name])
        self.ln1 = enn.LayerNorm(weight=get_params("ln_1.weight"), bias=get_params("ln_1.bias"))
        self.attn = CausalSelfAttention(config, state_dict=attn_params)
        self.ln2 = enn.LayerNorm(weight=get_params("ln_2.weight"), bias=get_params("ln_2.bias"))
        self.expand_fc = enn.Linear(weight=get_params("mlp.c_fc.weight"), bias=get_params("mlp.c_fc.bias"))
        self.reduce_fc = enn.Linear(weight=get_params("mlp.c_proj.weight"), bias=get_params("mlp.c_proj.bias"))
        self.dropout = nn.Dropout(p=config.transformer_dropout)

    @named_scope("gpt2.TransformerBlock")
    def __call__(self, data, *, key: PRNGKeyArray):
        """data: embedded single sentence, an [L,D] array."""
        attn_key, dropout_key = jr.split(key, 2)

        x = jax.vmap(self.ln1)(data)                # [L,D], apply layer norm to each token (last coordinate)
        x = self.attn(x, key=attn_key)              # [L,D]
        x += data                                   # [L,D]

        y = jax.vmap(self.ln2)(x)                   # [L,D]
        y = jax.vmap(self.expand_fc)(y)             # [L,D] -> [L,4D]
        y = jax.nn.gelu(y)                          # [L,4D]
        y = jax.vmap(self.reduce_fc)(y)             # [L,4D] -> [L,D]
        y = self.dropout(y, key=dropout_key)        # [L,D]
        output = x + y                              # [L,D]
        return output


class GPT(eqx.Module):
    context_length: int = eqx.field(static=True)
    token_embedding: nn.Embedding
    position_embedding: nn.Embedding
    dropout: nn.Dropout
    transformer_blocks: eqx.Module
    ln: nn.LayerNorm
    head: nn.Linear

    def __init__(self, vocab_size, config, state_dict: Optional[StateDict] = None, *, key: Optional[PRNGKeyArray] = None):
        """Initializes GPT model based on config.
        
        Args:
            vocab_size: vocabulary size of the tokenizer. this will be directly loaded from tokenizer instead of config.
            config: a dictionary that specifies (
                    dim, num_heads, bias, num_blocks, context_length, 
                    gpt_dropout, transformer_dropout, attn_dropout, attn_linear_dropout
                ).
            key: random key used for initialization.
            state_dict: pytorch model.state_dict(). If state_dict is provided, the model is intialized using provided params.
        
        Note:
            If state_dict is provided, we assume config matches the pytorch config.
            We *do not* check whether two configs match.
            However, the extended Linear and LayerNorm checks dimensionality.
        """
        self.context_length = config.context_length

        if state_dict is None:
            self._init_default(vocab_size, config, key=key)
        else:
            self._init_from_torch(config, state_dict=state_dict)
    
    def _init_default(self, vocab_size, config, *, key: PRNGKeyArray):
        """Initializes GPT with random initialization."""
        token_key, position_key, transformer_key, head_key = jr.split(key, 4)
        transformer_key = jr.split(transformer_key, config.num_blocks)

        self.token_embedding = nn.Embedding(vocab_size, config.dim, key=token_key)
        self.position_embedding = nn.Embedding(config.context_length, config.dim, key=position_key)
        self.dropout = nn.Dropout(config.gpt_dropout)
        self.transformer_blocks = nn.Sequential([
            TransformerBlock(config, key=transformer_key[i]) for i in range(config.num_blocks)
        ])
        self.ln = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, vocab_size, use_bias=config.head_bias, key=head_key)

    def _init_from_torch(self, config, *, state_dict: StateDict):
        """Initializes GPT with provided pytorch params."""
        # Pre-process state_dict: split transformer blocks into list.
        pattern = re.compile(r'^transformer\.h\.(\d+)\.')
        num_blocks = float("-inf")
        for k in state_dict.keys():
            match = pattern.match(k)
            if match:
                n = int(match.group(1))
                if n+1 > num_blocks:
                    num_blocks = n+1
        if config.num_blocks != num_blocks:
            raise ValueError("number of transformer blocks must match.")
        
        transformer_params = [{} for _ in range(num_blocks)]
        for k in state_dict.keys():
            match = pattern.match(k)
            if match:
                n = int(match.group(1))
                new_k = pattern.sub('', k)
                transformer_params[n].update({new_k: state_dict[k]})

        # Construct model.
        def get_params(name):
            return jnp.array(state_dict[name])
        self.token_embedding = nn.Embedding(weight=get_params("transformer.wte.weight"))
        self.position_embedding = nn.Embedding(weight=get_params("transformer.wpe.weight"))
        self.dropout = nn.Dropout(config.gpt_dropout)
        self.transformer_blocks = nn.Sequential([
            TransformerBlock(config, state_dict=transformer_params[i]) for i in range(config.num_blocks)
        ])
        self.ln = enn.LayerNorm(weight=get_params("transformer.ln_f.weight"), bias=get_params("transformer.ln_f.bias"))
        self.head = enn.Linear(weight=get_params("lm_head.weight"))

    @named_scope("gpt2.GPT")
    def __call__(self, input, *, key: PRNGKeyArray):
        """input is single input, a [L,] sequence of tokenized sentence."""
        L = len(input)
        assert L <= self.context_length, f"Input size {L} exceeds context length{self.context_length}."

        transformer_key, dropout_key = jr.split(key, 2)

        tokens = jax.vmap(self.token_embedding)(input)      # [L,D]
        # for position embedding, there's a more clever way than using jax.vmap(...)([1,2,...,L]).
        # note that the positional embedding of the i-th token is always the i-th row of the lookup table.
        # in other words, the position embedded data is just the first L rows of the lookup table.
        positions = self.position_embedding.weight[:L, :]   # [L,D]

        x = tokens + positions                              # [L,D]
        x = self.dropout(x, key=dropout_key)                # [L,D]
        x = self.transformer_blocks(x, key=transformer_key) # [L,D]
        x = jax.vmap(self.ln)(x)                            # [L,D]
        logits = jax.vmap(self.head)(x)                     # [L,D] -> [L,vocab_size], sequence of one-hot vectors
        return logits