"""Commonly used util functions."""

import jax
from jax import numpy as jnp
from typing import Tuple
from jaxtyping import Array
import ml_dtypes


def merge_dicts(*to_merge):
    """Merges a list of dictionaries into one."""
    result = {}
    for d in to_merge:
        result.update(d)
    return result


def get_accuracy(logits: Array, batch: Tuple[Array, Array], ignore_index: int = -100):
    input, target = batch # [N, L],  [N, L]
    predictions = jnp.argmax(logits, axis=2) # [N, L, C] -> [N, L]
    return jnp.sum(predictions == target) / jnp.sum(target != ignore_index)


def get_dtype(dtype: str):
    registry = {
        "bfloat16": ml_dtypes.bfloat16,
        "float16": jnp.float16,
    }
    return registry[dtype.lower()]