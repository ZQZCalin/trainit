"""We extend the __init__ method of Linear and LayerNorm layers from equinox by allowing initialization from
user-specified weight and bias. Embedding already has this feature and is untouched.
"""

import jax
from jax import numpy as jnp
from jax import random as jr
from equinox import nn
import math
from typing import Union, Optional, Sequence, Literal
from jaxtyping import Array, Float, PRNGKeyArray
import warnings


def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32
    

class Linear(nn.Linear):
    def __init__(
        self,
        in_features: Optional[Union[int, Literal["scalar"]]] = None,
        out_features: Optional[Union[int, Literal["scalar"]]] = None,
        use_bias: bool = True,
        weight: Optional[Float[Array, "out_features in_features"]] = None,
        bias: Optional[Float[Array, "out_features"]] = None,
        dtype=None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        """Extends equinox.nn.Linear by allowing user-specified weight and bias.
        Only overrides default initialization if weight is not None.
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        if weight is None:
            if in_features is None or out_features is None or key is None:
                raise ValueError(
                    "Must provide `eqx.nn.Linear(in_features=..., out_features=..., "
                    "key=...)` if not providing the weight and bias directly."
                )
            wkey, bkey = jr.split(key, 2)
            in_features_ = 1 if in_features == "scalar" else in_features
            out_features_ = 1 if out_features == "scalar" else out_features
            lim = 1 / math.sqrt(in_features_)

            self.weight = jr.uniform(
                wkey, (out_features_, in_features_), minval=-lim, maxval=lim, dtype=dtype
            )
            if use_bias:
                self.bias = jr.uniform(
                    bkey, (out_features_,), minval=-lim, maxval=lim, dtype=dtype
                )
            else:
                self.bias = None
        else:
            use_bias = bias is not None
            if weight.ndim != 2:
                raise ValueError("weight must have shape (out_features, in_features)")
            if in_features is None:
                in_features: int = weight.shape[1]
            if out_features is None:
                out_features: int = weight.shape[0]
            if weight.shape != (out_features, in_features):
                raise ValueError("weight must have shape (out_features, in_features)")
            if use_bias and bias.shape != (out_features,):
                raise ValueError("bias must have shape (out_features,)")
            self.weight = weight
            self.bias = bias
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias


class LayerNorm(nn.LayerNorm):
    def __init__(
        self,
        shape: Optional[Union[int, Sequence[int]]] = None,
        eps: float = 1e-5,
        use_weight: bool = True,
        use_bias: bool = True,
        weight: Optional[Float[Array, "*shape"]] = None,
        bias: Optional[Float[Array, "*shape"]] = None,
        dtype=None,
        *,
        elementwise_affine: Optional[bool] = None,
    ):
        """Extends equinox.nn.Linear by allowing user-specified weight and bias.
        Only overrides default initialization if weight is not None.
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        if weight is None:
            if isinstance(shape, int):
                shape = (shape,)
            else:
                shape = tuple(shape)
            if elementwise_affine is not None:
                use_weight = elementwise_affine
                use_bias = elementwise_affine
                warnings.warn(
                    "LayerNorm(elementwise_affine=...) is deprecated "
                    "in favour of LayerNorm(use_weight=...) and LayerNorm(use_bias=...)"
                )
            self.weight = jnp.ones(shape, dtype=dtype) if use_weight else None
            self.bias = jnp.zeros(shape, dtype=dtype) if use_bias else None
        else:
            use_weight = True
            use_bias = bias is not None
            if shape is None:
                shape: tuple = weight.shape
            if weight.shape != shape:
                raise ValueError("weight must have shape *shape.")
            if use_bias and bias.shape != shape:
                raise ValueError("bias must have shape *shape.")
            self.weight = weight
            self.bias = bias            
        self.use_weight = use_weight
        self.use_bias = use_bias
        self.shape = shape
        self.eps = eps


# To load paramters from Pytorch models, we extend basic equinox modules with the _load_params method.
# We will not modify the static fields (e.g., use_weight, use_bias), and these fields have higher priority
# than input parameters. For example, if a Linear layer sets use_bias = False, then the bias will remain None
# even if the input bias is not None.
# class Linear(nn.Linear):
#     """Extends equinox.nn.Linear with _load_params."""
#     def _load_params(self, weight: Array, bias: Optional[Array] = None) -> nn.Linear:
#         assert weight.shape == self.weight.shape, \
#             f"input weight shape {weight.shape} does not match current weight shape {self.weight.shape}"
#         if self.use_bias != (bias is not None):
#             warnings.warn(f"Input bias does not match use_bias={self.use_bias}, and it will be neglected.")
#             bias = self.bias
#         else:
#             if self.use_bias:
#                 assert bias.shape == self.bias.shape, \
#                     f"input weight shape {bias.shape} does not match current weight shape {self.bias.shape}"
#         return eqx.tree_at(lambda m: (m.weight, m.bias), self, (weight, bias))


# class Embedding(nn.Embedding):
#     """Extends equinox.nn.Embedding with _load_params."""
#     def _load_params(self, weight: Array):
#         assert weight.shape == self.weight.shape, \
#             f"input weight shape {weight.shape} does not match current weight shape {self.weight.shape}"
#         return eqx.tree_at(lambda m: m.weight, self, weight)


# class LayerNorm(nn.LayerNorm):
#     """Extends equinox.nn.LayerNorm with _load_params."""
#     def _load_params(self, weight: Optional[Array] = None, bias: Optional[Array] = None):
#         if self.use_weight != (weight is not None):
#             warnings.warn(f"Input weight does not match use_weight={self.use_weight}, and it will be neglected.")
#             weight = self.weight
#         else:
#             if self.use_weight:
#                 assert weight.shape == self.weight.shape, \
#                     f"input weight shape {weight.shape} does not match current weight shape {self.weight.shape}"
#         if self.use_bias != (bias is not None):
#             warnings.warn(f"Input bias does not match use_bias={self.use_bias}, and it will be neglected.")
#             bias = self.bias
#         else:
#             if self.use_bias:
#                 assert bias.shape == self.bias.shape, \
#                     f"input weight shape {bias.shape} does not match current weight shape {self.bias.shape}"
#         return eqx.tree_at(lambda m: (m.weight, m.bias), self, (weight, bias))
