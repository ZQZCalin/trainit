"""Util functions."""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
from jax import Array
from optax import GradientTransformation
from typing import Tuple
import ml_dtypes


# Other util functions.
def merge_non_zero_dict(target, source):
    """Merges non-zero items in source dictionary into target dictionary.
    This is a mutable operation.
    """
    for key, value in source.items():
        if not value == 0:
            target[key] = value


# Util functions for tree manipulation. 
def zero_tree(tree):
    """Returns an all-zero tree with the same structure as the input."""
    return jtu.tree_map(jnp.zeros_like, tree)


def tree_add(tree1, tree2):
    return jtu.tree_map(lambda x,y: x+y, tree1, tree2)


def tree_subtract(tree1, tree2):
    return jtu.tree_map(lambda x,y: x-y, tree1, tree2)


def tree_multiply(tree1, tree2):
    return jtu.tree_map(lambda x,y: x*y, tree1, tree2)


def tree_dot(tree1, tree2):
    return jtu.tree_reduce(
        lambda x,y: x+y,
        jtu.tree_map(lambda x,y: jnp.dot(x,y), tree1, tree2)
    )


def negative_tree(tree):
    """A `jtu.tree_map`-broadcasted version of tree -> -tree."""
    return jtu.tree_map(lambda t: -t, tree)


def tree_scalar_multiply(tree, scalar):
    return jtu.tree_map(lambda x: scalar*x, tree)


def tree_l1_norm(tree):
    """Returns the l1 norm of the vectorized tree."""
    return jtu.tree_reduce(
        lambda x, y: x + y,
        jtu.tree_map(lambda x: jnp.sum(jnp.abs(x)), tree)
    )


def tree_l2_norm(tree):
    """Returns the l2 norm of the vectorized tree."""
    return jnp.sqrt(
        jtu.tree_reduce(
            lambda x, y: x + y, jtu.tree_map(lambda x: jnp.sum(x * x), tree)
        )
    )


# TODO: deprecated, to be removed
def tree_norm(tree):
    """Returns the l2 norm of the vectorized tree."""
    return tree_l2_norm(tree)


def is_zero_tree(tree):
    """Checks if a tree only has zero entries."""
    return jtu.tree_reduce(
        lambda x, y: x & y, jtu.tree_map(lambda x: jnp.all(x == 0), tree)
    )


def is_finite_tree(tree):
    """Returns whether a tree is finite."""
    leaves = jtu.tree_flatten(tree)[0]
    return jnp.all(
        jnp.array([jnp.all(jnp.isfinite(node)) for node in leaves]))


def tree_normalize(tree):
    # Use jax.lax.cond to avoid trace issue.
    return jax.lax.cond(
        is_zero_tree(tree),
        true_fun=lambda _: tree,
        false_fun=lambda _: tree_scalar_multiply(tree, 1/tree_norm(tree)),
        operand=None,
    )


def tree_inner_product(tree1, tree2):
    leaves1, _ = jtu.tree_flatten(tree1)
    leaves2, _ = jtu.tree_flatten(tree2)
    return sum(jnp.sum(a * b) for a, b in zip(leaves1, leaves2))


def tree_cosine_similarity(tree1, tree2):
    """Returns the cosine similarity of two trees."""
    return tree_inner_product(tree_normalize(tree1), tree_normalize(tree2))


def tree_norm_direction_decomposition(tree):
    """Decomposes the norm and the direction of a tree.

    Returns:
        The norm of a tree (1d array) and the normalized tree.
        If the tree is all zeros, then return 0 as the norm and an all-zero tree.
    """
    def true_fun(_):
        return jnp.zeros([], jnp.float32), tree
    def false_fun(_):
        norm = tree_norm(tree)
        return norm, tree_scalar_multiply(tree, 1/norm)
    return jax.lax.cond(
        is_zero_tree(tree), true_fun, false_fun, operand=None)
    # norm = tree_norm(tree)
    # # NOTE: we need to return jax.Array to make sure both branches returns the
    # # same data structure and thus avoid lax.cond issue
    # if norm == 0:
    #     return jnp.zeros([], jnp.float32), tree
    # return norm, tree_scalar_multiply(tree, 1/norm)


def random_unit_vector(key, tree):
    """Constructs a pytree of same structure as input whose leaves is a random unit vector.

    Returns:
        New PRNGKey and a uniform random vector on the unit sphere.
    """
    # Construct a pytree of random keys.
    key, new_key = jr.split(key)
    keys = jr.split(key, num=len(jtu.tree_leaves(tree)))
    keys_tree = jtu.tree_unflatten(jtu.tree_structure(tree), keys)
    # Sample Gaussian vector.
    normal_vector = jtu.tree_map(
        lambda t, k: jr.normal(k, shape=t.shape, dtype=t.dtype), 
        tree, keys_tree
    )
    return new_key, tree_normalize(normal_vector)


def check_tree_structures_match(tree1, tree2):
    """Check whether tree1 and tree2 have the same tree structure. 
        Raises error when structures do not match.
    """
    if jtu.tree_structure(tree1) != jtu.tree_structure(tree2):
        raise ValueError("Input Pytrees do not have the same structure")


# ===============================================
# Other util functions
# ===============================================

def merge_dicts(*to_merge):
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


# TODO: This is hella slow. Needs better solution
def log_optax(base_optimizer, log_fn):
    def init_fn(params):
        return base_optimizer.init(params)

    def update_fn(updates, state, params):
        log_fn(updates, state, params)
        return base_optimizer.update(updates, state, params)

    return GradientTransformation(init_fn, update_fn)


# basically the same as the pytorch function cross_entropy
def softmax_cross_entropy(
    input,
    target,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    label_smoothing=0.0,
    axis=None,
):
    """Computes softmax cross entropy between sets of logits and integer labels.
    Measures the probability error in discrete classification tasks in which
    the classes are mutually exclusive (each entry is in exactly one class).
    For example, each CIFAR-10 image is labeled with one and only one label:
    an image can be a dog or a truck, but not both.
    References:
    [Goodfellow et al, 2016](http://www.deeplearningbook.org/contents/prob.html)
    Args:
    logits: Unnormalized log probabilities, with shape `[..., num_classes]`.
    labels: Integers specifying the correct class for each input, with shape
        `[...]`.
    Returns:
    Cross entropy between each prediction and the corresponding target
    distributions, with shape `[...]`.
    """
    # This is like jnp.take_along_axis(jax.nn.log_softmax(...), ...) except that
    # we avoid subtracting the normalizer from all values, just from the values
    # for the correct labels.

    if axis is None:
        axis = input.ndim - 1
    if axis < 0:
        axis = input.ndim + axis

    C = input.shape[axis]

    if weight is not None:
        weight_shape = (
            (1,) * axis + (input.shape[axis],) + (1,) * (input.ndim - axis - 1)
        )
        weight = weight.reshape(weight_shape)

    if isinstance(target, int) or target.ndim != input.ndim:
        no_ignore = jax.lax.stop_gradient(target != ignore_index)
        logits_max = jnp.max(
            input, axis=axis, keepdims=True
        )  # , where=no_ignore, initial=-jnp.inf)
        logits = input - jax.lax.stop_gradient(logits_max)

        broadcast_shape = logits.shape[:axis] + (1,) + logits.shape[axis + 1 :]

        log_normalizers = jnp.log(
            jnp.sum(
                jnp.exp(logits), axis=axis, where=no_ignore.reshape(broadcast_shape)
            )
        )

        labels_no_ignore = jnp.where(no_ignore, target, 0)

        label_logits = jnp.take_along_axis(
            logits, labels_no_ignore[..., None], axis=axis
        )[..., 0]

        if label_smoothing != 0 or weight is not None:
            one_hot_labels = jax.nn.one_hot(labels_no_ignore, num_classes=C, axis=axis)
            target_probs = (
                one_hot_labels * (1.0 - label_smoothing)
                + jnp.ones_like(one_hot_labels) / C * label_smoothing
            )

            if weight is not None:
                target_probs = target_probs * weight
                log_normalizers = log_normalizers * jnp.sum(target_probs, axis=axis)

            losses = -(
                jnp.sum(
                    target_probs * logits,
                    where=no_ignore.reshape(broadcast_shape),
                    axis=axis,
                )
                - log_normalizers
            )
        else:
            label_logits = jnp.take_along_axis(
                logits, labels_no_ignore[..., None], axis=axis
            )[..., 0]
            losses = log_normalizers - label_logits

        losses = jnp.where(no_ignore, losses, 0.0)
    else:
        target_probs = (
            target * (1.0 - label_smoothing)
            + jnp.ones_like(target) / C * label_smoothing
        )

        logits_max = jnp.max(input, axis=axis, keepdims=True)
        logits = input - jax.lax.stop_gradient(logits_max)

        log_normalizers = jnp.log(jnp.sum(jnp.exp(logits), axis=axis))

        if weight is not None:
            target_probs = target_probs * weight
            log_normalizers = log_normalizers * jnp.sum(
                target_probs * weight, axis=axis
            )

        losses = -(jnp.sum(target_probs * logits, axis=axis) - log_normalizers)

        no_ignore = None

    if reduction == "none":
        return losses
    if reduction == "mean":
        return jnp.mean(losses, where=no_ignore)
    if reduction == "sum":
        return jnp.sum(losses, where=no_ignore)