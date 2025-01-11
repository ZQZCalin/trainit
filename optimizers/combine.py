"""Extends/simplifies some combine methods in optax."""

import jax
import jax.tree_util as jtu
import optax
from typing import Any, Union, Mapping, Hashable, Callable, NamedTuple
from jaxtyping import PyTree


class MaskedState(NamedTuple):
    """Maintains inner transform state for masked transformations."""
    inner_state: Any


class MaskedNode(NamedTuple):
    """A node used to mask out unspecified parts of a tree.

    This node is ignored when mapping functions across the tree e.g. using
    `jax.tree_util.tree_map` since it is a container without children. It can
    therefore be used to mask out parts of a tree.
    """


def masked(
        inner: optax.GradientTransformation,
        mask: PyTree,
) -> optax.GradientTransformation:
    """A simplified `optax.masked` wrapper.

    Turns off support for GradientTransformationExtraArgs,
    and callable function type of `mask` argument.
    """

    def mask_pytree(pytree, mask_tree):
        return jax.tree_util.tree_map(
            lambda m, p: p if m else MaskedNode(), mask_tree, pytree
        )

    def init_fn(params):
        masked_params = mask_pytree(params, mask)
        return MaskedState(inner_state=inner.init(masked_params))

    def update_fn(updates, state, params=None):
        masked_updates = mask_pytree(updates, mask)
        masked_params = None if params is None else mask_pytree(params, mask)

        new_masked_updates, new_inner_state = inner.update(
            masked_updates, state.inner_state, masked_params)

        new_updates = jtu.tree_map(
            lambda m, new_u, old_u: new_u if m else old_u,
            mask, new_masked_updates, updates)
        return new_updates, MaskedState(inner_state=new_inner_state)

    return optax.GradientTransformation(init_fn, update_fn)


class MultiTransformState(NamedTuple):
    inner_states: Mapping[Hashable, optax.OptState]


def multi_transform(
        transforms: Mapping[Hashable, optax.GradientTransformation],
        param_labels: Union[PyTree, Callable[[PyTree], PyTree]],
) -> optax.GradientTransformation:
    """Adapts from `optax.multi_transform` and `optax.masked`.
    
    An issue of the current optax implementation is that
    any callable params will be be called in the `masked`
    function, which is unintended. For this reason, we
    implement a simplified multi_transform that builds on
    a `masked` wrapper with pytree mask only (excluding
    mask functions). This fits better for equinox models
    where params are callable objects.
    """

    def make_mask(labels, group):
        return jtu.tree_map(lambda label: label == group, labels)

    def init_fn(params):
        labels = param_labels(params) if callable(param_labels) else param_labels

        label_set = set(jax.tree_util.tree_leaves(labels))
        if not label_set.issubset(transforms.keys()):
            raise ValueError('Some parameters have no corresponding transformation.\n'
                            f'Parameter labels: {list(sorted(label_set))} \n'
                            f'Transforms keys: {list(sorted(transforms.keys()))} \n')

        inner_states = {
            group: masked(tx, make_mask(labels, group)).init(params)
            for group, tx in transforms.items()
        }
        return MultiTransformState(inner_states)

    def update_fn(updates, state, params=None):
        labels = param_labels(updates) if callable(param_labels) else param_labels
        new_inner_state = {}
        for group, tx in transforms.items():
            masked_tx = masked(tx, make_mask(labels, group))
            updates, new_inner_state[group] = masked_tx.update(
                updates, state.inner_states[group], params)
        return updates, MultiTransformState(new_inner_state)

    return optax.GradientTransformation(init_fn, update_fn)