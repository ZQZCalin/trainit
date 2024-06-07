'''
plumbing log (or other) data through a jax.jit is a huge pain
This is intended to make it a bit easier.

The typical setup is that you have some kind of train_step function that takes as input
a train_state pytree (or more than one train_state pytree) containing model parameters, optimizer
state etc.
Then train_step returns a new train_state pytree.

Your problem is that you might want to log some data from inside the model or optimizer. Normally,
you would return this data as part  of the train_state pytree. But where is it in the pytree?
In order to actually do the logging, you'd need to have some monstrosity like

log_data(train_state.model.trunk.layers[2].transformer.attention.activations_to_log)

Worse, if your model architecture changes slightly, you might have to go back and 
change all these log  calls.

Instead, we'll introduce a simple way to get around this:
any data that you'll want to log, you should wrap in a Log class defined here.
There's nothing fancy about the Log class, it's literally just a pytree-compatible lable for
the given node.

Then, after returning the pytree, you can functions like:
map_logs(func, train_state) to apply func to all the values in train_state that were wrapped in Log
    objects (map_logs will unwrap them for you)
list_logs(train_state) to get a list of all the (unwrapped) objects that were wrapped as logs.

So, you could do
map_logs(log_data, train_state)
to automatically call log_data on all the log stuff.
'''


import jax
from typing import NamedTuple, Callable
from jaxtyping import PyTree
from jax import tree_util as jtu
from jax import numpy as jnp
import equinox as eqx


class Log(eqx.Module):
    data: PyTree
    def __init__(self, data):
        self.data=data
    

# class LoggedState(eqx.Module):
#     _state: PyTree
#     _log_data: PyTree

#     def __init__(self, state: PyTree, log_data: PyTree):
#         self._state = state
#         self._log_data = log_data

#     def __iter__(self):
#         yield self._state
#         yield self._log_data

#     def get_state(self):
#         return self._state

#     def get_logs(self):
#         return self._log_data

#     def __getattr__(self, name):
#         return getattr(self._state, name)


# def map_logs(fn: Callable, tree: PyTree, state_fn: Callable = lambda x: x):
#     def map_fn(logged_state):
#         if not isinstance(logged_state, LoggedState):
#             return state_fn(logged_state)
#         state = logged_state.get_state()
#         log_data = logged_state.get_logs()

#         state = map_logs(fn, state)
#         log_data = fn(log_data)
#         return LoggedState(state, log_data)

#     return jtu.tree_map(map_fn, tree, is_leaf=lambda x: isinstance(x, LoggedState))


# def prune_logs(tree: PyTree):
#     def map_fn(logged_state):
#         if not isinstance(logged_state, LoggedState):
#             return logged_state
#         else:
#             return prune_logs(logged_state.state)

#     pruned = jtu.tree_map(map_fn, tree, is_leaf=lambda x: isinstance(x, LoggedState))
#     logs = filter_logs(tree)
#     return pruned, logs


# def filter_logs(tree: PyTree):
#     return map_logs(lambda x: x, tree, state_fn=lambda x: None)


# def list_of_logs(tree: PyTree):
#     result = []
#     map_logs(result.append, tree)
#     return result


# def set_all_logs(tree: PyTree, value=None):
#     return map_logs(lambda x: value, tree)


def map_logs(fn: Callable, tree: PyTree, state_fn: Callable = lambda x: x):
    def map_fn(value):
        if not isinstance(value, Log):
            return state_fn(value)
        return Log(fn(value.data))

    return jtu.tree_map(map_fn, tree, is_leaf=lambda x: isinstance(x, Log))


def filter_logs(tree: PyTree):
    return map_logs(lambda x: x, tree, state_fn=lambda x: None)


def list_of_logs(tree: PyTree, keep_none=False):
    result = []
    def append(log):
        if log is not None or keep_none:
            result.append(log)
        return log
    map_logs(append, tree)
    return result


def set_all_logs(tree: PyTree, value=None):
    return map_logs(lambda x: value, tree)