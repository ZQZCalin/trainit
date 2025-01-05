"""Base functions and data types."""

import equinox as eqx
import optax
import torch

from typing import List, Tuple, Optional, NamedTuple, Union, Iterable, Callable
from jaxtyping import Array, PRNGKeyArray
from jaxamp import DynamicScalerState

import loggers


language_models = [
    "gpt",
    "bert",
    "llama",
]

vision_models = [
    "vgg",
    "resnet",
]

basic_models = [
    "linear",
]

lm_datasets = [
    "pile",
    "c4",
]

cv_datasets = [
    "cifar10",
    "cifar100",
    "imagenet",
]

classification_tasks = [
    "pile",
    "c4",
    "cifar10",
    "cifar100",
    "imagenet",
]