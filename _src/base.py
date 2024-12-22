"""Base functions and data types."""

import torch
from typing import Union, Iterable

DataLoader = Union[torch.utils.data.DataLoader, Iterable]