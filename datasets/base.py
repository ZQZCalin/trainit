import torch
from typing import List, Tuple, Union, Iterable
from jaxtyping import Array


DataLoader = Union[torch.utils.data.DataLoader, Iterable]
DataBatch = Tuple[Array, Array]