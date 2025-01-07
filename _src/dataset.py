"""Dataset loaders."""

from omegaconf import DictConfig
from typing import Any, List
from loadit import LoadIt, chunk_shuffle

from dataloaders import DataLoader
from dataloaders import get_lm_loader_next_token
from _src.base import lm_datasets, cv_datasets
from _src.model import init_tokenizer


def load_lm_data(
        config: DictConfig, 
        model_config: DictConfig,
        seed: int = 42,
        split: str = "train",
    ) -> DataLoader:
    """Loads LLM datasets (c4 or pile).

    Args:
        config: config.dataset.
        model_config: config.model.
        seed: random seed for loadit.chunk_shuffle.
        split: specifies train/validation/test split.

    Returns:
        `torch.utils.data.DataLoader` or `Iterable`.
    """
    if config.use_loadit:
        if config.loadit_path is None:
            raise ValueError("invalid config: dataset.loadit_path cannot be None.")
        loader = LoadIt(config.loadit_path)
        if config.shuffle_buffer_size > 0:
            # TODO: either specify max_samples in config, or set length to full dataset length.
            # current implementation is vulnerable to batch size change 
            # NOTE: for this reason, I will just set length to None so we shuffle the entire dataset.
            # max_steps = config.train.max_steps
            # length = max_steps * config.total_batch_size
            # length = None
            chunk_size = config.shuffle_buffer_size
            length = config.loadit_length
            loader = chunk_shuffle(loader, chunk_size=chunk_size, length=length, seed=seed)
    else:
        if config.name not in lm_datasets:
            raise ValueError(f"invalid config: dataset.name '{config.name}' is not included in {lm_datasets}")
        tokenizer = init_tokenizer(model_config)
        loader = get_lm_loader_next_token(
            tokenizer,
            split=split,
            batch_size=config.batch_size,
            max_length=model_config.context_length,
            shuffle_buffer_size=config.shuffle_buffer_size,
            pad_to_multiple_of=model_config.context_length,
            num_workers=config.dataloader_workers,
            dataset=config.name,
        )
    return loader


def load_cv_data(
          
) -> DataLoader:
    """Loads CV datasets."""
    raise NotImplementedError


# TODO: maybe at a later stage, we need to include other tasks 
# such as basic regression/classification, and synthetic problems.
def load_other_data():
	"""..."""
	raise NotImplementedError


def init_dataloader(
		config: DictConfig,
) -> DataLoader:
	"""Initializes dataloader.
	
	Args:
		config: global_config.
	
	Returns:
		A `DataLoader` object.
	"""
	name = config.dataset.name
	if name in lm_datasets:
		return load_lm_data(
            config=config.dataset,
            model_config=config.model,
            seed=config.dataset.seed,
        )
	if name in cv_datasets:
		return load_cv_data()
	raise ValueError(f"invalid config: dataset.name == '{name}' is not supported.")