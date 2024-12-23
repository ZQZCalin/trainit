"""Dataset loaders."""

from omegaconf import DictConfig
from typing import Any
from loadit import LoadIt, chunk_shuffle
from datasets import get_lm_loader_next_token
from model import init_tokenizer
import base


def load_lm_data(
        config: DictConfig, 
        model_config: DictConfig,
        seed: int = 42,
        split: str = "train",
    ) -> base.DataLoader:
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
            length = None
            loader = chunk_shuffle(loader, chunk_size=config.shuffle_buffer_size, length=length, seed=seed)
    else:
        if config.name not in ["c4", "pile"]:
            raise ValueError(f"invalid config: dataset.name '{config.name}' is not 'c4' or 'pile'.")
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


def init_dataloader(config: DictConfig) -> base.DataLoader:
    """Initializes a dataloader.
    
    Args:
        config: global_config.

    Returns:
        A `base.DataLoader` object.
    """
    name = config.dataset.name
    if name == "c4" or name == "pile":
        loader = load_lm_data(
            config=config.dataset,
            model_config=config.model,
            seed=config.dataset.seed,
        )
    else:
        raise ValueError(f"invalid config: dataset.name cannot be '{name}'")
    return loader