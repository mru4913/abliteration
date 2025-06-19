#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from pathlib import Path
from typing import List, Tuple, Union

from datasets import disable_caching, load_dataset

from ..config.config import DatasetConfig, LocalDatasetConfig


class DatasetManager:
    """Manages dataset loading and processing for abliteration."""

    def __init__(self, config: Union[DatasetConfig, LocalDatasetConfig]) -> None:
        """Initialize the dataset manager.

        Args:
            config: Dataset configuration
        """
        self.config = config
        if isinstance(config, DatasetConfig) and config.cache_dir:
            disable_caching()

    def load_instructions(self, n_instructions: int) -> Tuple[List[str], List[str]]:
        """Load and split instructions from the dataset.

        Returns:
            Tuple of (train_instructions, test_instructions)
        """
        if isinstance(self.config, LocalDatasetConfig):
            # Load from local text file
            dataset_path = Path(self.config.dataset_path)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

            with open(dataset_path) as f:
                data = f.readlines()

            # Remove empty lines and whitespace
            data = [line.strip() for line in data if line.strip()]

            # Sample from local data
            if len(data) < n_instructions:
                raise ValueError(f"Dataset too small, needs at least {n_instructions} entries")

            indices = random.sample(range(len(data)), n_instructions)
            train_instructions = [data[i] for i in indices[:n_instructions]]
            test_instructions = [data[i] for i in indices[n_instructions:]]

        else:
            # Load from HuggingFace datasets
            dataset = load_dataset(
                self.config.dataset_id,
                split=self.config.split,
                cache_dir=self.config.cache_dir,
            )

            # Ensure dataset has enough entries
            if len(dataset) < n_instructions:
                raise ValueError(f"Dataset too small, needs at least {n_instructions * 2} entries but only has {len(dataset)}")

            indices = random.sample(range(len(dataset)), n_instructions)
            train_instructions = [dataset[i][self.config.column] for i in indices[:n_instructions]]
            test_instructions = [dataset[i][self.config.column] for i in indices[n_instructions:]]

        return train_instructions, test_instructions
