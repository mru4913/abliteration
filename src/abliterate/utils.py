#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import torch

HF_DATASETS_CACHE = "/tmp/datasets_cache"


def create_tmp_hf_datasets_folder() -> Path:
    """Create a temporary folder for storing intermediate files.

    Returns:
        Path: Path object pointing to the created temporary directory

    Example:
        >>> tmp_path = create_tmp_hf_datasets_folder()
        >>> print(tmp_path)
        /tmp/abliterate_datasets
    """
    # Create a fixed directory name
    tmp_dir = Path("/tmp/abliterate_datasets")

    os.environ["HF_DATASETS_CACHE"] = str(tmp_dir)

    # Ensure directory exists and has proper permissions
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created temporary directory: {tmp_dir}")

    return tmp_dir


def turn_off_gradients() -> None:
    torch.set_grad_enabled(False)
    torch.inference_mode()
