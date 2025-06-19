#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import torch


class BaseModel(ABC):
    """Base class for all models that can be abliterated."""

    @abstractmethod
    def __init__(self, model_id: str, **kwargs) -> None:
        """Initialize the model.

        Args:
            model_id: The model identifier or path
            **kwargs: Additional model-specific arguments
        """
        pass

    @abstractmethod
    def set_target_layer(self, layer_idx: int) -> int:
        """Get the target layer for abliteration."""
        pass

    @abstractmethod
    def tokenize(self, instruction: str, system_prompt: str = "") -> torch.Tensor:
        """Tokenize an instruction with system prompt."""
        pass

    @abstractmethod
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1,
        use_cache: bool = False,
        return_dict_in_generate: bool = True,
        output_hidden_states: bool = True,
    ) -> Dict[str, Any]:
        """Generate outputs from the model.

        Args:
            input_ids: Input token ids
            max_new_tokens: Maximum number of new tokens to generate
            use_cache: Whether to use KV cache
            return_dict_in_generate: Whether to return a dictionary
            output_hidden_states: Whether to output hidden states

        Returns:
            Dictionary containing generation outputs
        """
        pass

    @abstractmethod
    def get_hidden_states(self, outputs: Dict[str, Any]) -> torch.Tensor:
        """Extract hidden states from model outputs.

        Args:
            outputs: Model outputs dictionary

        Returns:
            Hidden states tensor
        """
        pass

    @abstractmethod
    def modify_weights(self, refusal_dir: torch.Tensor, weight: float = 1.0, modify_function: Callable = None) -> Dict[str, bool]:
        """Modify model weights using refusal direction.

        Args:
            refusal_dir: Refusal direction vector
            weight: Weight for modification

        Returns:
            Dictionary containing statistics about weight modification
        """
        pass
