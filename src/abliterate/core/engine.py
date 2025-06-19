#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from ..config.config import AbliterationConfig
from ..datasets.manager import DatasetManager
from ..models.base import BaseModel


class AbliterationEngine:
    """Core engine for performing model abliteration."""

    def __init__(self, config: AbliterationConfig) -> None:
        """Initialize the abliteration engine.

        Args:
            config: Abliteration configuration
        """
        self.config = config

    @staticmethod
    def _orthogonalize_matrix(matrix: torch.Tensor, vec: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
        """Orthogonalize a matrix against a vector.

        Args:
            matrix: Matrix to orthogonalize
            vec: Vector to orthogonalize against
            weight: Weight for orthogonalization

        Returns:
            Orthogonalized matrix
        """
        vec = vec.view(-1).to(matrix.device)

        if matrix.shape[-1] == vec.shape[0]:
            proj = torch.einsum("...d,d->...", matrix, vec).unsqueeze(-1) * vec.unsqueeze(0)
            return matrix - weight * proj
        elif matrix.shape[0] == vec.shape[0]:
            proj = torch.einsum("d...,d->...", matrix, vec).unsqueeze(0) * vec.unsqueeze(-1)
            return matrix - weight * proj
        else:
            raise ValueError(f"Matrix shape {matrix.shape} incompatible with vector shape {vec.shape}")

    @staticmethod
    def _calculate_refusal_direction(target_hidden: List[torch.Tensor], baseline_hidden: List[torch.Tensor]) -> torch.Tensor:
        """Calculate the refusal direction vector.

        Args:
            target_hidden: Hidden states from target dataset
            baseline_hidden: Hidden states from baseline dataset

        Returns:
            Normalized refusal direction vector
        """
        target_mean = torch.stack(target_hidden).mean(dim=0)
        baseline_mean = torch.stack(baseline_hidden).mean(dim=0)
        refusal_dir = target_mean - baseline_mean
        refusal_dir = refusal_dir / refusal_dir.norm()

        del target_mean, baseline_mean
        torch.cuda.empty_cache()

        return refusal_dir

    def abliterate(self, model: BaseModel, generate_kwargs: Dict[str, Any] = {}) -> Tuple[BaseModel, Dict[str, Any]]:
        """Perform abliteration on the model.

        Args:
            model: Model to abliterate
            generate_kwargs: Additional keyword arguments for generation

        Returns:
            Dictionary containing abliteration statistics
        """
        model.set_target_layer(self.config.target_layer)

        # Load datasets
        target_manager = DatasetManager(self.config.target_dataset)
        baseline_manager = DatasetManager(self.config.baseline_dataset)

        # Generate outputs
        target_instructions, _ = target_manager.load_instructions(n_instructions=self.config.n_instructions)
        baseline_instructions, _ = baseline_manager.load_instructions(n_instructions=self.config.n_instructions)

        # Default generation parameters
        default_generate_kwargs = {
            "max_new_tokens": 1,
            "use_cache": False,
            "return_dict_in_generate": True,
            "output_hidden_states": True,
        }

        # Merge with user-provided kwargs
        final_generate_kwargs = {**default_generate_kwargs, **generate_kwargs}

        target_outputs = [
            model.generate(
                input_ids=model.tokenize(instruction, self.config.target_prompt),
                **final_generate_kwargs,
            )
            for instruction in tqdm(target_instructions, desc="Generating target outputs")
        ]

        baseline_outputs = [
            model.generate(
                input_ids=model.tokenize(instruction, self.config.baseline_prompt),
                **final_generate_kwargs,
            )
            for instruction in tqdm(baseline_instructions, desc="Generating baseline outputs")
        ]

        # Extract hidden states
        target_hidden = [model.get_hidden_states(output) for output in target_outputs]
        baseline_hidden = [model.get_hidden_states(output) for output in baseline_outputs]

        # Calculate refusal direction
        refusal_dir = self._calculate_refusal_direction(target_hidden, baseline_hidden)

        # Modify model weights using refusal direction
        stats = model.modify_weights(refusal_dir, self.config.refusal_weight, modify_function=self._orthogonalize_matrix)

        del target_outputs, baseline_outputs, target_hidden, baseline_hidden

        del refusal_dir

        return model, stats
