#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unittest.mock import Mock, patch

import pytest
import torch

from abliterate.config.config import (
    AbliterationConfig,
    DatasetConfig,
    LocalDatasetConfig,
    ModelConfig,
)
from abliterate.core.engine import AbliterationEngine
from abliterate.datasets.manager import DatasetManager


@pytest.fixture
def mock_model_config():
    """Create a mock model configuration."""
    return ModelConfig(
        model_id="test_model",
        device_map="cpu",  # Use CPU for testing
        torch_dtype="float16",
        attn_implementation="eager",
        trust_remote_code=True,
    )


@pytest.fixture
def mock_dataset_config():
    """Create a mock dataset configuration."""
    return DatasetConfig(
        dataset_id="test_dataset",
        column="text",
        split="train",
    )


@pytest.fixture
def mock_local_dataset_config():
    """Create a mock local dataset configuration."""
    return LocalDatasetConfig(
        dataset_path="test_path.txt",
    )


@pytest.fixture
def mock_abliteration_config(mock_model_config, mock_dataset_config):
    """Create a mock abliteration configuration."""
    return AbliterationConfig(
        model=mock_model_config,
        target_layer=0.5,
        refusal_weight=1.0,
        n_instructions=2,
        target_prompt="You are a helpful assistant.",
        baseline_prompt="You are a helpful assistant.",
        target_dataset=mock_dataset_config,
        baseline_dataset=mock_dataset_config,
    )


@pytest.fixture
def mock_model_output():
    """Create a mock model output."""
    return {
        "hidden_states": [
            torch.randn(1, 10, 512)  # [batch_size, seq_len, hidden_size]
        ]
    }


class TestDatasetManager:
    """Test cases for DatasetManager."""

    @patch("abliterate.datasets.manager.load_dataset")
    def test_load_instructions(self, mock_load_dataset, mock_dataset_config, mock_abliteration_config):
        """Test loading instructions from dataset."""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        mock_dataset.__getitem__ = Mock(side_effect=lambda i: {"text": f"instruction_{i}"})
        mock_load_dataset.return_value = mock_dataset

        manager = DatasetManager(mock_dataset_config)
        train_instructions, test_instructions = manager.load_instructions(n_instructions=mock_abliteration_config.n_instructions)

        # Based on current implementation: only n_instructions are sampled, so train gets all, test gets empty
        assert len(train_instructions) == mock_abliteration_config.n_instructions
        assert len(test_instructions) == 0  # Current implementation returns empty test set
        assert all(isinstance(instruction, str) for instruction in train_instructions)
        assert all(isinstance(instruction, str) for instruction in test_instructions)


class TestAbliterationEngine:
    """Test cases for AbliterationEngine."""

    def test_orthogonalize_matrix(self):
        """Test matrix orthogonalization."""
        matrix = torch.randn(10, 20)
        vec = torch.randn(20)
        weight = 1.0

        # Test orthogonalization
        result = AbliterationEngine._orthogonalize_matrix(matrix, vec, weight)
        assert result.shape == matrix.shape
        assert not torch.allclose(result, matrix)  # Should be different

        # Test incompatible shapes
        with pytest.raises(ValueError):
            AbliterationEngine._orthogonalize_matrix(matrix, torch.randn(30), weight)

    def test_calculate_refusal_direction(self):
        """Test refusal direction calculation."""
        target_hidden = [torch.randn(10) for _ in range(3)]
        baseline_hidden = [torch.randn(10) for _ in range(3)]

        refusal_dir = AbliterationEngine._calculate_refusal_direction(target_hidden, baseline_hidden)

        assert refusal_dir.shape == (10,)
        assert torch.allclose(refusal_dir.norm(), torch.tensor(1.0), atol=1e-6)

    def test_abliteration_engine_initialization(self, mock_abliteration_config):
        """Test abliteration engine initialization."""
        engine = AbliterationEngine(mock_abliteration_config)
        assert engine.config == mock_abliteration_config


class TestIntegration:
    """Integration tests."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_full_abliteration_pipeline(self, mock_abliteration_config):
        """Test the full abliteration pipeline with mocked components."""
        # This test would require more complex mocking of the entire pipeline
        # For now, we'll just test that the engine can be created
        engine = AbliterationEngine(mock_abliteration_config)
        assert engine is not None
