#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from abliterate.utils import create_tmp_hf_datasets_folder


class LocalDatasetConfig(BaseModel):
    """Configuration for local dataset loading."""

    dataset_path: str


class DatasetConfig(BaseModel):
    """Configuration for dataset loading."""

    dataset_id: str
    column: str
    split: str = "train"
    cache_dir: str = Field(default_factory=lambda: str(create_tmp_hf_datasets_folder()))


class ModelConfig(BaseModel):
    """Configuration for model loading."""

    model_id: str
    device_map: str = "auto"
    torch_dtype: Optional[str] = None
    attn_implementation: Optional[str] = None
    trust_remote_code: bool = True

    @field_validator("attn_implementation")
    def validate_attn_implementation(cls, v: str) -> str:
        """Validate that attn_implementation is a valid value."""
        if v not in ["eager", "flash_attention", "flash_attention_2"]:
            raise ValueError("attn_implementation must be one of: eager, flash_attention, flash_attention_2")
        return v


class AbliterationConfig(BaseSettings):
    """Configuration for the abliteration process."""

    model: ModelConfig
    target_layer: float = Field(default=0.65, gt=0.0, lt=1.0)
    refusal_weight: float = Field(default=1.0, ge=0.0, le=2.0)
    n_instructions: int = Field(default=128, gt=0)

    target_prompt: str = "You are a helpful assistant."
    baseline_prompt: str = "You are a helpful assistant."

    target_dataset: DatasetConfig | LocalDatasetConfig
    baseline_dataset: DatasetConfig | LocalDatasetConfig

    model_config = SettingsConfigDict(
        env_prefix="ABLITERATE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        json_schema_extra={
            "example": {
                "model": {
                    "model_id": "Qwen/Qwen-7B",
                    "device_map": "auto",
                    "torch_dtype": "float16",
                    "attn_implementation": "eager",
                    "trust_remote_code": True,
                },
                "target_layer": 0.65,
                "refusal_weight": 1.0,
                "n_instructions": 128,
                "target_prompt": "You are a helpful assistant.",
                "baseline_prompt": "You are a helpful assistant.",
                "target_dataset": {
                    "dataset_id": "mlabonne/harmful_behaviors",
                    "column": "text",
                    "n_instructions": 128,
                    "split": "train",
                },
                "baseline_dataset": {
                    "dataset_id": "mlabonne/harmless_alpaca",
                    "column": "text",
                    "n_instructions": 128,
                    "split": "train",
                },
            }
        },
    )

    @field_validator("target_layer")
    def validate_target_layer(cls, v: float) -> float:
        """Validate that target_layer is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("target_layer must be between 0 and 1")
        return v

    @field_validator("refusal_weight")
    def validate_refusal_weight(cls, v: float) -> float:
        """Validate that refusal_weight is between 0 and 2."""
        if not 0 <= v <= 2:
            raise ValueError("refusal_weight must be between 0 and 2")
        return v
