#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from ..config.config import ModelConfig
from .base import BaseModel


class DefaultModel(BaseModel):
    """Default model implementation for abliteration."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the default model.

        Args:
            config: Model configuration
        """
        self.target_layer = None

        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            device_map=config.device_map,
            torch_dtype=getattr(torch, config.torch_dtype) if config.torch_dtype else torch.float16,
            attn_implementation=config.attn_implementation,
            trust_remote_code=config.trust_remote_code,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=config.trust_remote_code)

    def set_target_layer(self, layer_idx: int) -> int:
        """Get the target layer for abliteration."""
        self.target_layer = int(layer_idx * len(self.model.model.layers))
        return self.target_layer

    def tokenize(self, instruction: str, system_prompt: str = "") -> torch.Tensor:
        """Tokenize an instruction with system prompt.

        Args:
            instruction: Instruction text
            system_prompt: System prompt text

        Returns:
            Tokenized input tensor
        """
        return self.tokenizer.apply_chat_template(
            conversation=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
            ],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1,
        use_cache: bool = False,
        do_sample: bool = True,
        return_dict_in_generate: bool = True,
        output_hidden_states: bool = True,
        streamer: TextStreamer = None,
        **kwargs,
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
        with torch.no_grad():  # Ensure no gradients are computed
            return self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                do_sample=do_sample,
                return_dict_in_generate=return_dict_in_generate,
                output_hidden_states=output_hidden_states,
                streamer=streamer,
                **kwargs,
            )

    def get_hidden_states(self, outputs: Dict[str, Any]) -> torch.Tensor:
        """Extract hidden states from model outputs.

        Args:
            layer_idx: Index of the layer to extract states from
            outputs: Model outputs dictionary

        Returns:
            Hidden states tensor
        """
        return outputs["hidden_states"][0][self.target_layer][:, -1, :]

    def modify_weights(self, refusal_dir: torch.Tensor, weight: float = 1.0, modify_function: Callable = None) -> Dict[str, bool]:
        """Modify model weights using refusal direction.

        Args:
            refusal_dir: Refusal direction vector
            weight: Weight for modification

        Returns:
            Dictionary containing statistics about weight modification
        """
        stats = {"embed_tokens": False, "attention_o_proj": 0, "mlp_proj": 0}

        # Embed tokens
        if hasattr(self.model.model, "embed_tokens"):
            self.model.model.embed_tokens.weight.data = modify_function(self.model.model.embed_tokens.weight.data, refusal_dir, weight)
            stats["embed_tokens"] = True

        # Layer projections
        for layer in self.model.model.layers:
            # Attention output projection
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                layer.self_attn.o_proj.weight.data = modify_function(layer.self_attn.o_proj.weight.data, refusal_dir, weight)
                stats["attention_o_proj"] += 1

            # MLP projection
            if hasattr(layer, "mlp"):
                proj_name = "down_proj" if hasattr(layer.mlp, "down_proj") else "c_proj" if hasattr(layer.mlp, "c_proj") else None
                if proj_name:
                    getattr(layer.mlp, proj_name).weight.data = modify_function(getattr(layer.mlp, proj_name).weight.data, refusal_dir, weight)
                    stats["mlp_proj"] += 1

        if not stats["embed_tokens"] and stats["attention_o_proj"] == 0 and stats["mlp_proj"] == 0:
            raise RuntimeError("Failed to modify any model weights. Model not abliterated.")

        return stats
