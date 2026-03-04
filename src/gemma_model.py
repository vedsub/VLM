from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional, Tuple

import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from .vit_model import SiglipVisionConfig, SiglipVisionModel


class KVCache:
    """Layer-wise KV cache for autoregressive decoding."""

    def __init__(self, num_layers: int, max_cache_len: int | None = None) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}.")
        if max_cache_len is not None and max_cache_len <= 0:
            raise ValueError(f"max_cache_len must be > 0, got {max_cache_len}.")

        self.num_layers = int(num_layers)
        self.max_cache_len = max_cache_len
        self.key_cache: list[Optional[torch.Tensor]] = [None] * self.num_layers
        self.value_cache: list[Optional[torch.Tensor]] = [None] * self.num_layers

    def __len__(self) -> int:
        return self.get_seq_length()

    @classmethod
    def from_config(cls, config: "GemmaConfig", max_cache_len: int | None = None) -> "KVCache":
        return cls(num_layers=config.num_hidden_layers, max_cache_len=max_cache_len)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        self._validate_layer_idx(layer_idx)
        layer_cache = self.key_cache[layer_idx]
        return 0 if layer_cache is None else int(layer_cache.shape[-2])

    def _validate_layer_idx(self, layer_idx: int) -> None:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"Invalid layer_idx={layer_idx}. Expected in [0, {self.num_layers}).")

    def get_layer(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        self._validate_layer_idx(layer_idx)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for one layer.

        Args:
            layer_idx: Decoder layer index.
            key_states: (batch, num_kv_heads, new_seq_len, head_dim)
            value_states: (batch, num_kv_heads, new_seq_len, head_dim)
        """
        self._validate_layer_idx(layer_idx)

        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError(
                "key_states and value_states must both be 4D tensors with shape "
                "(batch, num_kv_heads, seq_len, head_dim)."
            )
        if key_states.shape != value_states.shape:
            raise ValueError(
                f"key/value shape mismatch: {tuple(key_states.shape)} vs {tuple(value_states.shape)}."
            )

        cached_keys = self.key_cache[layer_idx]
        cached_values = self.value_cache[layer_idx]

        if cached_keys is None or cached_values is None:
            new_keys = key_states
            new_values = value_states
        else:
            if cached_keys.shape[0] != key_states.shape[0]:
                raise ValueError(
                    f"Batch size mismatch for cache update at layer {layer_idx}: "
                    f"{cached_keys.shape[0]} vs {key_states.shape[0]}."
                )
            if cached_keys.shape[1] != key_states.shape[1] or cached_keys.shape[3] != key_states.shape[3]:
                raise ValueError(
                    f"KV head/head_dim mismatch for cache update at layer {layer_idx}: "
                    f"cached={tuple(cached_keys.shape)}, incoming={tuple(key_states.shape)}."
                )
            new_keys = torch.cat([cached_keys, key_states], dim=-2)
            new_values = torch.cat([cached_values, value_states], dim=-2)

        if self.max_cache_len is not None and new_keys.shape[-2] > self.max_cache_len:
            new_keys = new_keys[:, :, -self.max_cache_len :, :]
            new_values = new_values[:, :, -self.max_cache_len :, :]

        self.key_cache[layer_idx] = new_keys
        self.value_cache[layer_idx] = new_values
        return new_keys, new_values

    def reset(self) -> None:
        for i in range(self.num_layers):
            self.key_cache[i] = None
            self.value_cache[i] = None

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "KVCache":
        for i in range(self.num_layers):
            if self.key_cache[i] is not None:
                self.key_cache[i] = self.key_cache[i].to(device=device, dtype=dtype)
            if self.value_cache[i] is not None:
                self.value_cache[i] = self.value_cache[i].to(device=device, dtype=dtype)
        return self

    def reorder_cache(self, beam_idx: torch.LongTensor) -> "KVCache":
        if beam_idx.ndim != 1:
            raise ValueError(f"beam_idx must be 1D, got shape {tuple(beam_idx.shape)}.")
        for i in range(self.num_layers):
            if self.key_cache[i] is not None:
                self.key_cache[i] = self.key_cache[i].index_select(0, beam_idx)
            if self.value_cache[i] is not None:
                self.value_cache[i] = self.value_cache[i].index_select(0, beam_idx)
        return self


@dataclass
class GemmaConfig:
    vocab_size: int = 257216
    hidden_size: int = 2048
    intermediate_size: int = 16384
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    num_image_tokens: int = 256
    head_dim: int | None = None
    hidden_act: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 1
    model_type: str = "gemma"
    torch_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads, got "
                f"{self.hidden_size} and {self.num_attention_heads}."
            )
        computed_head_dim = self.hidden_size // self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = computed_head_dim
        elif self.head_dim != computed_head_dim:
            raise ValueError(
                f"head_dim mismatch: expected {computed_head_dim} from hidden_size/"
                f"num_attention_heads, got {self.head_dim}."
            )
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                "num_key_value_heads cannot exceed num_attention_heads, got "
                f"{self.num_key_value_heads} > {self.num_attention_heads}."
            )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "GemmaConfig":
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_image_tokens": self.num_image_tokens,
            "head_dim": self.head_dim,
            "hidden_act": self.hidden_act,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "attention_bias": self.attention_bias,
            "attention_dropout": self.attention_dropout,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "model_type": self.model_type,
            "torch_dtype": self.torch_dtype,
        }


@dataclass
class PaliGemmaConfig:
    vision_config: SiglipVisionConfig = field(default_factory=SiglipVisionConfig)
    text_config: GemmaConfig = field(default_factory=GemmaConfig)
    ignore_index: int = -100
    image_token_index: int = 257152
    projection_dim: int | None = None
    hidden_size: int | None = None
    vocab_size: int | None = None
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 1
    model_type: str = "paligemma"
    torch_dtype: str = "float32"
    transformers_version: str | None = None
    architectures: list[str] | None = None
    _name_or_path: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.vision_config, dict):
            self.vision_config = SiglipVisionConfig.from_dict(self.vision_config)
        if isinstance(self.text_config, dict):
            self.text_config = GemmaConfig.from_dict(self.text_config)

        if self.hidden_size is None:
            self.hidden_size = self.text_config.hidden_size
        if self.vocab_size is None:
            self.vocab_size = self.text_config.vocab_size
        if self.projection_dim is None:
            self.projection_dim = self.hidden_size

        if self.hidden_size != self.text_config.hidden_size:
            raise ValueError(
                f"hidden_size mismatch: top-level {self.hidden_size}, "
                f"text_config.hidden_size {self.text_config.hidden_size}."
            )
        if self.vocab_size != self.text_config.vocab_size:
            raise ValueError(
                f"vocab_size mismatch: top-level {self.vocab_size}, "
                f"text_config.vocab_size {self.text_config.vocab_size}."
            )
        if self.text_config.num_image_tokens != self.vision_config.num_image_tokens:
            raise ValueError(
                "num_image_tokens mismatch: "
                f"text_config={self.text_config.num_image_tokens}, "
                f"vision_config={self.vision_config.num_image_tokens}."
            )

    @property
    def num_image_tokens(self) -> int:
        return self.text_config.num_image_tokens

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "PaliGemmaConfig":
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        if "text_config" in filtered and isinstance(filtered["text_config"], dict):
            filtered["text_config"] = GemmaConfig.from_dict(filtered["text_config"])
        if "vision_config" in filtered and isinstance(filtered["vision_config"], dict):
            filtered["vision_config"] = SiglipVisionConfig.from_dict(filtered["vision_config"])
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        output = {
            "vision_config": self.vision_config.to_dict(),
            "text_config": self.text_config.to_dict(),
            "ignore_index": self.ignore_index,
            "image_token_index": self.image_token_index,
            "projection_dim": self.projection_dim,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "model_type": self.model_type,
            "torch_dtype": self.torch_dtype,
        }
        if self.transformers_version is not None:
            output["transformers_version"] = self.transformers_version
        if self.architectures is not None:
            output["architectures"] = self.architectures
        if self._name_or_path is not None:
            output["_name_or_path"] = self._name_or_path
        return output
