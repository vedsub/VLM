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


class GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        # Gemma-style RMSNorm keeps zero-initialized weights and applies (1 + weight).
        self.weight = nn.Parameter(torch.zeros(self.hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return hidden_states.to(input_dtype) * (1.0 + self.weight.to(input_dtype))


def _build_gemma_activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized in {"gelu", "gelu_new"}:
        return nn.GELU()
    if normalized in {"gelu_fast", "gelu_pytorch_tanh"}:
        return nn.GELU(approximate="tanh")
    if normalized in {"relu"}:
        return nn.ReLU()
    raise ValueError(f"Unsupported activation function: {name}")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {x.shape[-1]}.")
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query_states = (query_states * cos) + (_rotate_half(query_states) * sin)
    key_states = (key_states * cos) + (_rotate_half(key_states) * sin)
    return query_states, key_states


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {dim}.")
        self.dim = int(dim)
        self.base = float(base)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.LongTensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2:
            raise ValueError(
                f"position_ids must be 2D (batch, seq_len), got shape {tuple(position_ids.shape)}."
            )
        freqs = torch.einsum(
            "bs,d->bsd",
            position_ids.to(device=self.inv_freq.device, dtype=self.inv_freq.dtype),
            self.inv_freq,
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        return cos, sin


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep <= 0:
        raise ValueError(f"n_rep must be > 0, got {n_rep}.")
    if hidden_states.ndim != 4:
        raise ValueError("hidden_states must be 4D (batch, num_kv_heads, seq_len, head_dim).")
    if n_rep == 1:
        return hidden_states
    batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = int(config.head_dim or (config.hidden_size // config.num_attention_heads))
        self.hidden_size = config.hidden_size
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        q_out_dim = self.num_heads * self.head_dim
        kv_out_dim = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(self.hidden_size, q_out_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, kv_out_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, kv_out_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(q_out_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = GemmaRotaryEmbedding(dim=self.head_dim, base=config.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        kv_cache: KVCache | None = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, query_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(
            batch_size, query_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            batch_size, query_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            batch_size, query_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if position_ids is None:
            past_len = kv_cache.get_seq_length(self.layer_idx) if kv_cache is not None else 0
            position_ids = (
                torch.arange(
                    past_len,
                    past_len + query_len,
                    device=hidden_states.device,
                    dtype=torch.long,
                )
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
        cos, sin = self.rotary_emb(position_ids=position_ids, dtype=query_states.dtype)
        query_states, key_states = _apply_rotary_pos_emb(
            query_states=query_states,
            key_states=key_states,
            cos=cos,
            sin=sin,
        )

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(self.layer_idx, key_states, value_states)

        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale
        key_len = key_states.shape[-2]

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask.to(dtype=attn_weights.dtype)
        else:
            key_positions = torch.arange(key_len, device=hidden_states.device).view(1, 1, key_len)
            query_positions = position_ids.unsqueeze(-1)
            causal_mask = key_positions <= query_positions
            min_val = torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights.masked_fill(~causal_mask.unsqueeze(1), min_val)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, query_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            return attn_output, None
        return attn_output, attn_probs


class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = _build_gemma_activation(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = self.act_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gated * up)


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GemmaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        kv_cache: KVCache | None = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            output_attentions=output_attentions,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        if not output_attentions:
            return hidden_states, None
        return hidden_states, attn_weights


@dataclass
class GemmaModelOutput:
    last_hidden_state: torch.Tensor
    hidden_states: Tuple[torch.Tensor, ...] | None = None
    attentions: Tuple[torch.Tensor, ...] | None = None


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config=config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_scale = math.sqrt(config.hidden_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def _prepare_4d_causal_attention_mask(
        self,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        key_value_length: int,
    ) -> torch.Tensor:
        batch_size, query_length, _ = hidden_states.shape

        if attention_mask.ndim == 4:
            if attention_mask.dtype == torch.bool:
                min_value = torch.finfo(hidden_states.dtype).min
                additive_mask = torch.zeros_like(
                    attention_mask,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                return additive_mask.masked_fill(~attention_mask.to(device=hidden_states.device), min_value)
            return attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)

        if attention_mask.ndim != 2:
            raise ValueError(
                f"attention_mask must be 2D or 4D, got shape {tuple(attention_mask.shape)}."
            )

        if attention_mask.shape[0] != batch_size:
            raise ValueError(
                f"attention_mask batch mismatch: expected {batch_size}, got {attention_mask.shape[0]}."
            )
        if attention_mask.shape[1] not in {query_length, key_value_length}:
            raise ValueError(
                "attention_mask length mismatch: expected either "
                f"{query_length} (current tokens) or {key_value_length} (past+current), "
                f"got {attention_mask.shape[1]}."
            )

        if attention_mask.shape[1] == query_length and key_value_length > query_length:
            past_visible = torch.ones(
                (batch_size, key_value_length - query_length),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([past_visible, attention_mask], dim=-1)

        key_positions = torch.arange(key_value_length, device=hidden_states.device).view(1, 1, key_value_length)
        query_positions = position_ids.unsqueeze(-1)
        causal_mask = key_positions <= query_positions

        padding_mask = attention_mask[:, None, None, :].to(device=hidden_states.device, dtype=torch.bool)
        full_mask = causal_mask.unsqueeze(1) & padding_mask

        min_value = torch.finfo(hidden_states.dtype).min
        additive_mask = torch.zeros(
            (batch_size, 1, query_length, key_value_length),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        additive_mask = additive_mask.masked_fill(~full_mask, min_value)
        return additive_mask

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> GemmaModelOutput | Tuple[torch.Tensor, ...]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds * self.embed_scale
        batch_size, seq_length, _ = hidden_states.shape

        if position_ids is None:
            past_seen_tokens = kv_cache.get_seq_length(0) if kv_cache is not None else 0
            position_ids = (
                torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + seq_length,
                    device=hidden_states.device,
                    dtype=torch.long,
                )
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        if position_ids.shape != (batch_size, seq_length):
            raise ValueError(
                "position_ids must have shape "
                f"({batch_size}, {seq_length}), got {tuple(position_ids.shape)}."
            )

        past_seen_tokens = kv_cache.get_seq_length(0) if kv_cache is not None else 0
        key_value_length = past_seen_tokens + seq_length
        causal_attention_mask = None
        if attention_mask is not None:
            causal_attention_mask = self._prepare_4d_causal_attention_mask(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                position_ids=position_ids,
                key_value_length=key_value_length,
            )

        all_hidden_states: Tuple[torch.Tensor, ...] | None = () if output_hidden_states else None
        all_self_attns: Tuple[torch.Tensor, ...] | None = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, self_attn_weights = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=causal_attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_self_attns = all_self_attns + (self_attn_weights,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            outputs: list[Any] = [hidden_states]
            if all_hidden_states is not None:
                outputs.append(all_hidden_states)
            if all_self_attns is not None:
                outputs.append(all_self_attns)
            return tuple(outputs)

        return GemmaModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


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
