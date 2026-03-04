from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class QuickGELU(nn.Module):
    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states * torch.sigmoid(1.702 * hidden_states)


def _build_activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized in {"gelu", "gelu_new"}:
        return nn.GELU()
    if normalized in {"gelu_fast", "gelu_pytorch_tanh"}:
        return nn.GELU(approximate="tanh")
    if normalized in {"quick_gelu", "quickgelu"}:
        return QuickGELU()
    if normalized == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation function: {name}")


@dataclass
class SiglipVisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: Optional[int] = None
    patch_size: int = 14
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    initializer_range: float = 0.02
    num_image_tokens: Optional[int] = None
    projection_dim: Optional[int] = 2048
    projector_hidden_act: str = "gelu_fast"
    vision_use_head: bool = False
    model_type: str = "siglip_vision_model"

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads, got "
                f"hidden_size={self.hidden_size}, num_attention_heads={self.num_attention_heads}"
            )

        if self.image_size is not None and self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})."
            )

        if self.num_image_tokens is None and self.image_size is None:
            raise ValueError("Either num_image_tokens or image_size must be provided.")

        if self.num_image_tokens is None and self.image_size is not None:
            patches_per_dim = self.image_size // self.patch_size
            self.num_image_tokens = patches_per_dim * patches_per_dim

        if self.image_size is None and self.num_image_tokens is not None:
            side = int(math.sqrt(self.num_image_tokens))
            if side * side != self.num_image_tokens:
                raise ValueError(
                    "num_image_tokens must be a perfect square when image_size is not provided."
                )
            self.image_size = side * self.patch_size

        if self.image_size is not None and self.num_image_tokens is not None:
            expected_tokens = (self.image_size // self.patch_size) ** 2
            if expected_tokens != self.num_image_tokens:
                raise ValueError(
                    "image_size and num_image_tokens are inconsistent: "
                    f"expected {expected_tokens} tokens from image_size={self.image_size} "
                    f"and patch_size={self.patch_size}, got num_image_tokens={self.num_image_tokens}."
                )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SiglipVisionConfig":
        valid_fields = {f.name for f in fields(cls)}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_config)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SiglipVisionModelOutput:
    last_hidden_state: Tensor
    pooler_output: Optional[Tensor] = None
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    attentions: Optional[Tuple[Tensor, ...]] = None


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )
        self.num_positions = config.num_image_tokens
        self.position_embedding = nn.Embedding(self.num_positions, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def _interpolate_position_embeddings(self, height: int, width: int) -> Tensor:
        num_positions = self.num_positions
        grid_size = int(math.sqrt(num_positions))
        if grid_size * grid_size != num_positions:
            raise ValueError("Cannot interpolate positional embeddings for non-square patch grids.")

        position_embeddings = self.position_embedding.weight.reshape(
            1, grid_size, grid_size, self.config.hidden_size
        ).permute(0, 3, 1, 2)
        position_embeddings = F.interpolate(
            position_embeddings,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        return position_embeddings.permute(0, 2, 3, 1).reshape(1, height * width, -1)

    def forward(self, pixel_values: Tensor, interpolate_pos_encoding: bool = False) -> Tensor:
        if pixel_values.ndim != 4:
            raise ValueError(
                f"pixel_values must be 4D (batch, channels, height, width), got {pixel_values.ndim}D."
            )

        patch_embeds = self.patch_embedding(pixel_values)
        height, width = patch_embeds.shape[-2:]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        if embeddings.shape[1] == self.num_positions:
            position_embeddings = self.position_embedding(self.position_ids)
        elif interpolate_pos_encoding:
            position_embeddings = self._interpolate_position_embeddings(height, width)
        else:
            expected = int(math.sqrt(self.num_positions)) * self.config.patch_size
            raise ValueError(
                f"Input image produced {embeddings.shape[1]} patches, but model expects "
                f"{self.num_positions}. Use image_size={expected} or set "
                "interpolate_pos_encoding=True."
            )

        embeddings = embeddings + position_embeddings.to(
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        return self.dropout(embeddings)


class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: Tensor,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            return attn_output, None
        return attn_output, attn_probs


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation_fn = _build_activation(config.hidden_act)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.dropout = config.hidden_dropout

    def forward(
        self,
        hidden_states: Tensor,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
        )
        hidden_states = residual + F.dropout(hidden_states, p=self.dropout, training=self.training)

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + F.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states, attn_weights


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, ...]], Optional[Tuple[Tensor, ...]]]:
        hidden_states = inputs_embeds
        all_hidden_states: Optional[Tuple[Tensor, ...]] = () if output_hidden_states else None
        all_attentions: Optional[Tuple[Tensor, ...]] = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, attn_weights = encoder_layer(
                hidden_states=hidden_states,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, all_hidden_states, all_attentions


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.use_head = config.vision_use_head

        if self.use_head:
            output_dim = config.projection_dim or config.hidden_size
            self.head = nn.Linear(config.hidden_size, output_dim)

    def forward(
        self,
        pixel_values: Tensor,
        interpolate_pos_encoding: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> SiglipVisionModelOutput | Tuple[Tensor, ...]:
        hidden_states = self.embeddings(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        hidden_states, all_hidden_states, all_attentions = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        last_hidden_state = self.post_layernorm(hidden_states)

        pooler_output = None
        if self.use_head:
            pooled = last_hidden_state.mean(dim=1)
            pooler_output = self.head(pooled)

        if not return_dict:
            outputs = [last_hidden_state]
            if pooler_output is not None:
                outputs.append(pooler_output)
            if all_hidden_states is not None:
                outputs.append(all_hidden_states)
            if all_attentions is not None:
                outputs.append(all_attentions)
            return tuple(outputs)

        return SiglipVisionModelOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                module.bias.data.zero_()
            return

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: Tensor,
        interpolate_pos_encoding: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> SiglipVisionModelOutput | Tuple[Tensor, ...]:
        return self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
