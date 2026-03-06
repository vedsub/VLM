from __future__ import annotations

import torch

from src.gemma_model import (
    GemmaConfig,
    GemmaDecoderLayer,
    GemmaMLP,
    GemmaModel,
    GemmaRMSNorm,
    KVCache,
    PaliGemmaConfig,
    PaliGemmaMultiModalProjector,
    merge_text_and_image_embeddings,
)
from src.vit_model import SiglipVisionConfig


def test_gemma_rms_norm_matches_manual_formula() -> None:
    norm = GemmaRMSNorm(hidden_size=4, eps=1e-6)
    with torch.no_grad():
        norm.weight.zero_()

    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float32)
    out = norm(x)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    expected = x * torch.rsqrt(variance + 1e-6)
    assert torch.allclose(out, expected, atol=1e-6)


def test_gemma_mlp_uses_gate_up_down_projections() -> None:
    config = GemmaConfig(
        hidden_size=2,
        intermediate_size=2,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        hidden_act="gelu_pytorch_tanh",
    )
    mlp = GemmaMLP(config)
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(torch.eye(2))
        mlp.up_proj.weight.copy_(2.0 * torch.eye(2))
        mlp.down_proj.weight.copy_(torch.eye(2))

    x = torch.tensor([[[1.0, -1.0]]], dtype=torch.float32)
    expected = torch.nn.functional.gelu(x, approximate="tanh") * (2.0 * x)
    out = mlp(x)
    assert torch.allclose(out, expected, atol=1e-6)


def test_gemma_decoder_layer_pre_norm_attention_and_cache() -> None:
    config = GemmaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_dropout=0.0,
    )
    layer = GemmaDecoderLayer(config=config, layer_idx=0)
    cache = KVCache(num_layers=1)

    hidden_states = torch.randn(2, 3, 16)
    out, attn = layer(hidden_states=hidden_states, kv_cache=cache, output_attentions=True)
    assert out.shape == (2, 3, 16)
    assert attn is not None
    assert attn.shape == (2, 4, 3, 3)
    assert cache.get_seq_length(0) == 3

    next_hidden_states = torch.randn(2, 1, 16)
    out, attn = layer(hidden_states=next_hidden_states, kv_cache=cache, output_attentions=True)
    assert out.shape == (2, 1, 16)
    assert attn is not None
    assert attn.shape == (2, 4, 1, 4)
    assert cache.get_seq_length(0) == 4


def test_gemma_model_stacks_decoder_layers_with_embeddings() -> None:
    config = GemmaConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_dropout=0.0,
    )
    model = GemmaModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    outputs = model(input_ids=input_ids, output_attentions=True, output_hidden_states=True)

    assert outputs.last_hidden_state.shape == (2, 5, 16)
    assert outputs.hidden_states is not None
    assert len(outputs.hidden_states) == config.num_hidden_layers + 1
    assert outputs.attentions is not None
    assert len(outputs.attentions) == config.num_hidden_layers
    assert outputs.attentions[0] is not None
    assert outputs.attentions[0].shape == (2, 4, 5, 5)


def test_gemma_model_cache_aware_decoding_step() -> None:
    config = GemmaConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_dropout=0.0,
    )
    model = GemmaModel(config)
    cache = KVCache(num_layers=config.num_hidden_layers)

    prompt_ids = torch.randint(0, config.vocab_size, (2, 3))
    outputs = model(input_ids=prompt_ids, kv_cache=cache, output_attentions=True)
    assert outputs.last_hidden_state.shape == (2, 3, 16)
    assert cache.get_seq_length(0) == 3
    assert cache.get_seq_length(1) == 3

    next_ids = torch.randint(0, config.vocab_size, (2, 1))
    attention_mask = torch.ones(2, 4)
    outputs = model(
        input_ids=next_ids,
        attention_mask=attention_mask,
        kv_cache=cache,
        output_attentions=True,
    )
    assert outputs.last_hidden_state.shape == (2, 1, 16)
    assert cache.get_seq_length(0) == 4
    assert cache.get_seq_length(1) == 4
    assert outputs.attentions is not None
    assert outputs.attentions[0] is not None
    assert outputs.attentions[0].shape == (2, 4, 1, 4)


def test_multimodal_projector_aligns_vision_and_text_hidden_sizes() -> None:
    config = PaliGemmaConfig(
        vision_config=SiglipVisionConfig(
            hidden_size=12,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=3,
            num_key_value_heads=3,
            image_size=28,
            patch_size=14,
            num_image_tokens=4,
            vision_use_head=False,
            projection_dim=16,
        ),
        text_config=GemmaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_image_tokens=4,
        ),
        hidden_size=16,
        projection_dim=16,
    )
    projector = PaliGemmaMultiModalProjector(config)
    vision_features = torch.randn(2, 4, 12)
    projected = projector(vision_features)
    assert projected.shape == (2, 4, 16)


def test_merge_text_and_image_embeddings_replaces_image_token_slots() -> None:
    image_token_index = 99
    input_ids = torch.tensor(
        [
            [99, 99, 5, 6, 7],
            [99, 99, 8, 9, 10],
        ],
        dtype=torch.long,
    )
    inputs_embeds = torch.zeros(2, 5, 4, dtype=torch.float32)
    image_features = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]],
            [[-1.0, -2.0, -3.0, -4.0], [-10.0, -20.0, -30.0, -40.0]],
        ],
        dtype=torch.float32,
    )

    merged = merge_text_and_image_embeddings(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_features=image_features,
        image_token_index=image_token_index,
    )

    assert torch.allclose(merged[0, 0], image_features[0, 0])
    assert torch.allclose(merged[0, 1], image_features[0, 1])
    assert torch.allclose(merged[1, 0], image_features[1, 0])
    assert torch.allclose(merged[1, 1], image_features[1, 1])
    assert torch.allclose(merged[:, 2:, :], torch.zeros(2, 3, 4))
