from __future__ import annotations

import torch

from src.gemma_model import (
    GemmaConfig,
    GemmaDecoderLayer,
    GemmaMLP,
    GemmaModel,
    GemmaRMSNorm,
    KVCache,
)


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
