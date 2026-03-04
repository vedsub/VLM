from __future__ import annotations

import torch

from src.gemma_model import GemmaConfig, KVCache


def test_kv_cache_update_and_sequence_growth() -> None:
    cache = KVCache(num_layers=2)

    k1 = torch.randn(1, 1, 3, 8)
    v1 = torch.randn(1, 1, 3, 8)
    out_k, out_v = cache.update(layer_idx=0, key_states=k1, value_states=v1)
    assert out_k.shape == (1, 1, 3, 8)
    assert out_v.shape == (1, 1, 3, 8)
    assert cache.get_seq_length(0) == 3

    k2 = torch.randn(1, 1, 2, 8)
    v2 = torch.randn(1, 1, 2, 8)
    out_k, out_v = cache.update(layer_idx=0, key_states=k2, value_states=v2)
    assert out_k.shape == (1, 1, 5, 8)
    assert out_v.shape == (1, 1, 5, 8)
    assert cache.get_seq_length(0) == 5
    assert len(cache) == 5
    assert torch.allclose(out_k[:, :, :3], k1)
    assert torch.allclose(out_v[:, :, :3], v1)


def test_kv_cache_respects_max_cache_len() -> None:
    cache = KVCache(num_layers=1, max_cache_len=4)
    k1 = torch.arange(1 * 1 * 3 * 2, dtype=torch.float32).reshape(1, 1, 3, 2)
    v1 = k1 + 100
    k2 = torch.arange(1 * 1 * 3 * 2, dtype=torch.float32).reshape(1, 1, 3, 2) + 1000
    v2 = k2 + 100

    out_k, out_v = cache.update(0, k1, v1)
    assert out_k.shape[-2] == 3
    out_k, out_v = cache.update(0, k2, v2)
    assert out_k.shape[-2] == 4
    assert out_v.shape[-2] == 4
    assert cache.get_seq_length(0) == 4
    expected_k = torch.cat([k1, k2], dim=-2)[:, :, -4:, :]
    expected_v = torch.cat([v1, v2], dim=-2)[:, :, -4:, :]
    assert torch.allclose(out_k, expected_k)
    assert torch.allclose(out_v, expected_v)


def test_kv_cache_reorder_cache_for_beam_search() -> None:
    cache = KVCache(num_layers=1)
    k = torch.tensor(
        [
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[10.0, 20.0], [30.0, 40.0]]],
        ]
    )  # (2, 1, 2, 2)
    v = k + 0.5
    cache.update(0, k, v)

    beam_idx = torch.tensor([1, 0], dtype=torch.long)
    cache.reorder_cache(beam_idx)
    out_k, out_v = cache.get_layer(0)
    assert out_k is not None and out_v is not None
    assert torch.allclose(out_k[0], k[1])
    assert torch.allclose(out_k[1], k[0])
    assert torch.allclose(out_v[0], v[1])
    assert torch.allclose(out_v[1], v[0])


def test_kv_cache_from_config_and_reset() -> None:
    config = GemmaConfig(num_hidden_layers=3)
    cache = KVCache.from_config(config)
    assert cache.num_layers == 3

    cache.update(1, torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4))
    assert cache.get_seq_length(1) == 2
    cache.reset()
    assert cache.get_seq_length(1) == 0
