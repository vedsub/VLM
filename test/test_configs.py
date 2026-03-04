from __future__ import annotations

import json

from src.gemma_model import GemmaConfig, PaliGemmaConfig
from src.vit_model import SiglipVisionConfig


def test_gemma_config_from_dict() -> None:
    config_dict = {
        "hidden_size": 2048,
        "intermediate_size": 16384,
        "num_hidden_layers": 18,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "num_image_tokens": 256,
        "vocab_size": 257216,
        "torch_dtype": "float32",
    }
    config = GemmaConfig.from_dict(config_dict)

    assert config.hidden_size == 2048
    assert config.num_hidden_layers == 18
    assert config.num_image_tokens == 256
    assert config.vocab_size == 257216
    assert config.head_dim == 256


def test_paligemma_config_from_local_json() -> None:
    with open("paligemma-weights/config.json", "r", encoding="utf-8") as f:
        raw = json.load(f)

    config = PaliGemmaConfig.from_dict(raw)

    assert isinstance(config.text_config, GemmaConfig)
    assert isinstance(config.vision_config, SiglipVisionConfig)
    assert config.image_token_index == 257152
    assert config.hidden_size == 2048
    assert config.projection_dim == 2048
    assert config.text_config.num_image_tokens == 256
    assert config.vision_config.num_image_tokens == 256
