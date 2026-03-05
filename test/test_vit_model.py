import torch

from src.vit_model import (
    RotaryEmbedding,
    SiglipAttention,
    SiglipVisionConfig,
    SiglipVisionModel,
    apply_rotary_pos_emb,
    rotate_half,
)


def test_siglip_vision_forward_shapes_without_head() -> None:
    config = SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=32,
        patch_size=16,
        vision_use_head=False,
    )
    model = SiglipVisionModel(config)
    pixel_values = torch.randn(2, 3, 32, 32)

    outputs = model(pixel_values)

    assert outputs.last_hidden_state.shape == (2, 4, 32)
    assert outputs.pooler_output is None


def test_siglip_vision_forward_with_flags_and_head() -> None:
    config = SiglipVisionConfig(
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=3,
        num_attention_heads=6,
        image_size=32,
        patch_size=16,
        vision_use_head=True,
        projection_dim=24,
    )
    model = SiglipVisionModel(config)
    pixel_values = torch.randn(1, 3, 32, 32)

    outputs = model(pixel_values, output_hidden_states=True, output_attentions=True)
    tuple_outputs = model(pixel_values, return_dict=False)

    assert outputs.last_hidden_state.shape == (1, 4, 48)
    assert outputs.pooler_output is not None
    assert outputs.pooler_output.shape == (1, 24)
    assert outputs.hidden_states is not None
    assert len(outputs.hidden_states) == config.num_hidden_layers + 1
    assert outputs.attentions is not None
    assert len(outputs.attentions) == config.num_hidden_layers
    assert tuple_outputs[0].shape == (1, 4, 48)


def test_siglip_vision_accepts_variable_patch_sequence_without_abs_pos_embeddings() -> None:
    config = SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=32,
        patch_size=16,
        vision_use_head=False,
    )
    model = SiglipVisionModel(config)
    pixel_values = torch.randn(1, 3, 48, 48)

    outputs = model(pixel_values)

    assert outputs.last_hidden_state.shape == (1, 9, 32)
    assert outputs.pooler_output is None


def test_rotate_half() -> None:
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    out = rotate_half(x)
    expected = torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]])
    assert torch.allclose(out, expected)


def test_apply_rotary_pos_emb_changes_nonzero_positions() -> None:
    q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]]])
    k = q.clone()
    rotary = RotaryEmbedding(dim=4)
    position_ids = torch.tensor([[0, 1]], dtype=torch.long)
    cos, sin = rotary(position_ids=position_ids, dtype=q.dtype)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos=cos, sin=sin)

    assert torch.allclose(q_rot[:, :, 0, :], q[:, :, 0, :], atol=1e-6)
    assert torch.allclose(k_rot[:, :, 0, :], k[:, :, 0, :], atol=1e-6)
    assert not torch.allclose(q_rot[:, :, 1, :], q[:, :, 1, :], atol=1e-6)
    assert not torch.allclose(k_rot[:, :, 1, :], k[:, :, 1, :], atol=1e-6)


def test_siglip_attention_grouped_query_attention_shapes() -> None:
    config = SiglipVisionConfig(
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=1,
        num_attention_heads=6,
        num_key_value_heads=2,
        image_size=32,
        patch_size=16,
    )
    attn = SiglipAttention(config)
    hidden_states = torch.randn(2, 4, 48)
    attn_output, attn_probs = attn(hidden_states, output_attentions=True)

    assert attn.k_proj.out_features == 16
    assert attn.v_proj.out_features == 16
    assert attn_output.shape == (2, 4, 48)
    assert attn_probs is not None
    assert attn_probs.shape == (2, 6, 4, 4)


def test_siglip_config_rejects_invalid_num_key_value_heads() -> None:
    try:
        SiglipVisionConfig(
            hidden_size=48,
            intermediate_size=96,
            num_hidden_layers=1,
            num_attention_heads=6,
            num_key_value_heads=8,
            image_size=32,
            patch_size=16,
        )
    except ValueError as exc:
        assert "cannot exceed" in str(exc)
    else:
        raise AssertionError("Expected ValueError when num_key_value_heads > num_attention_heads.")

    try:
        SiglipVisionConfig(
            hidden_size=48,
            intermediate_size=96,
            num_hidden_layers=1,
            num_attention_heads=6,
            num_key_value_heads=4,
            image_size=32,
            patch_size=16,
        )
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError when num_attention_heads is not divisible by num_key_value_heads."
        )
