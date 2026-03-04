import torch

from src.vit_model import SiglipVisionConfig, SiglipVisionModel


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
