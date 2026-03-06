from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.paligemma_processor import PaliGemmaProcessor


def _dummy_image(size: int = 224) -> Image.Image:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[..., 0] = 120
    arr[..., 1] = 50
    arr[..., 2] = 200
    return Image.fromarray(arr, mode="RGB")


def test_build_prompt_format() -> None:
    processor = PaliGemmaProcessor.from_pretrained("paligemma-weights")
    prompt = processor.build_prompt("What instrument is shown in this picture?")

    assert prompt.startswith("<image>" * processor.image_seq_length + "<bos>")
    assert prompt.endswith("\n")


def test_processor_loads_model_config_token_fields() -> None:
    processor = PaliGemmaProcessor.from_pretrained("paligemma-weights")

    assert processor.image_seq_length == 256
    assert processor.image_token_index == 257152
    assert processor.image_token_id == 257152
    assert processor.bos_token_id == 2
    assert processor.eos_token_id == 1
    assert processor.pad_token_id == 0


def test_build_prompt_normalizes_existing_markers() -> None:
    processor = PaliGemmaProcessor.from_pretrained("paligemma-weights")
    raw = "<image><image><bos>What instrument is shown in this picture?\n"
    prompt = processor.build_prompt(raw)

    expected_prefix = "<image>" * processor.image_seq_length + "<bos>"
    assert prompt.startswith(expected_prefix)
    assert prompt.count("<image>") == processor.image_seq_length
    assert prompt.count("<bos>") == 1


def test_processor_outputs_multimodal_tensors() -> None:
    processor = PaliGemmaProcessor.from_pretrained("paligemma-weights")
    image = _dummy_image()

    model_inputs = processor(
        text="What instrument is shown in this picture?",
        images=image,
        return_tensors="pt",
    )

    input_ids = model_inputs["input_ids"]
    pixel_values = model_inputs["pixel_values"]

    assert input_ids.shape[0] == 1
    assert pixel_values.shape == (1, 3, 224, 224)
    assert (input_ids[0, : processor.image_seq_length] == processor.image_token_id).all()
    assert int(input_ids[0, processor.image_seq_length]) == processor.tokenizer.bos_token_id


def test_processor_broadcasts_single_image_for_multiple_prompts() -> None:
    processor = PaliGemmaProcessor.from_pretrained("paligemma-weights")
    image = _dummy_image()

    model_inputs = processor(
        text=["caption en", "caption fr"],
        images=image,
        return_tensors="pt",
    )

    assert model_inputs["input_ids"].shape[0] == 2
    assert model_inputs["pixel_values"].shape[0] == 2


def test_processor_allows_custom_image_token_count() -> None:
    processor = PaliGemmaProcessor.from_pretrained("paligemma-weights")
    image = _dummy_image()
    image_token_count = 8

    model_inputs = processor(
        text="caption en",
        images=image,
        return_tensors="pt",
        image_token_count=image_token_count,
    )

    input_ids = model_inputs["input_ids"]
    assert (input_ids[0, :image_token_count] == processor.image_token_id).all()
    assert int(input_ids[0, image_token_count]) == processor.tokenizer.bos_token_id


def test_processor_builds_text_image_padding_masks_from_input_ids() -> None:
    processor = PaliGemmaProcessor.from_pretrained("paligemma-weights")
    image = _dummy_image()

    model_inputs = processor(
        text=["short", "this is a longer prompt to force padding"],
        images=image,
        return_tensors="pt",
        padding="longest",
    )

    input_ids = model_inputs["input_ids"]
    text_mask = model_inputs["text_mask"]
    image_mask = model_inputs["image_mask"]
    padding_mask = model_inputs["padding_mask"]

    assert text_mask.shape == input_ids.shape
    assert image_mask.shape == input_ids.shape
    assert padding_mask.shape == input_ids.shape

    expected_image = input_ids == processor.image_token_id
    expected_padding = input_ids == processor.pad_token_id
    expected_text = (~expected_image) & (~expected_padding)

    assert torch.equal(image_mask, expected_image)
    assert torch.equal(padding_mask, expected_padding)
    assert torch.equal(text_mask, expected_text)
