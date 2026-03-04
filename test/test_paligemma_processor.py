from __future__ import annotations

import numpy as np
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
