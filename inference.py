from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, PaliGemmaForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local PaliGemma inference from a downloaded model directory."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("paligemma-weights"),
        help="Local path containing model weights/tokenizer files.",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        required=True,
        help="Path to an input image.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device. 'auto' picks cuda if available, else cpu.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print token-by-token debug logs during decoding.",
    )
    parser.add_argument(
        "--problematic-token-id",
        type=int,
        default=1,
        help="If argmax selects this token, use next-best token and print a warning.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def choose_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def load_model_and_tokenizer(
    model_path: Path,
    device: torch.device,
) -> tuple[PaliGemmaForConditionalGeneration, AutoTokenizer, AutoProcessor, dict[str, list[str]]]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    processor = AutoProcessor.from_pretrained(str(model_path))

    # Keep tokenizer source explicit and local as requested.
    if hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer

    model_and_info = PaliGemmaForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=choose_dtype(device),
        local_files_only=True,
        output_loading_info=True,
    )
    if isinstance(model_and_info, tuple):
        model, loading_info = model_and_info
    else:
        model = model_and_info
        loading_info = {"missing_keys": [], "unexpected_keys": []}
    model.to(device)
    model.eval()
    return model, tokenizer, processor, loading_info


def _move_inputs_to_device(
    inputs: dict[str, torch.Tensor],
    device: torch.device,
    model_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            continue
        if torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=model_dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def _verbose_decode(
    model: PaliGemmaForConditionalGeneration,
    tokenizer: AutoTokenizer,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    problematic_token_id: int | None,
) -> str:
    input_ids = inputs["input_ids"]
    if input_ids.shape[0] != 1:
        raise ValueError("Verbose decoding currently supports batch_size=1.")
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    pixel_values = inputs.get("pixel_values")
    past_key_values = None
    generated_token_ids: list[int] = []
    eos_token_id = tokenizer.eos_token_id

    for step in range(max_new_tokens):
        model_inputs: dict[str, torch.Tensor] = {
            "attention_mask": attention_mask,
            "use_cache": True,
            "return_dict": True,
        }
        if past_key_values is None:
            model_inputs["input_ids"] = input_ids
            if pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values
        else:
            model_inputs["input_ids"] = torch.tensor(
                [[generated_token_ids[-1]]],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            model_inputs["past_key_values"] = past_key_values

        with torch.inference_mode():
            outputs = model(**model_inputs)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        topk = torch.topk(logits, k=2, dim=-1).indices[0]
        next_token_id = int(topk[0].item())
        if problematic_token_id is not None and next_token_id == problematic_token_id:
            print(
                f"Warning: Generated problematic token {problematic_token_id}. "
                "Trying next best token."
            )
            next_token_id = int(topk[1].item())

        generated_token_ids.append(next_token_id)
        decoded_piece = tokenizer.decode([next_token_id], skip_special_tokens=False)
        print(f"Step {step}: Generated token ID: {next_token_id}, Decoded: {decoded_piece!r}")
        print(f"Position IDs: tensor([[{float(attention_mask.shape[-1])}]])")

        if eos_token_id is not None and next_token_id == int(eos_token_id):
            break

        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=-1,
        )
    return tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()


def run_inference(
    model: PaliGemmaForConditionalGeneration,
    tokenizer: AutoTokenizer,
    processor: AutoProcessor,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    verbose: bool,
    problematic_token_id: int | None,
) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    image = Image.open(image_path).convert("RGB")
    processor_inputs = processor(images=image, text=prompt, return_tensors="pt")
    model_inputs = _move_inputs_to_device(processor_inputs, device=device, model_dtype=model.dtype)

    if verbose:
        return _verbose_decode(
            model=model,
            tokenizer=tokenizer,
            inputs=model_inputs,
            max_new_tokens=max_new_tokens,
            problematic_token_id=problematic_token_id,
        )

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )

    input_len = int(model_inputs["input_ids"].shape[-1])
    new_token_ids = generated_ids[0, input_len:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"device :  {device.type}")
    print("loading model :")

    model, tokenizer, processor, loading_info = load_model_and_tokenizer(
        model_path=args.model_path,
        device=device,
    )
    missing_keys = loading_info.get("missing_keys", [])
    unexpected_keys = loading_info.get("unexpected_keys", [])
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    output_text = run_inference(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        image_path=args.image_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=device,
        verbose=args.verbose,
        problematic_token_id=args.problematic_token_id,
    )
    print(output_text)


if __name__ == "__main__":
    main()
