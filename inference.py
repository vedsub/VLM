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
) -> tuple[PaliGemmaForConditionalGeneration, AutoTokenizer, AutoProcessor]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    processor = AutoProcessor.from_pretrained(str(model_path))

    # Keep tokenizer source explicit and local as requested.
    if hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=choose_dtype(device),
    )
    model.to(device)
    model.eval()
    return model, tokenizer, processor


def run_inference(
    model: PaliGemmaForConditionalGeneration,
    tokenizer: AutoTokenizer,
    processor: AutoProcessor,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    input_len = int(inputs["input_ids"].shape[-1])
    new_token_ids = generated_ids[0, input_len:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    model, tokenizer, processor = load_model_and_tokenizer(
        model_path=args.model_path,
        device=device,
    )
    output_text = run_inference(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        image_path=args.image_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    print(output_text)


if __name__ == "__main__":
    main()
