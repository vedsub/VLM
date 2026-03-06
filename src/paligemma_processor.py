from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, PreTrainedTokenizerBase


ImageInput = Image.Image | np.ndarray | torch.Tensor
TextInput = str | Sequence[str]


class PaliGemmaProcessor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        image_size: int = 224,
        image_seq_length: int = 256,
        image_mean: Sequence[float] = (0.5, 0.5, 0.5),
        image_std: Sequence[float] = (0.5, 0.5, 0.5),
        rescale_factor: float = 1.0 / 255.0,
        do_resize: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
        do_convert_rgb: bool | None = True,
        image_token: str = "<image>",
        image_token_index: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.image_seq_length = image_seq_length
        self.image_mean = np.asarray(image_mean, dtype=np.float32)
        self.image_std = np.asarray(image_std, dtype=np.float32)
        self.rescale_factor = float(rescale_factor)
        self.do_resize = bool(do_resize)
        self.do_rescale = bool(do_rescale)
        self.do_normalize = bool(do_normalize)
        self.do_convert_rgb = True if do_convert_rgb is None else bool(do_convert_rgb)
        self.image_token = image_token
        self.image_token_index = image_token_index
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        if hasattr(self.tokenizer, "add_bos_token"):
            self.tokenizer.add_bos_token = False
        if hasattr(self.tokenizer, "add_eos_token"):
            self.tokenizer.add_eos_token = False

        if self.image_token_index is not None:
            tokenized_image_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
            if int(tokenized_image_id) != int(self.image_token_index):
                raise ValueError(
                    "Tokenizer image token id does not match model config: "
                    f"tokenizer({self.image_token})={tokenized_image_id}, "
                    f"config.image_token_index={self.image_token_index}."
                )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        **kwargs: Any,
    ) -> "PaliGemmaProcessor":
        model_path = Path(pretrained_model_name_or_path)
        processor_keys = {
            "image_size",
            "image_seq_length",
            "image_mean",
            "image_std",
            "rescale_factor",
            "do_resize",
            "do_rescale",
            "do_normalize",
            "do_convert_rgb",
            "image_token",
        }
        processor_overrides = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in processor_keys}

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), **kwargs)

        preprocessor_config: dict[str, Any] = {}
        model_config: dict[str, Any] = {}

        preprocessor_config_path = model_path / "preprocessor_config.json"
        if preprocessor_config_path.exists():
            with preprocessor_config_path.open("r", encoding="utf-8") as f:
                preprocessor_config = json.load(f)

        model_config_path = model_path / "config.json"
        if model_config_path.exists():
            with model_config_path.open("r", encoding="utf-8") as f:
                model_config = json.load(f)

        size_obj = preprocessor_config.get("size", {})
        image_size = int(size_obj.get("height", size_obj.get("width", 224)))
        text_num_image_tokens = model_config.get("text_config", {}).get("num_image_tokens")
        vision_num_image_tokens = model_config.get("vision_config", {}).get("num_image_tokens")
        if (
            text_num_image_tokens is not None
            and vision_num_image_tokens is not None
            and int(text_num_image_tokens) != int(vision_num_image_tokens)
        ):
            raise ValueError(
                "Model config mismatch: text_config.num_image_tokens "
                f"({text_num_image_tokens}) != vision_config.num_image_tokens "
                f"({vision_num_image_tokens})."
            )
        image_seq_length = int(
            preprocessor_config.get(
                "image_seq_length",
                text_num_image_tokens
                if text_num_image_tokens is not None
                else model_config.get("vision_config", {}).get("num_image_tokens", 256),
            )
        )

        return cls(
            tokenizer=tokenizer,
            image_size=processor_overrides.get("image_size", image_size),
            image_seq_length=processor_overrides.get("image_seq_length", image_seq_length),
            image_mean=processor_overrides.get(
                "image_mean",
                preprocessor_config.get("image_mean", [0.5, 0.5, 0.5]),
            ),
            image_std=processor_overrides.get(
                "image_std",
                preprocessor_config.get("image_std", [0.5, 0.5, 0.5]),
            ),
            rescale_factor=processor_overrides.get(
                "rescale_factor",
                preprocessor_config.get("rescale_factor", 1.0 / 255.0),
            ),
            do_resize=processor_overrides.get("do_resize", preprocessor_config.get("do_resize", True)),
            do_rescale=processor_overrides.get(
                "do_rescale", preprocessor_config.get("do_rescale", True)
            ),
            do_normalize=processor_overrides.get(
                "do_normalize", preprocessor_config.get("do_normalize", True)
            ),
            do_convert_rgb=processor_overrides.get(
                "do_convert_rgb", preprocessor_config.get("do_convert_rgb", True)
            ),
            image_token=processor_overrides.get("image_token", "<image>"),
            image_token_index=model_config.get("image_token_index"),
            bos_token_id=model_config.get("bos_token_id"),
            eos_token_id=model_config.get("eos_token_id"),
            pad_token_id=model_config.get("pad_token_id"),
        )

    @property
    def image_token_id(self) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        if token_id is None or token_id < 0:
            raise ValueError(f"Tokenizer does not recognize image token {self.image_token!r}.")
        return int(token_id)

    @property
    def bos_token(self) -> str:
        return self.tokenizer.bos_token or "<bos>"

    def _strip_prompt_markers(self, prompt: str) -> str:
        clean = prompt.rstrip("\n")
        clean = clean.replace(self.image_token, "")
        clean = clean.replace(self.bos_token, "")
        return clean.strip()

    def _ensure_text_list(self, text: TextInput) -> list[str]:
        if isinstance(text, str):
            return [text]
        if isinstance(text, Sequence) and text and all(isinstance(t, str) for t in text):
            return list(text)
        raise TypeError("text must be a string or a non-empty sequence of strings.")

    def _ensure_image_list(self, images: ImageInput | Sequence[ImageInput]) -> list[ImageInput]:
        if isinstance(images, (Image.Image, np.ndarray, torch.Tensor)):
            return [images]
        if isinstance(images, Sequence) and images:
            return list(images)
        raise TypeError("images must be an image or a non-empty sequence of images.")

    def _align_batch(self, texts: list[str], images: list[ImageInput]) -> tuple[list[str], list[ImageInput]]:
        if len(texts) == len(images):
            return texts, images
        if len(texts) == 1 and len(images) > 1:
            return texts * len(images), images
        if len(images) == 1 and len(texts) > 1:
            return texts, images * len(texts)
        raise ValueError(
            f"Mismatched batch sizes: got {len(texts)} text prompts and {len(images)} images."
        )

    def _to_pil_image(self, image: ImageInput) -> Image.Image:
        if isinstance(image, Image.Image):
            pil = image
        elif isinstance(image, torch.Tensor):
            array = image.detach().cpu().numpy()
            if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
                array = np.transpose(array, (1, 2, 0))
            pil = Image.fromarray(self._to_uint8(array))
        elif isinstance(image, np.ndarray):
            pil = Image.fromarray(self._to_uint8(image))
        else:
            raise TypeError(
                "Unsupported image type. Expected PIL.Image.Image, numpy.ndarray, or torch.Tensor."
            )

        if self.do_convert_rgb:
            pil = pil.convert("RGB")
        return pil

    def _to_uint8(self, array: np.ndarray) -> np.ndarray:
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.ndim != 3:
            raise ValueError(f"Expected image array with 2 or 3 dims, got shape {array.shape}.")

        if np.issubdtype(array.dtype, np.floating):
            max_val = float(np.max(array)) if array.size else 0.0
            if max_val <= 1.0:
                array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
        return array

    def _process_single_image(self, image: ImageInput) -> torch.Tensor:
        pil = self._to_pil_image(image)
        if self.do_resize:
            resampling = getattr(Image, "Resampling", Image)
            pil = pil.resize((self.image_size, self.image_size), resample=resampling.BICUBIC)

        array = np.asarray(pil, dtype=np.float32)
        if self.do_rescale:
            array = array * self.rescale_factor
        if self.do_normalize:
            array = (array - self.image_mean) / self.image_std

        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array)

    def process_images(
        self,
        images: ImageInput | Sequence[ImageInput],
        return_tensors: str = "pt",
    ) -> torch.Tensor | np.ndarray:
        image_list = self._ensure_image_list(images)
        pixel_values = torch.stack([self._process_single_image(img) for img in image_list], dim=0)

        if return_tensors == "pt":
            return pixel_values
        if return_tensors == "np":
            return pixel_values.numpy()
        raise ValueError(f"Unsupported return_tensors={return_tensors!r}. Expected 'pt' or 'np'.")

    def build_input_masks(
        self,
        input_ids: torch.Tensor | np.ndarray,
    ) -> dict[str, torch.Tensor | np.ndarray]:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be 2D with shape (batch_size, seq_length), got {tuple(input_ids.shape)}."
            )

        image_token_id = int(self.image_token_id)
        pad_token_id = (
            int(self.pad_token_id)
            if self.pad_token_id is not None
            else (
                int(self.tokenizer.pad_token_id)
                if getattr(self.tokenizer, "pad_token_id", None) is not None
                else None
            )
        )

        image_mask = input_ids == image_token_id
        if pad_token_id is None:
            if isinstance(input_ids, torch.Tensor):
                padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            else:
                padding_mask = np.zeros_like(input_ids, dtype=bool)
        else:
            padding_mask = input_ids == pad_token_id
        text_mask = (~image_mask) & (~padding_mask)

        return {
            "text_mask": text_mask,
            "image_mask": image_mask,
            "padding_mask": padding_mask,
        }

    def build_prompt(self, prompt: str, image_token_count: int | None = None) -> str:
        token_count = self.image_seq_length if image_token_count is None else int(image_token_count)
        if token_count <= 0:
            raise ValueError(f"image_token_count must be > 0, got {token_count}.")

        clean_prompt = self._strip_prompt_markers(prompt)
        image_prefix = self.image_token * token_count
        return f"{image_prefix}{self.bos_token}{clean_prompt}\n"

    def __call__(
        self,
        text: TextInput,
        images: ImageInput | Sequence[ImageInput],
        return_tensors: str = "pt",
        padding: str | bool = "longest",
        truncation: bool = False,
        max_length: int | None = None,
        image_token_count: int | None = None,
        **tokenizer_kwargs: Any,
    ) -> dict[str, Any]:
        texts = self._ensure_text_list(text)
        image_list = self._ensure_image_list(images)
        texts, image_list = self._align_batch(texts, image_list)

        prompts = [self.build_prompt(prompt, image_token_count=image_token_count) for prompt in texts]
        tokenized = self.tokenizer(
            prompts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=False,
            return_tensors=return_tensors,
            **tokenizer_kwargs,
        )
        token_masks = self.build_input_masks(tokenized["input_ids"])
        tokenized.update(token_masks)
        tokenized["pixel_values"] = self.process_images(image_list, return_tensors=return_tensors)
        return tokenized

    def batch_decode(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.decode(*args, **kwargs)
