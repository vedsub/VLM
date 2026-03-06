#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-paligemma-weights}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
DEVICE="${DEVICE:-auto}"
PROBLEMATIC_TOKEN_ID="${PROBLEMATIC_TOKEN_ID:-1}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash src/launch_inference.sh <image_path> [prompt...]"
  exit 1
fi

IMAGE_PATH="$1"
shift
PROMPT="${*:-describe this image}"

if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in python3 python /usr/bin/python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

if [[ "${IMAGE_PATH}" =~ ^[A-Za-z]:\\ ]] && command -v wslpath >/dev/null 2>&1; then
  IMAGE_PATH="$(wslpath "${IMAGE_PATH}")"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Error: python/python3 not found in PATH."
  exit 1
fi

has_required_pkgs() {
  local bin="$1"
  "${bin}" - <<'PY' >/dev/null 2>&1
import torch
import transformers
from PIL import Image
PY
}

if ! has_required_pkgs "${PYTHON_BIN}"; then
  for fallback in /usr/bin/python3 python3 python; do
    if command -v "${fallback}" >/dev/null 2>&1 && has_required_pkgs "${fallback}"; then
      PYTHON_BIN="${fallback}"
      break
    fi
  done
fi

if ! has_required_pkgs "${PYTHON_BIN}"; then
  echo "Error: required packages missing for ${PYTHON_BIN}."
  echo "Install with uv (no pip needed):"
  echo "  uv pip install --python ${PYTHON_BIN} torch transformers pillow"
  exit 1
fi

"${PYTHON_BIN}" inference.py \
  --model-path "${MODEL_PATH}" \
  --image-path "${IMAGE_PATH}" \
  --prompt "${PROMPT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --device "${DEVICE}" \
  --problematic-token-id "${PROBLEMATIC_TOKEN_ID}" \
  --verbose
