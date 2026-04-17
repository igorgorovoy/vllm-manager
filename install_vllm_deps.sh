#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"

echo "[1/4] Creating virtual environment at ${VENV_DIR}..."
python3 -m venv "${VENV_DIR}"

echo "[2/4] Updating pip tools in venv..."
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel

echo "[3/4] Installing dependencies from ${REQUIREMENTS}..."
"${VENV_DIR}/bin/python" -m pip install -U -r "${REQUIREMENTS}"

echo "[4/4] Verifying tools..."
"${VENV_DIR}/bin/python" -m pip show vllm >/dev/null
"${VENV_DIR}/bin/hf" --help >/dev/null

echo "Done. vLLM dependencies installed in ${VENV_DIR}."
