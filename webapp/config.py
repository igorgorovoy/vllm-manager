import os
from pathlib import Path

# CUDA library paths for GB10 (aarch64 + CUDA 13.0)
CUDA_LD_PATH = ":".join([
    "/usr/lib/aarch64-linux-gnu/libcusparseLt/13",
    "/usr/lib/aarch64-linux-gnu/nvshmem/13",
    "/usr/local/cuda-13.0/targets/sbsa-linux/lib",
    "/usr/lib/aarch64-linux-gnu",
])

CONFIG_DIR = Path.home() / ".config" / "vllm-models"
CONFIG_FILE = CONFIG_DIR / "models.json"
SERVERS_FILE = CONFIG_DIR / "servers.json"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
TOOLS_DIR = Path("/home/igogo/vllm-tools")
VENV_PY = TOOLS_DIR / ".venv" / "bin" / "python"
VENV_HFCLI = TOOLS_DIR / ".venv" / "bin" / "hf"
WEBAPP_PORT = 7860
