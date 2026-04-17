# Hardware, CUDA, PyTorch — platform specifics

This file describes the specifics of the machine (`spark-98a2`) — **NVIDIA DGX Spark** based on **GB10 Grace-Blackwell**. All non-standard decisions in `vllm-tools` (`LD_LIBRARY_PATH` paths, `cu130` wheels, memory checks) follow directly from these specifics.

## 1. Hardware

| Component | Value |
|---|---|
| Platform | NVIDIA DGX Spark |
| GPU | NVIDIA GB10 (Grace-Blackwell) |
| Compute capability | **sm_121** (12.1) |
| Graphics memory | **unified with CPU** — 121 GB LPDDR5X |
| CPU | ARM Cortex-X925 + Cortex-A725 (Grace), 20 cores, NUMA node 0 |
| Architecture | **aarch64 / arm64** (NOT x86_64, NOT Tegra) |
| OS | Ubuntu, kernel 6.17 nvidia |
| NVIDIA driver | 580.142 |
| CUDA runtime | **13.0** (nvcc 13.0.88) |

### Note 1.1. Unified memory
`nvidia-smi` shows `Memory-Usage: Not Supported` — this is **not a bug**. GB10 has no dedicated VRAM: CPU and GPU share the same LPDDR5X pool. To query available GPU memory, use:

```python
import torch
free, total = torch.cuda.mem_get_info(0)   # in bytes
```

That's exactly what `webapp/core.py:get_gpu_free_memory_gb` does. `free -h` also shows the entire pool — don't think of it as "system" RAM separate from GPU. Any large model in vLLM consumes the same memory the Linux kernel sees. That's why `start_server` in `core.py:316` refuses to launch if `model_size > free_gb`.

### Note 1.2. sbsa, not tegra
The CUDA target is `sbsa-linux` (*Server Base System Architecture*), not `tegra-linux`. The correct path is:

```
/usr/local/cuda-13.0/targets/sbsa-linux/lib
```

Jetson tutorials with `aarch64-linux-gnu/tegra/...` **do not work** here — that's a different ABI.

### Note 1.3. This machine is not a typical ARM server
GB10 is a SoC with tight CPU↔GPU coupling via NVLink-C2C. Consequences:
- `host→device` copies are nearly free (same pool).
- `pin_memory=True` in DataLoader gives no speedup — it's already in the "pinned" pool.
- `tensor_parallel_size > 1` makes no sense: there is only one GPU.

## 2. PyTorch

Installed: **`torch==2.10.0+cu130`**, linked against CUDA 13.0.

### Note 2.1. sm_121 vs PyTorch's official matrix
PyTorch 2.10 officially supports `sm_80 … sm_120`. GB10 is `sm_121` → on `import torch` you get:

```
UserWarning: Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0).
```

**This is not an error** — kernels run via PTX JIT or fall back to sm_120. Works reliably for inference (verified with vLLM). But:
- **Do not ignore** this warning for custom CUDA extensions — they may lack a PTX fallback.
- When a torch release with official `sm_121` appears — upgrade; performance can improve noticeably.
- `TORCH_CUDA_ARCH_LIST="12.1"` when compiling extensions — only if the toolchain understands it (check `nvcc --list-gpu-arch`).

### Note 2.2. `+cu130` wheels are mandatory
All binary packages with CUDA extensions (torch, torchaudio, torchvision, flashinfer, xformers, bitsandbytes, apex) **must** be:
1. an aarch64 build,
2. built against **cu130** (not cu121/cu124).

On PyPI not everything is available immediately. If `pip install` pulls an `x86_64` or `cu121` wheel — import will fail with `undefined symbol: cuXXX_v2` or `Illegal instruction`. In those cases look at:
- PyTorch index: `https://download.pytorch.org/whl/cu130`
- direct vLLM releases: they publish `+cu130` variants.

### Note 2.3. Python 3.12
The venv uses **Python 3.12.3**. Not every cu130 wheel has `cp313` builds — check before upgrading Python.

## 3. vLLM

Installed: **`vllm==0.19.0+cu130`**.

### Note 3.1. `LD_LIBRARY_PATH` is required
Before running Python with `import torch` or `import vllm` you must prepend these paths (in order):

```
/usr/lib/aarch64-linux-gnu/libcusparseLt/13
/usr/lib/aarch64-linux-gnu/nvshmem/13
/usr/local/cuda-13.0/targets/sbsa-linux/lib
/usr/lib/aarch64-linux-gnu
```

Without them: **silent import failures** (`OSError: libcusparseLt.so.0: cannot open shared object file` or similar for nvshmem). `run_webapp.py` and `webapp/core.py:start_server` already do this — don't remove it.

### Note 3.2. Supporting libs
- **FlashInfer** `0.6.6` — installed together with `flashinfer-cubin` (precompiled cubins). Runs on sm_121 via PTX.
- **Triton** `3.6.0` — JIT compilation of custom kernels; cache in `~/.triton/`.
- **xformers** — not installed; vLLM on GB10 uses the FlashInfer backend by default.

### Note 3.3. Only one server at a time
Unified memory + a single GPU → two models cannot be held simultaneously. `core.start_server` enforces this (see the `if running: raise RuntimeError` check). `vllm_models.py stop` does SIGTERM + SIGKILL fallback + cleanup of `VLLM::EngineCore` workers — do not touch this logic.

### Note 3.4. Useful ENV vars
| Variable | When to set |
|---|---|
| `VLLM_USE_V1=1` | enabled by default in 0.19, no need to set manually |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | only when hunting ghost processes |
| `TORCH_CUDA_ARCH_LIST=12.1` | when compiling custom ops — otherwise PTX fallback |
| `HF_HUB_ENABLE_HF_TRANSFER=1` | faster model downloads from HF (requires `hf_transfer` package) |
| `CUDA_VISIBLE_DEVICES=0` | belt-and-braces, even with a single GPU |

## 4. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `libcusparseLt.so.0: cannot open` | `LD_LIBRARY_PATH` missing `/usr/lib/aarch64-linux-gnu/libcusparseLt/13` | Run via `run_webapp.py` or the CLI; don't start Python directly |
| `Illegal instruction` on torch import | x86_64 wheel instead of aarch64 | `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu130` |
| `undefined symbol: cuTensorMapEncodeTiled` | wheel built against cu121/cu124 while runtime is 13.0 | Reinstall with the `+cu130` tag |
| `CUDA out of memory` for a small model | memory held by another process (unified!) | `free -h`, kill orphan `VLLM::EngineCore` (`vllm_models.py stop`) |
| `UserWarning ... cuda capability 12.1 ... (8.0) - (12.0)` | normal for sm_121 on torch 2.10 | Ignore for inference; for custom ops recompile with `TORCH_CUDA_ARCH_LIST=12.1` |
| `nvidia-smi` shows `Memory-Usage: Not Supported` | this is unified memory, not a bug | Use `torch.cuda.mem_get_info` or `free -h` |
| vLLM hangs at startup for 10 min | Triton / FlashInfer cache compilation on first run of a model | Wait; caches live in `~/.triton` and `~/.cache/flashinfer` |

## 5. Useful commands

```bash
# Full GPU picture
nvidia-smi
/home/igogo/gpu-control.sh status
/home/igogo/gpu-control.sh health

# Free GPU memory (this way, not via nvidia-smi)
/home/igogo/vllm-tools/.venv/bin/python -c \
  "import torch; f,t=torch.cuda.mem_get_info(0); print(f'{f/(1024**3):.1f} / {t/(1024**3):.1f} GB')"

# Verify that torch sees the GPU with the correct capability
/home/igogo/vllm-tools/.venv/bin/python -c \
  "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"

# Clean up all vLLM processes (including orphan engine core workers)
/home/igogo/vllm-tools/vllm_models.py stop
```
