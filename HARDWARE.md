# Hardware, CUDA, PyTorch — platform specifics

This file describes the specifics of the machine (`spark-98a2`) — an **NVIDIA DGX Spark** based on **GB10 Grace-Blackwell**. Every non-standard decision in `vllm-tools` (`LD_LIBRARY_PATH` paths, `cu130` wheels, memory checks) follows directly from these specifics.

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
`nvidia-smi` shows `Memory-Usage: Not Supported` — this is **not an error**. GB10 has no discrete VRAM: CPU and GPU share the same LPDDR5X pool. To check available GPU memory, use:

```python
import torch
free, total = torch.cuda.mem_get_info(0)   # bytes
```

That is exactly what `webapp/core.py:get_gpu_free_memory_gb` does. `free -h` also shows the whole pool — don't assume it's "system" RAM separate from the GPU. Any large model in vLLM eats the same memory the Linux kernel sees. That's why `start_server` in `core.py:316` refuses to launch when `model_size > free_gb`.

### Note 1.2. sbsa, not tegra
The CUDA target is `sbsa-linux` (*Server Base System Architecture*), not `tegra-linux`. The correct path is:

```
/usr/local/cuda-13.0/targets/sbsa-linux/lib
```

Jetson tutorials pointing at `aarch64-linux-gnu/tegra/...` **do not apply** — that's a different ABI.

### Note 1.3. This is not a typical ARM server
GB10 is an SoC with a tight CPU↔GPU NVLink-C2C bond. Consequences:
- `host→device` copies are essentially free (same pool).
- `pin_memory=True` in DataLoader doesn't speed anything up — memory is already in the "pinned" pool.
- `tensor_parallel_size > 1` is pointless: there is only one GPU.

## 2. PyTorch

Installed: **`torch==2.10.0+cu130`**, linked against CUDA 13.0.

### Note 2.1. sm_121 vs PyTorch's official matrix
PyTorch 2.10 officially supports `sm_80 … sm_120`. GB10 = `sm_121` → on `import torch` you'll see:

```
UserWarning: Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0).
```

**This is not an error** — kernels are launched via PTX JIT or fall back to sm_120. Works stably for inference (vLLM verified). However:
- **Don't ignore** this warning for custom CUDA extensions — they may not have a PTX fallback.
- If a torch build with official `sm_121` appears — upgrade; performance may rise noticeably.
- `TORCH_CUDA_ARCH_LIST="12.1"` when compiling extensions — only if the toolchain understands it (check `nvcc --list-gpu-arch`).

### Note 2.2. `+cu130` wheels — mandatory
Every binary package with CUDA extensions (torch, torchaudio, torchvision, flashinfer, xformers, bitsandbytes, apex) **must** be:
1. an aarch64 build,
2. built against **cu130** (not cu121/cu124).

PyPI does not carry everything immediately. If `pip install` pulls an `x86_64` or `cu121` wheel — import will fail with `undefined symbol: cuXXX_v2` or `Illegal instruction`. In those cases look at:
- the PyTorch index: `https://download.pytorch.org/whl/cu130`
- direct vLLM releases: they publish `+cu130` variants.

### Note 2.3. Python 3.12
The venv runs on **Python 3.12.3**. Not every cu130 wheel has `cp313` builds yet — double-check before upgrading Python.

## 3. vLLM

Installed: **`vllm==0.19.0+cu130`**.

### Note 3.1. `LD_LIBRARY_PATH` is mandatory
Before starting Python with `import torch` or `import vllm`, these paths must be present (in this order):

```
/usr/lib/aarch64-linux-gnu/libcusparseLt/13
/usr/lib/aarch64-linux-gnu/nvshmem/13
/usr/local/cuda-13.0/targets/sbsa-linux/lib
/usr/lib/aarch64-linux-gnu
```

Without them: **silent import failures** (`OSError: libcusparseLt.so.0: cannot open shared object file` or similar from nvshmem). `run_webapp.py` and `webapp/core.py:start_server` already handle this — don't strip it out.

### Note 3.2. Supporting libs
- **FlashInfer** `0.6.6` — installed together with `flashinfer-cubin` (precompiled cubins). Works on sm_121 via PTX.
- **Triton** `3.6.0` — JIT-compiles custom kernels; cache at `~/.triton/`.
- **xformers** — not installed; vLLM on GB10 defaults to the FlashInfer backend.

### Note 3.3. Multiple servers allowed (current state)
Originally the webapp/CLI enforced a single vLLM server at a time. That constraint has been **removed** — both frontends now allow multiple concurrent servers on different ports (see the co-hosting section in the README). Guards that remain: reject if the same `model_name` is already running, reject on port collision, and (webapp only) reject if model weight size would exceed free unified memory. `vllm_models.py stop` still does SIGTERM + SIGKILL fallback + cleanup of orphan `VLLM::EngineCore` workers — don't touch that logic.

### Note 3.4. Useful ENV vars
| Variable | When to set |
|---|---|
| `VLLM_USE_V1=1` | Enabled by default in 0.19; no need to set manually |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | Only when fighting ghost processes |
| `TORCH_CUDA_ARCH_LIST=12.1` | When compiling custom ops — otherwise PTX fallback |
| `HF_HUB_ENABLE_HF_TRANSFER=1` | Faster model downloads from HF (requires `hf_transfer`) |
| `CUDA_VISIBLE_DEVICES=0` | Insurance, even with a single GPU |

## 4. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `libcusparseLt.so.0: cannot open` | `LD_LIBRARY_PATH` lacks `/usr/lib/aarch64-linux-gnu/libcusparseLt/13` | Launch via `run_webapp.py` or the CLI; don't run Python directly |
| `Illegal instruction` on `import torch` | x86_64 wheel instead of aarch64 | `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu130` |
| `undefined symbol: cuTensorMapEncodeTiled` | Wheel built for cu121/cu124 but runtime is 13.0 | Reinstall with the `+cu130` tag |
| `CUDA out of memory` with a small model | Memory held by another process (unified!) | `free -h`; kill orphan `VLLM::EngineCore` (`vllm_models.py stop`) |
| `UserWarning ... cuda capability 12.1 ... (8.0) - (12.0)` | Normal for sm_121 on torch 2.10 | Ignore for inference; for custom ops, recompile with `TORCH_CUDA_ARCH_LIST=12.1` |
| `nvidia-smi` shows `Memory-Usage: Not Supported` | Unified memory, not a bug | Use `torch.cuda.mem_get_info` or `free -h` |
| vLLM hangs for 10 min on startup | Triton / FlashInfer cache compilation on first model load | Wait; caches live at `~/.triton` and `~/.cache/flashinfer` |

## 5. Handy commands

```bash
# Full GPU picture
nvidia-smi
./scripts/gpu-control.sh status
./scripts/gpu-control.sh health

# Free GPU memory (use this, not nvidia-smi)
./.venv/bin/python -c \
  "import torch; f,t=torch.cuda.mem_get_info(0); print(f'{f/(1024**3):.1f} / {t/(1024**3):.1f} GB')"

# Verify torch sees the GPU with the right capability
./.venv/bin/python -c \
  "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"

# Clean up every vLLM process (including orphan engine-core workers)
./vllm_models.py stop
```
