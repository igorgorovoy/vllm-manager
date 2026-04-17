# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

vLLM management tooling for a **single-host deployment** on an NVIDIA DGX Spark (GB10 Grace-Blackwell, aarch64, CUDA 13.0). Two front ends share one backend:

- `vllm_models.py` — standalone CLI for model profiles, HF downloads, and single-server lifecycle (writes a legacy `server.pid`).
- `webapp/` — FastAPI + SSE + Alpine.js UI (`/` static + `/api/*`) that supersedes the CLI with multi-server tracking via `servers.json`. `webapp/core.py:_migrate_legacy_pid` absorbs the CLI's old `server.pid` on first read.

The code edits live on macOS but **everything runs on `spark-98a2`**. Paths like `/home/igogo/vllm-tools/.venv` are hardcoded in `vllm_models.py` and `webapp/config.py` — treat this repo as deployed-in-place, not portable.

## Host constraints that shape the code

Read `HARDWARE.md` before touching GPU/memory/launch logic. The non-obvious parts:

- **Unified memory** (CPU+GPU share one 121 GB LPDDR5X pool). `nvidia-smi` reports `Memory-Usage: Not Supported` — this is not a bug. Free GPU memory comes from `torch.cuda.mem_get_info`, spawned as a subprocess in `webapp/core.py:get_gpu_free_memory_gb` because the webapp process itself must not import torch.
- **Only one vLLM server can run at a time.** `core.start_server` enforces this (`list_servers()` then `raise`). Do not relax this — there is one GPU and shared memory.
- **`LD_LIBRARY_PATH` is mandatory** before any `import torch`/`import vllm`. The exact path list lives in `webapp/config.py:CUDA_LD_PATH` and is re-applied in `run_webapp.py` (for the webapp itself) and injected into the subprocess env in `core.start_server` (for spawned vLLM servers). Do not reorder or strip it — missing `libcusparseLt/13` or `nvshmem/13` causes silent import failures.
- **sm_121 on torch 2.10 (`sm_80–sm_120` officially)** — the PyTorch capability warning at startup is expected. Custom CUDA extensions without PTX fallback may break; pure vLLM inference works.
- **All wheels must be `aarch64 + cu130`.** See `HARDWARE.md §2.2` before adding a dependency.
- **`stop_all` kills both the tracked PID and any orphan `VLLM::EngineCore` workers** (`vllm_models.py:find_vllm_pids`, cmdline markers). Leave this dual-kill alone — engine core children survive a plain SIGTERM of the parent.

## Commands

Install / reset the venv (runs on the Linux host):
```bash
./install_vllm_deps.sh            # creates .venv, installs requirements.txt
```

Model profile CLI (state in `~/.config/vllm-models/models.json`):
```bash
./vllm_models.py list
./vllm_models.py add <name> <hf-repo> --dtype auto --max-model-len 8192 \
                 --tensor-parallel-size 1 --gpu-memory-utilization 0.9
./vllm_models.py pull <name>
./vllm_models.py serve <name> --port 8000
./vllm_models.py status
./vllm_models.py stop                       # SIGTERM + SIGKILL + orphan reap
./vllm_models.py restart <name> --port 8000
```

Webapp (FastAPI on `:7860`, auto-reload):
```bash
python3 run_webapp.py
```

Free-GPU sanity check (the only correct way on unified memory):
```bash
.venv/bin/python -c "import torch; f,t=torch.cuda.mem_get_info(0); print(f'{f/1024**3:.1f}/{t/1024**3:.1f} GB')"
```

No test suite, linter, or formatter is configured in this repo.

## Webapp architecture

Entry: `webapp/main.py` mounts six routers under `/api/*` and serves `static/index.html` at `/`.

- `webapp/core.py` — **all business logic and I/O**. Config persistence (`models.json`, `servers.json`), HF cache inspection, subprocess spawn for `vllm serve`, nvidia-smi parsing, and a minimal prometheus parser for vLLM's `/metrics` endpoint (`get_inference_metrics`). Routers are thin wrappers around these functions.
- `webapp/models.py` — Pydantic DTOs for requests/responses. `ModelProfile` vs `ModelProfileCreate` vs `ModelProfileResponse` differ by whether `name`, `downloaded`, and `size_gb` are present.
- `webapp/routers/`:
  - `profiles.py` — CRUD on model profiles.
  - `downloads.py` — starts an async `hf download` subprocess; stdout/stderr piped into an `asyncio.Queue` and streamed to the browser via SSE (`/api/downloads/{name}/stream`). A single `active_downloads` dict in `core.py` gates concurrent downloads per model.
  - `servers.py` — start/stop/health plus `/startup-stream` which `tail -f`s the log and polls `/health` until the vLLM server is ready.
  - `logs.py` — tail and live-stream a server's log file.
  - `gpu.py` — one-shot and SSE-streamed `SystemStats` (GPU + memory + disk + inference metrics).
  - `chat.py` — proxies to the running vLLM's OpenAI-compatible `/v1/chat/completions`, auto-discovering the model id from `/v1/models` if the client did not pass one. Supports streaming via SSE.
- `webapp/static/` — single-page Alpine.js + Chart.js UI. Chart instances and EventSource handles live on a `_nonReactive` object because Alpine's proxy recursion breaks Chart.js's circular refs.

### External vs tracked servers

`list_servers()` merges two sources:
1. `servers.json` entries (webapp-started, keyed `server-<epoch>`).
2. Any `vllm.entrypoints.openai.api_server` process owned by the current user that isn't already tracked, surfaced as `external-<pid>`.

`stop_server` branches on the `external-` prefix to avoid touching `servers.json` for externally-launched processes. `list_servers` also garbage-collects stopped tracked entries on every call.

## Conventions

- Config state is **not in-repo**; it lives under `~/.config/vllm-models/` and `~/.cache/huggingface/hub/`. Do not add fixtures or sample configs that would shadow it.
- Do not add a second concurrent vLLM server path. One GPU, one server — enforced in `core.start_server`.
- When adding new subprocess spawns that will import torch/vllm, always merge `CUDA_LD_PATH` into the child env (see `core.start_server` for the pattern).
