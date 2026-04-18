# File Classification & Benchmarking Architecture

_Session notes — 2026-04-17 → 2026-04-18_

## Context

Working on two related tracks on the DGX Spark (GB10, 128GB unified memory):

1. Establish reliable performance metrics for installed vLLM models.
2. Design an automatic file classifier to sort a large personal archive into a hierarchical taxonomy from `jd.yaml` (~400 leaf classes, EN/UK/RU, files from short titles to multi-KB PDF/text).

---

## Part 1 — Model benchmarking

### Starting point

`webapp/core.py:get_inference_metrics` reads `/metrics` from a running vLLM in real time but does **not persist history**. There was no historical performance data across models; comparing them meant running ad-hoc tests with no methodology.

### Methodology decisions

Chose `vllm bench serve` over a hand-rolled script:
- Standard tool used by the vLLM team — reproducible numbers.
- Handles warmup, steady-state measurement, percentile latency correctly.
- JSON output, integrates cleanly with a driver script.

Workload: `--dataset-name random`, 512 input / 256 output tokens, `--num-prompts 100`, `--request-rate inf`. Rationale: fixed synthetic token counts make cross-model throughput directly comparable (ShareGPT would give a more realistic length distribution, but different tokenizers produce different token counts from the same text, muddying throughput comparisons).

Quality improvements layered in three iterations:

| Version | Change | Why |
|---|---|---|
| v1 | First pass | Establish baseline — revealed suspicious Qwen3.6 vs Qwen3.5 gap |
| v2 | `NUM_WARMUPS=5`; aligned Qwen3.6 profile (`dtype=bfloat16`, `max_model_len=32768` like Qwen3.5) | Removed JIT/compile cold-start + dtype asymmetry as confounds |
| v3 | `NUM_REPEATS=3` + median aggregation | Replaced single-sample numbers with something you can actually trust |

### Driver script

`/home/igogo/vllm-tools/bench_models.sh`:
- Iterates models from `~/.config/vllm-models/models.json`.
- For each: starts server via `vllm_models.py serve`, waits for `/health`, runs `vllm bench serve` N times inside one server session (steady-state, no reload between repeats), computes median across runs via `jq`, appends one row to `~/.config/vllm-models/benchmarks/summary.csv`, stops server.
- Keeps per-run JSONs for audit (`BENCH_KEEP=20` retention).
- Flag `--if-idle`: exits 0 if any vLLM process is already running — used by the scheduled timer so it never kills an active server.
- Env overrides: `NUM_PROMPTS`, `INPUT_LEN`, `OUTPUT_LEN`, `REQUEST_RATE`, `NUM_WARMUPS`, `NUM_REPEATS`, `ONLY=a,b,c`, `BENCH_KEEP`.

### Scheduled regular runs

Weekly systemd user timer:
- `~/.config/systemd/user/vllm-bench.service` — oneshot, calls `bench_models.sh --if-idle`, 3h timeout.
- `~/.config/systemd/user/vllm-bench.timer` — `OnCalendar=Sun 04:00`, `RandomizedDelaySec=30m`, `Persistent=true`.
- Verified both sides of the idle guard work: runs proceed when GPU is idle, skip cleanly (exit 0) when a vLLM process is detected.
- Monitor: `journalctl --user -u vllm-bench.service`, `systemctl --user list-timers vllm-bench.timer`.

### Latest results (v3, median of 3, 2026-04-18)

| Model | output tok/s | total tok/s | TTFT median | TTFT p99 | TPOT median |
|---|---:|---:|---:|---:|---:|
| **glm47-flash** | **337.2** | **1011.5** | **1.98 s** | 2.30 s | **289 ms** |
| qwen35-coder-35b | 273.4 | 820.3 | 8.04 s | 17.22 s | 325 ms |
| Qwen3.6-35B-A3B | 270.9 | 812.7 | 8.18 s | 16.96 s | 331 ms |
| gemma4-31b-it | 81.5 | 244.5 | 22.38 s | 33.26 s | 910 ms |

Takeaways:
- **GLM-4.7-Flash** is the throughput winner by a meaningful margin, and its TTFT collapsed once warmup was added and steady-state reached — the earlier 7s TTFT was startup artefact.
- **Qwen3.5-coder-35B and Qwen3.6-35B-A3B are essentially tied** (within 1%). The v1 gap was entirely from `dtype=auto` + `max_model_len=8192` + no warmup on Qwen3.6.
- **Gemma-4-31B** is a dense model competing with MoE 35B/A3B models (which activate only 3B params per token). The ~3× gap is architectural, not a config issue — expected behaviour.

Variance across 3 repeats is ~1% on throughput, confirming the bench is reproducible.

---

## Part 2 — File classification architecture

### Problem

Taxonomy lives in `/Users/gorovoy/__SYNC__/projects/personal/agentic-ai-landing-zone/jd.yaml` on the user's Mac. It's:
- **Hierarchical**: top-level categories (10 Finance, 20 Work, 46 DevOps, 50 AI, 55 Woodcarving, …), each with 2-3 levels of sub-categories.
- **~400 leaf classes** total.
- **Multilingual content**: EN (tech), UK and RU (personal docs, receipts, personal projects).
- **Mixed granularity**: abstract ("Observability") alongside very specific ("Karpenter Setup", "Sharpening").
- **Has known duplicates**: Elasticsearch appears under both Databases/Search Engines and DevOps; ClickHouse similarly.
- **No labeled training data.**

Goal: sort existing files on disk into `XX.YY.ZZ/` folders matching leaf classes.

### Why fine-tuning encoders is wrong here

- 400 classes × ~50 labeled examples each = 20K manual annotations required upfront.
- Class overlap would confuse any classifier; current `jd.yaml` has literal duplicates.
- Adding a new class would require re-training.

### Chosen architecture: embedding retrieval + LLM rerank

**Stage 1 — semantic retrieval**:
1. Flatten `jd.yaml` into label strings with hierarchical context (e.g. `"10.06.03 Finance / Expenses / Food — grocery receipts, restaurant bills"`).
2. Embed all ~400 labels once → store in local vector index (FAISS in-process; Qdrant only if this becomes a long-lived service).
3. For each file: take `filename + first 4KB content`, embed, get top-5 similar classes by cosine.

**Stage 2 — LLM disambiguation** (only when needed):
- If `top1_score - top2_score > 0.15` → accept top-1 directly (cheap path, most files).
- Otherwise → small LLM gets the file excerpt + top-5 candidates + descriptions, picks one with confidence. Expected to fire on ~20-30% of files.

### Model choices

**Embedder** (the critical decision — must handle EN + UK + RU + long documents):

- **jina-embeddings-v3** (570M, 8K context, 100+ languages) — chosen default. Task-specific prompting, handles multilingual well, long enough context to avoid chunking for typical files.
- Alternatives considered: BGE-M3 (also 8K, dense+sparse+multivec), mE5-large (faster but only 512 ctx — worse for longer docs).
- **Not served via vLLM.** Use Hugging Face **Text Embeddings Inference (TEI)** — purpose-built for embeddings, 2-3× more efficient than routing through vLLM's generative pipeline.

**Reranker LLM**:
- **Gemma-3-4B-it** — best UK/RU support in the 1-5B range.
- Alternatives: Qwen3-4B (faster, slightly weaker on UK/RU), Qwen3-1.7B (if rerank path is rarely hit and speed matters).

### Known issues to fix in `jd.yaml` before deployment

- Resolve duplicate classes (Elasticsearch in two trees, ClickHouse in two trees, IAM in two trees).
- Either remove `47 Observability` (empty subtree) or merge with `46.13 Monitoring`.
- Consider classifying only to level-2 initially (e.g. `10.06 Expenses`, not `10.06.03 Food`) — long-tail leaves will rarely match confidently; deepen gradually.

### Expected throughput on GB10

- Embedder (TEI, batch): ~1000-2000 texts/sec.
- kNN across 400 vectors: effectively instant.
- LLM rerank: ~40 files/sec when it fires.
- Weighted average (~70% files handled by embedding alone): **~300-800 files/sec**.

---

## Part 3 — Running 3 models concurrently on GB10

Co-hosting coder + reranker + embedder in unified memory:

| Service | Port | Engine | `gpu_memory_utilization` | Approx footprint |
|---|---:|---|---:|---:|
| coder (Qwen3.5-35B-A3B) | 8000 | vLLM | 0.70 | ~85 GB |
| reranker (Gemma-3-4B) | 8001 | vLLM | 0.15 | ~18 GB |
| embedder (jina-v3) | 8002 | **TEI (Docker)** | — | ~5 GB |
| **Σ used** | | | | **~108 GB / 121 GB** |
| OS, webapp, buffers | | | | ~13 GB |

Fits with tight but workable headroom.

### Tradeoffs to keep in mind

- **Compute bottleneck, not memory.** One Blackwell chip → SM contention when coder is in a hot batch and classifier is running. Coding latency rises (e.g. 300 ms → 400 ms TPOT) during bulk classification.
- **Memory bandwidth shared.** Both vLLM engines compete for LPDDR5X bandwidth; embedder is cheap, reranker is the competitor.
- **Recommended operating mode**: run large file-classification imports as a **batch job overnight**. Keep embedder + reranker always on (near-zero idle cost); only run coder when actively coding. No meaningful impact on either workload this way.

### Current CLI limitation

`vllm_models.py` tracks a single server via one PID file — can't start 2+ through the CLI. Two paths forward:
- Use the **webapp** (already supports multi-server via `servers.json` + `webapp/core.py:start_server`).
- Or launch the second vLLM engine directly with `python -m vllm.entrypoints.openai.api_server …` and bypass the CLI's single-server bookkeeping.

Bringing `vllm_models.py` up to parity with the webapp (multi-server tracking) is a reasonable follow-up if CLI-first deployment matters.

---

## Small-models recommendation for classification (general)

Tabulated for reference; specifics above apply to this taxonomy.

### Encoders (fastest + most accurate when you have labels)

| Model | Params | When |
|---|---:|---|
| ModernBERT-large | 395M | English, 8K context, current SOTA encoder |
| DeBERTa-v3-large | 435M | English, classic leader on GLUE/MNLI |
| **mDeBERTa-v3-base** | 278M | **Multilingual incl. UK — best speed/quality tradeoff** |
| XLM-RoBERTa-large | 560M | Multilingual, slightly higher accuracy than mDeBERTa |
| DistilBERT / TinyBERT | 66M / 14M | CPU/edge |

### Small generative LLMs (zero/few-shot classification)

| Model | Params | Notes |
|---|---:|---|
| **Gemma-3-1B-it / 4B-it** | 1B / 4B | Strong UK/RU, current default for multilingual small-LLM work |
| Qwen3-1.7B / 3-4B | 1.7B / 4B | Slightly faster than Gemma-3, weaker on UK/RU |
| Phi-4-mini | 3.8B | Strong reasoning, weaker multilingual |
| SmolLM2-1.7B | 1.7B | Fastest in class, mostly English |
| Llama-3.2-3B | 3B | Stable but weaker on classification than Qwen3 |
