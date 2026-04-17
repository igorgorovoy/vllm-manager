# vLLM model management scripts

## Files

- `vllm_models.py` - CLI for model profiles, downloads, and server lifecycle.

## Quick start

```bash
cd /home/igogo/vllm-tools
chmod +x vllm_models.py
./vllm_models.py list
```

## Add model profiles

```bash
./vllm_models.py add qwen-coder Qwen/Qwen2.5-Coder-32B-Instruct --dtype auto --max-model-len 8192 --tensor-parallel-size 1 --gpu-memory-utilization 0.9
./vllm_models.py add gemma4 google/gemma-3-27b-it --dtype bfloat16 --max-model-len 8192 --tensor-parallel-size 1 --gpu-memory-utilization 0.9
```

## Download configured model from Hugging Face

```bash
./vllm_models.py pull qwen-coder
```

## Serve / stop

```bash
./vllm_models.py serve qwen-coder --port 8000
./vllm_models.py status
./vllm_models.py stop
```

## Notes

- Config is stored in `~/.config/vllm-models/models.json`.
- Server PID/logs are stored in `~/.config/vllm-models/`.
- Requires `python3`, `vllm`, and `huggingface-cli`.
