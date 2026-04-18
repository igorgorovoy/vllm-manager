import asyncio
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import psutil

from webapp.config import (
    CONFIG_DIR,
    CONFIG_FILE,
    CUDA_LD_PATH,
    HF_CACHE,
    SERVERS_FILE,
    VENV_HFCLI,
    VENV_PY,
)
from webapp.models import (
    DiskStats,
    GpuStats,
    InferenceMetrics,
    MemoryStats,
    ModelProfile,
    ModelProfileResponse,
    ServerInfo,
    SystemStats,
)


# --- Config ---

def ensure_config():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(json.dumps({"models": {}}, indent=2) + "\n")


def load_config() -> dict:
    ensure_config()
    return json.loads(CONFIG_FILE.read_text())


def save_config(cfg: dict):
    ensure_config()
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2) + "\n")


def ensure_servers_config():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not SERVERS_FILE.exists():
        SERVERS_FILE.write_text(json.dumps({"servers": {}}, indent=2) + "\n")


def load_servers() -> dict:
    ensure_servers_config()
    return json.loads(SERVERS_FILE.read_text())


def save_servers(data: dict):
    ensure_servers_config()
    SERVERS_FILE.write_text(json.dumps(data, indent=2) + "\n")


# --- Model Profiles ---

def _hf_cache_dir(repo: str) -> Path:
    return HF_CACHE / f"models--{repo.replace('/', '--')}"


def is_downloaded(repo: str) -> bool:
    d = _hf_cache_dir(repo)
    if not d.exists():
        return False
    incomplete = list(d.rglob("*.incomplete"))
    return len(incomplete) == 0


def get_model_size_gb(repo: str) -> float | None:
    d = _hf_cache_dir(repo)
    if not d.exists():
        return None
    try:
        result = subprocess.run(
            ["du", "-sb", str(d)], capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            size_bytes = int(result.stdout.split()[0])
            return round(size_bytes / (1024**3), 1)
    except Exception:
        pass
    return None


def list_models() -> list[ModelProfileResponse]:
    cfg = load_config()
    models = []
    for name, data in cfg.get("models", {}).items():
        repo = data["repo"]
        models.append(ModelProfileResponse(
            name=name,
            repo=repo,
            dtype=data["dtype"],
            max_model_len=data["max_model_len"],
            tensor_parallel_size=data["tensor_parallel_size"],
            gpu_memory_utilization=data["gpu_memory_utilization"],
            downloaded=is_downloaded(repo),
            size_gb=get_model_size_gb(repo),
        ))
    return models


def add_model(name: str, repo: str, dtype: str, max_model_len: int,
              tensor_parallel_size: int, gpu_memory_utilization: float):
    cfg = load_config()
    cfg.setdefault("models", {})[name] = {
        "repo": repo,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    save_config(cfg)


def remove_model(name: str):
    cfg = load_config()
    if name not in cfg.get("models", {}):
        raise KeyError(f"Model '{name}' not found")
    del cfg["models"][name]
    save_config(cfg)


def get_model(name: str) -> dict:
    cfg = load_config()
    model = cfg.get("models", {}).get(name)
    if not model:
        raise KeyError(f"Model '{name}' not found")
    return model


# --- Downloads ---

active_downloads: dict[str, asyncio.Queue] = {}


async def download_model(name: str, queue: asyncio.Queue):
    model = get_model(name)
    repo = model["repo"]

    cmd = [str(VENV_HFCLI), "download", repo]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def read_stream(stream):
        buffer = ""
        while True:
            chunk = await stream.read(256)
            if not chunk:
                break
            buffer += chunk.decode(errors="replace")
            while "\n" in buffer or "\r" in buffer:
                for sep in ["\n", "\r"]:
                    if sep in buffer:
                        line, buffer = buffer.split(sep, 1)
                        line = line.strip()
                        if line:
                            await queue.put({"event": "progress", "data": line})

    await asyncio.gather(
        read_stream(proc.stdout),
        read_stream(proc.stderr),
    )

    rc = await proc.wait()
    if rc == 0:
        await queue.put({"event": "complete", "data": f"Model '{name}' downloaded"})
    else:
        await queue.put({"event": "error", "data": f"Download failed (exit code {rc})"})
    await queue.put(None)  # sentinel


# --- Server Management ---

def _is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _migrate_legacy_pid():
    """Import legacy single-PID server.pid into servers.json."""
    legacy_pid = CONFIG_DIR / "server.pid"
    legacy_log = CONFIG_DIR / "server.log"
    if not legacy_pid.exists():
        return
    try:
        pid = int(legacy_pid.read_text().strip())
        if _is_pid_running(pid):
            data = load_servers()
            sid = f"server-legacy"
            data["servers"][sid] = {
                "model_name": "unknown",
                "repo": "unknown",
                "port": 8000,
                "pid": pid,
                "log_file": str(legacy_log),
                "started_at": "unknown",
                "gpu_memory_utilization": 0.9,
            }
            save_servers(data)
        legacy_pid.unlink(missing_ok=True)
    except Exception:
        legacy_pid.unlink(missing_ok=True)


def _detect_external_vllm_servers() -> list[dict]:
    """Detect vLLM processes started outside the webapp."""
    external = []
    try:
        for proc in psutil.process_iter(["pid", "cmdline"]):
            cmdline = proc.info.get("cmdline") or []
            cmd_str = " ".join(cmdline)
            if "vllm.entrypoints.openai.api_server" not in cmd_str:
                continue
            # Extract --model and --port from cmdline
            model_name = "unknown"
            port = 8000
            for i, arg in enumerate(cmdline):
                if arg == "--model" and i + 1 < len(cmdline):
                    model_name = cmdline[i + 1]
                if arg == "--port" and i + 1 < len(cmdline):
                    try:
                        port = int(cmdline[i + 1])
                    except ValueError:
                        pass
            external.append({
                "pid": proc.info["pid"],
                "model_name": model_name,
                "port": port,
            })
    except Exception:
        pass
    return external


def list_servers() -> list[ServerInfo]:
    _migrate_legacy_pid()
    data = load_servers()
    servers = []
    tracked_pids = set()

    for sid, info in data.get("servers", {}).items():
        pid = info["pid"]
        running = _is_pid_running(pid)
        tracked_pids.add(pid)
        servers.append(ServerInfo(
            server_id=sid,
            model_name=info["model_name"],
            repo=info.get("repo", "unknown"),
            port=info["port"],
            pid=pid,
            log_file=info["log_file"],
            started_at=info.get("started_at", "unknown"),
            status="running" if running else "stopped",
            gpu_memory_utilization=info.get("gpu_memory_utilization", 0.9),
        ))

    # Detect external vLLM processes not tracked by webapp
    for ext in _detect_external_vllm_servers():
        if ext["pid"] not in tracked_pids:
            servers.append(ServerInfo(
                server_id=f"external-{ext['pid']}",
                model_name=ext["model_name"],
                repo=ext["model_name"],
                port=ext["port"],
                pid=ext["pid"],
                log_file="",
                started_at="external",
                status="running",
                gpu_memory_utilization=0.9,
            ))

    # Clean stopped servers from config
    to_remove = [s.server_id for s in servers if s.status == "stopped"]
    if to_remove:
        for sid in to_remove:
            data["servers"].pop(sid, None)
        save_servers(data)
        servers = [s for s in servers if s.status != "stopped"]
    return servers


def get_gpu_free_memory_gb() -> float | None:
    """Get free GPU memory in GB using torch.cuda."""
    try:
        result = subprocess.run(
            [str(VENV_PY), "-c",
             "import torch; f,t=torch.cuda.mem_get_info(0); print(f'{f/(1024**3):.1f}')"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def start_server(model_name: str, port: int,
                 gpu_memory_utilization: float | None = None) -> ServerInfo:
    model = get_model(model_name)
    gpu_mem = gpu_memory_utilization or model["gpu_memory_utilization"]

    # Block only if the same model is already running (multi-server is allowed).
    running = list_servers()
    for s in running:
        if s.model_name == model_name:
            raise RuntimeError(
                f"Model '{model_name}' is already running as {s.server_id} "
                f"on port {s.port} (pid={s.pid})"
            )
        if s.port == port:
            raise RuntimeError(
                f"Port {port} already used by {s.server_id} ({s.model_name})"
            )

    # Port conflict (catches non-vLLM listeners too).
    for conn in psutil.net_connections(kind="tcp"):
        if conn.laddr.port == port and conn.status == "LISTEN":
            raise RuntimeError(f"Port {port} is already in use")

    # Check GPU memory — model needs at least size_gb * 1.1 free.
    model_size = get_model_size_gb(model["repo"])
    if model_size:
        free_gb = get_gpu_free_memory_gb()
        if free_gb is not None and model_size > free_gb:
            raise RuntimeError(
                f"Not enough GPU memory: model needs ~{model_size} GB "
                f"but only {free_gb:.0f} GB free"
            )

    sid = f"server-{int(time.time())}"
    log_path = CONFIG_DIR / f"{sid}.log"

    cmd = [
        str(VENV_PY), "-m", "vllm.entrypoints.openai.api_server",
        "--model", model["repo"],
        "--dtype", str(model["dtype"]),
        "--port", str(port),
        "--max-model-len", str(model["max_model_len"]),
        "--tensor-parallel-size", str(model["tensor_parallel_size"]),
        "--gpu-memory-utilization", str(gpu_mem),
    ]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = CUDA_LD_PATH + ":" + env.get("LD_LIBRARY_PATH", "")

    log_file = log_path.open("a")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, env=env)

    info = {
        "model_name": model_name,
        "repo": model["repo"],
        "port": port,
        "pid": proc.pid,
        "log_file": str(log_path),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_memory_utilization": gpu_mem,
    }

    data = load_servers()
    data["servers"][sid] = info
    save_servers(data)

    return ServerInfo(server_id=sid, status="starting", **info)


def stop_server(server_id: str) -> bool:
    # Handle external servers (not in config)
    if server_id.startswith("external-"):
        try:
            pid = int(server_id.split("-", 1)[1])
            if _is_pid_running(pid):
                os.kill(pid, signal.SIGTERM)
            return True
        except (ValueError, OSError):
            raise KeyError(f"Server '{server_id}' not found")

    data = load_servers()
    info = data.get("servers", {}).get(server_id)
    if not info:
        raise KeyError(f"Server '{server_id}' not found")

    pid = info["pid"]
    if _is_pid_running(pid):
        os.kill(pid, signal.SIGTERM)

    data["servers"].pop(server_id, None)
    save_servers(data)
    return True


# --- GPU Stats ---

def get_gpu_stats() -> GpuStats:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,"
             "power.draw,pstate,clocks.current.graphics",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return GpuStats(
                name=parts[0],
                temperature_c=_safe_int(parts[1]),
                utilization_gpu_pct=_safe_int(parts[2]),
                power_draw_w=_safe_float(parts[3]),
                performance_state=parts[4] if parts[4] != "[N/A]" else None,
                clock_speed_mhz=_safe_int(parts[5]),
            )
    except Exception:
        pass
    return GpuStats(name="unknown")


def get_memory_stats() -> MemoryStats:
    mem = psutil.virtual_memory()
    return MemoryStats(
        total_gb=round(mem.total / (1024**3), 1),
        used_gb=round(mem.used / (1024**3), 1),
        available_gb=round(mem.available / (1024**3), 1),
        percent=mem.percent,
    )


def get_disk_stats() -> DiskStats:
    disk = psutil.disk_usage("/")
    hf_cache_gb = 0.0
    if HF_CACHE.exists():
        try:
            result = subprocess.run(
                ["du", "-sb", str(HF_CACHE)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                hf_cache_gb = round(int(result.stdout.split()[0]) / (1024**3), 1)
        except Exception:
            pass
    return DiskStats(
        total_gb=round(disk.total / (1024**3), 1),
        used_gb=round(disk.used / (1024**3), 1),
        free_gb=round(disk.free / (1024**3), 1),
        hf_cache_gb=hf_cache_gb,
    )


_prev_gen_tokens: dict[int, tuple[float, float]] = {}  # port -> (timestamp, total)


def get_inference_metrics(port: int) -> InferenceMetrics | None:
    """Fetch and parse vLLM /metrics endpoint."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=2) as resp:
            text = resp.read().decode()
    except Exception:
        return None

    def _parse_metric(name: str, text: str, aggregate: bool = False) -> float:
        """Extract the value of a prometheus metric. If aggregate=True, sum all label variants."""
        val = 0.0
        for line in text.split("\n"):
            if line.startswith(name + "{") or line.startswith(name + " "):
                parts = line.rsplit(" ", 1)
                if len(parts) == 2:
                    try:
                        v = float(parts[1])
                        if aggregate:
                            val += v
                        else:
                            val = v
                    except ValueError:
                        pass
        return val

    def _parse_histogram_sum_count(name: str, text: str) -> tuple[float, float]:
        s = _parse_metric(name + "_sum", text)
        c = _parse_metric(name + "_count", text)
        return s, c

    gen_tokens = _parse_metric("vllm:generation_tokens_total", text)
    prompt_tokens = _parse_metric("vllm:prompt_tokens_total", text)
    requests_running = int(_parse_metric("vllm:num_requests_running", text))
    requests_waiting = int(_parse_metric("vllm:num_requests_waiting", text))
    kv_cache = _parse_metric("vllm:kv_cache_usage_perc", text) * 100
    req_success = _parse_metric("vllm:request_success_total", text, aggregate=True)

    # TTFT
    ttft_sum, ttft_count = _parse_histogram_sum_count(
        "vllm:time_to_first_token_seconds", text)
    avg_ttft_ms = (ttft_sum / ttft_count * 1000) if ttft_count > 0 else 0

    # Time per output token
    tpot_sum, tpot_count = _parse_histogram_sum_count(
        "vllm:request_time_per_output_token_seconds", text)
    avg_tpot_ms = (tpot_sum / tpot_count * 1000) if tpot_count > 0 else 0

    # Prefix cache hit rate
    cache_hits = _parse_metric("vllm:prefix_cache_hits_total", text)
    cache_queries = _parse_metric("vllm:prefix_cache_queries_total", text)
    cache_hit_pct = (cache_hits / cache_queries * 100) if cache_queries > 0 else 0

    # Tokens/sec (delta based)
    now = time.time()
    prev = _prev_gen_tokens.get(port)
    tokens_per_sec = 0.0
    if prev:
        dt = now - prev[0]
        if dt > 0:
            tokens_per_sec = (gen_tokens - prev[1]) / dt
    _prev_gen_tokens[port] = (now, gen_tokens)

    return InferenceMetrics(
        tokens_per_sec=round(tokens_per_sec, 1),
        total_generation_tokens=int(gen_tokens),
        total_prompt_tokens=int(prompt_tokens),
        total_requests=int(req_success),
        requests_running=requests_running,
        requests_waiting=requests_waiting,
        kv_cache_usage_pct=round(kv_cache, 1),
        avg_ttft_ms=round(avg_ttft_ms, 1),
        avg_tpot_ms=round(avg_tpot_ms, 1),
        cache_hit_rate_pct=round(cache_hit_pct, 1),
    )


def get_system_stats() -> SystemStats:
    # Try to get inference metrics from any running server
    inference = None
    servers = list_servers()
    for srv in servers:
        if srv.status == "running":
            inference = get_inference_metrics(srv.port)
            if inference:
                break

    return SystemStats(
        gpu=get_gpu_stats(),
        memory=get_memory_stats(),
        disk=get_disk_stats(),
        inference=inference,
    )


def _safe_int(val: str) -> int | None:
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _safe_float(val: str) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
