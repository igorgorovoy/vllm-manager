#!/usr/bin/env python3
"""Manage vLLM model profiles and run multiple concurrent servers.

State lives in ~/.config/vllm-models/:
  - models.json  profile definitions
  - servers.json registry of running servers (shared with webapp)
  - server-<ts>.log per-server stdout/stderr
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


CONFIG_DIR = Path.home() / ".config" / "vllm-models"
CONFIG_FILE = CONFIG_DIR / "models.json"
SERVERS_FILE = CONFIG_DIR / "servers.json"
LEGACY_PID_FILE = CONFIG_DIR / "server.pid"
LEGACY_LOG_FILE = CONFIG_DIR / "server.log"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
TOOLS_DIR = Path("/home/igogo/vllm-tools")
VENV_PY = TOOLS_DIR / ".venv" / "bin" / "python"
VENV_HFCLI = TOOLS_DIR / ".venv" / "bin" / "hf"

# CUDA library paths required by vllm subprocess on GB10 aarch64 (must match webapp/config.py).
CUDA_LD_PATH = ":".join([
    "/usr/lib/aarch64-linux-gnu/libcusparseLt/13",
    "/usr/lib/aarch64-linux-gnu/nvshmem/13",
    "/usr/local/cuda-13.0/targets/sbsa-linux/lib",
    "/usr/lib/aarch64-linux-gnu",
])

VLLM_CMDLINE_MARKERS = ("vllm.entrypoints.openai.api_server", "VLLM::EngineCore")


# --- Models config ---

def ensure_config():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(json.dumps({"models": {}}, indent=2) + "\n", encoding="utf-8")


def load_config():
    ensure_config()
    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg):
    ensure_config()
    with CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def get_model(name):
    cfg = load_config()
    model = cfg.get("models", {}).get(name)
    if not model:
        print(f"Model '{name}' is not configured", file=sys.stderr)
        sys.exit(1)
    return model


# --- Servers config (shared on-disk state with webapp) ---

def ensure_servers_config():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not SERVERS_FILE.exists():
        SERVERS_FILE.write_text(json.dumps({"servers": {}}, indent=2) + "\n", encoding="utf-8")


def load_servers():
    ensure_servers_config()
    with SERVERS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_servers(data):
    ensure_servers_config()
    with SERVERS_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def migrate_legacy_pid():
    """Import the old single-server PID file into servers.json on first run."""
    if not LEGACY_PID_FILE.exists():
        return
    try:
        pid = int(LEGACY_PID_FILE.read_text(encoding="utf-8").strip())
        if is_pid_running(pid):
            data = load_servers()
            data["servers"]["server-legacy"] = {
                "model_name": "unknown",
                "repo": "unknown",
                "port": 8000,
                "pid": pid,
                "log_file": str(LEGACY_LOG_FILE),
                "started_at": "legacy",
                "gpu_memory_utilization": 0.9,
            }
            save_servers(data)
    except (ValueError, OSError):
        pass
    LEGACY_PID_FILE.unlink(missing_ok=True)


# --- Process helpers ---

def is_pid_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def find_vllm_procs():
    """Return [(pid, cmdline), ...] for every vllm-related process owned by current user."""
    procs = []
    try:
        my_uid = os.getuid()
    except AttributeError:
        return procs
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            if os.stat(f"/proc/{pid}").st_uid != my_uid:
                continue
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").decode(errors="ignore").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        if any(m in cmdline for m in VLLM_CMDLINE_MARKERS):
            procs.append((pid, cmdline))
    return procs


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        try:
            return s.connect_ex(("127.0.0.1", port)) == 0
        except OSError:
            return False


def kill_pid(pid, label=""):
    if not is_pid_running(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    print(f"SIGTERM {label}pid={pid}")
    for _ in range(60):
        if not is_pid_running(pid):
            return
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGKILL)
        print(f"SIGKILL {label}pid={pid} (did not exit in 30s)")
    except ProcessLookupError:
        pass


# --- Server state ---

def list_running():
    """Return tracked servers (from servers.json, live PIDs) + externally-started api_server procs."""
    migrate_legacy_pid()
    data = load_servers()
    tracked = data.get("servers", {})
    result = []
    tracked_pids = set()
    to_remove = []

    for sid, info in tracked.items():
        pid = info["pid"]
        if is_pid_running(pid):
            tracked_pids.add(pid)
            result.append({
                "server_id": sid,
                "model_name": info.get("model_name", "unknown"),
                "repo": info.get("repo", "unknown"),
                "port": info["port"],
                "pid": pid,
                "log_file": info.get("log_file", ""),
                "started_at": info.get("started_at", "unknown"),
                "source": "tracked",
            })
        else:
            to_remove.append(sid)

    if to_remove:
        for sid in to_remove:
            tracked.pop(sid, None)
        save_servers(data)

    # External api_server processes (skip EngineCore children — they'll be grouped under their parent).
    for pid, cmdline in find_vllm_procs():
        if pid in tracked_pids:
            continue
        if "vllm.entrypoints.openai.api_server" not in cmdline:
            continue
        port, model = 8000, "unknown"
        tokens = cmdline.split()
        for i, t in enumerate(tokens):
            if t == "--port" and i + 1 < len(tokens):
                try:
                    port = int(tokens[i + 1])
                except ValueError:
                    pass
            if t == "--model" and i + 1 < len(tokens):
                model = tokens[i + 1]
        result.append({
            "server_id": f"external-{pid}",
            "model_name": model,
            "repo": model,
            "port": port,
            "pid": pid,
            "log_file": "",
            "started_at": "external",
            "source": "external",
        })

    return result


# --- Commands ---

def command_add(args):
    cfg = load_config()
    cfg["models"][args.name] = {
        "repo": args.repo,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    save_config(cfg)
    print(f"Added model '{args.name}' -> {args.repo}")


def command_remove(args):
    cfg = load_config()
    if args.name not in cfg["models"]:
        print(f"Model '{args.name}' does not exist", file=sys.stderr)
        sys.exit(1)
    del cfg["models"][args.name]
    save_config(cfg)
    print(f"Removed model '{args.name}'")


def command_list(_args):
    cfg = load_config()
    models = cfg.get("models", {})
    if not models:
        print("No models configured")
        return
    for name, data in models.items():
        print(
            f"{name}: repo={data['repo']} dtype={data['dtype']} "
            f"tp={data['tensor_parallel_size']} max_len={data['max_model_len']} "
            f"gpu_mem_util={data['gpu_memory_utilization']}"
        )


def command_pull(args):
    if not VENV_HFCLI.exists():
        print(f"Missing {VENV_HFCLI}. Run /home/igogo/vllm-tools/install_vllm_deps.sh", file=sys.stderr)
        sys.exit(1)
    model = get_model(args.name)
    cmd = [str(VENV_HFCLI), "download", model["repo"]]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Model '{args.name}' downloaded (or already cached) in {HF_CACHE}")


def command_serve(args):
    if not VENV_PY.exists():
        print(f"Missing {VENV_PY}. Run /home/igogo/vllm-tools/install_vllm_deps.sh", file=sys.stderr)
        sys.exit(1)
    model = get_model(args.name)

    if port_in_use(args.port):
        print(f"Port {args.port} is already in use", file=sys.stderr)
        sys.exit(1)

    running = list_running()
    for s in running:
        if s["model_name"] == args.name:
            print(
                f"Model '{args.name}' is already running as {s['server_id']} "
                f"on port {s['port']} (pid={s['pid']})",
                file=sys.stderr,
            )
            sys.exit(1)

    gpu_mem = args.gpu_memory_utilization
    if gpu_mem is None:
        gpu_mem = model["gpu_memory_utilization"]

    sid = f"server-{int(time.time())}"
    log_path = CONFIG_DIR / f"{sid}.log"

    cmd = [
        str(VENV_PY), "-m", "vllm.entrypoints.openai.api_server",
        "--model", model["repo"],
        "--dtype", str(model["dtype"]),
        "--port", str(args.port),
        "--max-model-len", str(model["max_model_len"]),
        "--tensor-parallel-size", str(model["tensor_parallel_size"]),
        "--gpu-memory-utilization", str(gpu_mem),
    ]
    if model.get("trust_remote_code"):
        cmd.append("--trust-remote-code")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = CUDA_LD_PATH + ":" + env.get("LD_LIBRARY_PATH", "")

    ensure_config()
    log_file = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, env=env)

    info = {
        "model_name": args.name,
        "repo": model["repo"],
        "port": args.port,
        "pid": proc.pid,
        "log_file": str(log_path),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_memory_utilization": gpu_mem,
    }

    data = load_servers()
    data["servers"][sid] = info
    save_servers(data)

    print(f"Started vLLM for '{args.name}' on port {args.port} (pid={proc.pid}, id={sid})")
    print(f"Logs: {log_path}")


def command_stop(args):
    servers = list_running()
    target = args.target

    if target is None or target == "all":
        if not servers:
            print("No vLLM processes found")
            return
        stopped = []
        for s in servers:
            kill_pid(s["pid"], label=f"{s['server_id']} ")
            stopped.append(s["pid"])
        data = load_servers()
        data["servers"] = {}
        save_servers(data)
        # Reap orphan EngineCore workers not covered above.
        known = set(stopped)
        for pid, _ in find_vllm_procs():
            if pid in known:
                continue
            kill_pid(pid, label="orphan ")
            stopped.append(pid)
        print(f"Stopped vLLM process(es): {stopped}")
        return

    matches = [s for s in servers if s["server_id"] == target or s["model_name"] == target]
    if not matches:
        print(f"No running server matches '{target}'", file=sys.stderr)
        sys.exit(1)

    data = load_servers()
    for s in matches:
        kill_pid(s["pid"], label=f"{s['server_id']} ")
        data["servers"].pop(s["server_id"], None)
    save_servers(data)
    print(f"Stopped: {[s['server_id'] for s in matches]}")


def command_restart(args):
    servers = list_running()
    data = load_servers()
    for s in servers:
        if s["model_name"] == args.name:
            kill_pid(s["pid"], label=f"{s['server_id']} ")
            data["servers"].pop(s["server_id"], None)
    save_servers(data)
    command_serve(args)


def command_status(_args):
    servers = list_running()
    if not servers:
        print("No vLLM servers running")
        return
    header = f"{'SERVER-ID':<22} {'MODEL':<24} {'PORT':<6} {'PID':<8} {'STARTED':<22} SOURCE"
    print(header)
    for s in servers:
        print(f"{s['server_id']:<22} {s['model_name']:<24} {s['port']:<6} "
              f"{s['pid']:<8} {s['started_at']:<22} {s['source']}")


def main():
    parser = argparse.ArgumentParser(description="Manage vLLM model profiles (multi-server)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="Add/update model profile")
    p_add.add_argument("name")
    p_add.add_argument("repo", help="HF model repo, e.g. Qwen/Qwen3.5-35B-A3B")
    p_add.add_argument("--dtype", default="auto")
    p_add.add_argument("--max-model-len", type=int, default=8192)
    p_add.add_argument("--tensor-parallel-size", type=int, default=1)
    p_add.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p_add.set_defaults(func=command_add)

    p_remove = sub.add_parser("remove", help="Remove model profile")
    p_remove.add_argument("name")
    p_remove.set_defaults(func=command_remove)

    p_list = sub.add_parser("list", help="List configured model profiles")
    p_list.set_defaults(func=command_list)

    p_pull = sub.add_parser("pull", help="Download model from Hugging Face")
    p_pull.add_argument("name")
    p_pull.set_defaults(func=command_pull)

    p_serve = sub.add_parser("serve", help="Start a vLLM OpenAI API server (multi-server allowed)")
    p_serve.add_argument("name")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--gpu-memory-utilization", type=float, default=None,
                         help="Override profile's gpu_memory_utilization (useful when co-hosting)")
    p_serve.set_defaults(func=command_serve)

    p_stop = sub.add_parser(
        "stop",
        help="Stop server(s). Target = server-id or model-name; omit to stop all + reap orphans.",
    )
    p_stop.add_argument("target", nargs="?")
    p_stop.set_defaults(func=command_stop)

    p_restart = sub.add_parser("restart", help="Stop the named model (if running) then serve it")
    p_restart.add_argument("name")
    p_restart.add_argument("--port", type=int, default=8000)
    p_restart.add_argument("--gpu-memory-utilization", type=float, default=None)
    p_restart.set_defaults(func=command_restart)

    p_status = sub.add_parser("status", help="List running vLLM servers (tracked + external)")
    p_status.set_defaults(func=command_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
