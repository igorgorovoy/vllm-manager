#!/usr/bin/env python3
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


CONFIG_DIR = Path.home() / ".config" / "vllm-models"
CONFIG_FILE = CONFIG_DIR / "models.json"
PID_FILE = CONFIG_DIR / "server.pid"
LOG_FILE = CONFIG_DIR / "server.log"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
TOOLS_DIR = Path("/home/igogo/vllm-tools")
VENV_PY = TOOLS_DIR / ".venv" / "bin" / "python"
VENV_HFCLI = TOOLS_DIR / ".venv" / "bin" / "hf"


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


def require_tool(tool_name):
    from shutil import which

    if which(tool_name) is None:
        print(f"Missing required tool: {tool_name}", file=sys.stderr)
        sys.exit(1)


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


def get_model(name):
    cfg = load_config()
    model = cfg.get("models", {}).get(name)
    if not model:
        print(f"Model '{name}' is not configured", file=sys.stderr)
        sys.exit(1)
    return model


def command_pull(args):
    if not VENV_HFCLI.exists():
        print(f"Missing {VENV_HFCLI}. Run /home/igogo/vllm-tools/install_vllm_deps.sh", file=sys.stderr)
        sys.exit(1)
    model = get_model(args.name)
    cmd = [
        str(VENV_HFCLI),
        "download",
        model["repo"],
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Model '{args.name}' downloaded (or already cached) in {HF_CACHE}")


def is_pid_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def command_serve(args):
    if not VENV_PY.exists():
        print(f"Missing {VENV_PY}. Run /home/igogo/vllm-tools/install_vllm_deps.sh", file=sys.stderr)
        sys.exit(1)
    model = get_model(args.name)

    if PID_FILE.exists():
        old_pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        if is_pid_running(old_pid):
            print(f"vLLM server already running with PID {old_pid}", file=sys.stderr)
            sys.exit(1)
        PID_FILE.unlink(missing_ok=True)

    cmd = [
        str(VENV_PY),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model["repo"],
        "--dtype",
        str(model["dtype"]),
        "--port",
        str(args.port),
        "--max-model-len",
        str(model["max_model_len"]),
        "--tensor-parallel-size",
        str(model["tensor_parallel_size"]),
        "--gpu-memory-utilization",
        str(model["gpu_memory_utilization"]),
    ]

    ensure_config()
    log = LOG_FILE.open("a", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)
    PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    print(f"Started vLLM for '{args.name}' on port {args.port} (pid={proc.pid})")
    print(f"Logs: {LOG_FILE}")


VLLM_CMDLINE_MARKERS = ("vllm.entrypoints.openai.api_server", "VLLM::EngineCore")


def find_vllm_pids():
    pids = []
    try:
        my_uid = os.getuid()
    except AttributeError:
        return pids
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
            pids.append(pid)
    return pids


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


def stop_all():
    stopped = []
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = None
        PID_FILE.unlink(missing_ok=True)
        if pid is not None and is_pid_running(pid):
            kill_pid(pid, label="tracked ")
            stopped.append(pid)
    for pid in find_vllm_pids():
        if pid in stopped:
            continue
        kill_pid(pid, label="orphan ")
        stopped.append(pid)
    return stopped


def command_stop(_args):
    stopped = stop_all()
    if not stopped:
        print("No vLLM processes found")
    else:
        print(f"Stopped vLLM process(es): {stopped}")


def command_restart(args):
    stop_all()
    command_serve(args)


def command_status(_args):
    if not PID_FILE.exists():
        print("vLLM server: stopped")
        return
    pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    if is_pid_running(pid):
        print(f"vLLM server: running (pid={pid})")
    else:
        print("vLLM server: stopped (stale pid file)")


def main():
    parser = argparse.ArgumentParser(description="Manage models for vLLM")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="Add/update model profile")
    p_add.add_argument("name")
    p_add.add_argument("repo", help="HF model repo, e.g. Qwen/Qwen2.5-Coder-32B-Instruct")
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

    p_serve = sub.add_parser("serve", help="Start vLLM OpenAI API server")
    p_serve.add_argument("name")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=command_serve)

    p_stop = sub.add_parser("stop", help="Stop vLLM server + reap any orphan vllm processes")
    p_stop.set_defaults(func=command_stop)

    p_restart = sub.add_parser("restart", help="Stop (incl. orphans) then serve")
    p_restart.add_argument("name")
    p_restart.add_argument("--port", type=int, default=8000)
    p_restart.set_defaults(func=command_restart)

    p_status = sub.add_parser("status", help="Show vLLM server status")
    p_status.set_defaults(func=command_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
