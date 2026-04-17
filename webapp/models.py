from pydantic import BaseModel


class ModelProfile(BaseModel):
    repo: str
    dtype: str = "auto"
    max_model_len: int = 8192
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9


class ModelProfileCreate(BaseModel):
    name: str
    repo: str
    dtype: str = "auto"
    max_model_len: int = 8192
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9


class ModelProfileResponse(BaseModel):
    name: str
    repo: str
    dtype: str
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    downloaded: bool
    size_gb: float | None = None


class ServerStartRequest(BaseModel):
    model_name: str
    port: int = 8000
    gpu_memory_utilization: float | None = None


class ServerInfo(BaseModel):
    server_id: str
    model_name: str
    repo: str
    port: int
    pid: int
    log_file: str
    started_at: str
    status: str  # "running" | "stopped" | "starting"
    gpu_memory_utilization: float


class GpuStats(BaseModel):
    name: str
    temperature_c: int | None = None
    utilization_gpu_pct: int | None = None
    power_draw_w: float | None = None
    performance_state: str | None = None
    clock_speed_mhz: int | None = None


class MemoryStats(BaseModel):
    total_gb: float
    used_gb: float
    available_gb: float
    percent: float


class DiskStats(BaseModel):
    total_gb: float
    used_gb: float
    free_gb: float
    hf_cache_gb: float


class InferenceMetrics(BaseModel):
    tokens_per_sec: float = 0
    total_generation_tokens: int = 0
    total_prompt_tokens: int = 0
    total_requests: int = 0
    requests_running: int = 0
    requests_waiting: int = 0
    kv_cache_usage_pct: float = 0
    avg_ttft_ms: float = 0
    avg_tpot_ms: float = 0
    cache_hit_rate_pct: float = 0


class SystemStats(BaseModel):
    gpu: GpuStats
    memory: MemoryStats
    disk: DiskStats
    inference: InferenceMetrics | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    port: int = 8000
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = True
