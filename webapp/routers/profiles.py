from fastapi import APIRouter, HTTPException

from webapp.core import add_model, get_model_size_gb, is_downloaded, list_models, remove_model
from webapp.models import ModelProfileCreate, ModelProfileResponse

router = APIRouter(prefix="/api/profiles", tags=["profiles"])


@router.get("", response_model=list[ModelProfileResponse])
async def get_profiles():
    return list_models()


@router.post("", response_model=ModelProfileResponse)
async def create_profile(data: ModelProfileCreate):
    add_model(
        name=data.name,
        repo=data.repo,
        dtype=data.dtype,
        max_model_len=data.max_model_len,
        tensor_parallel_size=data.tensor_parallel_size,
        gpu_memory_utilization=data.gpu_memory_utilization,
    )
    return ModelProfileResponse(
        name=data.name,
        repo=data.repo,
        dtype=data.dtype,
        max_model_len=data.max_model_len,
        tensor_parallel_size=data.tensor_parallel_size,
        gpu_memory_utilization=data.gpu_memory_utilization,
        downloaded=is_downloaded(data.repo),
        size_gb=get_model_size_gb(data.repo),
    )


@router.delete("/{name}")
async def delete_profile(name: str):
    try:
        remove_model(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"ok": True, "deleted": name}
