import asyncio

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from webapp.core import get_system_stats
from webapp.models import SystemStats

router = APIRouter(prefix="/api/gpu", tags=["gpu"])


@router.get("/stats", response_model=SystemStats)
async def gpu_stats():
    return get_system_stats()


@router.get("/stream")
async def gpu_stream():
    async def event_generator():
        while True:
            stats = get_system_stats()
            yield {"data": stats.model_dump_json()}
            await asyncio.sleep(2)

    return EventSourceResponse(event_generator())
