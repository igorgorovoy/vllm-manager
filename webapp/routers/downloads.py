import asyncio

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from webapp.core import active_downloads, download_model, get_model

router = APIRouter(prefix="/api/downloads", tags=["downloads"])


@router.post("/{name}/start")
async def start_download(name: str):
    try:
        get_model(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if name in active_downloads:
        raise HTTPException(status_code=409, detail=f"Download already in progress for '{name}'")

    queue = asyncio.Queue()
    active_downloads[name] = queue
    asyncio.create_task(_run_download(name, queue))
    return {"ok": True, "message": f"Download started for '{name}'"}


async def _run_download(name: str, queue: asyncio.Queue):
    try:
        await download_model(name, queue)
    finally:
        active_downloads.pop(name, None)


@router.get("/{name}/stream")
async def stream_download(name: str):
    queue = active_downloads.get(name)
    if not queue:
        raise HTTPException(status_code=404, detail=f"No active download for '{name}'")

    async def event_generator():
        while True:
            msg = await queue.get()
            if msg is None:
                break
            yield {"event": msg["event"], "data": msg["data"]}

    return EventSourceResponse(event_generator())


@router.get("/active")
async def list_active_downloads():
    return {"downloads": list(active_downloads.keys())}
