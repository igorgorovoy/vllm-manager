import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from webapp.core import list_servers

router = APIRouter(prefix="/api/logs", tags=["logs"])


@router.get("/{server_id}")
async def get_logs(server_id: str, lines: int = Query(200, ge=1, le=5000)):
    srv = _find_server(server_id)
    log_path = Path(srv.log_file)
    if not log_path.exists():
        return {"lines": []}

    proc = await asyncio.create_subprocess_exec(
        "tail", "-n", str(lines), str(log_path),
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    return {"lines": stdout.decode(errors="replace").splitlines()}


@router.get("/{server_id}/stream")
async def stream_logs(server_id: str):
    srv = _find_server(server_id)
    log_path = Path(srv.log_file)

    async def event_generator():
        while not log_path.exists():
            await asyncio.sleep(0.5)

        proc = await asyncio.create_subprocess_exec(
            "tail", "-n", "50", "-f", str(log_path),
            stdout=asyncio.subprocess.PIPE,
        )
        try:
            while True:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=300)
                if not line:
                    break
                text = line.decode(errors="replace").strip()
                if text:
                    yield {"data": text}
        except asyncio.TimeoutError:
            pass
        finally:
            proc.kill()
            await proc.wait()

    return EventSourceResponse(event_generator())


def _find_server(server_id: str):
    servers = list_servers()
    srv = next((s for s in servers if s.server_id == server_id), None)
    if not srv:
        raise HTTPException(status_code=404, detail="Server not found")
    return srv
