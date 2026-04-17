import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from webapp.core import list_servers, start_server, stop_server
from webapp.models import ServerInfo, ServerStartRequest

router = APIRouter(prefix="/api/servers", tags=["servers"])


@router.get("", response_model=list[ServerInfo])
async def get_servers():
    return list_servers()


@router.post("", response_model=ServerInfo)
async def create_server(req: ServerStartRequest):
    try:
        return start_server(req.model_name, req.port, req.gpu_memory_utilization)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.delete("/{server_id}")
async def delete_server(server_id: str):
    try:
        stop_server(server_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"ok": True, "stopped": server_id}


@router.get("/{server_id}/health")
async def server_health(server_id: str):
    servers = list_servers()
    srv = next((s for s in servers if s.server_id == server_id), None)
    if not srv:
        raise HTTPException(status_code=404, detail="Server not found")

    import httpx
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://localhost:{srv.port}/health", timeout=3)
            return {"healthy": resp.status_code == 200, "port": srv.port}
    except Exception:
        return {"healthy": False, "port": srv.port}


@router.get("/{server_id}/startup-stream")
async def startup_stream(server_id: str):
    servers = list_servers()
    srv = next((s for s in servers if s.server_id == server_id), None)
    if not srv:
        raise HTTPException(status_code=404, detail="Server not found")

    log_path = Path(srv.log_file)

    async def event_generator():
        import httpx

        while not log_path.exists():
            await asyncio.sleep(0.5)

        proc = await asyncio.create_subprocess_exec(
            "tail", "-n", "50", "-f", str(log_path),
            stdout=asyncio.subprocess.PIPE,
        )
        try:
            ready = False
            while not ready:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=600)
                if not line:
                    break
                text = line.decode(errors="replace").strip()
                if text:
                    yield {"event": "log", "data": text}

                # Check if vLLM is ready
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(
                            f"http://localhost:{srv.port}/health", timeout=2
                        )
                        if resp.status_code == 200:
                            yield {"event": "ready", "data": f"Server ready on port {srv.port}"}
                            ready = True
                except Exception:
                    pass
        except asyncio.TimeoutError:
            yield {"event": "error", "data": "Startup timed out (10 min)"}
        finally:
            proc.kill()
            await proc.wait()

    return EventSourceResponse(event_generator())
