import httpx
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from webapp.models import ChatRequest

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/completions")
async def chat_completions(req: ChatRequest):
    base_url = f"http://localhost:{req.port}"

    # Discover model name if not provided
    model_name = req.model
    if not model_name:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{base_url}/v1/models", timeout=5)
                data = resp.json()
                model_name = data["data"][0]["id"]
        except Exception:
            raise HTTPException(
                status_code=502,
                detail=f"Cannot reach vLLM server on port {req.port}",
            )

    payload = {
        "model": model_name,
        "messages": [m.model_dump() for m in req.messages],
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "stream": req.stream,
    }

    if not req.stream:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                timeout=httpx.Timeout(300.0, connect=10.0),
            )
            return resp.json()

    async def event_generator():
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=payload,
                timeout=httpx.Timeout(300.0, connect=10.0),
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk.strip() == "[DONE]":
                            yield {"data": "[DONE]"}
                            break
                        yield {"data": chunk}

    return EventSourceResponse(event_generator())


@router.get("/models")
async def list_chat_models(port: int = 8000):
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://localhost:{port}/v1/models", timeout=5)
            return resp.json()
    except Exception:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot reach vLLM server on port {port}",
        )
