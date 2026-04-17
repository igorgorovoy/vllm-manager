from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from webapp.routers import chat, downloads, gpu, logs, profiles, servers

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="vLLM Manager", version="1.0.0")

app.include_router(profiles.router)
app.include_router(downloads.router)
app.include_router(servers.router)
app.include_router(gpu.router)
app.include_router(logs.router)
app.include_router(chat.router)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))
