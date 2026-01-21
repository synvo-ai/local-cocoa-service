from fastapi import APIRouter
from pydantic import BaseModel
from core.spawn_manager import get_spawn_manager
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/spawns", tags=["spawns"])

class SpawnStatusResponse(BaseModel):
    alias: str
    running: bool

@router.get("/status/{alias}")
async def get_spawn_status(alias: str):
    manager = get_spawn_manager()
    running = manager.is_running(alias)
    return SpawnStatusResponse(alias=alias, running=running)

@router.get("/status")
async def get_all_spawns_status():
    manager = get_spawn_manager()
    # This only returns spawns managed by this manager
    # We might want to combine this with health check results if needed
    results = []
    for alias in ["embedding", "reranker", "vlm", "whisper"]:
        results.append({
            "alias": alias,
            "running": manager.is_running(alias)
        })
    return results

@router.post("/stop-all")
async def stop_all_spawns():
    manager = get_spawn_manager()
    await manager.stop_all()
    return {"status": "ok", "message": "All spawns stop initiated"}

@router.post("/start-all")
async def start_all_spawns():
    manager = get_spawn_manager()
    await manager.start_all_spawns()
    return {"status": "ok", "message": "All spawns start initiated"}
