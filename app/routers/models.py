from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from core.model_manager import get_model_manager, ModelType
from core.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


class ModelPathUpdate(BaseModel):
    model_path: Optional[str] = None
    mmproj_path: Optional[str] = None


class AllModelsConfigUpdate(BaseModel):
    """Update all model paths at once."""
    vlm_model: Optional[str] = None
    vlm_mmproj: Optional[str] = None
    embedding_model: Optional[str] = None
    rerank_model: Optional[str] = None
    whisper_model: Optional[str] = None


class ModelStartRequest(BaseModel):
    model_path: Optional[str] = None


class ModelStatusResponse(BaseModel):
    type: str
    state: str
    port: int
    last_accessed: Optional[float] = None
    idle_seconds_remaining: Optional[float] = None


@router.get("/status")
async def get_all_models_status():
    """Get status of all managed models."""
    manager = get_model_manager()
    
    return manager.get_status()


@router.get("/status/{model_type}")
async def get_model_status(model_type: ModelType):
    """Get status of a specific model."""
    manager = get_model_manager()
    status = manager.get_status()
    if model_type.value not in status:
        raise HTTPException(status_code=404, detail=f"Model type {model_type} not found")
    return {
        "type": model_type.value,
        **status[model_type.value]
    }


@router.post("/start-all")
async def start_all_models():
    """Manually start all configured models."""
    manager = get_model_manager()
    await manager.start_all_models()
    return {"status": "ok", "message": "All models start initiated"}


@router.post("/stop-all")
async def stop_all_models():
    """Manually stop all running models."""
    manager = get_model_manager()
    manager.stop_all_models()
    return {"status": "ok", "message": "All models stop initiated"}


@router.post("/{model_type}/start")
async def start_model(model_type: ModelType, request: Optional[ModelStartRequest] = None):
    """Manually start a specific model."""
    manager = get_model_manager()
    model_path = request.model_path if request else None
    try:
        await manager.ensure_model(model_type, model_path=model_path)
        return {"status": "started", "model": model_type}
    except Exception as e:
        logger.error(f"Failed to start model {model_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_type}/stop")
async def stop_model(model_type: ModelType):
    """Manually stop a specific model."""
    manager = get_model_manager()
    manager.stop_model(model_type)
    return {"status": "stopped", "model": model_type}


@router.patch("/config")
async def update_all_models_config(update: AllModelsConfigUpdate):
    """Update all model file paths at once."""
    if update.vlm_model is not None:
        settings.paths.vlm_model = update.vlm_model
    if update.vlm_mmproj is not None:
        settings.paths.vlm_mmproj = update.vlm_mmproj
    if update.embedding_model is not None:
        settings.paths.embedding_model = update.embedding_model
    if update.rerank_model is not None:
        settings.paths.rerank_model = update.rerank_model
    if update.whisper_model is not None:
        settings.paths.whisper_model = update.whisper_model
    return {
        "vlm_model": settings.paths.vlm_model,
        "vlm_mmproj": settings.paths.vlm_mmproj,
        "embedding_model": settings.paths.embedding_model,
        "rerank_model": settings.paths.rerank_model,
        "whisper_model": settings.paths.whisper_model
    }


@router.patch("/{model_type}/config")
async def update_model_config(model_type: ModelType, update: ModelPathUpdate):
    """Update specific model file paths."""
    if model_type == ModelType.VISION:
        if update.model_path is not None:
            settings.paths.vlm_model = update.model_path
        if update.mmproj_path is not None:
            settings.paths.vlm_mmproj = update.mmproj_path
        return {"model_path": settings.paths.vlm_model, "mmproj_path": settings.paths.vlm_mmproj}
    elif model_type == ModelType.EMBEDDING:
        if update.model_path is not None:
            settings.paths.embedding_model = update.model_path
        return {"model_path": settings.paths.embedding_model}
    elif model_type == ModelType.RERANK:
        if update.model_path is not None:
            settings.paths.rerank_model = update.model_path
        return {"model_path": settings.paths.rerank_model}
    elif model_type == ModelType.WHISPER:
        if update.model_path is not None:
            settings.paths.whisper_model = update.model_path
        return {"model_path": settings.paths.whisper_model}
    
    raise HTTPException(status_code=400, detail=f"Unsupported model type for config update: {model_type}")
