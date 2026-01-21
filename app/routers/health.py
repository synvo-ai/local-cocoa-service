from __future__ import annotations

import asyncio
import time
import httpx
from fastapi import APIRouter

from core.config import settings
from core.context import get_indexer, get_storage
from core.models import HealthResponse, ServiceStatus

router = APIRouter(tags=["health"])

# OPTIMIZATION: Cache service health checks to reduce network overhead during bulk indexing
# Previous behavior: Every /health call would check all 3 services (Embedding, Reranker, VLM)
# This caused ~3 network requests every 1.5 seconds, competing with indexing I/O
_service_cache: dict[str, tuple[ServiceStatus, float]] = {}
_SERVICE_CACHE_TTL = 10.0  # Cache service status for 10 seconds


async def check_service(name: str, url: str | None, use_cache: bool = True) -> ServiceStatus:
    if not url:
        return ServiceStatus(name=name, status="unknown", details="URL not configured")

    # Check cache first
    now = time.perf_counter()
    cache_key = f"{name}:{url}"
    if use_cache and cache_key in _service_cache:
        cached_status, cached_time = _service_cache[cache_key]
        if now - cached_time < _SERVICE_CACHE_TTL:
            return cached_status

    start = now
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            # Try /health first
            target = url.rstrip("/") + "/health"
            try:
                response = await client.get(target)
                if response.status_code == 404:
                    raise httpx.HTTPStatusError("Not Found", request=response.request, response=response)
            except httpx.HTTPStatusError:
                # Fallback to root
                target = url.rstrip("/")
                response = await client.get(target)

            if 200 <= response.status_code < 500:  # Accept even 4xx as "online" (service is reachable)
                latency = (time.perf_counter() - start) * 1000
                result = ServiceStatus(name=name, status="online", latency_ms=latency)
            else:
                result = ServiceStatus(name=name, status="offline", details=f"HTTP {response.status_code}")
    except Exception as e:
        result = ServiceStatus(name=name, status="offline", details=str(e))

    # Update cache
    _service_cache[cache_key] = (result, time.perf_counter())
    return result


@router.get("/health", response_model=HealthResponse)
async def read_health() -> HealthResponse:
    storage = get_storage()
    indexer = get_indexer()

    # Run blocking DB call in thread pool
    loop = asyncio.get_running_loop()
    files, folders = await loop.run_in_executor(None, storage.counts)

    status = "ready" if files else "idle"
    progress = indexer.status()
    if progress.status in ("running", "paused"):
        status = "indexing"
    message = progress.last_error if progress.last_error else progress.message
    if progress.status == "paused":
        message = message or "Indexing paused."

    # Check services
    # Core services (required for basic functionality)
    checks = [
        check_service("Embedding", settings.endpoints.embedding_url),
        check_service("Reranker", settings.endpoints.rerank_url),
    ]
    if settings.endpoints.vision_url:
        check_service("Vision/LLM", settings.endpoints.vision_url)
    
    # Optional services (Whisper is optional - only check if explicitly enabled)
    # Whisper being offline should not cause degraded status for file management
    optional_checks = []
    if settings.endpoints.transcribe_url:
        optional_checks.append(check_service("Whisper", settings.endpoints.transcribe_url))

    # Gather core service statuses
    core_services = await asyncio.gather(*checks)
    
    # Gather optional service statuses separately
    optional_services = await asyncio.gather(*optional_checks) if optional_checks else []
    
    # Combine all services for reporting
    all_services = list(core_services) + list(optional_services)

    # Downgrade status only if CORE services are offline
    # Optional services (like Whisper) being offline should not affect overall status
    if any(s.status == "offline" for s in core_services):
        status = "degraded"
        if not message:
            message = "Some AI services are offline."

    return HealthResponse(
        status=status,
        indexed_files=files,
        watched_folders=folders,
        message=message,
        services=all_services
    )
