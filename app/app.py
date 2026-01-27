import sys
from pathlib import Path

# Ensure app root is in path. to simplify all imports, packages inside app folder can be directlyimported
sys.path.append(str(Path(__file__).parent))

# Core Routers
from services.drive import files_router as files
from services.drive import folders_router as folders
from services.indexer import router as index
from services.search import router as search
from services.memory import router as memory
from routers import chat, health, settings as settings_router, security, plugins as plugins_router, language as language_router, events as events_router, spawns as spawns_router

from core.context import get_indexer
from core.config import settings
from core.auth import verify_api_key, ensure_local_key

# Plugin system
from plugins import init_all_plugins

import asyncio
import logging
import datetime as dt

# Force ProactorEventLoop on Windows to avoid "too many file descriptors" error
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

app = FastAPI(title="Local Cocoa Service", version="1.0.0", dependencies=[Depends(verify_api_key)])

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include core routers
app.include_router(health.router)
app.include_router(folders)
app.include_router(index.router)
app.include_router(files)
app.include_router(search.router)
app.include_router(memory)
app.include_router(chat.router)
app.include_router(settings_router.router)
app.include_router(security.router)
app.include_router(plugins_router.router)
app.include_router(language_router.router)
app.include_router(events_router.router)
app.include_router(spawns_router.router)

_poll_task: asyncio.Task | None = None
_startup_refresh_task: asyncio.Task | None = None
_summary_task: asyncio.Task | None = None
_staged_scheduler_task: asyncio.Task | None = None

SUMMARY_FOLDER = Path.home() / "local-cocoa-activity-summaries"


def _track_task(task: asyncio.Task, name: str) -> None:
    def _finalise(completed: asyncio.Task) -> None:
        try:
            completed.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s task failed: %s", name, exc)

    task.add_done_callback(_finalise)


async def _poll_loop(interval: int) -> None:
    """Background loop that periodically scans for new/changed files.

    Uses staged indexing (refresh_staged) instead of legacy refresh() to avoid
    conflicts and ensure consistent behavior with the staged indexing pipeline.
    """
    indexer = get_indexer()
    while True:
        await asyncio.sleep(interval)

        # Skip if already running to avoid conflicts
        if indexer.status().status == "running":
            logger.debug("Poll loop: indexer already running, skipping")
            continue

        # Use staged indexing which handles file discovery and processing
        # This is consistent with _staged_scheduler_loop and avoids the legacy refresh() path
        try:
            await indexer.refresh_staged(skip_pending_registration=False, reindex=False)
        except Exception as e:
            logger.warning("Poll loop refresh_staged failed: %s", e)


async def _staged_scheduler_loop(max_wait: int = 60) -> None:
    """Background loop that processes pending files using staged indexing.

    Uses event-based triggering for immediate response, with a max wait timeout
    to catch any files that might have been missed.
    """
    indexer = get_indexer()
    pending_event = indexer.get_pending_files_event()

    while True:
        try:
            # Wait for either:
            # 1. The pending files event to be signaled (immediate trigger)
            # 2. Timeout (fallback check every max_wait seconds)
            try:
                await asyncio.wait_for(pending_event.wait(), timeout=max_wait)
                pending_event.clear()  # Reset for next signal
            except asyncio.TimeoutError:
                pass  # Timeout is fine, we'll check anyway

            # Trigger staged indexing if there are pending files
            await indexer.trigger_staged_if_pending()

        except Exception as e:
            logger.warning("Staged scheduler loop error: %s", e)
            await asyncio.sleep(5)  # Brief pause on error


@app.on_event("startup")
async def on_startup() -> None:
    global _poll_task, _startup_refresh_task, _summary_task, _staged_scheduler_task
    # Initialize plugin system and load plugins at startup. Ensures plugin routes are available before the first request
    init_all_plugins(app)

    # Ensure local-key exists and is written to file for frontend
    ensure_local_key(settings.runtime_root)

    indexer = get_indexer()
    if settings.refresh_on_startup:
        # Launch initial refresh without blocking server startup so the API becomes responsive quickly.
        # Use staged indexing for consistency with the rest of the system.
        _startup_refresh_task = asyncio.create_task(
            indexer.refresh_staged(skip_pending_registration=False, reindex=False)
        )
        _track_task(_startup_refresh_task, "startup-refresh")
    if settings.poll_interval_seconds > 0:
        _poll_task = asyncio.create_task(_poll_loop(settings.poll_interval_seconds))
        _track_task(_poll_task, "poll-loop")

    # Start staged scheduler loop (checks every 10 seconds for pending files)
    _staged_scheduler_task = asyncio.create_task(_staged_scheduler_loop(10))
    _track_task(_staged_scheduler_task, "staged-scheduler")

    # Start managed model spawns (VLM, Embedding, etc.)
    from core.spawn_manager import get_spawn_manager
    manager = get_spawn_manager()
    asyncio.create_task(manager.start_all_spawns())


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global _poll_task, _startup_refresh_task, _summary_task, _staged_scheduler_task
    # Stop managed model spawns
    from core.spawn_manager import get_spawn_manager
    manager = get_spawn_manager()
    await manager.stop_all()

    if _poll_task:
        _poll_task.cancel()
        try:
            await _poll_task
        except asyncio.CancelledError:
            # Expected when cancelling task on shutdown - safe to ignore
            pass
        _poll_task = None
    if _startup_refresh_task:
        _startup_refresh_task.cancel()
        try:
            await _startup_refresh_task
        except asyncio.CancelledError:
            # Expected when cancelling task on shutdown - safe to ignore
            pass
        _startup_refresh_task = None
    if _summary_task:
        _summary_task.cancel()
        try:
            await _summary_task
        except asyncio.CancelledError:
            # Expected when cancelling task on shutdown - safe to ignore
            pass
        _summary_task = None
    if _staged_scheduler_task:
        _staged_scheduler_task.cancel()
        try:
            await _staged_scheduler_task
        except asyncio.CancelledError:
            # Expected when cancelling task on shutdown - safe to ignore
            pass
        _staged_scheduler_task = None
