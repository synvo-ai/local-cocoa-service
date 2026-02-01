import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any
from pydantic import BaseModel

from .config import settings

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    EMBEDDING = "embedding"
    RERANK = "rerank"
    VISION = "vision"  # Corresponds to VLM
    WHISPER = "whisper"


class ModelState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


class ModelInstance:
    def __init__(self, model_type: ModelType):
        self.type = model_type
        self.process: Optional[subprocess.Popen] = None
        self.state = ModelState.STOPPED
        self.last_accessed = 0.0
        self.error_message: Optional[str] = None
        self.port = self._get_port()

    def _get_port(self) -> int:
        if self.type == ModelType.EMBEDDING:
            return int(settings.endpoints.embedding_port)
        elif self.type == ModelType.RERANK:
            return int(settings.endpoints.rerank_port)
        elif self.type == ModelType.VISION:
            return int(settings.endpoints.vision_port)
        elif self.type == ModelType.WHISPER:
            return int(settings.endpoints.transcribe_port)
        return 0

    def touch(self):
        """Update last accessed timestamp."""
        self.last_accessed = time.time()


class ModelManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.models: Dict[ModelType, ModelInstance] = {
            ModelType.EMBEDDING: ModelInstance(ModelType.EMBEDDING),
            ModelType.RERANK: ModelInstance(ModelType.RERANK),
            ModelType.VISION: ModelInstance(ModelType.VISION),
            ModelType.WHISPER: ModelInstance(ModelType.WHISPER),
        }
        self._manager_task: Optional[asyncio.Task] = None
        self.llama_server_bin = str((Path(settings.paths.llama_server_path) / "llama-server").with_suffix(".exe" if settings.is_win else ""))
        self.whisper_server_bin = str((Path(settings.paths.whisper_server_path) / "whisper-server").with_suffix(".exe" if settings.is_win else ""))
        
        self._initialized = True
        logger.info("ModelManager initialized (models will start on-demand)")

    def start_manager_loop(self):
        if not settings.model_manager.enabled:
            logger.info("ModelManager disabled by config")
            return

        if self._manager_task is None:
            self._manager_task = asyncio.create_task(self._monitor_loop())
            logger.info("ModelManager monitoring loop started")

    async def _monitor_loop(self):
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                self._check_idle_models()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ModelManager loop: {e}")
                await asyncio.sleep(30)

    def _check_idle_models(self):
        now = time.time()
        timeout = settings.model_manager.idle_timeout_seconds

        for model in self.models.values():
            if model.state == ModelState.RUNNING and model.process:
                # Check if process is actually alive
                if model.process.poll() is None:
                    # Process alive and idle too long -> hibernate
                    if now - model.last_accessed > timeout:
                        logger.info(f"Model {model.type} idle for {now - model.last_accessed:.1f}s. Hibernating...")
                        self.stop_model(model.type)
                else:
                    # Process died unexpectedly
                    logger.warning(f"Model {model.type} process died unexpectedly")
                    model.state = ModelState.STOPPED
                    model.process = None

    def get_status(self) -> Dict[str, dict]:
        now = time.time()
        timeout = settings.model_manager.idle_timeout_seconds
        status: Dict[str, dict] = {}

        for model_type, model in self.models.items():
            idle_remaining = None
            if model.state == ModelState.RUNNING and model.last_accessed:
                idle_remaining = max(0, timeout - (now - model.last_accessed))

            status[model_type.value] = {
                "state": model.state.value,
                "last_accessed": model.last_accessed or None,
                "idle_timeout_seconds": timeout,
                "idle_seconds_remaining": idle_remaining,
                "port": model.port
            }

        return status

    def get_model_instance(self, model_type: ModelType) -> ModelInstance:
        """Get the model instance for a specific model type."""
        return self.models[model_type]

    async def ensure_model(self, model_type: ModelType, model_path: Optional[str] = None):
        """Ensure the requested model is running. If not, start it and wait until ready."""
        if not settings.model_manager.enabled:
            return

        model = self.models[model_type]
        model.touch()  # Update access time

        if model.state == ModelState.RUNNING:
            # Check if process is actually alive
            if model.process and model.process.poll() is None:
                return  # Healthy
            logger.warning(f"Model {model_type} process died. Restarting.")
            model.state = ModelState.STOPPED

        if model.state == ModelState.STARTING:
            # Another request is already starting this model - wait for it
            for _ in range(300):  # 30s max
                if model.state == ModelState.RUNNING:
                    return
                if model.state == ModelState.ERROR:
                    raise RuntimeError(f"Model {model_type} failed to start: {model.error_message}")
                await asyncio.sleep(0.1)
            raise RuntimeError(f"Model {model_type} startup timeout")

        # Before starting a new process, check if there's already a healthy service on the port
        if await self._health_check_model(model, timeout=2):
            logger.info(f"Model {model_type} already running on port {model.port}")
            model.state = ModelState.RUNNING
            return

        # Start the model
        await self._start_model_process(model, model_path)

        if model.state != ModelState.RUNNING:
            raise RuntimeError(f"Model {model_type} failed to start: {model.error_message}")

    async def _start_model_process(self, model: ModelInstance, model_path: Optional[str] = None):
        if model.state == ModelState.STARTING:
            return  # Already starting

        model.state = ModelState.STARTING
        try:
            cmd = self._build_command(model, model_path)
            logger.info(f"Starting model {model.type} on port {model.port}")
            logger.debug(f"Command: {' '.join(cmd)}")

            # Start process
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NO_WINDOW

            model.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=creationflags,
                bufsize=1
            )
            
            # Start background task to log output if needed
            asyncio.create_task(self._log_output(model.type.value, model.process))

            # Wait for port to be ready
            if await self._wait_for_port(model.port):
                # Port is open, now verify the server is actually ready to serve requests
                if await self._health_check_model(model):
                    model.state = ModelState.RUNNING
                    model.touch()
                    logger.info(f"Model {model.type} started successfully")
                else:
                    model.state = ModelState.ERROR
                    model.error_message = "Model server not responding to health checks"
                    self._kill_process(model.process)
                    logger.error(f"Failed to start model {model.type}: Health check failed")
            else:
                model.state = ModelState.ERROR
                model.error_message = "Timed out waiting for port"
                self._kill_process(model.process)
                logger.error(f"Failed to start model {model.type}: Timeout")

        except Exception as e:
            model.state = ModelState.ERROR
            model.error_message = str(e)
            logger.exception(f"Exception starting model {model.type}")

    def _kill_process(self, process: subprocess.Popen):
        if not process:
            return
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/pid", str(process.pid), "/T", "/F"], check=False, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                process.terminate()
                process.wait(timeout=5)
        except Exception as e:
            logger.error(f"Error killing process: {e}")

    def stop_model(self, model_type: ModelType):
        model = self.models[model_type]
        if model.process:
            logger.info(f"Stopping model {model_type}...")
            self._kill_process(model.process)
            model.process = None
        model.state = ModelState.STOPPED

    def stop_all_models(self):
        """Stop all running model processes. Called on application shutdown."""
        logger.info("Stopping all model processes...")
        for model_type in self.models.keys():
            self.stop_model(model_type)
        logger.info("All model processes stopped.")

    def _build_command(self, model: ModelInstance, model_path_override: Optional[str] = None) -> list[str]:
        # Use provided path or fallback to settings
        def get_path(key: str, default_path: Path) -> str:
            path = model_path_override or str(default_path)
            if not path or path == ".":
                 # Some defaults might be empty or placeholders
                 pass
            
            # If relative, join with model_root_path
            p = Path(path)
            if not p.is_absolute():
                p = settings.paths.model_root_path / p
            
            if not p.exists():
                raise FileNotFoundError(f"Model file not found at {p}")
            return str(p)

        binary_path = self.llama_server_bin
        host = "127.0.0.1"
        threads = os.getenv("LLAMA_THREADS", "4")
        ngl = os.getenv("LLAMA_NGL", "999")
        ctx = str(settings.llm_context_tokens)

        if model.type == ModelType.EMBEDDING:
            model_path = get_path("embedding", settings.paths.embedding_model)
            return [
                binary_path,
                "-m", model_path,
                "--embedding",
                "--pooling", "cls",
                "--host", host,
                "--port", str(model.port),
                "-c", "8192", # Embeddings usually need smaller context unless specific
                "--threads", threads,
                "-ngl", ngl,
                "--log-disable"
            ]

        elif model.type == ModelType.RERANK:
            model_path = get_path("rerank", settings.paths.rerank_model)
            ubatch = os.getenv("RERANK_N_UBATCH", "2048")
            return [
                binary_path,
                "-m", model_path,
                "--reranking",
                "--host", host,
                "--port", str(model.port),
                "--ubatch-size", ubatch,
                "--threads", threads,
                "-ngl", ngl,
                "--log-disable"
            ]

        elif model.type == ModelType.VISION:
            model_path = get_path("vlm", settings.paths.vlm_model)
            mmproj_path = get_path("vlm_mmproj", settings.paths.vlm_mmproj)
            cmd = [
                binary_path,
                "-m", model_path,
                "--mmproj", mmproj_path,
                "--host", host,
                "--port", str(model.port),
                "-c", ctx,
                "--threads", threads,
                "-ngl", ngl
            ]
            cache_ram = os.getenv("LLAMA_PROMPT_CACHE", "0")
            if cache_ram != "0" and cache_ram != "false":
                cmd.extend(["--cache-ram", cache_ram])
            return cmd

        elif model.type == ModelType.WHISPER:
            binary_path = self.whisper_server_bin
            if not os.path.exists(binary_path):
                raise FileNotFoundError(f"Whisper server binary not found at {binary_path}")
            model_path = get_path("whisper", settings.paths.whisper_model)
            return [
                binary_path,
                "-m", model_path,
                "--host", host,
                "--port", str(model.port),
                "-t", threads,
                "--convert"
            ]

        return []

    async def _wait_for_port(self, port: int, timeout: int = 30) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                reader, writer = await asyncio.open_connection('127.0.0.1', port)
                writer.close()
                await writer.wait_closed()
                return True
            except (ConnectionRefusedError, OSError):
                await asyncio.sleep(0.5)
        return False

    async def _health_check_model(self, model: ModelInstance, timeout: int = 30) -> bool:
        import httpx
        start_time = time.time()
        url = f"http://127.0.0.1:{model.port}/health"

        while time.time() - start_time < timeout:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(url)
                    if response.status_code == 200:
                        return True
                    if response.status_code != 503:
                        logger.warning(f"Model {model.type} health check got {response.status_code}")
            except httpx.RequestError:
                pass
            await asyncio.sleep(1)
        return False

    async def _log_output(self, alias: str, process: subprocess.Popen):
        async def read_stream(stream):
            while True:
                line = await asyncio.get_event_loop().run_in_executor(None, stream.readline)
                if not line:
                    break
                logger.debug(f"[{alias}] {line.strip()}")

        if process.stdout:
            asyncio.create_task(read_stream(process.stdout))
        if process.stderr:
            asyncio.create_task(read_stream(process.stderr))

    async def start_all_models(self) -> None:
        """Starts all configured models."""
        logger.info("Starting all models...")
        for model_type in self.models.keys():
            try:
                await self.ensure_model(model_type)
            except Exception as e:
                logger.error(f"Failed to start model {model_type}: {e}")

    def is_running(self, model_type: ModelType) -> bool:
        model = self.models.get(model_type)
        return model is not None and model.state == ModelState.RUNNING and model.process is not None and model.process.poll() is None


def get_model_manager() -> ModelManager:
    return ModelManager()
