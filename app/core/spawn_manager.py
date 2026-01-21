import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any
from pydantic import BaseModel

from .config import settings

logger = logging.getLogger(__name__)

class SpawnConfig(BaseModel):
    alias: str
    model_path: str
    port: int
    context_size: int
    threads: int
    ngl: int
    type: Literal["embedding", "reranking", "vlm", "completion", "whisper"]
    mmproj_path: Optional[str] = None
    batch_size: Optional[int] = None
    ubatch_size: Optional[int] = None
    parallel: Optional[int] = None
    log_path: Optional[str] = None

class SpawnManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SpawnManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.processes: Dict[str, subprocess.Popen] = {}
        self.llama_server_bin = str((Path(settings.llama_server_path) / "llama-server").with_suffix(".exe" if settings.is_win else ""))
        self.whisper_server_bin = str((Path(settings.whisper_server_path) / "whisper-server").with_suffix(".exe" if settings.is_win else ""))

        self.descriptors: List[Dict[str, Any]] = []
        self._load_descriptors()
        self._initialized = True

    def _load_descriptors(self):
        if settings.models_config_path and settings.models_config_path.exists():
            try:
                with open(settings.models_config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.descriptors = data.get('models', [])
            except Exception as e:
                logger.error(f"Failed to load models config from {settings.models_config_path}: {e}")

    def get_descriptor(self, model_id: str) -> Optional[Dict[str, Any]]:
        for d in self.descriptors:
            if d.get('id') == model_id:
                return d
        return None

    def get_model_path(self, model_id: str) -> str:
        descriptor = self.get_descriptor(model_id)
        if not descriptor:
            # Fallback for known defaults if not in config
            if "embedding" in model_id: return str(settings.model_root_path / "Qwen3-Embedding-0.6B-Q4_K_M.gguf")
            if "reranker" in model_id: return str(settings.model_root_path / "bge-reranker-v2-m3-q8_0.gguf")
            if "vlm" in model_id: return str(settings.model_root_path / "qwenvl" / "Qwen3VL-2B-Instruct-Q4_K_M.gguf")
            return ""

        return str(settings.model_root_path / descriptor.get('relativePath', ''))

    async def start_all_spawns(self) -> None:
        """Starts all configured model spawns."""
        logger.info("Starting all model spawns...")
        
        # 1. VLM
        try:
            model_id = settings.active_model_id
            descriptor = self.get_descriptor(model_id)
            mmproj_path = None
            if descriptor and (descriptor.get('type') == 'vlm' or descriptor.get('id') == 'vlm'):
                mmproj_id = descriptor.get('mmprojId') or 'vlm-mmproj'
                mmproj_path = self.get_model_path(mmproj_id)

            await self.start_spawn(SpawnConfig(
                alias='vlm',
                model_path=self.get_model_path(model_id),
                port=settings.endpoints.vision_port,
                context_size=settings.llm_context_tokens,
                threads=4,
                ngl=999,
                type='vlm',
                mmproj_path=mmproj_path,
                log_path=settings.vlm_log_path
            ))
        except Exception as e:
            logger.error(f"Failed to start VLM spawn: {e}")

        # 2. Embedding
        try:
            await self.start_spawn(SpawnConfig(
                alias='embedding',
                model_path=self.get_model_path(settings.active_embedding_model_id),
                port=settings.endpoints.embedding_port,
                context_size=8192,
                threads=4,
                ngl=999,
                type='embedding',
                batch_size=8192,
                ubatch_size=512,
                parallel=4,
                log_path=settings.embed_log_path
            ))
        except Exception as e:
            logger.error(f"Failed to start Embedding spawn: {e}")

        # 3. Reranker
        try:
            await self.start_spawn(SpawnConfig(
                alias='reranker',
                model_path=self.get_model_path(settings.active_reranker_model_id),
                port=settings.endpoints.rerank_port,
                context_size=4096,
                threads=2,
                ngl=999,
                type='reranking',
                ubatch_size=2048,
                log_path=settings.rerank_log_path
            ))
        except Exception as e:
            logger.error(f"Failed to start Reranker spawn: {e}")

        # 4. Whisper
        try:
            await self.start_spawn(SpawnConfig(
                alias='whisper',
                model_path=self.get_model_path(settings.active_audio_model_id),
                port=settings.endpoints.transcribe_port,
                context_size=0,
                threads=4,
                ngl=0,
                type='whisper',
                log_path=settings.whisper_log_path
            ))
        except Exception as e:
            logger.error(f"Failed to start Whisper spawn: {e}")

    async def start_spawn(self, config: SpawnConfig) -> None:
        if config.alias in self.processes:
            # Check if process is still running
            if self.processes[config.alias].poll() is None:
                logger.info(f"Spawn {config.alias} is already running.")
                return
            else:
                del self.processes[config.alias]

        binary_path = self.llama_server_bin
        args: List[str] = []

        if config.type == "whisper":
            binary_path = self.whisper_server_bin
            if not os.path.exists(binary_path):
                logger.warning(f"[SpawnManager] whisper-server binary not found at {binary_path}. Skipping {config.alias}.")
                return
            if not os.path.exists(config.model_path):
                logger.warning(f"[SpawnManager] Model file not found at {config.model_path}. Skipping {config.alias}.")
                return

            args.extend([
                "-m", config.model_path,
                "--host", "127.0.0.1",
                "--port", str(config.port),
                "-t", str(config.threads),
                "--convert"
            ])
            
            # TODO: whisper-server doesn't support log-file yet
            # if config.log_path:
            #     args.extend(["--log-file", config.log_path])
        else:
            # Llama Spawns
            if not os.path.exists(self.llama_server_bin):
                logger.error(f"llama-server binary not found at {self.llama_server_bin}")
                return
            if not os.path.exists(config.model_path):
                logger.warning(f"[SpawnManager] Model file not found at {config.model_path}. Skipping {config.alias}.")
                return

            args.extend([
                "-m", config.model_path,
                "--host", "127.0.0.1",
                "--port", str(config.port),
                "-c", str(config.context_size),
                "-t", str(config.threads),
                "-ngl", str(config.ngl)
            ])

            if config.log_path:
                args.extend(["--log-file", config.log_path])

            if config.type == "embedding":
                args.append("--embedding")
            elif config.type == "reranking":
                args.append("--reranking")
            elif config.type == "vlm" and config.mmproj_path:
                if os.path.exists(config.mmproj_path):
                    args.extend(["--mmproj", config.mmproj_path])
                else:
                    logger.warning(f"[SpawnManager] mmproj file not found at {config.mmproj_path}")

            if config.batch_size:
                args.extend(["-b", str(config.batch_size)])
            if config.ubatch_size:
                args.extend(["-ub", str(config.ubatch_size)])
            if config.parallel:
                args.extend(["-np", str(config.parallel)])

        logger.info(f"[SpawnManager] Starting {config.alias}")
        logger.info(f"[SpawnManager] Binary: {binary_path}")
        logger.info(f"[SpawnManager] Args: {' '.join(args)}")

        # Prepare environment
        env = os.environ.copy()
        
        try:
            # On Windows, we might need creationflags to hide the window
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NO_WINDOW

            process = subprocess.Popen(
                [binary_path] + args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                creationflags=creationflags,
                text=True,
                bufsize=1
            )
            
            self.processes[config.alias] = process
            
            # Start background tasks to log output
            asyncio.create_task(self._log_output(config.alias, process))
            
        except Exception as e:
            logger.error(f"[SpawnManager] Failed to start {config.alias}: {e}")

    async def _log_output(self, alias: str, process: subprocess.Popen):
        # We need to read from both stdout and stderr
        # Since we used text=True, we can read lines
        
        async def read_stream(stream, is_stderr: bool):
            while True:
                line = await asyncio.get_event_loop().run_in_executor(None, stream.read, 1024)
                if not line:
                    break
                # Process lines
                if is_stderr:
                    logger.info(f"[{alias}] {line.strip()}")
                else:
                    logger.info(f"[{alias}] {line.strip()}")

        try:
            # Simplified logging for now to avoid complexity with asyncio/threading mix
            pass
        except Exception as e:
            logger.error(f"[{alias}] Error reading output: {e}")
        finally:
            # We can't easily wait for it without blocking or more complex logic
            # For now, let's just let it run.
            pass

    async def stop_spawn(self, alias: str) -> None:
        if alias in self.processes:
            process = self.processes[alias]
            logger.info(f"Stopping {alias}...")
            try:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/pid", str(process.pid), "/T", "/F"], check=False, creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    process.send_signal(signal.SIGKILL)
                process.wait(timeout=5)
            except Exception as e:
                logger.error(f"[SpawnManager] Error stopping {alias}: {e}")
            finally:
                if alias in self.processes:
                    del self.processes[alias]

    async def stop_all(self) -> None:
        aliases = list(self.processes.keys())
        for alias in aliases:
            await self.stop_spawn(alias)

    def is_running(self, alias: str) -> bool:
        return alias in self.processes and self.processes[alias].poll() is None

def get_spawn_manager() -> SpawnManager:
    return SpawnManager()
