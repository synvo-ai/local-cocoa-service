from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import shutil

import httpx

from .api import default_base_url, load_api_key


@dataclass
class LaunchResult:
    started: bool
    api_key: str | None
    base_url: str


class ServiceLauncher:
    def __init__(
        self,
        *,
        project_root: Path,
        base_url: str | None = None,
        python_executable: str | None = None,
    ) -> None:
        self.project_root = project_root
        self.base_url = (base_url or default_base_url()).rstrip("/")
        self.python_executable = python_executable or self._resolve_python_executable(project_root)
        self.process: subprocess.Popen[str] | None = None
        self.api_key: str | None = None
        self._stdout_tail: deque[str] = deque(maxlen=30)
        self._stderr_tail: deque[str] = deque(maxlen=30)
        self._threads: list[threading.Thread] = []

    @staticmethod
    def _resolve_python_executable(project_root: Path) -> str:
        venv_python = project_root / ".venv" / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)

        active_venv = os.getenv("VIRTUAL_ENV")
        if active_venv:
            active_python = Path(active_venv) / "bin" / "python"
            if active_python.exists():
                return str(active_python)

        path_python = shutil.which("python")
        if path_python:
            return path_python

        return sys.executable

    def ensure_running(self, *, api_key: str | None = None, timeout: float = 30.0) -> LaunchResult:
        resolved_key = load_api_key(api_key)
        if resolved_key and self._is_ready(resolved_key):
            self.api_key = resolved_key
            return LaunchResult(started=False, api_key=resolved_key, base_url=self.base_url)

        if self._is_service_listening():
            return LaunchResult(started=False, api_key=resolved_key, base_url=self.base_url)

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        cmd = [self.python_executable, str(self.project_root / "main.py")]
        self.process = subprocess.Popen(
            cmd,
            cwd=self.project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self._start_reader(self.process.stdout, self._stdout_tail, capture_token=True)
        self._start_reader(self.process.stderr, self._stderr_tail, capture_token=False)

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(self._failure_message())

            if self.api_key and self._is_ready(self.api_key):
                return LaunchResult(started=True, api_key=self.api_key, base_url=self.base_url)

            fallback_key = load_api_key(None)
            if fallback_key and self._is_ready(fallback_key):
                self.api_key = fallback_key
                return LaunchResult(started=True, api_key=fallback_key, base_url=self.base_url)

            time.sleep(0.25)

        raise RuntimeError(self._failure_message(prefix="Timed out waiting for service startup."))

    def stop(self) -> None:
        if not self.process:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None

    def _start_reader(
        self,
        stream: subprocess.PIPE[str] | None,
        sink: deque[str],
        *,
        capture_token: bool,
    ) -> None:
        if stream is None:
            return

        def _reader() -> None:
            for line in iter(stream.readline, ""):
                cleaned = line.rstrip()
                if cleaned:
                    sink.append(cleaned)
                    if capture_token and "SERVER_SESSION_TOKEN:" in cleaned:
                        token = cleaned.split("SERVER_SESSION_TOKEN:", 1)[1].strip()
                        if token:
                            self.api_key = token

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        self._threads.append(thread)

    def _is_service_listening(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=2.0)
        except httpx.HTTPError:
            return False
        return response.status_code in {200, 403}

    def _is_ready(self, api_key: str) -> bool:
        try:
            response = httpx.get(
                f"{self.base_url}/health",
                headers={"X-API-Key": api_key},
                timeout=2.0,
            )
        except httpx.HTTPError:
            return False
        return response.is_success

    def _failure_message(self, *, prefix: str = "Service failed to start.") -> str:
        stdout_lines = "\n".join(self._stdout_tail)
        stderr_lines = "\n".join(self._stderr_tail)
        parts = [prefix]
        if stdout_lines:
            parts.append(f"stdout:\n{stdout_lines}")
        if stderr_lines:
            parts.append(f"stderr:\n{stderr_lines}")
        return "\n\n".join(parts)
