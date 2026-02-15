"""System resource monitoring endpoint.

Provides real-time CPU, GPU, memory, and battery status so the frontend
can display system health and the throttle monitor can auto-pause indexing.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import subprocess
import time
from typing import Optional

import psutil
from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["system"])

# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class SystemResourceStatus(BaseModel):
    """Snapshot of system resource utilisation."""
    cpu_percent: float = 0.0
    cpu_core_count: int = 1
    gpu_percent: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    on_battery: bool = False
    battery_percent: Optional[float] = None
    # llama-cpp process resource usage (so UI can show "system minus our backend")
    llama_cpu_percent: float = 0.0
    llama_memory_mb: float = 0.0
    # Throttle state (set by the throttle monitor, reflected here for convenience)
    throttled: bool = False
    throttle_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Cache – avoid hammering psutil every request
# ---------------------------------------------------------------------------
_cache: Optional[SystemResourceStatus] = None
_cache_ts: float = 0.0
_CACHE_TTL = 2.0  # seconds

# External throttle state (written by throttle.py)
_throttle_state: dict = {"throttled": False, "reason": None}


def set_throttle_state(throttled: bool, reason: Optional[str] = None) -> None:
    """Called by the throttle monitor to publish state."""
    _throttle_state["throttled"] = throttled
    _throttle_state["reason"] = reason


def get_throttle_state() -> dict:
    return dict(_throttle_state)


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

def _collect_cpu() -> float:
    """Non-blocking CPU percent (averaged over a very short interval)."""
    return psutil.cpu_percent(interval=0)


def _collect_memory() -> dict:
    vm = psutil.virtual_memory()
    return {
        "percent": vm.percent,
        "used_gb": round(vm.used / (1024 ** 3), 2),
        "total_gb": round(vm.total / (1024 ** 3), 2),
        "available_gb": round(vm.available / (1024 ** 3), 2),
    }


def _collect_battery() -> dict:
    bat = psutil.sensors_battery()
    if bat is None:
        return {"on_battery": False, "percent": None}
    return {
        "on_battery": not bat.power_plugged,
        "percent": round(bat.percent, 1),
    }


def _collect_llama_processes() -> dict:
    """Find llama-server / llama-cli processes and sum their resource usage."""
    cpu = 0.0
    mem_mb = 0.0
    llama_names = {"llama-server", "llama-cli", "llama_server", "llama_cli"}
    try:
        for proc in psutil.process_iter(["name", "cpu_percent", "memory_info"]):
            try:
                name = (proc.info["name"] or "").lower().replace(".exe", "")
                if any(n in name for n in llama_names):
                    cpu += proc.info.get("cpu_percent") or 0.0
                    mem_info = proc.info.get("memory_info")
                    if mem_info:
                        mem_mb += mem_info.rss / (1024 ** 2)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass
    return {"cpu": round(cpu, 1), "mem_mb": round(mem_mb, 1)}


def _collect_gpu_macos() -> dict:
    """Best-effort GPU utilisation on macOS via ioreg."""
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            capture_output=True, text=True, timeout=3,
        )
        text = result.stdout
        # Parse "Device Utilization %" or "GPU Core Utilization" values
        import re
        util_match = re.search(r'"Device Utilization %"\s*=\s*(\d+)', text)
        if util_match:
            return {"gpu_percent": float(util_match.group(1)), "gpu_memory_percent": None}
        # Fallback: try GPU Activity Monitor
        util_match = re.search(r'"GPU Activity\(%\)"\s*=\s*(\d+)', text)
        if util_match:
            return {"gpu_percent": float(util_match.group(1)), "gpu_memory_percent": None}
    except Exception:
        pass
    return {"gpu_percent": None, "gpu_memory_percent": None}


def _collect_gpu_nvidia() -> dict:
    """GPU utilisation via nvidia-smi (Linux/Windows)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                return {
                    "gpu_percent": float(parts[0].strip()),
                    "gpu_memory_percent": float(parts[1].strip()),
                }
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return {"gpu_percent": None, "gpu_memory_percent": None}


def _collect_gpu() -> dict:
    system = platform.system()
    if system == "Darwin":
        return _collect_gpu_macos()
    else:
        return _collect_gpu_nvidia()


def _build_snapshot() -> SystemResourceStatus:
    cpu = _collect_cpu()
    mem = _collect_memory()
    bat = _collect_battery()
    gpu = _collect_gpu()
    llama = _collect_llama_processes()
    ts = get_throttle_state()

    return SystemResourceStatus(
        cpu_percent=cpu,
        cpu_core_count=psutil.cpu_count(logical=True) or 1,
        gpu_percent=gpu["gpu_percent"],
        gpu_memory_percent=gpu["gpu_memory_percent"],
        memory_percent=mem["percent"],
        memory_used_gb=mem["used_gb"],
        memory_total_gb=mem["total_gb"],
        memory_available_gb=mem["available_gb"],
        on_battery=bat["on_battery"],
        battery_percent=bat["percent"],
        llama_cpu_percent=llama["cpu"],
        llama_memory_mb=llama["mem_mb"],
        throttled=ts["throttled"],
        throttle_reason=ts["reason"],
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get("/status", response_model=SystemResourceStatus)
async def get_system_status() -> SystemResourceStatus:
    """Return current system resource utilisation (cached 2 s)."""
    global _cache, _cache_ts
    now = time.perf_counter()
    if _cache is not None and (now - _cache_ts) < _CACHE_TTL:
        return _cache

    loop = asyncio.get_running_loop()
    snapshot = await loop.run_in_executor(None, _build_snapshot)
    _cache = snapshot
    _cache_ts = time.perf_counter()
    return snapshot


# ---------------------------------------------------------------------------
# Throttle configuration endpoints
# ---------------------------------------------------------------------------

# Reference to the live throttle monitor (set at startup from app.py)
_throttle_monitor: Optional["SystemThrottleMonitor"] = None  # noqa: F821


def set_throttle_monitor(monitor) -> None:
    """Called once at startup so the config endpoints can access the monitor."""
    global _throttle_monitor
    _throttle_monitor = monitor


class ThrottleConfigResponse(BaseModel):
    enabled: bool = True
    cpu_max_percent: float = 80.0
    memory_min_available_gb: float = 2.0
    memory_max_percent: float = 90.0
    gpu_max_percent: float = 85.0
    on_battery_pause: bool = True
    cooldown_seconds: float = 30.0


@router.get("/throttle-config", response_model=ThrottleConfigResponse)
async def get_throttle_config() -> ThrottleConfigResponse:
    """Return the current auto-throttle thresholds."""
    if _throttle_monitor is None:
        return ThrottleConfigResponse()
    cfg = _throttle_monitor.config
    return ThrottleConfigResponse(
        enabled=cfg.enabled,
        cpu_max_percent=cfg.cpu_max_percent,
        memory_min_available_gb=cfg.memory_min_available_gb,
        memory_max_percent=cfg.memory_max_percent,
        gpu_max_percent=cfg.gpu_max_percent,
        on_battery_pause=cfg.on_battery_pause,
        cooldown_seconds=cfg.cooldown_seconds,
    )


@router.patch("/throttle-config", response_model=ThrottleConfigResponse)
async def update_throttle_config(body: dict) -> ThrottleConfigResponse:
    """Update one or more throttle thresholds at runtime."""
    if _throttle_monitor is None:
        return ThrottleConfigResponse()
    _throttle_monitor.update_config(**body)
    return await get_throttle_config()


# ---------------------------------------------------------------------------
# Throttle override endpoints
# ---------------------------------------------------------------------------

class ThrottleOverrideRequest(BaseModel):
    duration_minutes: float = 30.0


class ThrottleOverrideResponse(BaseModel):
    active: bool = False
    remaining_seconds: float = 0.0


@router.post("/throttle-override", response_model=ThrottleOverrideResponse)
async def activate_throttle_override(body: ThrottleOverrideRequest = ThrottleOverrideRequest()) -> ThrottleOverrideResponse:
    """Temporarily bypass the auto-throttle for *duration_minutes*.

    Immediately resumes any stages that the throttle had paused.
    """
    if _throttle_monitor is None:
        return ThrottleOverrideResponse()
    result = _throttle_monitor.set_override(body.duration_minutes)
    return ThrottleOverrideResponse(**result)


@router.delete("/throttle-override", response_model=ThrottleOverrideResponse)
async def cancel_throttle_override() -> ThrottleOverrideResponse:
    """Cancel an active throttle override early."""
    if _throttle_monitor is None:
        return ThrottleOverrideResponse()
    result = _throttle_monitor.cancel_override()
    return ThrottleOverrideResponse(**result)


@router.get("/throttle-override", response_model=ThrottleOverrideResponse)
async def get_throttle_override_status() -> ThrottleOverrideResponse:
    """Check whether a throttle override is active."""
    if _throttle_monitor is None:
        return ThrottleOverrideResponse()
    result = _throttle_monitor.get_override_status()
    return ThrottleOverrideResponse(**result)
