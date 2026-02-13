"""Automatic throttle monitor for the indexer.

Periodically checks system resources and pauses/resumes the heavy indexing
stages (fast_embed, deep) when the machine is under pressure or on battery.

Thresholds are configurable and stored alongside regular settings.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ThrottleConfig:
    """Tunable thresholds – values can be updated at runtime from the settings API."""
    cpu_max_percent: float = 80.0           # Pause when CPU > X% (excluding llama)
    memory_min_available_gb: float = 2.0    # Pause when available RAM < X GB
    memory_max_percent: float = 90.0        # Pause when RAM usage > X%
    gpu_max_percent: float = 85.0           # Pause when GPU utilisation > X%
    on_battery_pause: bool = True           # Pause when on battery
    cooldown_seconds: float = 30.0          # Seconds conditions must be clear before resuming
    poll_interval: float = 10.0             # How often to check (seconds)
    enabled: bool = True                    # Master switch


class SystemThrottleMonitor:
    """Background monitor that auto-pauses/resumes indexing stages."""

    def __init__(self, scheduler: "TwoRoundScheduler", config: Optional[ThrottleConfig] = None) -> None:  # noqa: F821
        self.scheduler = scheduler
        self.config = config or ThrottleConfig()
        self._task: Optional[asyncio.Task] = None
        self._stop = False

        # Track whether we paused each stage so we only resume what *we* paused
        self._paused_by_throttle: dict[str, bool] = {
            "fast_embed": False,
            "deep": False,
        }
        # Timestamp of when conditions last cleared
        self._clear_since: Optional[float] = None

        # ── Override state ──
        # When active, the throttle monitor skips pausing and resumes stages.
        self._override_active: bool = False
        self._override_until: Optional[float] = None  # monotonic timestamp

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> asyncio.Task:
        """Launch the background polling task."""
        self._stop = False
        self._task = asyncio.create_task(self._loop())
        logger.info("System throttle monitor started (poll=%.0fs)", self.config.poll_interval)
        return self._task

    def stop(self) -> None:
        self._stop = True
        if self._task:
            self._task.cancel()
            self._task = None

    # ------------------------------------------------------------------
    # Config update (callable from settings API)
    # ------------------------------------------------------------------

    def update_config(self, **kwargs: object) -> ThrottleConfig:
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        return self.config

    # ------------------------------------------------------------------
    # Override – temporarily bypass throttle
    # ------------------------------------------------------------------

    def set_override(self, duration_minutes: float = 30.0) -> dict:
        """Temporarily bypass the throttle for *duration_minutes*.

        Immediately resumes any stages that the throttle had paused.
        The override auto-expires after the specified duration.
        """
        self._override_active = True
        self._override_until = time.monotonic() + duration_minutes * 60
        self._do_resume()  # resume stages we paused
        logger.info(
            "Throttle override activated for %.0f min", duration_minutes,
        )
        return self.get_override_status()

    def cancel_override(self) -> dict:
        """Cancel an active override early."""
        self._override_active = False
        self._override_until = None
        logger.info("Throttle override cancelled")
        return self.get_override_status()

    def get_override_status(self) -> dict:
        remaining = 0.0
        if self._override_active and self._override_until is not None:
            remaining = max(0.0, self._override_until - time.monotonic())
            if remaining == 0:
                self._override_active = False
                self._override_until = None
        return {
            "active": self._override_active,
            "remaining_seconds": round(remaining, 0),
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        from routers.system_status import _build_snapshot, set_throttle_state

        while not self._stop:
            try:
                await asyncio.sleep(self.config.poll_interval)

                if not self.config.enabled:
                    set_throttle_state(False, None)
                    continue

                # ── Check override ──
                if self._override_active:
                    if self._override_until is not None and time.monotonic() >= self._override_until:
                        # Override expired
                        self._override_active = False
                        self._override_until = None
                        logger.info("Throttle override expired, resuming monitoring")
                    else:
                        # Override still active – skip throttle logic, keep stages running
                        set_throttle_state(False, None)
                        continue

                loop = asyncio.get_running_loop()
                snapshot = await loop.run_in_executor(None, _build_snapshot)

                reason = self._evaluate(snapshot)

                if reason:
                    self._clear_since = None
                    set_throttle_state(True, reason)
                    self._do_pause(reason)
                else:
                    # Conditions clear – start cooldown timer
                    now = time.monotonic()
                    if self._clear_since is None:
                        self._clear_since = now

                    elapsed = now - self._clear_since
                    if elapsed >= self.config.cooldown_seconds:
                        set_throttle_state(False, None)
                        self._do_resume()
                    else:
                        # Still in cooldown
                        remaining = self.config.cooldown_seconds - elapsed
                        set_throttle_state(True, f"Cooling down ({remaining:.0f}s)")

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Throttle monitor error: %s", exc)
                await asyncio.sleep(5)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, snapshot) -> Optional[str]:
        """Return a throttle reason string, or None if system is fine."""
        cfg = self.config

        # Battery check
        if cfg.on_battery_pause and snapshot.on_battery:
            return "On battery power"

        # CPU check (subtract llama-cpp usage to get "other" system load)
        effective_cpu = max(0, snapshot.cpu_percent - snapshot.llama_cpu_percent)
        if effective_cpu > cfg.cpu_max_percent:
            return f"CPU usage high ({effective_cpu:.0f}%)"

        # Memory checks
        if snapshot.memory_available_gb < cfg.memory_min_available_gb:
            return f"Low available memory ({snapshot.memory_available_gb:.1f} GB)"
        if snapshot.memory_percent > cfg.memory_max_percent:
            return f"Memory usage high ({snapshot.memory_percent:.0f}%)"

        # GPU check
        if snapshot.gpu_percent is not None and snapshot.gpu_percent > cfg.gpu_max_percent:
            return f"GPU usage high ({snapshot.gpu_percent:.0f}%)"

        return None

    # ------------------------------------------------------------------
    # Pause / Resume
    # ------------------------------------------------------------------

    def _do_pause(self, reason: str) -> None:
        """Pause heavy stages that we haven't already paused."""
        for stage in ("fast_embed", "deep"):
            if not self._paused_by_throttle[stage] and not self.scheduler.is_stage_paused(stage):
                self.scheduler.pause_stage(stage)
                self._paused_by_throttle[stage] = True
                logger.info("Throttle: paused %s – %s", stage, reason)

    def _do_resume(self) -> None:
        """Resume only stages that *we* paused (don't override user choice)."""
        for stage in ("fast_embed", "deep"):
            if self._paused_by_throttle[stage]:
                self.scheduler.resume_stage(stage)
                self._paused_by_throttle[stage] = False
                logger.info("Throttle: resumed %s (conditions cleared)", stage)
