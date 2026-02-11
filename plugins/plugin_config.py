"""
Plugin Config Helper
Provides a PluginConfig class that reads plugin.json and exposes
typed configuration values. Plugins should use this instead of
hardcoding plugin IDs, storage paths, etc.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from core.config import settings


class PluginConfig:
    """Plugin configuration loaded from plugin.json."""

    def __init__(self, plugin_dir: Path) -> None:
        manifest = plugin_dir / "plugin.json"
        with open(manifest, "r", encoding="utf-8") as f:
            self._data: Dict[str, Any] = json.load(f)
        self._plugin_dir = plugin_dir

    # ── core identifiers ──────────────────────────────────────

    @property
    def id(self) -> str:
        """Plugin ID from plugin.json (e.g. 'synvo_ai_mail')."""
        return self._data["id"]

    @property
    def name(self) -> str:
        """Human-readable plugin name."""
        return self._data["name"]

    @property
    def version(self) -> str:
        return self._data["version"]

    # ── paths ─────────────────────────────────────────────────

    @property
    def storage_root(self) -> Path:
        """
        Plugin-specific storage directory under the runtime root.
        Created automatically if it does not exist.
        """
        root = settings.paths.runtime_root / self.id
        root.mkdir(parents=True, exist_ok=True)
        return root

    @property
    def api_prefix(self) -> str:
        """API route prefix, e.g. '/plugins/synvo_ai_mail'."""
        return f"/plugins/{self.id}"

    # ── raw access ────────────────────────────────────────────

    @property
    def raw(self) -> Dict[str, Any]:
        """Full manifest dict for advanced usage."""
        return self._data


def load_plugin_config(backend_file: str) -> PluginConfig:
    """
    Convenience factory.  Call from any file inside ``<plugin>/backend/``:

        config = load_plugin_config(__file__)

    The helper walks up from the calling file to find the plugin root
    (the directory that contains ``plugin.json``).
    """
    current = Path(backend_file).resolve().parent
    # Walk up until we find plugin.json (max 3 levels)
    for _ in range(4):
        if (current / "plugin.json").exists():
            return PluginConfig(current)
        current = current.parent
    raise FileNotFoundError(
        f"Could not find plugin.json above {backend_file}"
    )
