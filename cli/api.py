from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx

from app.core.config import settings

DEV_SESSION_KEY_FILE = ".dev-session-key"
MAIL_PLUGIN_PREFIX = "/plugins/synvo_ai_mail"


class ApiError(RuntimeError):
    """Raised when the Local Cocoa API returns an error."""


def default_base_url() -> str:
    return f"http://{settings.endpoints.main_host}:{settings.endpoints.main_port}"


def default_runtime_root() -> Path:
    return settings.paths.runtime_root.resolve()


def default_dev_key_path() -> Path:
    return default_runtime_root() / DEV_SESSION_KEY_FILE


def load_api_key(explicit_key: str | None = None) -> str | None:
    if explicit_key:
        return explicit_key

    env_key = os.getenv("LOCAL_COCOA_API_KEY")
    if env_key:
        return env_key

    dev_key_path = default_dev_key_path()
    if dev_key_path.exists():
        try:
            return dev_key_path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    return None


class LocalCocoaClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        request_source: str = "external",
        timeout: float = 20.0,
    ) -> None:
        self.base_url = (base_url or default_base_url()).rstrip("/")
        self.api_key = load_api_key(api_key)
        self.request_source = request_source
        self.timeout = timeout

    def with_api_key(self, api_key: str) -> "LocalCocoaClient":
        return LocalCocoaClient(
            base_url=self.base_url,
            api_key=api_key,
            request_source=self.request_source,
            timeout=self.timeout,
        )

    def has_api_key(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ApiError(
                "No API key available. Set LOCAL_COCOA_API_KEY, run against a dev server with runtime/.dev-session-key, or start the service from this CLI."
            )
        return {
            "X-API-Key": self.api_key,
            "X-Request-Source": self.request_source,
        }

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        headers = kwargs.pop("headers", {})
        headers = {**self._headers(), **headers}
        timeout = kwargs.pop("timeout", self.timeout)

        try:
            with httpx.Client(timeout=timeout) as http_client:
                response = http_client.request(method, url, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(f"Request failed: {exc}") from exc

        if response.is_error:
            detail: str
            try:
                payload = response.json()
                detail = payload.get("detail") or payload.get("message") or str(payload)
            except ValueError:
                detail = response.text.strip() or response.reason_phrase
            raise ApiError(f"{response.status_code} {response.reason_phrase}: {detail}")

        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()
        return response.text

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def list_folders(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/folders")
        return payload.get("folders", [])

    def add_folder(self, path: str, *, label: str | None = None, scan_mode: str = "full") -> dict[str, Any]:
        return self._request(
            "POST",
            "/folders",
            json={"path": path, "label": label, "scan_mode": scan_mode},
        )

    def remove_folder(self, folder_id: str) -> None:
        self._request("DELETE", f"/folders/{folder_id}")

    def index_status(self) -> dict[str, Any]:
        return self._request("GET", "/index/status")

    def stage_progress(self) -> dict[str, Any]:
        return self._request("GET", "/index/stage-progress")

    def index_summary(self) -> dict[str, Any]:
        return self._request("GET", "/index/summary")

    def run_staged_index(self, *, folders: list[str] | None = None, files: list[str] | None = None, reindex: bool = False) -> dict[str, Any]:
        return self._request(
            "POST",
            "/index/run-staged",
            json={
                "mode": "reindex" if reindex else "rescan",
                "scope": "folder" if folders else "global",
                "folders": folders,
                "files": files,
            },
        )

    def start_semantic(self) -> dict[str, Any]:
        return self._request("POST", "/index/start-semantic")

    def stop_semantic(self) -> dict[str, Any]:
        return self._request("POST", "/index/stop-semantic")

    def start_deep(self) -> dict[str, Any]:
        return self._request("POST", "/index/start-deep")

    def stop_deep(self) -> dict[str, Any]:
        return self._request("POST", "/index/stop-deep")

    def deep_status(self) -> dict[str, Any]:
        return self._request("GET", "/index/deep-status")

    def providers_config(self) -> dict[str, Any]:
        return self._request("GET", "/providers/config")

    def update_providers(self, patch: dict[str, Any]) -> dict[str, Any]:
        return self._request("PUT", "/providers/config", json=patch)

    def models_status(self) -> dict[str, Any]:
        return self._request("GET", "/models/status")

    def update_models_config(self, patch: dict[str, Any]) -> dict[str, Any]:
        return self._request("PATCH", "/models/config", json=patch)

    def start_all_models(self) -> dict[str, Any]:
        return self._request("POST", "/models/start-all", timeout=300)

    def stop_all_models(self) -> dict[str, Any]:
        return self._request("POST", "/models/stop-all", timeout=120)

    def start_model(self, model_type: str) -> dict[str, Any]:
        return self._request("POST", f"/models/{model_type}/start", timeout=120)

    def stop_model(self, model_type: str) -> dict[str, Any]:
        return self._request("POST", f"/models/{model_type}/stop", timeout=60)

    def system_status(self) -> dict[str, Any]:
        return self._request("GET", "/system/status")

    def list_keys(self) -> list[dict[str, Any]]:
        return self._request("GET", "/security/keys")

    def create_key(self, name: str) -> dict[str, Any]:
        return self._request("POST", "/security/keys", params={"name": name})

    def delete_key(self, key: str) -> dict[str, Any]:
        return self._request("DELETE", f"/security/keys/{key}")

    def list_plugins(self) -> list[dict[str, Any]]:
        return self._request("GET", "/system/plugins")

    def settings(self) -> dict[str, Any]:
        return self._request("GET", "/settings/")

    def list_mail_accounts(self) -> list[dict[str, Any]]:
        return self._request("GET", f"{MAIL_PLUGIN_PREFIX}/accounts")

    def add_mail_account(
        self,
        *,
        label: str,
        protocol: str,
        host: str,
        port: int,
        username: str,
        password: str,
        use_ssl: bool,
        folder: str | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": label,
            "protocol": protocol,
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "use_ssl": use_ssl,
        }
        if folder is not None:
            payload["folder"] = folder
        return self._request("POST", f"{MAIL_PLUGIN_PREFIX}/accounts", json=payload)

    def remove_mail_account(self, account_id: str) -> None:
        self._request("DELETE", f"{MAIL_PLUGIN_PREFIX}/accounts/{account_id}")

    def sync_mail_account(self, account_id: str, *, limit: int = 100) -> dict[str, Any]:
        return self._request(
            "POST",
            f"{MAIL_PLUGIN_PREFIX}/accounts/{account_id}/sync",
            json={"limit": limit},
        )

    def start_outlook_auth(self, *, client_id: str = "", tenant_id: str = "") -> dict[str, Any]:
        return self._request(
            "POST",
            f"{MAIL_PLUGIN_PREFIX}/outlook/auth",
            json={"client_id": client_id, "tenant_id": tenant_id},
        )

    def get_outlook_auth_status(self, flow_id: str) -> dict[str, Any]:
        return self._request("GET", f"{MAIL_PLUGIN_PREFIX}/outlook/auth/{flow_id}")

    def complete_outlook_setup(self, *, flow_id: str, label: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"{MAIL_PLUGIN_PREFIX}/outlook/complete",
            json={"flow_id": flow_id, "label": label},
        )
