from __future__ import annotations

import json
import shlex
import time
import webbrowser
from pathlib import Path

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner

from .api import ApiError, LocalCocoaClient
from .launcher import ServiceLauncher

# ── Preset / model config (mirrors local-cocoa/config/) ─────
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "local-cocoa" / "config"


def _load_presets() -> dict:
    path = _CONFIG_DIR / "models.preset.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _load_model_descriptors() -> list[dict]:
    path = _CONFIG_DIR / "models.config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get("models", [])
    return []


def _model_root_path() -> Path:
    """Return the model weights root directory used by the service.

    The settings value may be relative; resolve it first against the service
    directory and then fall back to the parent workspace ``runtime/`` where
    the Electron app stores downloaded models.
    """
    from app.core.config import settings as _settings

    raw = Path(_settings.paths.model_root_path)
    # If it's already absolute and exists, use it directly.
    if raw.is_absolute() and raw.exists():
        return raw

    service_root = Path(__file__).resolve().parent.parent
    # Try resolving relative to the service directory first.
    candidate = (service_root / raw).resolve()
    if candidate.exists():
        return candidate

    # Fall back to the parent workspace runtime (Electron layout).
    workspace_root = service_root.parent
    candidate = (workspace_root / raw).resolve()
    if candidate.exists():
        return candidate

    # Last resort: return the resolved-to-service-root path.
    return (service_root / raw).resolve()


def _resolve_model_path(descriptor: dict, root: Path) -> Path:
    return root / descriptor.get("relativePath", "")


def _file_present(descriptor: dict, root: Path) -> bool:
    p = _resolve_model_path(descriptor, root)
    return p.exists() and p.stat().st_size > 1024

# ── Default constants mirroring the frontend UI ──────────────
OUTLOOK_DEFAULT_CLIENT_ID = "f0f434e5-80fb-4db9-823c-36707ec98470"
OUTLOOK_DEFAULT_TENANT_ID = "common"


def _default_mail_port(protocol: str, use_ssl: bool) -> int:
    if protocol == "imap":
        return 993 if use_ssl else 143
    return 995 if use_ssl else 110


def _prompt(console: Console, label: str, *, default: str = "", password: bool = False) -> str:
    """Prompt for a value with optional default."""
    suffix = f" [{default}]" if default else ""
    value = console.input(f"  {label}{suffix}: ", password=password).strip()
    return value if value else default


# ─────────────────────────────────────────────────────────────
#  Main dashboard shell with numbered-menu navigation
# ─────────────────────────────────────────────────────────────
class InteractiveShell:
    def __init__(
        self,
        *,
        client: LocalCocoaClient,
        console: Console,
        launcher: ServiceLauncher | None = None,
        owns_service: bool = False,
    ) -> None:
        self.client = client
        self.console = console
        self.launcher = launcher
        self.owns_service = owns_service
        self.running = True

    # ── Entry point ──────────────────────────────────────────
    def run(self) -> None:
        self._show_dashboard()
        while self.running:
            try:
                self._show_main_menu()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                break
        if self.owns_service and self.launcher:
            self.launcher.stop()

    # ── Dashboard (landing page) ─────────────────────────────
    def _show_dashboard(self) -> None:
        self.console.clear()
        try:
            health = self.client.health()
            system = self.client.system_status()
            index = self.client.index_status()
            summary = self.client.index_summary()
            providers = self.client.providers_config()
            deep = self.client.deep_status()
        except ApiError as exc:
            self.console.print(Panel(f"[red]Service unreachable:[/red] {exc}", title="Local Cocoa CLI", border_style="red"))
            return

        # -- Status block
        status_table = Table.grid(padding=(0, 2))
        status_table.add_column(style="bold")
        status_table.add_column()
        status_table.add_row("Health", f"[green]{health.get('status', 'ok')}[/green]")
        status_table.add_row("URL", self.client.base_url)
        status_table.add_row("API Key", "[green]loaded[/green]" if self.client.has_api_key() else "[yellow]none[/yellow]")
        status_table.add_row("CPU / Mem", f"{system.get('cpu_percent', '?')}% / {system.get('memory_percent', '?')}%")

        # -- Index block
        idx_table = Table.grid(padding=(0, 2))
        idx_table.add_column(style="bold")
        idx_table.add_column()
        idx_table.add_row("Status", str(index.get("status", "unknown")))
        idx_table.add_row("Indexed Files", str(summary.get("files_indexed", 0)))
        idx_table.add_row("Folders", str(summary.get("folders_indexed", 0)))
        idx_table.add_row("Deep Enabled", str(deep.get("deep_enabled", False)))

        # -- Provider block
        prov_table = Table.grid(padding=(0, 2))
        prov_table.add_column(style="bold")
        prov_table.add_column()
        prov_table.add_row("LLM", str(providers.get("llm_provider", "local")))
        prov_table.add_row("Rerank", str(providers.get("rerank_provider", "local")))
        remote_llm = providers.get("remote_llm") or {}
        if remote_llm.get("model"):
            prov_table.add_row("Remote LLM", str(remote_llm.get("model", "")))

        panels = Columns(
            [
                Panel(status_table, title="Service", border_style="cyan", expand=True),
                Panel(idx_table, title="Index", border_style="green", expand=True),
                Panel(prov_table, title="Providers", border_style="magenta", expand=True),
            ],
            equal=True,
        )
        self.console.print()
        self.console.print(Panel(panels, title="[bold]Local Cocoa CLI[/bold]", border_style="bright_blue", padding=(1, 2)))

    # ── Main menu ────────────────────────────────────────────
    MAIN_MENU_ITEMS = [
        ("1", "Dashboard",  "Refresh the main status dashboard"),
        ("2", "Folders",    "List, add, or remove indexed folders"),
        ("3", "Indexing",   "Run indexing, toggle semantic / deep"),
        ("4", "Providers",  "View or switch LLM / rerank providers"),
        ("5", "Models",     "Show local model runtime states"),
        ("6", "Mail",       "Manage email connectors"),
        ("7", "API Keys",   "List or create API keys"),
        ("8", "Plugins",    "Show loaded plugins"),
        ("9", "Watch",      "Live-refresh dashboard every 2 s"),
        ("0", "Quit",       "Exit the CLI"),
    ]

    def _show_main_menu(self) -> None:
        menu = Table.grid(padding=(0, 3))
        menu.add_column(style="bold cyan", justify="right")
        menu.add_column(style="bold")
        menu.add_column(style="dim")
        for key, label, desc in self.MAIN_MENU_ITEMS:
            menu.add_row(f"[{key}]", label, desc)
        self.console.print()
        self.console.print(menu)
        choice = self.console.input("\n[bold cyan]Select[/bold cyan] > ").strip()
        try:
            self._handle_main_choice(choice)
        except ApiError as exc:
            self.console.print(f"[red]{exc}[/red]")
        except Exception as exc:  # noqa: BLE001
            self.console.print(f"[red]Error:[/red] {exc}")

    def _handle_main_choice(self, choice: str) -> None:
        dispatch = {
            "1": self._show_dashboard,
            "2": self._folders_page,
            "3": self._indexing_page,
            "4": self._providers_page,
            "5": self._models_page,
            "6": self._mail_page,
            "7": self._keys_page,
            "8": self._plugins_page,
            "9": self._watch_status,
            "0": self._quit,
            "q": self._quit,
        }
        fn = dispatch.get(choice)
        if fn:
            fn()
        else:
            self.console.print(f"[yellow]Unknown choice:[/yellow] {choice}")

    def _quit(self) -> None:
        self.running = False

    # ── Helpers ──────────────────────────────────────────────
    def _pause(self) -> None:
        self.console.input("\n[dim]Press Enter to return …[/dim]")

    def _sub_menu(self, title: str, items: list[tuple[str, str]]) -> str:
        self.console.print(f"\n[bold underline]{title}[/bold underline]")
        menu = Table.grid(padding=(0, 3))
        menu.add_column(style="bold cyan", justify="right")
        menu.add_column()
        for key, label in items:
            menu.add_row(f"\\[{key}]", label)
        self.console.print(menu)
        return self.console.input("[bold cyan]Select[/bold cyan] > ").strip()

    # ── Folders page ─────────────────────────────────────────
    def _folders_page(self) -> None:
        folders = self.client.list_folders()
        table = Table(title="Registered Folders")
        table.add_column("ID")
        table.add_column("Label")
        table.add_column("Path")
        table.add_column("Mode")
        table.add_column("Indexed")
        for f in folders:
            table.add_row(
                str(f.get("id", "")),
                str(f.get("label", "")),
                str(f.get("path", "")),
                str(f.get("scan_mode", "full")),
                str(f.get("indexed_count", 0)),
            )
        self.console.print(table)

        choice = self._sub_menu("Folders", [("a", "Add folder"), ("r", "Remove folder"), ("b", "Back")])
        if choice == "a":
            path_raw = _prompt(self.console, "Folder path")
            if not path_raw:
                return
            path = str(Path(path_raw).expanduser().resolve())
            label = _prompt(self.console, "Label (optional)")
            result = self.client.add_folder(path, label=label or None)
            self.console.print(f"[green]Added:[/green] {result.get('label')} ({result.get('id')})")
            self._pause()
        elif choice == "r":
            fid = _prompt(self.console, "Folder ID to remove")
            if not fid:
                return
            self.client.remove_folder(fid)
            self.console.print(f"[green]Removed:[/green] {fid}")
            self._pause()

    # ── Indexing page ────────────────────────────────────────
    def _indexing_page(self) -> None:
        index = self.client.index_status()
        progress = self.client.stage_progress()
        deep = self.client.deep_status()

        info = Table.grid(padding=(0, 2))
        info.add_column(style="bold")
        info.add_column()
        info.add_row("Index Status", str(index.get("status", "unknown")))
        info.add_row("Deep Enabled", str(deep.get("deep_enabled", False)))
        self.console.print(Panel(info, title="Indexing"))

        stage_table = Table(title="Stage Progress")
        stage_table.add_column("Stage")
        stage_table.add_column("Done", justify="right")
        stage_table.add_column("Pending", justify="right")
        stage_table.add_column("Error", justify="right")
        stage_table.add_column("%", justify="right")
        for stage in ("fast_text", "fast_embed", "deep"):
            item = progress.get(stage, {})
            stage_table.add_row(
                stage,
                str(item.get("done", 0)),
                str(item.get("pending", 0)),
                str(item.get("error", 0)),
                f"{item.get('percent', 0)}%",
            )
        self.console.print(stage_table)

        choice = self._sub_menu("Indexing", [
            ("r", "Run staged indexing"),
            ("s", "Toggle semantic on/off"),
            ("d", "Toggle deep on/off"),
            ("b", "Back"),
        ])
        if choice == "r":
            result = self.client.run_staged_index()
            self.console.print(f"[green]{result.get('message', 'Indexing started.')}[/green]")
            self._pause()
        elif choice == "s":
            toggle = _prompt(self.console, "Semantic (on/off)")
            if toggle in {"on", "off"}:
                r = self.client.start_semantic() if toggle == "on" else self.client.stop_semantic()
                self.console.print(f"[green]{r.get('message', 'Updated.')}[/green]")
                self._pause()
        elif choice == "d":
            toggle = _prompt(self.console, "Deep (on/off)")
            if toggle in {"on", "off"}:
                r = self.client.start_deep() if toggle == "on" else self.client.stop_deep()
                self.console.print(f"[green]{r.get('message', 'Updated.')}[/green]")
                self._pause()

    # ── Providers page ───────────────────────────────────────
    def _providers_page(self) -> None:
        providers = self.client.providers_config()
        table = Table(title="Provider Configuration")
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_row("llm_provider", str(providers.get("llm_provider", "local")))
        table.add_row("rerank_provider", str(providers.get("rerank_provider", "local")))
        remote_llm = providers.get("remote_llm") or {}
        remote_rerank = providers.get("remote_rerank") or {}
        table.add_row("remote_llm.base_url", str(remote_llm.get("base_url", "")))
        table.add_row("remote_llm.model", str(remote_llm.get("model", "")))
        table.add_row("remote_rerank.base_url", str(remote_rerank.get("base_url", "")))
        table.add_row("remote_rerank.model", str(remote_rerank.get("model", "")))
        self.console.print(table)

        choice = self._sub_menu("Providers", [
            ("r", "Set remote LLM provider"),
            ("l", "Switch back to local"),
            ("b", "Back"),
        ])
        if choice == "r":
            base = _prompt(self.console, "Remote base URL")
            model = _prompt(self.console, "Model name")
            key = _prompt(self.console, "API key (optional)")
            if base and model:
                patch: dict = {
                    "llm_provider": "remote",
                    "remote_llm": {"base_url": base, "model": model, "api_key": key},
                }
                self.console.print_json(data=self.client.update_providers(patch))
            self._pause()
        elif choice == "l":
            self.console.print_json(data=self.client.update_providers({"llm_provider": "local"}))
            self._pause()

    # ── Models page ──────────────────────────────────────────
    def _models_page(self) -> None:
        models = self.client.models_status()
        table = Table(title="Running Models")
        table.add_column("Model")
        table.add_column("State")
        table.add_column("Port")
        for name, payload in models.items():
            state = str(payload.get("state", "unknown"))
            style = "green" if state == "running" else ("red" if state == "error" else "yellow")
            table.add_row(name, f"[{style}]{state}[/{style}]", str(payload.get("port", "-")))
        self.console.print(table)

        # Show available weight files
        try:
            root = _model_root_path()
            descriptors = _load_model_descriptors()
            if descriptors:
                wt = Table(title=f"Weight Files  ({root})")
                wt.add_column("ID")
                wt.add_column("Label")
                wt.add_column("Type")
                wt.add_column("Status")
                for d in descriptors:
                    present = _file_present(d, root)
                    tag = "[green]downloaded[/green]" if present else "[dim]missing[/dim]"
                    wt.add_row(d.get("id", ""), d.get("label", ""), d.get("type", ""), tag)
                self.console.print(wt)
        except Exception:
            pass

        choice = self._sub_menu("Models", [
            ("p", "Apply a preset (eco / balanced / pro)"),
            ("s", "Warm up all models"),
            ("x", "Stop all models"),
            ("b", "Back"),
        ])
        if choice == "p":
            self._apply_preset()
        elif choice == "s":
            with Live(Spinner("dots", text="Starting all models …"), console=self.console, transient=True):
                self.client.start_all_models()
            self.console.print("[green]All models started.[/green]")
            self._pause()
        elif choice == "x":
            with Live(Spinner("dots", text="Stopping all models …"), console=self.console, transient=True):
                self.client.stop_all_models()
            self.console.print("[green]All models stopped.[/green]")
            self._pause()

    def _apply_preset(self) -> None:
        preset_data = _load_presets()
        presets = preset_data.get("presets", {})
        descriptors = _load_model_descriptors()
        desc_by_id = {d["id"]: d for d in descriptors}

        try:
            root = _model_root_path()
        except Exception:
            root = None

        if not presets:
            self.console.print("[red]No presets found.[/red]")
            return

        # Show preset table with download status
        pt = Table(title="Available Presets")
        pt.add_column("Key", style="bold cyan")
        pt.add_column("Label")
        pt.add_column("VRAM")
        pt.add_column("Models")
        pt.add_column("Ready?")

        for key, info in presets.items():
            model_ids = info.get("models", {})
            # Collect all model ids including companion mmproj
            all_ids: list[str] = []
            for mid in model_ids.values():
                all_ids.append(mid)
                companion = desc_by_id.get(mid, {}).get("mmprojId")
                if companion:
                    all_ids.append(companion)

            if root:
                present = sum(1 for mid in all_ids if _file_present(desc_by_id.get(mid, {}), root))
                total = len(all_ids)
                ready = f"[green]{present}/{total}[/green]" if present == total else f"[yellow]{present}/{total}[/yellow]"
            else:
                ready = "[dim]?[/dim]"

            names = ", ".join(f"{k}={v}" for k, v in model_ids.items())
            pt.add_row(key, info.get("label", ""), info.get("estimatedVram", ""), names, ready)

        self.console.print(pt)

        choice = _prompt(self.console, "Preset to apply (eco/balanced/pro)")
        if choice not in presets:
            self.console.print("[yellow]Cancelled.[/yellow]")
            return

        selected = presets[choice]
        model_ids = selected["models"]

        # Check missing files
        missing: list[str] = []
        if root:
            for mid in model_ids.values():
                desc = desc_by_id.get(mid, {})
                if not _file_present(desc, root):
                    missing.append(mid)
                companion = desc.get("mmprojId")
                if companion and not _file_present(desc_by_id.get(companion, {}), root):
                    missing.append(companion)

        if missing:
            self.console.print(f"[yellow]Warning:[/yellow] Missing weight files: {', '.join(missing)}")
            self.console.print("[dim]Models will fail to start until these files are downloaded.[/dim]")
            proceed = _prompt(self.console, "Apply anyway? (y/n)", default="n")
            if proceed.lower() not in {"y", "yes"}:
                return

        # Resolve model IDs → absolute file paths and PATCH the backend
        patch: dict[str, str] = {}
        vlm_id = model_ids.get("vlm")
        if vlm_id and vlm_id in desc_by_id:
            vlm_desc = desc_by_id[vlm_id]
            if root:
                patch["vlm_model"] = str(_resolve_model_path(vlm_desc, root))
            companion = vlm_desc.get("mmprojId")
            if companion and companion in desc_by_id and root:
                patch["vlm_mmproj"] = str(_resolve_model_path(desc_by_id[companion], root))

        emb_id = model_ids.get("embedding")
        if emb_id and emb_id in desc_by_id and root:
            patch["embedding_model"] = str(_resolve_model_path(desc_by_id[emb_id], root))

        rerank_id = model_ids.get("reranker")
        if rerank_id and rerank_id in desc_by_id and root:
            patch["rerank_model"] = str(_resolve_model_path(desc_by_id[rerank_id], root))

        whisper_id = model_ids.get("whisper")
        if whisper_id and whisper_id in desc_by_id and root:
            patch["whisper_model"] = str(_resolve_model_path(desc_by_id[whisper_id], root))

        if patch:
            result = self.client.update_models_config(patch)
            self.console.print(f"[green]Applied preset '{choice}'.[/green]")
            self.console.print_json(data=result)

            reload_now = _prompt(
                self.console,
                "Stop currently running models so the new preset is used on next request? (y/n)",
                default="y",
            )
            if reload_now.lower() in {"y", "yes", ""}:
                with Live(Spinner("dots", text="Stopping running models …"), console=self.console, transient=True):
                    self.client.stop_all_models()
                self.console.print("[green]Preset applied. Models will start lazily on demand with the new weights.[/green]")
            else:
                self.console.print("[yellow]Preset saved. Existing running models keep their current process until restarted.[/yellow]")
        else:
            self.console.print("[yellow]No paths to update.[/yellow]")
        self._pause()

    # ── Mail page ────────────────────────────────────────────
    def _mail_page(self) -> None:
        accounts = self.client.list_mail_accounts()
        table = Table(title="Email Accounts")
        table.add_column("ID")
        table.add_column("Label")
        table.add_column("Protocol")
        table.add_column("Username")
        table.add_column("Messages", justify="right")
        table.add_column("Last Sync")
        for a in accounts:
            table.add_row(
                str(a.get("id", "")),
                str(a.get("label", "")),
                str(a.get("protocol", "")),
                str(a.get("username", "")),
                str(a.get("total_messages", 0)),
                str(a.get("last_synced_at", "") or "-"),
            )
        self.console.print(table)

        choice = self._sub_menu("Mail", [
            ("a", "Add standard IMAP/POP3 account"),
            ("o", "Add Outlook account"),
            ("s", "Sync an account"),
            ("r", "Remove an account"),
            ("b", "Back"),
        ])
        if choice == "a":
            self._mail_add_standard()
        elif choice == "o":
            self._mail_add_outlook()
        elif choice == "s":
            aid = _prompt(self.console, "Account ID to sync")
            if aid:
                result = self.client.sync_mail_account(aid)
                self.console.print(f"[green]{result.get('message', 'Sync done.')}[/green]")
                self._pause()
        elif choice == "r":
            aid = _prompt(self.console, "Account ID to remove")
            if aid:
                self.client.remove_mail_account(aid)
                self.console.print(f"[green]Removed: {aid}[/green]")
                self._pause()

    def _mail_add_standard(self) -> None:
        self.console.print("\n[bold underline]Add Standard Email Account[/bold underline]")
        label = _prompt(self.console, "Label")
        protocol = _prompt(self.console, "Protocol (imap/pop3)", default="imap").lower()
        if protocol not in {"imap", "pop3"}:
            self.console.print("[red]Invalid protocol.[/red]")
            return
        host = _prompt(self.console, "Mail server host")
        username = _prompt(self.console, "Username / email")
        password = _prompt(self.console, "Password / app password", password=True)
        ssl_str = _prompt(self.console, "Use SSL? (y/n)", default="y").lower()
        use_ssl = ssl_str in {"y", "yes", "true", "1"}
        port_str = _prompt(self.console, "Port", default=str(_default_mail_port(protocol, use_ssl)))
        port = int(port_str) if port_str.isdigit() else _default_mail_port(protocol, use_ssl)
        folder = _prompt(self.console, "IMAP folder", default="INBOX") if protocol == "imap" else None

        if not (label and host and username and password):
            self.console.print("[red]Missing required fields.[/red]")
            return

        result = self.client.add_mail_account(
            label=label,
            protocol=protocol,
            host=host,
            port=port,
            username=username,
            password=password,
            use_ssl=use_ssl,
            folder=folder,
        )
        self.console.print("[green]Account added.[/green]")
        self.console.print_json(data=result)
        self._pause()

    def _mail_add_outlook(self) -> None:
        self.console.print("\n[bold underline]Add Outlook Account (Device-Code Flow)[/bold underline]")
        label = _prompt(self.console, "Label", default="My Outlook")
        client_id = _prompt(self.console, "Client ID", default=OUTLOOK_DEFAULT_CLIENT_ID)
        tenant_id = _prompt(self.console, "Tenant ID", default=OUTLOOK_DEFAULT_TENANT_ID)

        started = self.client.start_outlook_auth(client_id=client_id, tenant_id=tenant_id)
        flow_id = str(started.get("flow_id", "")).strip()
        if not flow_id:
            raise ApiError("Outlook auth did not return a flow_id.")

        self.console.print(f"[cyan]Device-code auth started:[/cyan] {flow_id}")
        deadline = time.time() + 180
        code_shown = False

        while time.time() < deadline:
            status = self.client.get_outlook_auth_status(flow_id)
            state_value = str(status.get("status", ""))
            info = status.get("info") or {}

            if state_value == "code_ready":
                verification_uri = str(info.get("verification_uri", ""))
                user_code = str(info.get("user_code", ""))
                if not code_shown:
                    self.console.print(f"\n  [bold]Open:[/bold] {verification_uri}")
                    self.console.print(f"  [bold]Code:[/bold] {user_code}")
                    self.console.print("  [dim]Waiting for you to authorize in the browser …[/dim]\n")
                    if verification_uri:
                        try:
                            webbrowser.open(verification_uri)
                        except Exception:
                            pass
                    code_shown = True
            elif state_value == "authenticated":
                account = self.client.complete_outlook_setup(flow_id=flow_id, label=label.strip())
                self.console.print("[green]Outlook account connected![/green]")
                self.console.print_json(data=account)
                # Auto-sync after connect, matching the frontend behaviour
                account_id = str(account.get("id", ""))
                if account_id:
                    sync_result = self.client.sync_mail_account(account_id)
                    self.console.print(f"[green]Initial sync done. {sync_result.get('message', '')}[/green]")
                self._pause()
                return
            elif state_value == "error":
                raise ApiError(str(status.get("message") or "Outlook authentication failed."))

            time.sleep(2)

        raise ApiError("Timed out waiting for Outlook device-code authentication.")

    # ── Keys page ────────────────────────────────────────────
    def _keys_page(self) -> None:
        keys = self.client.list_keys()
        table = Table(title="API Keys")
        table.add_column("Name")
        table.add_column("Key")
        table.add_column("Active")
        table.add_column("System")
        for item in keys:
            table.add_row(
                str(item.get("name", "")),
                str(item.get("key", "")),
                str(item.get("is_active", False)),
                str(item.get("is_system", False)),
            )
        self.console.print(table)

        choice = self._sub_menu("API Keys", [("c", "Create key"), ("b", "Back")])
        if choice == "c":
            name = _prompt(self.console, "Key name")
            if name:
                created = self.client.create_key(name)
                self.console.print(f"[green]Created:[/green] {created.get('key')}")
                self._pause()

    # ── Plugins page ─────────────────────────────────────────
    def _plugins_page(self) -> None:
        plugins = self.client.list_plugins()
        table = Table(title="Loaded Plugins")
        table.add_column("Name")
        table.add_column("Version")
        table.add_column("Enabled")
        for p in plugins:
            table.add_row(
                str(p.get("name", "")),
                str(p.get("version", "")),
                str(p.get("enabled", True)),
            )
        self.console.print(table)
        self._pause()

    # ── Watch mode ───────────────────────────────────────────
    def _watch_status(self) -> None:
        self.console.print("[cyan]Live dashboard. Press Ctrl+C to stop.[/cyan]")
        try:
            while True:
                self._show_dashboard()
                time.sleep(2)
        except KeyboardInterrupt:
            self.console.print()
