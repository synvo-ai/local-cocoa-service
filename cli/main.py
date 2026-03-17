from __future__ import annotations

import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .api import ApiError, LocalCocoaClient
from .launcher import ServiceLauncher
from .shell import InteractiveShell

app = typer.Typer(
    add_completion=False,
    help="Standalone operator CLI for Local Cocoa Service.",
    rich_markup_mode="rich",
)
folders_app = typer.Typer(help="Folder management commands.")
index_app = typer.Typer(help="Indexing commands.")
keys_app = typer.Typer(help="API key commands.")
providers_app = typer.Typer(help="Provider configuration commands.")
mail_app = typer.Typer(help="Email account management commands.")

app.add_typer(folders_app, name="folders")
app.add_typer(index_app, name="index")
app.add_typer(keys_app, name="keys")
app.add_typer(providers_app, name="providers")
app.add_typer(mail_app, name="mail")

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTLOOK_DEFAULT_CLIENT_ID = "f0f434e5-80fb-4db9-823c-36707ec98470"
OUTLOOK_DEFAULT_TENANT_ID = "common"


@dataclass
class CliContext:
    client: LocalCocoaClient
    launcher: ServiceLauncher


def _default_mail_port(protocol: str, use_ssl: bool) -> int:
    if protocol == "imap":
        return 993 if use_ssl else 143
    return 995 if use_ssl else 110


def _render_mail_accounts(console: Console, accounts: list[dict[str, object]]) -> None:
    table = Table(title="Email Accounts")
    table.add_column("ID")
    table.add_column("Label")
    table.add_column("Protocol")
    table.add_column("Username")
    table.add_column("Messages")
    table.add_column("Status")
    for item in accounts:
        table.add_row(
            str(item.get("id", "")),
            str(item.get("label", "")),
            str(item.get("protocol", "")),
            str(item.get("username", "")),
            str(item.get("total_messages", 0)),
            str(item.get("last_sync_status", "") or "-"),
        )
    console.print(table)


def _lookup_root_option(ctx: typer.Context, key: str) -> str | None:
    current: typer.Context | None = ctx
    while current is not None:
        if isinstance(current.obj, dict) and key in current.obj:
            return current.obj.get(key)
        current = current.parent
    return None


def _get_context(
    ctx: typer.Context,
    *,
    ensure_started: bool,
    timeout: float = 30.0,
) -> CliContext:
    stored = ctx.obj
    if isinstance(stored, CliContext):
        if ensure_started:
            result = stored.launcher.ensure_running(api_key=stored.client.api_key, timeout=timeout)
            if result.api_key:
                stored.client = stored.client.with_api_key(result.api_key)
        return stored

    base_url = _lookup_root_option(ctx, "url")
    api_key = _lookup_root_option(ctx, "api_key")
    client = LocalCocoaClient(base_url=base_url, api_key=api_key)
    launcher = ServiceLauncher(project_root=PROJECT_ROOT, base_url=client.base_url)
    state = CliContext(client=client, launcher=launcher)

    if ensure_started:
        result = launcher.ensure_running(api_key=client.api_key, timeout=timeout)
        if result.api_key:
            state.client = client.with_api_key(result.api_key)

    ctx.obj = state
    return state


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    url: str = typer.Option(None, "--url", help="Override the service base URL."),
    api_key: str = typer.Option(None, "--api-key", help="Use a specific API key."),
) -> None:
    ctx.obj = {"url": url, "api_key": api_key}
    if ctx.invoked_subcommand is None:
        state = _get_context(ctx, ensure_started=True)
        shell = InteractiveShell(
            client=state.client,
            console=console,
            launcher=state.launcher,
            owns_service=state.launcher.process is not None,
        )
        shell.run()


@app.command("shell")
def shell_command(ctx: typer.Context) -> None:
    """Start the service if needed and enter the interactive shell."""
    state = _get_context(ctx, ensure_started=True)
    shell = InteractiveShell(
        client=state.client,
        console=console,
        launcher=state.launcher,
        owns_service=state.launcher.process is not None,
    )
    shell.run()


@app.command("serve")
def serve_command(
    ctx: typer.Context,
    stay_running: bool = typer.Option(True, "--stay-running/--exit-after-start", help="Keep the service process attached to this CLI process."),
) -> None:
    """Start the existing backend entrypoint from the CLI."""
    state = _get_context(ctx, ensure_started=True)
    console.print(f"[green]Service ready:[/green] {state.client.base_url}")
    if state.client.api_key:
        console.print(f"[green]API key:[/green] {state.client.api_key}")
    if stay_running and state.launcher.process is not None:
        try:
            state.launcher.process.wait()
        except KeyboardInterrupt:
            state.launcher.stop()


@app.command("status")
def status_command(ctx: typer.Context) -> None:
    """Show a high-level service snapshot."""
    state = _get_context(ctx, ensure_started=True)
    health = state.client.health()
    system = state.client.system_status()
    index = state.client.index_status()
    summary = state.client.index_summary()

    table = Table(title="Local Cocoa Status")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("base_url", state.client.base_url)
    table.add_row("health", str(health.get("status", "ok")))
    table.add_row("index_status", str(index.get("status", "unknown")))
    table.add_row("indexed_files", str(summary.get("files_indexed", 0)))
    table.add_row("folders", str(summary.get("folders_indexed", 0)))
    table.add_row("cpu_percent", str(system.get("cpu_percent", "n/a")))
    table.add_row("memory_percent", str(system.get("memory_percent", "n/a")))
    console.print(table)


@folders_app.command("list")
def folders_list(ctx: typer.Context) -> None:
    state = _get_context(ctx, ensure_started=True)
    folders = state.client.list_folders()
    table = Table(title="Folders")
    table.add_column("ID")
    table.add_column("Label")
    table.add_column("Path")
    table.add_column("Mode")
    for folder in folders:
        table.add_row(
            str(folder.get("id", "")),
            str(folder.get("label", "")),
            str(folder.get("path", "")),
            str(folder.get("scan_mode", "full")),
        )
    console.print(table)


@folders_app.command("add")
def folders_add(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Folder path to register."),
    label: str | None = typer.Option(None, "--label", help="Optional display label."),
    scan_mode: str = typer.Option("full", "--scan-mode", help="Folder scan mode: full or manual."),
) -> None:
    state = _get_context(ctx, ensure_started=True)
    created = state.client.add_folder(path, label=label, scan_mode=scan_mode)
    console.print(f"[green]Added:[/green] {created.get('label')} ({created.get('id')})")


@folders_app.command("remove")
def folders_remove(ctx: typer.Context, folder_id: str = typer.Argument(..., help="Folder ID to remove.")) -> None:
    state = _get_context(ctx, ensure_started=True)
    state.client.remove_folder(folder_id)
    console.print(f"[green]Removed:[/green] {folder_id}")


@index_app.command("status")
def index_status(ctx: typer.Context) -> None:
    state = _get_context(ctx, ensure_started=True)
    console.print_json(data=state.client.index_status())
    console.print_json(data=state.client.stage_progress())


@index_app.command("run")
def index_run(
    ctx: typer.Context,
    folder_ids: list[str] = typer.Argument(None, help="Optional folder IDs for targeted staged indexing."),
    reindex: bool = typer.Option(False, "--reindex", help="Force reindex mode."),
) -> None:
    state = _get_context(ctx, ensure_started=True)
    result = state.client.run_staged_index(folders=folder_ids or None, reindex=reindex)
    console.print_json(data=result)


@index_app.command("semantic")
def index_semantic(
    ctx: typer.Context,
    enabled: bool = typer.Option(..., "--on/--off", help="Enable or disable semantic embeddings."),
) -> None:
    state = _get_context(ctx, ensure_started=True)
    result = state.client.start_semantic() if enabled else state.client.stop_semantic()
    console.print_json(data=result)


@index_app.command("deep")
def index_deep(
    ctx: typer.Context,
    enabled: bool = typer.Option(..., "--on/--off", help="Enable or disable deep indexing."),
) -> None:
    state = _get_context(ctx, ensure_started=True)
    result = state.client.start_deep() if enabled else state.client.stop_deep()
    console.print_json(data=result)


@keys_app.command("list")
def keys_list(ctx: typer.Context) -> None:
    state = _get_context(ctx, ensure_started=True)
    table = Table(title="API Keys")
    table.add_column("Name")
    table.add_column("Key")
    table.add_column("Active")
    table.add_column("System")
    for item in state.client.list_keys():
        table.add_row(
            str(item.get("name", "")),
            str(item.get("key", "")),
            str(item.get("is_active", False)),
            str(item.get("is_system", False)),
        )
    console.print(table)


@keys_app.command("create")
def keys_create(ctx: typer.Context, name: str = typer.Argument(..., help="Human-readable key name.")) -> None:
    state = _get_context(ctx, ensure_started=True)
    console.print_json(data=state.client.create_key(name))


@providers_app.command("show")
def providers_show(ctx: typer.Context) -> None:
    state = _get_context(ctx, ensure_started=True)
    console.print_json(data=state.client.providers_config())


@providers_app.command("set-remote")
def providers_set_remote(
    ctx: typer.Context,
    base_url: str = typer.Argument(..., help="Remote OpenAI-compatible base URL."),
    model: str = typer.Argument(..., help="Remote model name."),
    api_key: str = typer.Option("", "--remote-api-key", help="Remote provider API key."),
    rerank_base_url: str | None = typer.Option(None, help="Optional remote rerank base URL."),
    rerank_model: str | None = typer.Option(None, help="Optional remote rerank model name."),
    rerank_api_key: str = typer.Option("", help="Optional remote rerank API key."),
) -> None:
    state = _get_context(ctx, ensure_started=True)
    patch = {
        "llm_provider": "remote",
        "remote_llm": {
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
        },
    }
    if rerank_base_url and rerank_model:
        patch["rerank_provider"] = "remote"
        patch["remote_rerank"] = {
            "base_url": rerank_base_url,
            "api_key": rerank_api_key,
            "model": rerank_model,
        }
    console.print_json(data=state.client.update_providers(patch))


@providers_app.command("set-local")
def providers_set_local(
    ctx: typer.Context,
    llm: bool = typer.Option(True, "--llm/--no-llm", help="Reset the LLM provider to local."),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Reset the rerank provider to local."),
) -> None:
    state = _get_context(ctx, ensure_started=True)
    patch: dict[str, str] = {}
    if llm:
        patch["llm_provider"] = "local"
    if rerank:
        patch["rerank_provider"] = "local"
    console.print_json(data=state.client.update_providers(patch))


@mail_app.command("list")
def mail_list(ctx: typer.Context) -> None:
    state = _get_context(ctx, ensure_started=True)
    _render_mail_accounts(console, state.client.list_mail_accounts())


@mail_app.command("add-standard")
def mail_add_standard(
    ctx: typer.Context,
    label: str = typer.Option(..., prompt=True, help="Connector label."),
    protocol: str = typer.Option("imap", help="Protocol: imap or pop3."),
    host: str = typer.Option(..., prompt=True, help="Mail server hostname."),
    username: str = typer.Option(..., prompt=True, help="Mailbox username."),
    password: str = typer.Option(..., prompt=True, hide_input=True, help="Mailbox password or app password."),
    use_ssl: bool = typer.Option(True, "--ssl/--no-ssl", help="Use SSL/TLS."),
    port: int | None = typer.Option(None, help="Override port. Default follows protocol + SSL."),
    folder: str = typer.Option("INBOX", help="IMAP folder to sync."),
) -> None:
    state = _get_context(ctx, ensure_started=True)
    protocol_value = protocol.strip().lower()
    if protocol_value not in {"imap", "pop3"}:
        raise ApiError("protocol must be either 'imap' or 'pop3'.")
    effective_port = port if port is not None else _default_mail_port(protocol_value, use_ssl)
    payload = state.client.add_mail_account(
        label=label.strip(),
        protocol=protocol_value,
        host=host.strip(),
        port=effective_port,
        username=username.strip(),
        password=password,
        use_ssl=use_ssl,
        folder=folder.strip() if protocol_value == "imap" else None,
    )
    console.print_json(data=payload)


@mail_app.command("add-outlook")
def mail_add_outlook(
    ctx: typer.Context,
    label: str = typer.Option("My Outlook", prompt=True, help="Display label for the Outlook account."),
    client_id: str = typer.Option(OUTLOOK_DEFAULT_CLIENT_ID, help="Microsoft app client ID."),
    tenant_id: str = typer.Option(OUTLOOK_DEFAULT_TENANT_ID, help="Microsoft tenant ID."),
    timeout: int = typer.Option(180, help="How long to wait for device auth, in seconds."),
    sync_after: bool = typer.Option(True, "--sync/--no-sync", help="Sync immediately after account creation."),
    open_browser: bool = typer.Option(True, "--open-browser/--no-open-browser", help="Attempt to open the Microsoft verification URL automatically."),
) -> None:
    state = _get_context(ctx, ensure_started=True)
    started = state.client.start_outlook_auth(client_id=client_id, tenant_id=tenant_id)
    flow_id = str(started.get("flow_id", "")).strip()
    if not flow_id:
        raise ApiError("Outlook auth did not return a flow_id.")

    console.print(f"[cyan]Started Outlook device auth:[/cyan] {flow_id}")
    deadline = time.time() + max(timeout, 10)
    code_shown = False

    while time.time() < deadline:
        status = state.client.get_outlook_auth_status(flow_id)
        state_value = str(status.get("status", ""))
        info = status.get("info") or {}

        if state_value == "code_ready":
            verification_uri = str(info.get("verification_uri", ""))
            user_code = str(info.get("user_code", ""))
            if not code_shown:
                console.print(f"[bold]Verification URL:[/bold] {verification_uri}")
                console.print(f"[bold]Device code:[/bold] {user_code}")
                if open_browser and verification_uri:
                    try:
                        webbrowser.open(verification_uri)
                    except Exception:
                        pass
                code_shown = True
        elif state_value == "authenticated":
            account = state.client.complete_outlook_setup(flow_id=flow_id, label=label.strip())
            console.print("[green]Outlook account connected.[/green]")
            console.print_json(data=account)
            if sync_after:
                sync_result = state.client.sync_mail_account(str(account.get("id", "")))
                console.print("[green]Initial sync completed.[/green]")
                console.print_json(data=sync_result)
            return
        elif state_value == "error":
            raise ApiError(str(status.get("message") or "Outlook authentication failed."))

        time.sleep(2)

    raise ApiError("Timed out waiting for Outlook authentication to complete.")


@mail_app.command("sync")
def mail_sync(
    ctx: typer.Context,
    account_id: str = typer.Argument(..., help="Email account ID."),
    limit: int = typer.Option(100, help="Max messages to fetch in one sync."),
) -> None:
    state = _get_context(ctx, ensure_started=True)
    console.print_json(data=state.client.sync_mail_account(account_id, limit=limit))


@mail_app.command("remove")
def mail_remove(ctx: typer.Context, account_id: str = typer.Argument(..., help="Email account ID to remove.")) -> None:
    state = _get_context(ctx, ensure_started=True)
    state.client.remove_mail_account(account_id)
    console.print(f"[green]Removed email account:[/green] {account_id}")


def main() -> int:
    try:
        if len(sys.argv) == 1:
            app(["shell"], standalone_mode=False)
        else:
            app(standalone_mode=False)
        return 0
    except ApiError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    except typer.Exit as exc:
        return int(exc.exit_code)
