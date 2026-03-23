from __future__ import annotations

import asyncio
import base64
import datetime as dt
import email
from email import policy
from email.message import EmailMessage
from email.utils import getaddresses, parsedate_to_datetime
import imaplib
import logging
import poplib
import smtplib
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

from markdownify import markdownify
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError

from core.config import settings
from plugins.plugin_config import load_plugin_config
from core.context import get_indexer
from services.indexer import Indexer
from .models import (
    EmailAccount,
    EmailAccountCreate,
    EmailAccountSummary,
    EmailMessageRecord,
    EmailMessageSummary,
    EmailMessageContent,
    EmailSendRequest,
    EmailSyncRequest,
    EmailSyncResult,
)
from services.storage import IndexStorage
from .storage import EmailMixin
from .outlook import (
    OUTLOOK_DEFAULT_CLIENT_ID,
    OUTLOOK_DEFAULT_TENANT_ID,
    OutlookService,
    OutlookAuthError,
    OutlookServiceError,
)

# Re-export for router
__all__ = [
    "EmailService",
    "EmailServiceError",
    "EmailSyncError",
    "EmailAuthError",
    "EmailAccountNotFound",
]

logger = logging.getLogger(__name__)

# Azure Identity's device-code flow legitimately receives 400 responses while waiting
# (e.g., "authorization_pending"). Keep the low-level HTTP logs quieter.
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)


# Use MSAL-friendly delegated scopes (do not include reserved values like offline_access).
# These must match what the Graph SDK/kiota auth provider requests, otherwise the token cache
# won't be hit and the app will keep prompting.
GRAPH_SCOPES: tuple[str, ...] = (
    "User.Read",
    "Mail.Read",
)


class EmailServiceError(Exception):
    """Base error for email service issues."""


class EmailSyncError(EmailServiceError):
    """Raised when a sync operation fails."""


class EmailAuthError(EmailSyncError):
    """Raised when syncing requires the user to re-authenticate."""


class EmailAccountNotFound(EmailServiceError):
    """Raised when an email account is missing."""


class EmailService(EmailMixin):
    MAX_SYNC_BATCH = 100

    def __init__(self, indexer: Indexer, plugin_id: str = "") -> None:
        # Initialize storage via inherited StorageBase
        super().__init__(plugin_id, db_path=settings.db_path)
        
        self.indexer = indexer
        _config = load_plugin_config(__file__)
        self._root = _config.storage_root
        self._root.mkdir(parents=True, exist_ok=True)
        self._outlook_service = OutlookService()

    async def start_outlook_auth(self, client_id: str, tenant_id: str) -> str:
        client_id = (client_id or "").strip() or OUTLOOK_DEFAULT_CLIENT_ID
        tenant_id = (tenant_id or "").strip() or OUTLOOK_DEFAULT_TENANT_ID
        return await self._outlook_service.start_auth(client_id, tenant_id)

    async def get_outlook_auth_status(self, flow_id: str) -> dict:
        try:
            return await self._outlook_service.get_auth_status(flow_id)
        except OutlookServiceError as e:
            return {'status': 'error', 'message': str(e)}

    async def complete_outlook_setup(self, flow_id: str, label: str) -> EmailAccountSummary:
        try:
            details = await self._outlook_service.complete_auth(flow_id)
        except OutlookServiceError as e:
            raise EmailServiceError(str(e)) from e

        now = dt.datetime.now(dt.timezone.utc)
        account = EmailAccount(
            id=uuid.uuid4().hex,
            label=label,
            protocol="outlook",
            host="graph.microsoft.com",
            port=443,
            username=details["username"] or "outlook_user",
            secret="",
            use_ssl=True,
            folder="INBOX",
            enabled=True,
            created_at=now,
            updated_at=now,
            client_id=details["client_id"],
            tenant_id=details["tenant_id"]
        )

        self.upsert_email_account(account)
        folder_path = self._ensure_account_folder(account)
        return self._to_summary(account, folder_path)

    def list_accounts(self) -> list[EmailAccountSummary]:
        summaries: list[EmailAccountSummary] = []
        accounts = self.list_email_accounts()
        for account in accounts:
            folder_path = self._account_spool_path(account.id)
            summaries.append(self._to_summary(account, folder_path))
        return summaries

    def perform_maintenance(self, account_id: Optional[str] = None) -> None:
        """
        Perform maintenance tasks: prune missing messages and ensure folders exist.
        If account_id is None, perform for all accounts.
        """
        accounts = [self.get_email_account(account_id)] if account_id else self.list_email_accounts()
        
        for account in accounts:
            if not account:
                continue
            removed = self.prune_missing_email_messages(account.id)
            if removed:
                logger.info("Pruned %d orphaned email message records for %s", removed, account.label)
            self._ensure_account_folder(account)
        
        if not account_id:
            logger.info("Universal email maintenance completed.")

    def add_account(self, payload: EmailAccountCreate) -> EmailAccountSummary:
        payload_label = payload.label.strip()
        if not payload_label:
            raise EmailServiceError("Account label is required.")

        now = dt.datetime.now(dt.timezone.utc)
        account = EmailAccount(
            id=uuid.uuid4().hex,
            label=payload_label,
            protocol=payload.protocol,
            host=payload.host.strip(),
            port=self._resolve_port(payload.protocol, payload.port, payload.use_ssl),
            username=payload.username.strip(),
            secret=self._encode_secret(payload.password),
            use_ssl=payload.use_ssl,
            folder=payload.folder.strip() if payload.protocol == "imap" and payload.folder else None,
            enabled=True,
            created_at=now,
            updated_at=now,
            last_synced_at=None,
            last_sync_status=None,
        )

        for existing in self.list_email_accounts():
            if existing.username == account.username and existing.host == account.host:
                raise EmailServiceError("An account with the same host and username already exists.")

        self.upsert_email_account(account)
        self.perform_maintenance(account.id)
        folder_path = self._account_spool_path(account.id)
        return self._to_summary(account, folder_path)

    def remove_account(self, account_id: str) -> None:
        account = self.get_email_account(account_id)
        if not account:
            raise EmailAccountNotFound("Email account not found.")
        folder_path = self._account_spool_path(account.id)
        folder_id = self._folder_id(account)

        # Remove vectors first
        try:
            self.indexer.vector_store.delete_by_filter(folder_id=folder_id)
        except Exception as exc:
            logger.warning("Failed to remove vectors for account %s: %s", account_id, exc)

        # Remove files from storage (files table)
        # This is important because files table has a foreign key to folders table
        # If we don't delete files first, we might violate FK constraints or leave orphaned records
        # if the FK isn't set to CASCADE (it is ON DELETE CASCADE in schema, but good to be explicit or rely on it)
        # The schema says: folder_id TEXT NOT NULL REFERENCES folders(id) ON DELETE CASCADE
        # So deleting the folder should cascade to files.

        self.delete_email_account(account_id)

        # Also remove the folder record from storage if it exists
        # This triggers the ON DELETE CASCADE for files and chunks
        with self.connect() as conn:
            exists = conn.execute("SELECT 1 FROM folders WHERE id = ?", (folder_id,)).fetchone()
            if exists:
                conn.execute("DELETE FROM folders WHERE id = ?", (folder_id,))

        if folder_path.exists():
            shutil.rmtree(folder_path, ignore_errors=True)

    async def sync_account(self, account_id: str, request: EmailSyncRequest) -> EmailSyncResult:
        account = self.get_email_account(account_id)
        if not account:
            raise EmailAccountNotFound("Email account not found.")

        self.perform_maintenance(account.id)
        folder_path = self._account_spool_path(account.id)

        if account.protocol == "outlook":
            return await self._sync_outlook(account, request, folder_path)

        try:
            batch_limit = self._sanitize_limit(request.limit)
            result = await asyncio.to_thread(self._sync_account_blocking, account, batch_limit, folder_path)
        except EmailSyncError as exc:
            self.update_email_account_sync(
                account.id, last_synced_at=dt.datetime.now(dt.timezone.utc), status=f"error: {exc}")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected email sync failure for %s", account.id)
            self.update_email_account_sync(
                account.id, last_synced_at=dt.datetime.now(dt.timezone.utc), status=f"error: {exc}")
            raise EmailSyncError("Unexpected failure while syncing account.") from exc

        if result.new_messages > 0:
            await self.indexer.refresh(folders=[result.folder_id])
            result.indexed = result.new_messages
        else:
            result.indexed = 0

        self.update_email_account_sync(account.id, last_synced_at=result.last_synced_at, status=result.status)
        return result

    async def _sync_outlook(self, account: EmailAccount, request: EmailSyncRequest, folder_path: Path) -> EmailSyncResult:
        try:
            if not (account.client_id or "").strip():
                account.client_id = OUTLOOK_DEFAULT_CLIENT_ID
                self.upsert_email_account(account)
            if not (account.tenant_id or "").strip():
                account.tenant_id = OUTLOOK_DEFAULT_TENANT_ID
                self.upsert_email_account(account)

            logger.info(
                "Syncing Outlook account %s client_id=%s tenant_id=%s",
                account.id,
                account.client_id,
                account.tenant_id,
            )

            limit = self._sanitize_limit(request.limit)

            # Delegate to clean service
            try:
                # Pass username if available to help select the right account from cache
                messages = await self._outlook_service.fetch_messages(
                    client_id=account.client_id,
                    tenant_id=account.tenant_id,
                    limit=limit,
                    username=account.username if account.username and "@" in account.username else None
                )
            except OutlookAuthError as e:
                raise EmailAuthError(str(e)) from e
            except OutlookServiceError as e:
                raise EmailSyncError(str(e)) from e

            existing_ids = set(self.list_email_message_ids(account.id))
            new_records = []

            for msg in messages:
                if msg.id in existing_ids:
                    continue

                # Extract basic fields
                subject = msg.subject or "(No Subject)"
                sender = "Unknown"
                if msg.from_ and msg.from_.email_address:
                    sender = f"{msg.from_.email_address.name or ''} <{msg.from_.email_address.address or ''}>".strip()

                date_val = msg.received_date_time

                to_recipients: list[str] = []
                cc_recipients: list[str] = []
                bcc_recipients: list[str] = []
                if msg.to_recipients:
                    to_recipients = [r.email_address.address for r in msg.to_recipients if r.email_address]
                if msg.cc_recipients:
                    cc_recipients = [r.email_address.address for r in msg.cc_recipients if r.email_address]
                if getattr(msg, 'bcc_recipients', None):
                    bcc_recipients = [r.email_address.address for r in msg.bcc_recipients if r.email_address]
                # Keep merged list for backward compatibility (memory/embedding)
                recipients = to_recipients + cc_recipients + bcc_recipients

                # Process body content
                body_content = ""
                if msg.body and msg.body.content:
                    body_content = self._render_body_markdown(
                        msg.body.content,
                        getattr(msg.body, "content_type", None),
                    )
                elif msg.body_preview:
                    body_content = msg.body_preview

                body_content = body_content.strip() or "_No body content available._"

                # Save to file using the same markdown writer as IMAP
                ts = int(date_val.timestamp()) if date_val else int(dt.datetime.now().timestamp())
                safe_id = "".join(c for c in msg.id if c.isalnum())
                file_name = f"{ts}_{safe_id}.md"
                file_path = folder_path / file_name
                self._write_markdown(file_path, subject, sender, to_recipients, cc_recipients, bcc_recipients, date_val, body_content)
                file_size = file_path.stat().st_size

                record = EmailMessageRecord(
                    id=uuid.uuid4().hex,
                    account_id=account.id,
                    external_id=msg.id,
                    subject=subject,
                    sender=sender,
                    recipients=recipients,
                    to_recipients=to_recipients,
                    cc_recipients=cc_recipients,
                    bcc_recipients=bcc_recipients,
                    sent_at=date_val or dt.datetime.now(dt.timezone.utc),
                    stored_path=file_path,
                    size=file_size,
                    created_at=dt.datetime.now(dt.timezone.utc),
                )
                new_records.append(record)

            new_count = len(new_records)
            for record in new_records:
                self.record_email_message(record)

            if new_count > 0:
                await self.indexer.refresh(folders=[self._folder_id(account)])

            total_messages = self.count_email_messages(account.id)
            last_synced = dt.datetime.now(dt.timezone.utc)
            message = "No new messages." if new_count == 0 else f"Fetched {new_count} message{'s' if new_count != 1 else ''}."

            self.update_email_account_sync(account.id, last_synced_at=last_synced, status="ok")

            return EmailSyncResult(
                account_id=account.id,
                folder_id=self._folder_id(account),
                folder_path=folder_path,
                new_messages=new_count,
                total_messages=total_messages,
                indexed=new_count,
                last_synced_at=last_synced,
                status="ok",
                message=message,
            )

        except EmailSyncError as exc:
            self.update_email_account_sync(
                account.id,
                last_synced_at=dt.datetime.now(dt.timezone.utc),
                status=f"error: {exc}",
            )
            raise
        except (ClientAuthenticationError, HttpResponseError) as exc:
            logger.exception("Outlook sync auth failed")
            self.update_email_account_sync(
                account.id,
                last_synced_at=dt.datetime.now(dt.timezone.utc),
                status=f"error: {exc}",
            )
            raise EmailAuthError(str(exc)) from exc
        except Exception as exc:
            logger.exception("Outlook sync failed")
            self.update_email_account_sync(
                account.id,
                last_synced_at=dt.datetime.now(dt.timezone.utc),
                status=f"error: {exc}",
            )
            raise EmailSyncError(f"Outlook sync failed: {exc}") from exc

    def list_messages(self, account_id: str, limit: int) -> list[EmailMessageSummary]:
        account = self.get_email_account(account_id)
        if not account:
            raise EmailAccountNotFound("Email account not found.")
        records = self.list_email_messages(account_id, limit)
        summaries: list[EmailMessageSummary] = []
        for record in records:
            summary = self._record_to_summary(record)
            if summary:
                summaries.append(summary)
        return summaries

    def get_message(self, message_id: str) -> EmailMessageContent:
        record = self.get_email_message(message_id)
        if not record:
            raise EmailServiceError("Email message not found.")
        markdown = self._read_markdown(record.stored_path)
        summary = self._record_to_summary(record)
        if not summary:
            raise EmailServiceError("Email file missing from disk.")
        return EmailMessageContent(**summary.model_dump(), markdown=markdown)

    # ── Sending ─────────────────────────────────────────────────────────

    # Common IMAP-to-SMTP host mapping for major providers
    _SMTP_HOST_MAP: dict[str, tuple[str, int]] = {
        "imap.gmail.com": ("smtp.gmail.com", 587),
        "imap.mail.yahoo.com": ("smtp.mail.yahoo.com", 587),
        "imap-mail.outlook.com": ("smtp-mail.outlook.com", 587),
        "outlook.office365.com": ("smtp.office365.com", 587),
        "imap.zoho.com": ("smtp.zoho.com", 587),
        "imap.qq.com": ("smtp.qq.com", 465),
        "imap.163.com": ("smtp.163.com", 465),
        "imap.126.com": ("smtp.126.com", 465),
    }

    def _resolve_smtp(self, account: EmailAccount) -> tuple[str, int]:
        """Derive SMTP host/port from the IMAP/POP3 host."""
        host = (account.host or "").lower().strip()
        if host in self._SMTP_HOST_MAP:
            return self._SMTP_HOST_MAP[host]
        # Generic fallback: replace imap/pop with smtp
        smtp_host = host.replace("imap.", "smtp.").replace("pop.", "smtp.").replace("pop3.", "smtp.")
        return smtp_host, 587

    async def send_email(self, request: EmailSendRequest) -> dict[str, str]:
        """Send an email using the credentials of the given account."""
        account = self.get_email_account(request.account_id)
        if not account:
            raise EmailAccountNotFound("Email account not found.")

        if account.protocol == "outlook":
            return await self._send_outlook(account, request)
        elif account.protocol in ("imap", "pop3"):
            return self._send_smtp(account, request)
        else:
            raise EmailServiceError(f"Sending not supported for protocol: {account.protocol}")

    def _send_smtp(self, account: EmailAccount, request: EmailSendRequest) -> dict[str, str]:
        """Send via SMTP using the account's stored credentials."""
        from plugins.synvo_ai_mail.backend.email_renderer import render_email_html

        smtp_host, smtp_port = self._resolve_smtp(account)
        password = self._decode_secret(account.secret)

        msg = EmailMessage()
        msg["From"] = account.username
        msg["To"] = ", ".join(request.to)
        msg["Subject"] = request.subject
        # Plain-text fallback
        msg.set_content(request.body)
        # Rich HTML alternative
        html_body = render_email_html(request.body)
        msg.add_alternative(html_body, subtype="html")

        try:
            if smtp_port == 465:
                server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30)
            else:
                server = smtplib.SMTP(smtp_host, smtp_port, timeout=30)
                server.starttls()
            server.login(account.username, password)
            server.send_message(msg)
            server.quit()
            logger.info("SMTP email sent from %s to %s", account.username, request.to)
            return {"status": "sent", "from": account.username, "to": request.to}
        except smtplib.SMTPAuthenticationError as exc:
            raise EmailServiceError(
                "SMTP authentication failed. For Gmail, you may need an App Password."
            ) from exc
        except Exception as exc:
            raise EmailServiceError(f"Failed to send email: {exc}") from exc

    async def _send_outlook(self, account: EmailAccount, request: EmailSendRequest) -> dict[str, str]:
        """Send via Microsoft Graph API."""
        outlook = OutlookService()
        client_id = account.client_id or OUTLOOK_DEFAULT_CLIENT_ID
        tenant_id = account.tenant_id or OUTLOOK_DEFAULT_TENANT_ID
        from plugins.synvo_ai_mail.backend.email_renderer import render_email_html

        html_body = render_email_html(request.body)
        await outlook.send_message(
            client_id=client_id,
            tenant_id=tenant_id,
            to_recipients=request.to,
            subject=request.subject,
            body=html_body,
            body_is_html=True,
            username=account.username,
        )
        return {"status": "sent", "from": account.username, "to": request.to}

    def _sync_account_blocking(self, account: EmailAccount, limit: int, folder_path: Path) -> EmailSyncResult:
        secret = self._decode_secret(account.secret)
        existing_ids = set(self.list_email_message_ids(account.id))
        if account.protocol == "imap":
            new_records = self._sync_imap(account, secret, folder_path, existing_ids, limit)
        elif account.protocol == "pop3":
            new_records = self._sync_pop3(account, secret, folder_path, existing_ids, limit)
        else:
            raise EmailSyncError(f"Unsupported protocol: {account.protocol}")

        new_count = len(new_records)
        for record in new_records:
            self.record_email_message(record)

        removed = self.prune_missing_email_messages(account.id)
        if removed:
            logger.info("Pruned %d orphaned email message records for %s", removed, account.label)
        total_messages = self.count_email_messages(account.id)
        last_synced = dt.datetime.now(dt.timezone.utc)
        message = "No new messages." if new_count == 0 else f"Fetched {new_count} message{'s' if new_count != 1 else ''}."

        return EmailSyncResult(
            account_id=account.id,
            folder_id=self._folder_id(account),
            folder_path=folder_path,
            new_messages=new_count,
            total_messages=total_messages,
            indexed=0,
            last_synced_at=last_synced,
            status="ok",
            message=message,
        )

    def _sync_imap(
        self,
        account: EmailAccount,
        password: str,
        spool_path: Path,
        existing_ids: set[str],
        limit: int,
    ) -> list[EmailMessageRecord]:
        client: imaplib.IMAP4 | imaplib.IMAP4_SSL
        if account.use_ssl:
            client = imaplib.IMAP4_SSL(account.host, account.port)
        else:
            client = imaplib.IMAP4(account.host, account.port)

        try:
            client.login(account.username, password)
            mailbox = account.folder or "INBOX"
            status, _ = client.select(mailbox, readonly=True)
            if status != "OK":
                raise EmailSyncError(f"Unable to select mailbox '{mailbox}'.")

            status, data = client.uid("search", None, "ALL")
            if status != "OK" or not data:
                return []
            raw_ids = data[0]
            if not raw_ids:
                return []
            uid_values = [uid for uid in raw_ids.split() if uid]
            uid_values = uid_values[-limit:]

            new_records: list[EmailMessageRecord] = []
            for uid in uid_values:
                status, fetched = client.uid("fetch", uid, "(RFC822)")
                if status != "OK" or not fetched:
                    continue
                message_bytes = fetched[0][1]
                if not message_bytes:
                    continue
                record = self._build_message_record(
                    account,
                    message_bytes,
                    existing_ids,
                    spool_path,
                    external_identifier=uid.decode("ascii", errors="ignore"),
                )
                if record:
                    new_records.append(record)
                    existing_ids.add(record.external_id)
            return new_records
        except EmailSyncError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise EmailSyncError(str(exc)) from exc
        finally:
            try:
                client.logout()
            except Exception:  # noqa: BLE001
                pass

    def _sync_pop3(
        self,
        account: EmailAccount,
        password: str,
        spool_path: Path,
        existing_ids: set[str],
        limit: int,
    ) -> list[EmailMessageRecord]:
        server: poplib.POP3 | poplib.POP3_SSL
        if account.use_ssl:
            server = poplib.POP3_SSL(account.host, account.port, timeout=30)
        else:
            server = poplib.POP3(account.host, account.port, timeout=30)

        try:
            server.user(account.username)
            server.pass_(password)
            response, listings, _ = server.uidl()
            if not listings or b"+OK" not in response:
                return []
            entries = [entry.decode("utf-8", errors="ignore") for entry in listings]
            indexed_entries = []
            for line in entries:
                parts = line.split()
                if len(parts) >= 2:
                    indexed_entries.append((int(parts[0]), parts[1]))
            indexed_entries = indexed_entries[-limit:]

            new_records: list[EmailMessageRecord] = []
            for index, external_id in indexed_entries:
                if external_id in existing_ids:
                    continue
                resp, lines, _ = server.retr(index)
                if b"+OK" not in resp or not lines:
                    continue
                message_bytes = b"\r\n".join(lines) + b"\r\n"
                record = self._build_message_record(
                    account,
                    message_bytes,
                    existing_ids,
                    spool_path,
                    external_identifier=external_id,
                )
                if record:
                    new_records.append(record)
                    existing_ids.add(record.external_id)
            return new_records
        except EmailSyncError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise EmailSyncError(str(exc)) from exc
        finally:
            try:
                server.quit()
            except Exception:  # noqa: BLE001
                pass

    def _build_message_record(
        self,
        account: EmailAccount,
        message_bytes: bytes,
        existing_ids: set[str],
        spool_path: Path,
        *,
        external_identifier: str,
    ) -> Optional[EmailMessageRecord]:
        message = email.message_from_bytes(message_bytes, policy=policy.default)
        external_id = message.get("Message-ID") or external_identifier
        if external_id in existing_ids:
            return None

        body_text = self._extract_text(message)
        sent_at = self._parse_date(message.get("Date"))
        subject = message.get("Subject")
        sender = message.get("From")
        separated = self._collect_recipients_separated(message)
        to_recipients = separated["to"]
        cc_recipients = separated["cc"]
        bcc_recipients = separated["bcc"]
        # Keep merged list for backward compatibility (memory/embedding)
        recipients = to_recipients + cc_recipients + bcc_recipients

        timestamp = (sent_at or dt.datetime.now(dt.timezone.utc)).strftime("%Y%m%d-%H%M%S")
        slug = self._slugify(subject or "message")
        filename = f"{timestamp}-{slug}-{uuid.uuid4().hex[:8]}.md"
        target_path = spool_path / filename
        self._write_markdown(target_path, subject, sender, to_recipients, cc_recipients, bcc_recipients, sent_at, body_text)

        return EmailMessageRecord(
            id=uuid.uuid4().hex,
            account_id=account.id,
            external_id=external_id,
            subject=subject,
            sender=sender,
            recipients=recipients,
            to_recipients=to_recipients,
            cc_recipients=cc_recipients,
            bcc_recipients=bcc_recipients,
            sent_at=sent_at,
            stored_path=target_path,
            size=len(message_bytes),
            created_at=dt.datetime.now(dt.timezone.utc),
        )

    def _ensure_account_folder(self, account: EmailAccount) -> Path:
        folder_path = self._account_spool_path(account.id)
        folder_path.mkdir(parents=True, exist_ok=True)

        folder_id = self._folder_id(account)
        now = dt.datetime.now(dt.timezone.utc)

        with self.connect() as conn:
            row = conn.execute(
                "SELECT id, label, created_at, last_indexed_at FROM folders WHERE id = ?",
                (folder_id,),
            ).fetchone()

            if row:
                conn.execute(
                    """UPDATE folders SET path=?, updated_at=?, enabled=1 WHERE id=?""",
                    (str(folder_path), now.isoformat(), folder_id),
                )
            else:
                conn.execute(
                    """INSERT INTO folders (id, path, label, created_at, updated_at, last_indexed_at, enabled)
                       VALUES (?, ?, ?, ?, ?, ?, 1)""",
                    (
                        folder_id,
                        str(folder_path),
                        f"Email · {account.label}",
                        now.isoformat(),
                        now.isoformat(),
                        None,
                    ),
                )
        return folder_path

    def _to_summary(self, account: EmailAccount, folder_path: Path) -> EmailAccountSummary:
        window = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=24)
        total_messages = self.count_email_messages(account.id)
        recent_new = self.count_email_messages_since(account.id, window)
        return EmailAccountSummary(
            id=account.id,
            label=account.label,
            protocol=account.protocol,
            host=account.host,
            port=account.port,
            username=account.username,
            use_ssl=account.use_ssl,
            folder=account.folder,
            enabled=account.enabled,
            created_at=account.created_at,
            updated_at=account.updated_at,
            last_synced_at=account.last_synced_at,
            last_sync_status=account.last_sync_status,
            total_messages=total_messages,
            recent_new_messages=recent_new,
            folder_id=self._folder_id(account),
            folder_path=folder_path,
        )

    def _account_spool_path(self, account_id: str) -> Path:
        return self._root / account_id

    @staticmethod
    def _resolve_port(protocol: str, port: int, use_ssl: bool) -> int:
        if port:
            return port
        if protocol == "imap":
            return 993 if use_ssl else 143
        if protocol == "pop3":
            return 995 if use_ssl else 110
        return port

    @staticmethod
    def _encode_secret(password: str) -> str:
        return base64.b64encode(password.encode("utf-8")).decode("ascii")

    @staticmethod
    def _decode_secret(secret: str) -> str:
        try:
            return base64.b64decode(secret.encode("ascii")).decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            raise EmailServiceError("Stored credentials are corrupted.") from exc

    @staticmethod
    def _collect_recipients_separated(message: EmailMessage) -> dict[str, list[str]]:
        """Return To, CC, BCC as separate lists."""
        result: dict[str, list[str]] = {"to": [], "cc": [], "bcc": []}
        for header, key in (("To", "to"), ("Cc", "cc"), ("Bcc", "bcc")):
            value = message.get(header)
            if value:
                addresses = getaddresses([value])
                result[key] = [addr for _, addr in addresses if addr]
        return result

    @staticmethod
    def _collect_recipients(message: EmailMessage) -> list[str]:
        """Return all recipients as a flat list (backward compat)."""
        separated = EmailService._collect_recipients_separated(message)
        return separated["to"] + separated["cc"] + separated["bcc"]

    @staticmethod
    def _parse_date(value: Optional[str]) -> Optional[dt.datetime]:
        if not value:
            return None
        try:
            parsed = parsedate_to_datetime(value)
            if parsed and parsed.tzinfo:
                return parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
            return parsed
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _extract_text(message: EmailMessage) -> str:
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_maintype() == "multipart":
                    continue
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    decoded = EmailService._decode_part(part)
                    if decoded:
                        return decoded
            for part in message.walk():
                if part.get_content_type() == "text/html":
                    html_text = EmailService._decode_part(part)
                    if html_text:
                        return EmailService._html_to_text(html_text)
            return ""
        content_type = message.get_content_type()
        payload = EmailService._decode_part(message)
        if content_type == "text/html":
            return EmailService._html_to_text(payload or "")
        return payload or ""

    @staticmethod
    def _decode_part(part: EmailMessage) -> str:
        try:
            payload = part.get_payload(decode=True)
            if payload is None:
                return ""
            charset = part.get_content_charset() or "utf-8"
            try:
                return payload.decode(charset, errors="replace")
            except LookupError:
                return payload.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _html_to_text(payload: str) -> str:
        text = re.sub(r"<br\s*/?>", "\n", payload, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _render_body_markdown(content: str, content_type: Any | None) -> str:
        if not content:
            return ""
        if EmailService._looks_like_html(content_type, content):
            return markdownify(
                content,
                heading_style="ATX",
                strip=["script", "style"],
            ).strip()
        return content

    @staticmethod
    def _looks_like_html(content_type: Any | None, content: str) -> bool:
        if content_type is not None:
            raw = getattr(content_type, "value", content_type)
            raw_str = str(raw)
            if "html" in raw_str.lower():
                return True
        return bool(re.search(r"<\w+[^>]*>", content))

    @staticmethod
    def _write_markdown(
        path: Path,
        subject: Optional[str],
        sender: Optional[str],
        to_recipients: Iterable[str],
        cc_recipients: Iterable[str],
        bcc_recipients: Iterable[str],
        sent_at: Optional[dt.datetime],
        body_text: str,
    ) -> None:
        to_list = list(to_recipients)
        cc_list = list(cc_recipients)
        bcc_list = list(bcc_recipients)
        lines = [
            f"# {subject.strip() if subject else 'Untitled message'}",
            "",
            f"- From: {sender or 'Unknown sender'}",
            f"- To: {', '.join(to_list) if to_list else 'Unknown recipients'}",
        ]
        if cc_list:
            lines.append(f"- CC: {', '.join(cc_list)}")
        if bcc_list:
            lines.append(f"- BCC: {', '.join(bcc_list)}")
        lines.extend([
            f"- Date: {sent_at.isoformat() if sent_at else 'Unknown'}",
            "",
            body_text.strip() or '_No body content available._',
        ])
        path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    @staticmethod
    def _slugify(value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
        return cleaned or "message"

    def _folder_id(self, account: EmailAccount) -> str:
        return f"email::{account.id}"

    def _sanitize_limit(self, requested: int | None) -> int:
        if requested is None or requested <= 0:
            return self.MAX_SYNC_BATCH
        return max(1, min(requested, self.MAX_SYNC_BATCH))

    def _record_to_summary(self, record: EmailMessageRecord) -> EmailMessageSummary | None:
        if not record.stored_path.exists():
            return None
        preview = self._extract_preview(record.stored_path)
        return EmailMessageSummary(
            id=record.id,
            account_id=record.account_id,
            subject=record.subject,
            sender=record.sender,
            recipients=record.recipients,
            to_recipients=record.to_recipients,
            cc_recipients=record.cc_recipients,
            bcc_recipients=record.bcc_recipients,
            sent_at=record.sent_at,
            stored_path=record.stored_path,
            size=record.size,
            created_at=record.created_at,
            preview=preview,
        )

    @staticmethod
    def _read_markdown(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise EmailServiceError("Email content file missing.") from exc

    @staticmethod
    def _extract_preview(path: Path, lines: int = 6) -> str | None:
        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        snippet = [line for line in content.splitlines() if line.strip()][:lines]
        return "\n".join(snippet) if snippet else None

    # ==================== Account-Level Memory Integration Methods (v2.5) ====================

    async def build_account_memory(
        self, 
        account_id: str, 
        user_id: str,
    ) -> dict:
        """
        为整个邮箱账户批量构建 Memory（一键构建）
        
        Args:
            account_id: 邮箱账户 ID
            user_id: 用户 ID
            
        Returns:
            构建结果 dict
        """
        import json
        from datetime import datetime, timezone
        
        account = self.get_email_account(account_id)
        if not account:
            raise EmailAccountNotFound("Email account not found.")
        
        # 获取该账户下所有邮件
        messages = self.list_email_messages(account_id, limit=1000)
        if not messages:
            return {
                "success": True,
                "message": "No messages to process",
                "account_id": account_id,
                "total_messages": 0,
                "memcells_created": 0,
                "episodes_created": 0,
                "event_logs_created": 0,
            }
        
        try:
            from services.memory.service import get_memory_service
            from services.memory.api_specs.memory_types import RawDataType, MemCell
            from services.memory.api_specs.memory_models import MemoryType
            from services.storage.memory import (
                MemCellRecord as StorageMemCellRecord,
                EpisodeRecord as StorageEpisodeRecord,
                EventLogRecord as StorageEventLogRecord,
            )
            
            memory_service = get_memory_service()
            group_id = f"email_account::{account_id}"
            
            total_memcells = 0
            total_episodes = 0
            total_event_logs = 0
            
            for record in messages:
                import hashlib
                now = datetime.now(timezone.utc).isoformat()
                # 使用稳定的 ID（基于 chunk_id 的 hash），这样相同邮件的 upsert 会更新而不是创建新记录
                chunk_id = f"email_account_{account_id}_{record.id}"
                memcell_id = hashlib.sha256(chunk_id.encode()).hexdigest()[:32]
                
                # 读取邮件内容
                try:
                    markdown = self._read_markdown(record.stored_path)
                except Exception as e:
                    logger.warning("Failed to read email %s: %s", record.id, e)
                    continue
                
                # 构建邮件元数据
                email_metadata = {
                    "account_id": account_id,
                    "message_id": record.id,
                    "subject": record.subject,
                    "sender": record.sender,
                    "recipients": record.recipients,
                    "sent_at": record.sent_at.isoformat() if record.sent_at else None,
                    "source": "email_account_build",
                }
                
                original_data_dict = {
                    "content": markdown,
                    "email_subject": record.subject,
                    "email_sender": record.sender,
                    "email_recipients": record.recipients,
                    "email_date": record.sent_at.isoformat() if record.sent_at else "",
                }
                
                # 创建 MemCell
                memcell = MemCell(
                    event_id=memcell_id,
                    user_id_list=[user_id],
                    original_data=[original_data_dict],
                    timestamp=datetime.now(timezone.utc),
                    summary=f"Email: {record.subject}" if record.subject else "Email message",
                    group_id=group_id,
                    participants=[record.sender] if record.sender else [],
                    type=RawDataType.DOCUMENT,
                )
                
                # 持久化 MemCell（使用稳定的 chunk_id）
                storage_memcell = StorageMemCellRecord(
                    id=memcell_id,
                    user_id=user_id,
                    original_data=json.dumps([original_data_dict]),
                    summary=f"Email: {record.subject}" if record.subject else "Email message",
                    subject=record.subject,
                    file_id=None,
                    chunk_id=chunk_id,
                    chunk_ordinal=0,
                    type="Document",
                    keywords=None,
                    timestamp=now,
                    metadata=email_metadata,
                )
                memory_service.storage.upsert_memcell(storage_memcell)
                total_memcells += 1
                
                # 提取情节记忆
                try:
                    logger.info("📧 [MEMORY] Extracting episode for email: %s", record.subject or record.id)
                    episode = await memory_service.memory_manager.extract_memory(
                        memcell=memcell,
                        memory_type=MemoryType.EPISODIC_MEMORY,
                        user_id=user_id,
                    )
                    
                    if episode:
                        logger.info("📧 [MEMORY] Episode extracted successfully for: %s", record.subject or record.id)
                        # 使用稳定的 Episode ID（基于 memcell_id）
                        episode_id = hashlib.sha256(f"{memcell_id}_episode".encode()).hexdigest()[:32]
                        storage_episode = StorageEpisodeRecord(
                            id=episode_id,
                            user_id=user_id,
                            summary=getattr(episode, "summary", record.subject or ""),
                            episode=getattr(episode, "episode", ""),
                            subject=getattr(episode, "subject", record.subject),
                            timestamp=now,
                            parent_memcell_id=memcell_id,
                            metadata=email_metadata,
                        )
                        memory_service.storage.upsert_episode(storage_episode)
                        total_episodes += 1
                        
                        # 提取事件日志
                        try:
                            logger.info("📧 [MEMORY] Extracting event log for episode: %s", episode_id)
                            event_log = await memory_service.memory_manager.extract_memory(
                                memcell=memcell,
                                memory_type=MemoryType.EVENT_LOG,
                                user_id=user_id,
                                episode_memory=episode,
                            )
                            if event_log:
                                facts = getattr(event_log, "atomic_fact", [])
                                if isinstance(facts, str):
                                    facts = [facts]
                                logger.info("📧 [MEMORY] EventLog extracted: %d facts", len(facts))
                                for idx_fact, fact in enumerate(facts):
                                    if fact and fact.strip():
                                        # 使用稳定的 EventLog ID（基于 episode_id + fact 索引）
                                        log_id = hashlib.sha256(f"{episode_id}_fact_{idx_fact}".encode()).hexdigest()[:32]
                                        storage_log = StorageEventLogRecord(
                                            id=log_id,
                                            user_id=user_id,
                                            atomic_fact=fact.strip(),
                                            timestamp=now,
                                            parent_episode_id=episode_id,
                                            metadata={"account_id": account_id, "message_id": record.id},
                                        )
                                        memory_service.storage.upsert_event_log(storage_log)
                                        total_event_logs += 1
                            else:
                                logger.warning("📧 [MEMORY] EventLog extraction returned None for episode: %s", episode_id)
                        except Exception as e:
                            logger.error("📧 [MEMORY] Event log extraction failed for email %s: %s", record.id, e, exc_info=True)
                    else:
                        logger.warning("📧 [MEMORY] Episode extraction returned None for: %s", record.subject or record.id)
                            
                except Exception as e:
                    logger.error("📧 [MEMORY] Episode extraction failed for email %s: %s", record.id, e, exc_info=True)
            
            logger.info("📧 [MEMORY] Account %s memory built: %d memcells, %d episodes, %d event logs", 
                       account_id, total_memcells, total_episodes, total_event_logs)
            
            return {
                "success": True,
                "message": f"Memory built successfully for {total_memcells} emails",
                "account_id": account_id,
                "total_messages": len(messages),
                "memcells_created": total_memcells,
                "episodes_created": total_episodes,
                "event_logs_created": total_event_logs,
            }
            
        except Exception as e:
            logger.error("Failed to build memory for account %s: %s", account_id, e)
            return {
                "success": False,
                "message": f"Failed to build memory: {str(e)}",
                "account_id": account_id,
                "total_messages": 0,
                "memcells_created": 0,
                "episodes_created": 0,
                "event_logs_created": 0,
            }

    async def build_account_memory_stream(self, account_id: str, user_id: str, force: bool = False):
        """
        流式构建 Memory，实时报告进度
        
        Args:
            account_id: 邮箱账户 ID
            user_id: 用户 ID
            force: 是否强制重建所有邮件的 Memory（忽略已打标的邮件）
        
        Yields:
            进度更新 dict，包含 type, current, total, email_subject, memory_result 等字段
        """
        import json
        from datetime import datetime, timezone
        
        account = self.get_email_account(account_id)
        if not account:
            raise EmailAccountNotFound("Email account not found.")
        
        # 获取该账户下所有邮件
        messages = self.list_email_messages(account_id, limit=1000)
        total_count = len(messages)
        
        # 初始进度
        yield {
            "type": "start",
            "account_id": account_id,
            "total": total_count,
            "force": force,
            "message": f"Starting memory build for {total_count} emails..." + (" (force rebuild)" if force else "")
        }
        
        if not messages:
            yield {
                "type": "complete",
                "account_id": account_id,
                "total": 0,
                "memcells_created": 0,
                "episodes_created": 0,
                "event_logs_created": 0,
                "skipped": 0,
                "message": "No messages to process"
            }
            return
        
        try:
            from services.memory.service import get_memory_service
            from services.memory.api_specs.memory_types import RawDataType, MemCell
            from services.memory.api_specs.memory_models import MemoryType
            from services.storage.memory import (
                MemCellRecord as StorageMemCellRecord,
                EpisodeRecord as StorageEpisodeRecord,
                EventLogRecord as StorageEventLogRecord,
            )
            
            memory_service = get_memory_service()
            group_id = f"email_account::{account_id}"
            
            total_memcells = 0
            total_episodes = 0
            total_event_logs = 0
            total_skipped = 0
            
            for idx, record in enumerate(messages):
                now = datetime.now(timezone.utc).isoformat()
                
                # 构建 chunk_id 来检查是否已处理
                chunk_id = f"email_account_{account_id}_{record.id}"
                
                # 检查是否已有 MemCell（除非是 force rebuild）
                if not force:
                    existing_memcells = memory_service.storage.get_memcells_by_chunk_id(chunk_id)
                    if existing_memcells:
                        total_skipped += 1
                        yield {
                            "type": "skipped",
                            "current": idx + 1,
                            "total": total_count,
                            "percentage": round((idx + 1) / total_count * 100, 1),
                            "email_id": record.id,
                            "email_subject": record.subject or "(No Subject)",
                            "email_sender": record.sender or "",
                            "message": f"Skipping already processed email: {record.subject or '(No Subject)'}",
                            "stats": {
                                "memcells": total_memcells,
                                "episodes": total_episodes,
                                "facts": total_event_logs,
                                "skipped": total_skipped,
                            },
                        }
                        continue
                
                # 使用稳定的 ID（基于 chunk_id 的 hash），这样相同邮件的 upsert 会更新而不是创建新记录
                import hashlib
                chunk_id = f"email_account_{account_id}_{record.id}"
                memcell_id = hashlib.sha256(chunk_id.encode()).hexdigest()[:32]
                
                # 当前邮件进度
                yield {
                    "type": "processing",
                    "current": idx + 1,
                    "total": total_count,
                    "percentage": round((idx + 1) / total_count * 100, 1),
                    "email_id": record.id,
                    "email_subject": record.subject or "(No Subject)",
                    "email_sender": record.sender or "",
                    "message": f"Processing email {idx + 1}/{total_count}: {record.subject or '(No Subject)'}"
                }
                
                # 读取邮件内容
                try:
                    markdown = self._read_markdown(record.stored_path)
                except Exception as e:
                    # 记录失败状态
                    self.update_email_memory_status(record.id, 'failed', f"Failed to read email: {str(e)}")
                    yield {
                        "type": "email_error",
                        "current": idx + 1,
                        "total": total_count,
                        "email_id": record.id,
                        "email_subject": record.subject or "(No Subject)",
                        "error": f"Failed to read email: {str(e)}"
                    }
                    continue
                
                # 构建邮件元数据
                email_metadata = {
                    "account_id": account_id,
                    "message_id": record.id,
                    "subject": record.subject,
                    "sender": record.sender,
                    "recipients": record.recipients,
                    "sent_at": record.sent_at.isoformat() if record.sent_at else None,
                    "source": "email_account_build",
                }
                
                original_data_dict = {
                    "content": markdown,
                    "email_subject": record.subject,
                    "email_sender": record.sender,
                    "email_recipients": record.recipients,
                    "email_date": record.sent_at.isoformat() if record.sent_at else "",
                }
                
                # 创建 MemCell
                memcell = MemCell(
                    event_id=memcell_id,
                    user_id_list=[user_id],
                    original_data=[original_data_dict],
                    timestamp=datetime.now(timezone.utc),
                    summary=f"Email: {record.subject}" if record.subject else "Email message",
                    group_id=group_id,
                    participants=[record.sender] if record.sender else [],
                    type=RawDataType.DOCUMENT,
                )
                
                # 持久化 MemCell（使用稳定的 chunk_id）
                storage_memcell = StorageMemCellRecord(
                    id=memcell_id,
                    user_id=user_id,
                    original_data=json.dumps([original_data_dict]),
                    summary=f"Email: {record.subject}" if record.subject else "Email message",
                    subject=record.subject,
                    file_id=None,
                    chunk_id=chunk_id,
                    chunk_ordinal=0,
                    type="Document",
                    keywords=None,
                    timestamp=now,
                    metadata=email_metadata,
                )
                memory_service.storage.upsert_memcell(storage_memcell)
                total_memcells += 1
                
                # 邮件结果
                email_result = {
                    "email_id": record.id,
                    "email_subject": record.subject or "(No Subject)",
                    "memcell_created": True,
                    "episode_created": False,
                    "episode_summary": None,
                    "facts_extracted": [],
                }
                
                # 提取情节记忆
                try:
                    episode = await memory_service.memory_manager.extract_memory(
                        memcell=memcell,
                        memory_type=MemoryType.EPISODIC_MEMORY,
                        user_id=user_id,
                    )
                    
                    if episode:
                        # 使用稳定的 Episode ID（基于 memcell_id）
                        episode_id = hashlib.sha256(f"{memcell_id}_episode".encode()).hexdigest()[:32]
                        episode_summary = getattr(episode, "summary", record.subject or "")
                        episode_content = getattr(episode, "episode", "")
                        
                        # 如果是 force rebuild，先删除旧的 event_logs（episode 会被 upsert 覆盖）
                        if force:
                            try:
                                memory_service.storage.delete_event_logs_by_episode(episode_id)
                            except Exception:
                                pass  # Ignore if delete fails
                        
                        storage_episode = StorageEpisodeRecord(
                            id=episode_id,
                            user_id=user_id,
                            summary=episode_summary,
                            episode=episode_content,
                            subject=getattr(episode, "subject", record.subject),
                            timestamp=now,
                            parent_memcell_id=memcell_id,
                            metadata=email_metadata,
                        )
                        memory_service.storage.upsert_episode(storage_episode)
                        total_episodes += 1
                        email_result["episode_created"] = True
                        email_result["episode_summary"] = episode_summary[:200] if episode_summary else None
                        
                        # 提取事件日志
                        try:
                            event_log = await memory_service.memory_manager.extract_memory(
                                memcell=memcell,
                                memory_type=MemoryType.EVENT_LOG,
                                user_id=user_id,
                                episode_memory=episode,
                            )
                            if event_log:
                                facts = getattr(event_log, "atomic_fact", [])
                                if isinstance(facts, str):
                                    facts = [facts]
                                for idx_fact, fact in enumerate(facts):
                                    if fact and fact.strip():
                                        # 使用稳定的 EventLog ID（基于 episode_id + fact 索引）
                                        log_id = hashlib.sha256(f"{episode_id}_fact_{idx_fact}".encode()).hexdigest()[:32]
                                        storage_log = StorageEventLogRecord(
                                            id=log_id,
                                            user_id=user_id,
                                            atomic_fact=fact.strip(),
                                            timestamp=now,
                                            parent_episode_id=episode_id,
                                            metadata={"account_id": account_id, "message_id": record.id},
                                        )
                                        memory_service.storage.upsert_event_log(storage_log)
                                        total_event_logs += 1
                                        email_result["facts_extracted"].append(fact.strip()[:100])
                        except Exception as e:
                            logger.warning("Event log extraction failed for email %s: %s", record.id, e)
                            # EventLog 失败不算整体失败，只是警告
                except Exception as e:
                    logger.warning("Episode extraction failed for email %s: %s", record.id, e)
                    # Episode 提取失败，记录为失败状态
                    self.update_email_memory_status(record.id, 'failed', f"Episode extraction failed: {str(e)}")
                    email_result["error"] = str(e)
                
                # 如果没有错误，标记为成功
                if "error" not in email_result:
                    self.update_email_memory_status(record.id, 'success')
                
                # 单封邮件处理完成
                yield {
                    "type": "email_complete",
                    "current": idx + 1,
                    "total": total_count,
                    "percentage": round((idx + 1) / total_count * 100, 1),
                    "stats": {
                        "memcells": total_memcells,
                        "episodes": total_episodes,
                        "facts": total_event_logs,
                        "skipped": total_skipped,
                    },
                    "email_result": email_result,
                }
            
            # 全部完成
            skip_msg = f", {total_skipped} skipped" if total_skipped > 0 else ""
            yield {
                "type": "complete",
                "account_id": account_id,
                "total": total_count,
                "memcells_created": total_memcells,
                "episodes_created": total_episodes,
                "event_logs_created": total_event_logs,
                "skipped": total_skipped,
                "message": f"Memory build complete: {total_memcells} memcells, {total_episodes} episodes, {total_event_logs} facts{skip_msg}"
            }
            
        except Exception as e:
            logger.error("Failed to build memory for account %s: %s", account_id, e)
            yield {
                "type": "error",
                "account_id": account_id,
                "message": f"Failed to build memory: {str(e)}"
            }

    async def retry_single_email(self, account_id: str, message_id: str, user_id: str):
        """
        重试单个失败的邮件打标
        
        Yields:
            进度更新 dict
        """
        import json
        import hashlib
        from datetime import datetime, timezone
        
        account = self.get_email_account(account_id)
        if not account:
            raise EmailAccountNotFound("Email account not found.")
        
        # 获取邮件记录
        record = self.get_email_message(message_id)
        if not record:
            yield {
                "type": "error",
                "message": f"Email message not found: {message_id}"
            }
            return
        
        yield {
            "type": "start",
            "message_id": message_id,
            "subject": record.subject or "(No Subject)",
            "message": f"Retrying email: {record.subject or '(No Subject)'}"
        }
        
        try:
            from services.memory.service import get_memory_service
            from services.memory.api_specs.memory_types import RawDataType, MemCell
            from services.memory.api_specs.memory_models import MemoryType
            from services.storage.memory import (
                MemCellRecord as StorageMemCellRecord,
                EpisodeRecord as StorageEpisodeRecord,
                EventLogRecord as StorageEventLogRecord,
            )
            
            memory_service = get_memory_service()
            group_id = f"email_account::{account_id}"
            
            now = datetime.now(timezone.utc).isoformat()
            chunk_id = f"email_account_{account_id}_{record.id}"
            memcell_id = hashlib.sha256(chunk_id.encode()).hexdigest()[:32]
            
            # 读取邮件内容
            try:
                markdown = self._read_markdown(record.stored_path)
            except Exception as e:
                self.update_email_memory_status(record.id, 'failed', f"Failed to read email: {str(e)}")
                yield {
                    "type": "error",
                    "message_id": message_id,
                    "error": f"Failed to read email: {str(e)}"
                }
                return
            
            # 构建邮件元数据
            email_metadata = {
                "account_id": account_id,
                "message_id": record.id,
                "subject": record.subject,
                "sender": record.sender,
                "recipients": record.recipients,
                "sent_at": record.sent_at.isoformat() if record.sent_at else None,
                "source": "email_retry",
            }
            
            original_data_dict = {
                "content": markdown,
                "email_subject": record.subject,
                "email_sender": record.sender,
                "email_recipients": record.recipients,
                "email_date": record.sent_at.isoformat() if record.sent_at else "",
            }
            
            # 创建 MemCell
            memcell = MemCell(
                event_id=memcell_id,
                user_id_list=[user_id],
                original_data=[original_data_dict],
                timestamp=datetime.now(timezone.utc),
                summary=f"Email: {record.subject}" if record.subject else "Email message",
                group_id=group_id,
                participants=[record.sender] if record.sender else [],
                type=RawDataType.DOCUMENT,
            )
            
            # 持久化 MemCell
            storage_memcell = StorageMemCellRecord(
                id=memcell_id,
                user_id=user_id,
                original_data=json.dumps([original_data_dict]),
                summary=f"Email: {record.subject}" if record.subject else "Email message",
                subject=record.subject,
                file_id=None,
                chunk_id=chunk_id,
                chunk_ordinal=0,
                type="Document",
                keywords=None,
                timestamp=now,
                metadata=email_metadata,
            )
            memory_service.storage.upsert_memcell(storage_memcell)
            
            yield {
                "type": "progress",
                "message_id": message_id,
                "step": "memcell",
                "message": "MemCell created"
            }
            
            # 提取情节记忆
            episode_id = hashlib.sha256(f"{memcell_id}_episode".encode()).hexdigest()[:32]
            episode_created = False
            facts_extracted = []
            
            try:
                # 先删除旧的 event_logs
                try:
                    memory_service.storage.delete_event_logs_by_episode(episode_id)
                except Exception:
                    pass
                
                episode = await memory_service.memory_manager.extract_memory(
                    memcell=memcell,
                    memory_type=MemoryType.EPISODIC_MEMORY,
                    user_id=user_id,
                )
                
                if episode:
                    episode_summary = getattr(episode, "summary", record.subject or "")
                    episode_content = getattr(episode, "episode", "")
                    
                    storage_episode = StorageEpisodeRecord(
                        id=episode_id,
                        user_id=user_id,
                        summary=episode_summary,
                        episode=episode_content,
                        subject=getattr(episode, "subject", record.subject),
                        timestamp=now,
                        parent_memcell_id=memcell_id,
                        metadata=email_metadata,
                    )
                    memory_service.storage.upsert_episode(storage_episode)
                    episode_created = True
                    
                    yield {
                        "type": "progress",
                        "message_id": message_id,
                        "step": "episode",
                        "message": "Episode extracted"
                    }
                    
                    # 提取事件日志
                    try:
                        event_log = await memory_service.memory_manager.extract_memory(
                            memcell=memcell,
                            memory_type=MemoryType.EVENT_LOG,
                            user_id=user_id,
                            episode_memory=episode,
                        )
                        if event_log:
                            facts = getattr(event_log, "atomic_fact", [])
                            if isinstance(facts, str):
                                facts = [facts]
                            for idx_fact, fact in enumerate(facts):
                                if fact and fact.strip():
                                    log_id = hashlib.sha256(f"{episode_id}_fact_{idx_fact}".encode()).hexdigest()[:32]
                                    storage_log = StorageEventLogRecord(
                                        id=log_id,
                                        user_id=user_id,
                                        atomic_fact=fact.strip(),
                                        timestamp=now,
                                        parent_episode_id=episode_id,
                                        metadata={"account_id": account_id, "message_id": record.id},
                                    )
                                    memory_service.storage.upsert_event_log(storage_log)
                                    facts_extracted.append(fact.strip()[:100])
                    except Exception as e:
                        logger.warning("Event log extraction failed for email %s: %s", record.id, e)
                        
            except Exception as e:
                logger.warning("Episode extraction failed for email %s: %s", record.id, e)
                self.update_email_memory_status(record.id, 'failed', f"Episode extraction failed: {str(e)}")
                yield {
                    "type": "error",
                    "message_id": message_id,
                    "error": f"Episode extraction failed: {str(e)}"
                }
                return
            
            # 标记为成功
            self.update_email_memory_status(record.id, 'success')
            
            yield {
                "type": "complete",
                "message_id": message_id,
                "subject": record.subject or "(No Subject)",
                "memcell_created": True,
                "episode_created": episode_created,
                "facts_count": len(facts_extracted),
                "message": f"Retry successful: {record.subject or '(No Subject)'}"
            }
            
        except Exception as e:
            logger.error("Failed to retry email %s: %s", message_id, e)
            self.update_email_memory_status(record.id, 'failed', str(e))
            yield {
                "type": "error",
                "message_id": message_id,
                "error": str(e)
            }

    async def get_account_memory_status(self, account_id: str, user_id: str) -> dict:
        """获取邮箱账户的 Memory 状态"""
        account = self.get_email_account(account_id)
        if not account:
            raise EmailAccountNotFound("Email account not found.")
        
        try:
            from services.memory.service import get_memory_service
            memory_service = get_memory_service()
            
            # 查找与该账户相关的 MemCells (通过 group_id 前缀匹配)
            group_id = f"email_account::{account_id}"
            memcells = memory_service.storage.get_memcells_by_group_id(group_id)
            
            if not memcells:
                return {
                    "account_id": account_id,
                    "is_built": False,
                    "memcell_count": 0,
                    "episode_count": 0,
                    "event_log_count": 0,
                    "last_built_at": None,
                }
            
            # 统计 episodes 和 event logs
            episode_count = 0
            event_log_count = 0
            latest_timestamp = None
            
            for memcell in memcells:
                episodes = memory_service.storage.get_episodes_by_memcell(memcell.id)
                episode_count += len(episodes)
                
                for ep in episodes:
                    logs = memory_service.storage.get_event_logs_by_episode(ep.id)
                    event_log_count += len(logs)
                
                # 跟踪最新时间戳
                if memcell.timestamp:
                    if latest_timestamp is None or memcell.timestamp > latest_timestamp:
                        latest_timestamp = memcell.timestamp
            
            return {
                "account_id": account_id,
                "is_built": True,
                "memcell_count": len(memcells),
                "episode_count": episode_count,
                "event_log_count": event_log_count,
                "last_built_at": latest_timestamp,
            }
            
        except Exception as e:
            logger.warning("Failed to get memory status for account %s: %s", account_id, e)
            return {
                "account_id": account_id,
                "is_built": False,
                "memcell_count": 0,
                "episode_count": 0,
                "event_log_count": 0,
                "last_built_at": None,
            }

    async def account_qa(
        self, 
        account_id: str, 
        question: str, 
        user_id: str,
    ) -> dict:
        """
        邮箱账户级别问答：基于该账户的所有邮件记忆进行问答
        
        Args:
            account_id: 邮箱账户 ID
            question: 用户问题
            user_id: 用户 ID
            
        Returns:
            QA 结果 dict
        """
        account = self.get_email_account(account_id)
        if not account:
            raise EmailAccountNotFound("Email account not found.")
        
        try:
            from core.context import get_llm_client
            from services.memory.service import get_memory_service
            
            llm_client = get_llm_client()
            memory_service = get_memory_service()
            sources = []
            
            # 获取该账户的邮件记忆 (通过 group_id 过滤)
            group_id = f"email_account::{account_id}"
            memcells = memory_service.storage.get_memcells_by_group_id(group_id)
            
            # 构建邮件上下文
            email_context = f"""## 邮箱信息
- 邮箱: {account.label} ({account.username})
- 已索引邮件数: {len(memcells)}
"""
            
            # 获取相关记忆 (基于问题检索)
            memory_context = ""
            if memcells:
                # 简单方法：遍历 memcells 找到相关的
                # 更好的方法是使用向量检索，但这里简化处理
                relevant_memories = []
                for memcell in memcells[:20]:  # 限制数量避免太长
                    try:
                        import json
                        data = json.loads(memcell.original_data) if isinstance(memcell.original_data, str) else memcell.original_data
                        if isinstance(data, list) and len(data) > 0:
                            content = data[0].get("content", "")[:500]  # 截断
                            subject = data[0].get("email_subject", "")
                            sender = data[0].get("email_sender", "")
                            relevant_memories.append({
                                "subject": subject,
                                "sender": sender,
                                "preview": content[:200],
                            })
                            sources.append({
                                "type": "email_memory",
                                "id": memcell.id,
                                "subject": subject,
                                "sender": sender,
                            })
                    except Exception:
                        pass
                
                if relevant_memories:
                    memory_context = "\n\n## 邮件记忆摘要\n"
                    for i, mem in enumerate(relevant_memories[:10], 1):
                        memory_context += f"{i}. **{mem['subject']}** (from: {mem['sender']})\n"
                        memory_context += f"   {mem['preview'][:100]}...\n\n"
            
            # 构建 prompt
            system_prompt = f"""你是一个智能邮件助手，帮助用户分析和理解他们的邮箱内容。
当前用户正在查询邮箱 "{account.label}" 中的邮件信息。
请基于提供的邮件记忆回答用户的问题。
回答要简洁准确，如果信息不足请明确说明。"""
            
            user_prompt = f"""
{email_context}
{memory_context}

## 用户问题
{question}

请基于上述邮件记忆回答用户的问题："""
            
            # 调用 LLM
            answer = await llm_client.chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            
            return {
                "answer": answer,
                "sources": sources[:10],  # 限制返回数量
                "account_id": account_id,
                "memories_used": len(sources),
            }
            
        except Exception as e:
            logger.error("Account QA failed for %s: %s", account_id, e)
            raise EmailServiceError(f"QA failed: {str(e)}") from e

    async def get_account_memory_details(
        self, 
        account_id: str, 
        user_id: str,
        limit: int = 50,
    ) -> dict:
        """
        获取邮箱账户的 Memory 详情：MemCells、Episodes 和 Facts 列表
        
        Args:
            account_id: 邮箱账户 ID
            user_id: 用户 ID
            limit: 最大返回数量
            
        Returns:
            包含 memcells、episodes、facts 列表的 dict
        """
        account = self.get_email_account(account_id)
        if not account:
            raise EmailAccountNotFound("Email account not found.")
        
        try:
            from services.memory.service import get_memory_service
            import json
            
            memory_service = get_memory_service()
            group_id = f"email_account::{account_id}"
            
            # 获取该账户的所有 MemCells
            memcells = memory_service.storage.get_memcells_by_group_id(group_id)
            
            memcells_list = []
            episodes_list = []
            facts_list = []
            
            for memcell in memcells[:limit]:
                # 从 memcell 的 original_data 中提取邮件信息
                email_subject = None
                email_sender = None
                email_preview = None
                try:
                    if memcell.original_data:
                        data = json.loads(memcell.original_data) if isinstance(memcell.original_data, str) else memcell.original_data
                        if isinstance(data, list) and len(data) > 0:
                            email_subject = data[0].get("email_subject")
                            email_sender = data[0].get("email_sender")
                            content = data[0].get("content", "")
                            email_preview = content[:200] if content else None
                except Exception:
                    pass
                
                # 添加 MemCell
                memcells_list.append({
                    "id": memcell.id,
                    "email_subject": email_subject or memcell.subject or "(No Subject)",
                    "email_sender": email_sender or "",
                    "preview": email_preview or memcell.summary or "",
                    "timestamp": memcell.timestamp,
                })
                
                # 获取 MemCell 关联的 Episodes
                episodes = memory_service.storage.get_episodes_by_memcell(memcell.id)
                
                for ep in episodes:
                    # 添加 Episode
                    episodes_list.append({
                        "id": ep.id,
                        "memcell_id": memcell.id,
                        "email_subject": email_subject or ep.subject,
                        "summary": ep.summary or "",
                        "episode": ep.episode or "",
                        "timestamp": ep.timestamp,
                    })
                    
                    # 获取 Episode 关联的 EventLogs (Facts)
                    event_logs = memory_service.storage.get_event_logs_by_episode(ep.id)
                    for log in event_logs:
                        if log.atomic_fact:
                            facts_list.append({
                                "id": log.id,
                                "episode_id": ep.id,
                                "email_subject": email_subject or ep.subject,
                                "fact": log.atomic_fact,
                                "timestamp": log.timestamp,
                            })
            
            result = {
                "account_id": account_id,
                "memcells": memcells_list,
                "episodes": episodes_list,
                "facts": facts_list,
                "total_memcells": len(memcells_list),
                "total_episodes": len(episodes_list),
                "total_facts": len(facts_list),
            }
            logger.info(
                "get_account_memory_details: account=%s, memcells=%d, episodes=%d, facts=%d",
                account_id, len(memcells_list), len(episodes_list), len(facts_list)
            )
            return result
            
        except Exception as e:
            logger.warning("Failed to get memory details for account %s: %s", account_id, e)
            return {
                "account_id": account_id,
                "memcells": [],
                "episodes": [],
                "facts": [],
                "total_memcells": 0,
                "total_episodes": 0,
                "total_facts": 0,
            }

# Lazy initialization of service
email_service_global: Optional[EmailService] = None

def get_email_service() -> EmailService:
    global email_service_global

    if email_service_global is None:
        raise RuntimeError("Email service not initialized")
    return email_service_global

def init_plugin_service(indexer: Indexer, plugin_id: str = "") -> EmailService:
    global email_service_global
    
    if email_service_global is None:
        try:
            # EmailService now inherits from StorageBase and initializes its own connection
            email_service_global = EmailService(indexer, plugin_id)
        except Exception as e:
            logger.warning(f"Failed to initialize global email service: {e}")
            
    if email_service_global:
        return email_service_global

    raise RuntimeError("Email service not initialized")
