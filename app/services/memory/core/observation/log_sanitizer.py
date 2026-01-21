"""
Log Sanitizer - Redacts sensitive information from log messages for privacy protection.

Supports redaction of:
- API keys and tokens
- Passwords and secrets
- Email addresses
- File paths with usernames
- IP addresses
- Credit card numbers
- JWT tokens
- Bearer tokens
"""

import re
import os
import logging
from typing import List, Tuple, Pattern, Optional
from functools import lru_cache


class LogSanitizer:
    """Sanitizes log messages by redacting sensitive information."""

    # Redaction placeholder
    REDACTED = "[REDACTED]"
    REDACTED_EMAIL = "[EMAIL_REDACTED]"
    REDACTED_PATH = "[PATH_REDACTED]"
    REDACTED_IP = "[IP_REDACTED]"
    REDACTED_CARD = "[CARD_REDACTED]"
    REDACTED_JWT = "[JWT_REDACTED]"

    def __init__(self, enabled: bool = True):
        """
        Initialize the log sanitizer.

        Args:
            enabled: Whether sanitization is enabled. Can be controlled via
                     LOG_SANITIZE_ENABLED environment variable.
        """
        env_enabled = os.getenv("LOG_SANITIZE_ENABLED", "true").lower()
        self._enabled = enabled and env_enabled in ("true", "1", "yes")
        self._patterns: List[Tuple[Pattern, str]] = self._compile_patterns()
        self._username = self._get_current_username()

    @staticmethod
    def _get_current_username() -> Optional[str]:
        """Get the current system username for path redaction."""
        try:
            import getpass
            return getpass.getuser()
        except Exception:
            return None

    def _compile_patterns(self) -> List[Tuple[Pattern, str]]:
        """Compile regex patterns for sensitive data detection."""
        patterns = []

        # API Keys - common patterns
        # Format: api_key=xxx, apikey=xxx, api-key=xxx, x-api-key: xxx
        patterns.append((
            re.compile(
                r'(?i)(api[-_]?key|apikey|x-api-key)\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{16,})["\']?',
                re.IGNORECASE
            ),
            r'\1=' + self.REDACTED
        ))

        # Bearer tokens
        patterns.append((
            re.compile(
                r'(?i)(bearer\s+)([a-zA-Z0-9_\-\.]+)',
                re.IGNORECASE
            ),
            r'\1' + self.REDACTED
        ))

        # Authorization headers
        patterns.append((
            re.compile(
                r'(?i)(authorization\s*[=:]\s*)["\']?([^"\'>\s]+)["\']?',
                re.IGNORECASE
            ),
            r'\1' + self.REDACTED
        ))

        # Password patterns
        # Format: password=xxx, passwd=xxx, pwd=xxx, pass=xxx
        patterns.append((
            re.compile(
                r'(?i)(password|passwd|pwd|pass)\s*[=:]\s*["\']?([^\s"\'&]+)["\']?',
                re.IGNORECASE
            ),
            r'\1=' + self.REDACTED
        ))

        # Secret/Token patterns
        # Format: secret=xxx, token=xxx, access_token=xxx, refresh_token=xxx
        patterns.append((
            re.compile(
                r'(?i)(secret|token|access_token|refresh_token|client_secret)\s*[=:]\s*["\']?([a-zA-Z0-9_\-\.]{8,})["\']?',
                re.IGNORECASE
            ),
            r'\1=' + self.REDACTED
        ))

        # JWT tokens (eyXXX.eyXXX.XXX format)
        patterns.append((
            re.compile(
                r'\beyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*\b'
            ),
            self.REDACTED_JWT
        ))

        # Email addresses
        patterns.append((
            re.compile(
                r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
            ),
            self.REDACTED_EMAIL
        ))

        # Credit card numbers (basic pattern - 13-19 digits with optional spaces/dashes)
        patterns.append((
            re.compile(
                r'\b(?:\d{4}[-\s]?){3,4}\d{1,4}\b'
            ),
            self.REDACTED_CARD
        ))

        # IPv4 addresses (optional - can be too aggressive)
        # Only redact if LOG_SANITIZE_IP is enabled
        if os.getenv("LOG_SANITIZE_IP", "false").lower() in ("true", "1", "yes"):
            patterns.append((
                re.compile(
                    r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
                ),
                self.REDACTED_IP
            ))

        # Private keys (BEGIN...PRIVATE KEY)
        patterns.append((
            re.compile(
                r'-----BEGIN[A-Z\s]*PRIVATE KEY-----[\s\S]*?-----END[A-Z\s]*PRIVATE KEY-----',
                re.MULTILINE
            ),
            self.REDACTED
        ))

        # AWS Access Key ID
        patterns.append((
            re.compile(
                r'(?i)(aws_access_key_id|aws_secret_access_key)\s*[=:]\s*["\']?([A-Za-z0-9/+=]{16,})["\']?',
                re.IGNORECASE
            ),
            r'\1=' + self.REDACTED
        ))

        # Connection strings with passwords
        patterns.append((
            re.compile(
                r'(?i)(mongodb|mysql|postgres|redis|amqp)://[^:]+:([^@]+)@',
                re.IGNORECASE
            ),
            r'\1://***:' + self.REDACTED + '@'
        ))

        return patterns

    @lru_cache(maxsize=256)
    def _sanitize_path_cached(self, path: str) -> str:
        """Cached version of path sanitization for performance."""
        if not self._username or self._username not in path:
            return path

        # Replace username in paths like /Users/username/ or /home/username/
        # macOS pattern
        sanitized = re.sub(
            rf'/Users/{re.escape(self._username)}/',
            '/Users/[USER]/',
            path
        )
        # Linux pattern
        sanitized = re.sub(
            rf'/home/{re.escape(self._username)}/',
            '/home/[USER]/',
            sanitized
        )
        # Windows pattern (C:\Users\username\)
        sanitized = re.sub(
            rf'[A-Za-z]:\\Users\\{re.escape(self._username)}\\',
            r'C:\\Users\\[USER]\\',
            sanitized,
            flags=re.IGNORECASE
        )

        return sanitized

    def sanitize(self, message: str) -> str:
        """
        Sanitize a log message by redacting sensitive information.

        Args:
            message: The log message to sanitize.

        Returns:
            The sanitized message with sensitive data redacted.
        """
        if not self._enabled or not message:
            return message

        sanitized = message

        # Apply all regex patterns
        for pattern, replacement in self._patterns:
            sanitized = pattern.sub(replacement, sanitized)

        # Sanitize file paths to remove usernames
        if self._username:
            sanitized = self._sanitize_path_cached(sanitized)

        return sanitized

    def enable(self) -> None:
        """Enable log sanitization."""
        self._enabled = True

    def disable(self) -> None:
        """Disable log sanitization."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if sanitization is enabled."""
        return self._enabled


class SanitizingFormatter(logging.Formatter):
    """
    A logging formatter that sanitizes log messages before output.

    Usage:
        formatter = SanitizingFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        sanitizer: Optional[LogSanitizer] = None
    ):
        super().__init__(fmt, datefmt)
        self._sanitizer = sanitizer or LogSanitizer()

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with sanitization."""
        # Sanitize the message
        original_msg = record.msg
        if isinstance(record.msg, str):
            record.msg = self._sanitizer.sanitize(record.msg)

        # Also sanitize args if they contain strings
        original_args = record.args
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self._sanitizer.sanitize(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    self._sanitizer.sanitize(str(arg)) if isinstance(arg, str) else arg
                    for arg in record.args
                )

        try:
            formatted = super().format(record)
            # Also sanitize the final formatted message (catches exception info etc.)
            return self._sanitizer.sanitize(formatted)
        finally:
            # Restore original message for any other handlers
            record.msg = original_msg
            record.args = original_args


class SanitizingFilter(logging.Filter):
    """
    A logging filter that sanitizes log messages.

    This is an alternative to SanitizingFormatter that works at the filter level,
    which means it sanitizes before the formatter processes the record.

    Usage:
        filter = SanitizingFilter()
        logger.addFilter(filter)
    """

    def __init__(self, name: str = '', sanitizer: Optional[LogSanitizer] = None):
        super().__init__(name)
        self._sanitizer = sanitizer or LogSanitizer()

    def filter(self, record: logging.LogRecord) -> bool:
        """Sanitize the record and allow it to pass."""
        if isinstance(record.msg, str):
            record.msg = self._sanitizer.sanitize(record.msg)

        # Sanitize args
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self._sanitizer.sanitize(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    self._sanitizer.sanitize(str(arg)) if isinstance(arg, str) else arg
                    for arg in record.args
                )

        return True


# Global sanitizer instance
_global_sanitizer: Optional[LogSanitizer] = None


def get_sanitizer() -> LogSanitizer:
    """Get the global log sanitizer instance."""
    global _global_sanitizer
    if _global_sanitizer is None:
        _global_sanitizer = LogSanitizer()
    return _global_sanitizer


def sanitize_log(message: str) -> str:
    """
    Convenience function to sanitize a log message.

    Args:
        message: The message to sanitize.

    Returns:
        The sanitized message.
    """
    return get_sanitizer().sanitize(message)

