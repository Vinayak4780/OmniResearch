"""Structured JSON logging with secret redaction."""

from __future__ import annotations

import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)

_REDACT_PATTERNS = [
    re.compile(r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?)(bearer\s+)?[^\s\"',}]+"),
    re.compile(r"(?i)((?:api[_-]?key|x-api-key|token|secret|password)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]+"),
    re.compile(r"\b(sk-[A-Za-z0-9_\-]{8,})\b"),
    re.compile(r"\b(nvapi-[A-Za-z0-9_\-]{8,})\b"),
    re.compile(r"\b(tvly-[A-Za-z0-9_\-]{8,})\b"),
]

_known_secrets: set[str] = set()


def register_secret(value: str | None) -> None:
    """Register a concrete secret value so it is scrubbed from any log line."""
    if value and len(value) >= 6:
        _known_secrets.add(value)


def redact(text: str) -> str:
    for s in _known_secrets:
        if s in text:
            text = text.replace(s, "***")
    for pat in _REDACT_PATTERNS:
        if pat.groups >= 2:
            text = pat.sub(lambda m: m.group(1) + "***", text)
        else:
            text = pat.sub("***", text)
    return text


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact(record.getMessage()),
        }
        if tid := trace_id_var.get():
            payload["trace_id"] = tid
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update({k: (redact(v) if isinstance(v, str) else v) for k, v in extra.items()})
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # httpx logs full URLs at INFO, which may include query-string credentials.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, msg, extra={"fields": fields})
