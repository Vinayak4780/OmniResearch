"""Helpers shared by adapters (header validation, image encoding, vector math)."""

from __future__ import annotations

import base64
import math
import re
from typing import Any

from app.providers.base import ImageInput, ProviderError, ProviderErrorKind

_HEADER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-]{0,63}$")
_FORBIDDEN_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "cookie", "proxy-authorization"}


def validate_headers(headers: dict[str, str]) -> dict[str, str]:
    """Static, non-secret headers. Credentials belong in secret fields, not here."""
    clean: dict[str, str] = {}
    for k, v in (headers or {}).items():
        if not _HEADER_NAME.match(k):
            raise ValueError(f"Invalid header name: {k!r}")
        if k.lower() in _FORBIDDEN_HEADERS:
            raise ValueError(f"Header {k!r} may not be set")
        if k.lower() in {"authorization", "x-api-key", "api-key"}:
            raise ValueError(f"Header {k!r} carries credentials; use the API key field instead")
        if not isinstance(v, str) or "\r" in v or "\n" in v or len(v) > 1024:
            raise ValueError(f"Invalid value for header {k!r}")
        clean[k] = v
    return clean


def validate_extra_body(body: dict[str, Any], *, reserved: set[str]) -> dict[str, Any]:
    """Extra request parameters must be plain JSON and may not override reserved fields."""

    def walk(x: Any, depth: int = 0) -> None:
        if depth > 5:
            raise ValueError("extra parameters nested too deeply")
        if isinstance(x, dict):
            for k, v in x.items():
                if not isinstance(k, str):
                    raise ValueError("extra parameter keys must be strings")
                walk(v, depth + 1)
        elif isinstance(x, list):
            for v in x:
                walk(v, depth + 1)
        elif not (x is None or isinstance(x, (str, int, float, bool))):
            raise ValueError("extra parameters must be JSON values")

    walk(body)
    clash = reserved & set(body)
    if clash:
        raise ValueError(f"extra parameters may not override: {sorted(clash)}")
    return body


def image_data_url(img: ImageInput) -> str:
    return f"data:{img.mime_type};base64,{base64.b64encode(img.data).decode()}"


def image_b64(img: ImageInput) -> str:
    return base64.b64encode(img.data).decode()


def l2_normalize(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec] if n > 0 else vec


def require(value: Any, provider_id: str, what: str) -> Any:
    if value in (None, "", []):
        raise ProviderError(ProviderErrorKind.CONFIG, f"{what} is not configured", provider_id=provider_id)
    return value


def invalid(provider_id: str, what: str) -> ProviderError:
    return ProviderError(ProviderErrorKind.INVALID_RESPONSE, f"Unexpected response format: {what}", provider_id=provider_id)


def batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]
