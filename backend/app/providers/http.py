"""Shared HTTP client for provider adapters.

Every outbound provider request goes through :class:`ProviderHttp`, which
 * validates the destination against the outbound policy (SSRF protection),
 * never follows redirects (so credentials are never forwarded to another host),
 * maps transport and HTTP failures to normalized :class:`ProviderError` kinds,
 * never puts request headers or bodies into error messages.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from app.core.netguard import OutboundPolicy, UnsafeURLError
from app.observability.logging import redact
from app.providers.base import ProviderError, ProviderErrorKind


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _body_excerpt(resp: httpx.Response, limit: int = 300) -> str:
    try:
        text = resp.text
    except Exception:
        return ""
    text = re.sub(r"\s+", " ", text)[:limit]
    return redact(text)


class ProviderHttp:
    def __init__(self, policy: OutboundPolicy, client: httpx.AsyncClient | None = None):
        self.policy = policy
        self._client = client or httpx.AsyncClient(follow_redirects=False, trust_env=False)

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client

    async def request(
        self,
        provider_id: str,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: Any = None,
        data: Any = None,
        files: Any = None,
        params: dict[str, Any] | None = None,
        timeout: float = 60.0,
    ) -> httpx.Response:
        try:
            await self.policy.validate(url, purpose="provider")
        except UnsafeURLError as exc:
            raise ProviderError(ProviderErrorKind.BLOCKED, str(exc), provider_id=provider_id) from exc
        try:
            resp = await self._client.request(
                method,
                url,
                headers=headers,
                json=json,
                data=data,
                files=files,
                params=params,
                timeout=timeout,
                follow_redirects=False,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(ProviderErrorKind.TIMEOUT, f"Request timed out after {timeout}s", provider_id=provider_id) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(ProviderErrorKind.NETWORK, f"Network error: {type(exc).__name__}", provider_id=provider_id) from exc

        status = resp.status_code
        if 300 <= status < 400:
            raise ProviderError(
                ProviderErrorKind.BAD_REQUEST,
                f"Endpoint returned redirect {status}; redirects are not followed for provider endpoints. Configure the final URL.",
                provider_id=provider_id,
                status_code=status,
            )
        if status in (401, 403):
            raise ProviderError(ProviderErrorKind.AUTH, f"Authentication failed (HTTP {status})", provider_id=provider_id, status_code=status)
        if status == 429:
            raise ProviderError(
                ProviderErrorKind.RATE_LIMIT,
                "Rate limited by provider",
                provider_id=provider_id,
                status_code=status,
                retry_after=_parse_retry_after(resp.headers.get("retry-after")),
            )
        if status in (408, 504):
            raise ProviderError(ProviderErrorKind.TIMEOUT, f"Upstream timeout (HTTP {status})", provider_id=provider_id, status_code=status)
        if status >= 500:
            raise ProviderError(
                ProviderErrorKind.UNAVAILABLE, f"Provider error HTTP {status}: {_body_excerpt(resp)}", provider_id=provider_id, status_code=status
            )
        if status == 404:
            raise ProviderError(ProviderErrorKind.CONFIG, f"Endpoint not found (HTTP 404) — check base URL/path: {_body_excerpt(resp)}", provider_id=provider_id, status_code=status)
        if status >= 400:
            raise ProviderError(
                ProviderErrorKind.BAD_REQUEST, f"Request rejected HTTP {status}: {_body_excerpt(resp)}", provider_id=provider_id, status_code=status
            )
        return resp

    async def request_json(self, provider_id: str, method: str, url: str, **kwargs: Any) -> Any:
        resp = await self.request(provider_id, method, url, **kwargs)
        try:
            return resp.json()
        except ValueError as exc:
            raise ProviderError(
                ProviderErrorKind.INVALID_RESPONSE, f"Expected JSON response, got: {_body_excerpt(resp, 120)}", provider_id=provider_id
            ) from exc

    async def aclose(self) -> None:
        await self._client.aclose()


def join_url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")
