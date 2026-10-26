"""Request dependencies: container access and admin authentication."""

from __future__ import annotations

import hmac
import ipaddress

from fastapi import HTTPException, Request

from app.core.container import Container


def container(request: Request) -> Container:
    return request.app.state.container


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    if host in ("localhost", "testclient"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def require_admin(request: Request) -> None:
    """Admin endpoints (providers, credentials, routes, reindexing, settings).

    With OMNI_ADMIN_TOKEN set: a matching ``Authorization: Bearer`` header is required.
    Without it: only loopback clients are allowed, so a default local install is not remotely administrable.
    """
    token = request.app.state.container.settings.admin_token
    if token:
        header = request.headers.get("authorization", "")
        supplied = header[7:] if header.lower().startswith("bearer ") else ""
        if not supplied or not hmac.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(status_code=401, detail="Admin token required", headers={"WWW-Authenticate": "Bearer"})
        return
    client = request.client.host if request.client else None
    if not _is_loopback(client):
        raise HTTPException(status_code=403, detail="Remote admin access requires OMNI_ADMIN_TOKEN to be configured")
