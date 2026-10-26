"""Outbound-destination controls: SSRF protection for provider endpoints and web fetching."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass, field
from urllib.parse import urlsplit


class UnsafeURLError(ValueError):
    """Raised when a URL targets a destination the policy does not allow."""


@dataclass
class OutboundPolicy:
    allow_private_hosts: list[str] = field(default_factory=list)
    provider_allowlist: list[str] = field(default_factory=list)
    blocklist: list[str] = field(default_factory=list)
    resolve_dns: bool = True  # tests disable resolution; literal IPs are still checked

    @staticmethod
    def _host_matches(host: str, patterns: list[str]) -> bool:
        host = host.lower().rstrip(".")
        for p in patterns:
            p = p.lower().strip().lstrip("*.").rstrip(".")
            if p and (host == p or host.endswith("." + p)):
                return True
        return False

    def check_syntax(self, url: str, *, purpose: str) -> tuple[str, str, int]:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            raise UnsafeURLError(f"Only http/https URLs are allowed (got {parts.scheme!r})")
        if not parts.hostname:
            raise UnsafeURLError("URL has no host")
        if parts.username or parts.password:
            raise UnsafeURLError("Credentials embedded in URLs are not allowed")
        host = parts.hostname.lower()
        if self._host_matches(host, self.blocklist):
            raise UnsafeURLError(f"Host {host} is blocked by policy")
        if purpose == "provider" and self.provider_allowlist and not self._host_matches(host, self.provider_allowlist):
            raise UnsafeURLError(f"Host {host} is not in the provider allowlist")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        return parts.scheme, host, port

    @staticmethod
    def _is_private(ip: ipaddress._BaseAddress) -> bool:
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )

    async def validate(self, url: str, *, purpose: str = "web") -> str:
        """Validate ``url`` and the addresses it resolves to. Returns the hostname.

        ``purpose`` is ``provider`` (configured AI/search endpoints) or ``web`` (agent-fetched pages).
        Agent-fetched pages may never reach private networks; provider endpoints may when the host is
        explicitly listed in ``allow_private_hosts`` (e.g. a local vLLM or Ollama server).
        """
        _, host, port = self.check_syntax(url, purpose=purpose)
        private_ok = purpose == "provider" and self._host_matches(host, self.allow_private_hosts)
        try:
            literal = ipaddress.ip_address(host)
            addrs = [literal]
        except ValueError:
            if not self.resolve_dns:
                if host in ("localhost",) and not private_ok:
                    raise UnsafeURLError(f"Host {host} is a loopback name; not allowed for {purpose}")
                return host
            loop = asyncio.get_running_loop()
            try:
                infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            except socket.gaierror as exc:
                raise UnsafeURLError(f"Cannot resolve host {host}") from exc
            addrs = [ipaddress.ip_address(info[4][0].split("%")[0]) for info in infos]
        for ip in addrs:
            if self._is_private(ip) and not private_ok:
                raise UnsafeURLError(f"Host {host} resolves to a non-public address ({ip}); not allowed for {purpose}")
        return host
