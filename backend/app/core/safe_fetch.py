"""Safe HTTP GET for untrusted URLs chosen by the agent or returned by search engines.

* scheme/host validated against the outbound policy for every redirect hop;
* the connected peer address is re-checked after connect (mitigates DNS rebinding);
* no cookies, no credentials, bounded redirects, bounded body size, timeouts;
* robots.txt is honoured when enabled; 401/402/403 responses are reported as access-restricted
  and never retried with other techniques.
"""

from __future__ import annotations

import asyncio
import ipaddress
import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from app.core.netguard import OutboundPolicy, UnsafeURLError

USER_AGENT = "OmniResearchBot/1.0 (+research assistant; respects robots.txt)"


class FetchError(Exception):
    def __init__(self, kind: str, message: str, status_code: int | None = None):
        super().__init__(message)
        self.kind = kind  # blocked | restricted | robots | too_large | timeout | network | http | unsupported_type
        self.status_code = status_code


@dataclass
class FetchResult:
    url: str
    final_url: str
    status_code: int
    content_type: str
    body: bytes
    truncated: bool
    fetched_at: str


class SafeFetcher:
    def __init__(
        self,
        policy: OutboundPolicy,
        *,
        max_bytes: int = 3_000_000,
        timeout_s: float = 20.0,
        max_redirects: int = 5,
        respect_robots: bool = True,
        client: httpx.AsyncClient | None = None,
    ):
        self.policy = policy
        self.max_bytes = max_bytes
        self.timeout_s = timeout_s
        self.max_redirects = max_redirects
        self.respect_robots = respect_robots
        self._client = client or httpx.AsyncClient(follow_redirects=False, trust_env=False, headers={"User-Agent": USER_AGENT})
        self._robots: dict[str, tuple[float, RobotFileParser | None]] = {}
        self._robots_lock = asyncio.Lock()

    async def _check_peer(self, resp: httpx.Response) -> None:
        stream = resp.extensions.get("network_stream")
        if stream is None:
            return
        try:
            addr = stream.get_extra_info("server_addr")
        except Exception:
            return
        if not addr:
            return
        try:
            ip = ipaddress.ip_address(str(addr[0]).split("%")[0])
        except ValueError:
            return
        if OutboundPolicy._is_private(ip):
            raise FetchError("blocked", f"Connected address {ip} is not public")

    async def _allowed_by_robots(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        async with self._robots_lock:
            cached = self._robots.get(origin)
            if cached and time.time() - cached[0] < 3600:
                rp = cached[1]
            else:
                rp = None
                try:
                    res = await self._get(origin + "/robots.txt", max_bytes=500_000, check_robots=False, max_redirects=2)
                    if res.status_code == 200:
                        rp = RobotFileParser()
                        rp.parse(res.body.decode("utf-8", "replace").splitlines())
                except FetchError:
                    rp = None
                self._robots[origin] = (time.time(), rp)
        return rp.can_fetch(USER_AGENT, url) if rp else True

    async def fetch(self, url: str, *, accept: str = "text/html,application/xhtml+xml,application/pdf,text/plain;q=0.8") -> FetchResult:
        if not await self._allowed_by_robots(url):
            raise FetchError("robots", "Disallowed by robots.txt")
        return await self._get(url, accept=accept)

    async def _get(self, url: str, *, accept: str = "*/*", max_bytes: int | None = None, check_robots: bool = True, max_redirects: int | None = None) -> FetchResult:
        limit = max_bytes or self.max_bytes
        current = url
        for _hop in range((max_redirects if max_redirects is not None else self.max_redirects) + 1):
            try:
                await self.policy.validate(current, purpose="web")
            except UnsafeURLError as exc:
                raise FetchError("blocked", str(exc)) from exc
            try:
                async with self._client.stream("GET", current, headers={"Accept": accept}, timeout=self.timeout_s) as resp:
                    await self._check_peer(resp)
                    if resp.status_code in (301, 302, 303, 307, 308):
                        loc = resp.headers.get("location")
                        if not loc:
                            raise FetchError("http", "Redirect without Location", resp.status_code)
                        current = urljoin(current, loc)
                        continue
                    if resp.status_code in (401, 402, 403, 407, 451):
                        raise FetchError("restricted", f"Access restricted (HTTP {resp.status_code}); not bypassed", resp.status_code)
                    if resp.status_code >= 400:
                        raise FetchError("http", f"HTTP {resp.status_code}", resp.status_code)
                    declared = resp.headers.get("content-length")
                    if declared and declared.isdigit() and int(declared) > limit * 4:
                        raise FetchError("too_large", f"Content-Length {declared} exceeds limit")
                    chunks: list[bytes] = []
                    size = 0
                    truncated = False
                    async for chunk in resp.aiter_bytes():
                        chunks.append(chunk)
                        size += len(chunk)
                        if size >= limit:
                            truncated = True
                            break
                    body = b"".join(chunks)[:limit]
                    return FetchResult(
                        url=url,
                        final_url=str(resp.url),
                        status_code=resp.status_code,
                        content_type=resp.headers.get("content-type", "").split(";")[0].strip().lower(),
                        body=body,
                        truncated=truncated,
                        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    )
            except httpx.TimeoutException as exc:
                raise FetchError("timeout", f"Timed out after {self.timeout_s}s") from exc
            except httpx.HTTPError as exc:
                raise FetchError("network", f"Network error: {type(exc).__name__}") from exc
        raise FetchError("http", "Too many redirects")

    async def aclose(self) -> None:
        await self._client.aclose()
