"""Mode C: custom HTTP APIs described entirely by validated configuration.

Safety model
------------
* Request bodies and query parameters are JSON *templates*. A string that is exactly ``{{var}}`` is
  replaced by the typed value of ``var``; ``{{var}}`` inside a longer string is replaced by its string
  form. Only a fixed whitelist of variables exists per capability. Nothing is evaluated or executed.
* Credentials are injected only through the ``auth`` block, never through templates.
* Response fields are selected by a minimal path language: ``a.b[0].c`` and ``items[*].field``.
* The path is static (no templating), so requests always go to the configured host.
"""

from __future__ import annotations

import re
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.providers.base import (
    Capability,
    ChatModelProvider,
    ChatRequest,
    ChatResponse,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResult,
    ProviderConfigBase,
    ProviderError,
    ProviderErrorKind,
    RerankingProvider,
    RerankRequest,
    RerankResponse,
    RerankResult,
    Usage,
    WebSearchProvider,
    WebSearchRequest,
    WebSearchResponse,
    WebSearchResult,
)
from app.providers.common import l2_normalize, validate_headers
from app.providers.http import join_url

_VAR = re.compile(r"\{\{\s*([a-z_][a-z0-9_]*)\s*\}\}")
_PATH_TOKEN = re.compile(r"([A-Za-z0-9_\-]+)|\[(\d+|\*)\]")

VARIABLES: dict[str, set[str]] = {
    "chat": {"messages", "prompt", "system", "model", "temperature", "max_tokens"},
    "embedding": {"texts", "text", "input_type", "model"},
    "reranking": {"query", "documents", "top_n", "model"},
    "web_search": {"query", "max_results", "model", "recency_days"},
}

REQUIRED_MAPPINGS: dict[str, set[str]] = {
    "chat": {"text"},
    "embedding": {"vectors"},
    "reranking": {"scores"},
    "web_search": {"items", "url"},
}


# --------------------------------------------------------------------------- template rendering


def template_variables(template: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(template, str):
        found.update(_VAR.findall(template))
    elif isinstance(template, dict):
        for k, v in template.items():
            if "{{" in str(k):
                raise ValueError("Template variables are not allowed in object keys")
            found |= template_variables(v)
    elif isinstance(template, list):
        for v in template:
            found |= template_variables(v)
    return found


def render_template(template: Any, variables: dict[str, Any]) -> Any:
    if isinstance(template, str):
        m = _VAR.fullmatch(template.strip())
        if m:
            return variables.get(m.group(1))

        def sub(mm: re.Match) -> str:
            v = variables.get(mm.group(1))
            if isinstance(v, (dict, list)):
                raise ValueError(f"Variable {mm.group(1)} is structured; use it as a whole value")
            return "" if v is None else str(v)

        return _VAR.sub(sub, template)
    if isinstance(template, dict):
        return {k: render_template(v, variables) for k, v in template.items()}
    if isinstance(template, list):
        return [render_template(v, variables) for v in template]
    return template


# --------------------------------------------------------------------------- response paths


def parse_path(path: str) -> list[str | int]:
    tokens: list[str | int] = []
    for part in path.split("."):
        if not part:
            raise ValueError(f"Invalid path {path!r}")
        consumed = 0
        for m in _PATH_TOKEN.finditer(part):
            if m.start() != consumed:
                raise ValueError(f"Invalid path segment {part!r}")
            consumed = m.end()
            if m.group(1) is not None:
                tokens.append(m.group(1))
            else:
                tokens.append("*" if m.group(2) == "*" else int(m.group(2)))
        if consumed != len(part):
            raise ValueError(f"Invalid path segment {part!r}")
    if not tokens:
        raise ValueError("Empty path")
    return tokens


def select(data: Any, path: str) -> Any:
    """Evaluate a path. ``[*]`` maps the rest of the path over a list and returns a list."""
    tokens = parse_path(path)

    def walk(node: Any, i: int) -> Any:
        if i == len(tokens):
            return node
        t = tokens[i]
        if t == "*":
            if not isinstance(node, list):
                raise KeyError(path)
            return [walk(x, i + 1) for x in node]
        if isinstance(t, int):
            if not isinstance(node, list) or t >= len(node):
                raise KeyError(path)
            return walk(node[t], i + 1)
        if not isinstance(node, dict) or t not in node:
            raise KeyError(path)
        return walk(node[t], i + 1)

    return walk(data, 0)


# --------------------------------------------------------------------------- configuration


class CustomEndpoint(BaseModel):
    method: Literal["GET", "POST", "PUT"] = "POST"
    path: str = Field(default="", description="Static path appended to base_url (no templates)")
    query_params: dict[str, Any] = Field(default_factory=dict, description="Template for query string")
    body_template: Any = Field(default=None, description="JSON template for the request body")
    response_mapping: dict[str, str] = Field(default_factory=dict, description="Normalized field → response path")
    model: str = ""
    normalize_vectors: bool = True

    @field_validator("path")
    @classmethod
    def _static_path(cls, v: str) -> str:
        if "{{" in v or "://" in v or ".." in v or v.startswith("//"):
            raise ValueError("path must be a static relative path")
        return v

    @field_validator("response_mapping")
    @classmethod
    def _paths(cls, v: dict[str, str]) -> dict[str, str]:
        for p in v.values():
            parse_path(p)
        return v


class CustomAuth(BaseModel):
    type: Literal["none", "bearer", "header", "query"] = "bearer"
    header_name: str = "Authorization"
    query_param: str = "api_key"

    @field_validator("header_name")
    @classmethod
    def _hn(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9\-]{1,64}", v):
            raise ValueError("invalid header name")
        return v


class CustomHTTPConfig(ProviderConfigBase):
    base_url: str = Field(..., description="Fixed API root; requests never leave this host")
    auth: CustomAuth = Field(default_factory=CustomAuth)
    extra_headers: dict[str, str] = Field(default_factory=dict)
    chat: CustomEndpoint | None = None
    embedding: CustomEndpoint | None = None
    reranking: CustomEndpoint | None = None
    web_search: CustomEndpoint | None = None

    @field_validator("extra_headers")
    @classmethod
    def _headers(cls, v: dict[str, str]) -> dict[str, str]:
        return validate_headers(v)

    @model_validator(mode="after")
    def _validate_templates(self) -> "CustomHTTPConfig":
        any_endpoint = False
        for cap in VARIABLES:
            ep: CustomEndpoint | None = getattr(self, cap)
            if ep is None:
                continue
            any_endpoint = True
            used = template_variables(ep.body_template) | template_variables(ep.query_params)
            unknown = used - VARIABLES[cap]
            if unknown:
                raise ValueError(f"{cap}: unknown template variables {sorted(unknown)}; allowed: {sorted(VARIABLES[cap])}")
            missing = REQUIRED_MAPPINGS[cap] - set(ep.response_mapping)
            if missing:
                raise ValueError(f"{cap}: response_mapping must define {sorted(missing)}")
        if not any_endpoint:
            raise ValueError("Configure at least one endpoint (chat, embedding, reranking or web_search)")
        return self


class CustomHTTPProvider(ChatModelProvider, EmbeddingProvider, RerankingProvider, WebSearchProvider):
    adapter = "custom_http"
    display_name = "Custom HTTP API (declarative)"
    protocol = "custom"
    capabilities = frozenset({Capability.CHAT, Capability.EMBEDDING, Capability.RERANKING, Capability.WEB_SEARCH})
    description = "Describe an HTTP API with a request template and response mapping. No code is executed from configuration."
    limitations = [
        "Only JSON request/response bodies are supported.",
        "Streaming, multipart uploads and multi-step auth flows need a dedicated adapter.",
        "Images are not supported by templates (use a native vision adapter).",
    ]
    config_model = CustomHTTPConfig
    secret_fields = ["api_key"]
    config: CustomHTTPConfig

    @property
    def active_capabilities(self) -> frozenset[Capability]:
        configured = {Capability(c) for c in VARIABLES if getattr(self.config, c) is not None}
        enabled = set(self.config.enabled_capabilities or configured)
        return frozenset(configured & enabled)

    def model_for(self, capability: Capability) -> str | None:
        ep = getattr(self.config, capability.value, None)
        return (ep.model or None) if ep else None

    def embedding_fingerprint(self) -> dict[str, Any]:
        ep = self.config.embedding
        return {"adapter": self.adapter, "endpoint": join_url(self.config.base_url, ep.path if ep else ""), "model": ep.model if ep else None, "normalize": ep.normalize_vectors if ep else None}

    async def _call(self, cap: str, variables: dict[str, Any]) -> Any:
        ep: CustomEndpoint | None = getattr(self.config, cap)
        if ep is None:
            raise ProviderError(ProviderErrorKind.UNSUPPORTED, f"No {cap} endpoint configured", provider_id=self.provider_id)
        variables = {**variables, "model": ep.model}
        headers = {"Accept": "application/json", **self.config.extra_headers}
        params = {k: v for k, v in render_template(ep.query_params, variables).items() if v is not None}
        key = self.secrets.get("api_key")
        auth = self.config.auth
        if auth.type != "none":
            if not key:
                raise ProviderError(ProviderErrorKind.CONFIG, "api_key is required by the auth configuration", provider_id=self.provider_id)
            if auth.type == "bearer":
                headers["Authorization"] = f"Bearer {key}"
            elif auth.type == "header":
                headers[auth.header_name] = key
            else:
                params[auth.query_param] = key
        body = render_template(ep.body_template, variables) if ep.body_template is not None and ep.method != "GET" else None
        return await self.http.request_json(
            self.provider_id, ep.method, join_url(self.config.base_url, ep.path), headers=headers, params=params or None, json=body, timeout=self.config.timeout_s
        ), ep

    def _get(self, data: Any, ep: CustomEndpoint, field: str, default: Any = ...) -> Any:
        path = ep.response_mapping.get(field)
        if path is None:
            if default is ...:
                raise ProviderError(ProviderErrorKind.CONFIG, f"response_mapping.{field} missing", provider_id=self.provider_id)
            return default
        try:
            return select(data, path)
        except KeyError:
            if default is not ...:
                return default
            raise ProviderError(ProviderErrorKind.INVALID_RESPONSE, f"Response has no value at '{path}' ({field})", provider_id=self.provider_id)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        msgs = [{"role": m.role, "content": m.content} for m in request.messages]
        variables = {
            "messages": msgs,
            "prompt": "\n\n".join(m.content for m in request.messages if m.role != "system"),
            "system": "\n\n".join(m.content for m in request.messages if m.role == "system"),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        data, ep = await self._call("chat", variables)
        text = self._get(data, ep, "text")
        if not isinstance(text, str):
            raise ProviderError(ProviderErrorKind.INVALID_RESPONSE, "Mapped chat text is not a string", provider_id=self.provider_id)
        return ChatResponse(
            text=text,
            model=ep.model or None,
            usage=Usage(input_tokens=self._get(data, ep, "input_tokens", None), output_tokens=self._get(data, ep, "output_tokens", None)),
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        data, ep = await self._call("embedding", {"texts": request.texts, "text": request.texts[0] if request.texts else "", "input_type": request.input_type})
        vecs = self._get(data, ep, "vectors")
        if not isinstance(vecs, list) or len(vecs) != len(request.texts) or not all(isinstance(v, list) for v in vecs):
            raise ProviderError(ProviderErrorKind.INVALID_RESPONSE, "Mapped vectors are not a list of float lists matching the input count", provider_id=self.provider_id)
        vecs = [l2_normalize([float(x) for x in v]) if ep.normalize_vectors else [float(x) for x in v] for v in vecs]
        return EmbeddingResponse(vectors=vecs, model=ep.model or None)

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        docs = [c.text for c in request.candidates]
        data, ep = await self._call("reranking", {"query": request.query, "documents": docs, "top_n": request.top_n or len(docs)})
        scores = self._get(data, ep, "scores")
        indices = self._get(data, ep, "indices", None)
        if not isinstance(scores, list):
            raise ProviderError(ProviderErrorKind.INVALID_RESPONSE, "Mapped scores are not a list", provider_id=self.provider_id)
        if indices is None:
            if len(scores) != len(docs):
                raise ProviderError(ProviderErrorKind.INVALID_RESPONSE, "Scores without indices must match candidate count", provider_id=self.provider_id)
            indices = list(range(len(docs)))
        pairs = sorted(((int(i), float(s)) for i, s in zip(indices, scores) if 0 <= int(i) < len(docs)), key=lambda p: p[1], reverse=True)
        n = request.top_n or len(pairs)
        return RerankResponse(results=[RerankResult(id=request.candidates[i].id, score=s, rank=r) for r, (i, s) in enumerate(pairs[:n], 1)], model=ep.model or None)

    async def search(self, request: WebSearchRequest) -> WebSearchResponse:
        data, ep = await self._call("web_search", {"query": request.query, "max_results": request.max_results, "recency_days": request.recency_days})
        items = self._get(data, ep, "items")
        if not isinstance(items, list):
            raise ProviderError(ProviderErrorKind.INVALID_RESPONSE, "Mapped items is not a list", provider_id=self.provider_id)
        out = []
        for it in items[: request.max_results]:
            try:
                url = select(it, ep.response_mapping["url"])
            except KeyError:
                continue
            def opt(field: str) -> Any:
                p = ep.response_mapping.get(field)
                if not p:
                    return None
                try:
                    return select(it, p)
                except KeyError:
                    return None
            out.append(WebSearchResult(url=str(url), title=str(opt("title") or ""), snippet=str(opt("snippet") or ""), published_at=opt("published_at"), source_engine="custom"))
        return WebSearchResponse(results=out, engine=f"custom:{self.provider_id}")

    async def health_check(self) -> HealthResult:
        t0 = time.perf_counter()
        checked = []
        for cap in self.active_capabilities:
            if cap == Capability.CHAT:
                from app.providers.base import ChatMessage

                await self.chat(ChatRequest(messages=[ChatMessage(role="user", content="Reply OK")], max_tokens=5))
            elif cap == Capability.EMBEDDING:
                await self.embed(EmbeddingRequest(texts=["health"]))
            elif cap == Capability.RERANKING:
                from app.providers.base import RerankCandidate

                await self.rerank(RerankRequest(query="a", candidates=[RerankCandidate(id="1", text="a"), RerankCandidate(id="2", text="b")]))
            elif cap == Capability.WEB_SEARCH:
                await self.search(WebSearchRequest(query="test", max_results=1))
            checked.append(cap.value)
        return HealthResult(ok=True, latency_ms=(time.perf_counter() - t0) * 1000, detail="OK", checked=checked)
