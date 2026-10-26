"""Mode A: OpenAI-compatible endpoints (NVIDIA NIM, OpenRouter, ModelScope, vLLM, SGLang, Ollama, ...).

Feature support differs between servers, so optional features (JSON mode, JSON schema, embedding
``dimensions``, ``input_type``, system role) are explicit configuration flags rather than assumptions.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import Field, field_validator

from app.providers.base import (
    Capability,
    ChatModelProvider,
    ChatRequest,
    ChatResponse,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResult,
    ImageEmbeddingProvider,
    ImageEmbeddingRequest,
    ProviderConfigBase,
    ProviderError,
    ProviderErrorKind,
    Usage,
    VisionProvider,
    VisionRequest,
    VisionResponse,
    ChatMessage,
)
from app.providers.common import batched, image_data_url, invalid, l2_normalize, require, validate_extra_body, validate_headers
from app.providers.http import join_url


class OpenAICompatibleConfig(ProviderConfigBase):
    base_url: str = Field(..., description="API root including version, e.g. https://integrate.api.nvidia.com/v1")
    chat_model: str = Field(default="", description="Model ID for chat completions")
    vision_model: str = Field(default="", description="Model ID for image inputs; set it to enable the vision capability")
    embedding_model: str = Field(default="", description="Model ID for /embeddings")
    image_embedding_model: str = Field(default="", description="Model ID for multimodal embeddings (experimental)")
    supports_json_mode: bool = Field(default=False, description="Send response_format={type: json_object}")
    supports_json_schema: bool = Field(default=False, description="Send response_format={type: json_schema}")
    supports_system_role: bool = Field(default=True, description="Disable for models that reject system messages")
    embedding_dimensions: int | None = Field(default=None, ge=1, le=16384, description="Send 'dimensions' (only if the server supports it)")
    embedding_input_type_field: str | None = Field(default=None, description="e.g. 'input_type' for NVIDIA NIM retrieval embedders")
    embedding_query_value: str = "query"
    embedding_document_value: str = "passage"
    normalize_embeddings: bool = Field(default=True, description="L2-normalize vectors client-side")
    embedding_batch_size: int = Field(default=32, ge=1, le=2048)
    image_embedding_format: str = Field(default="data_url", description="data_url | object_image (input: [{image: <data url>}])")
    extra_headers: dict[str, str] = Field(default_factory=dict, description="Static, non-secret headers (e.g. HTTP-Referer for OpenRouter)")
    extra_body: dict[str, Any] = Field(default_factory=dict, description="Additional JSON body parameters for chat requests")
    auth_header: str = Field(default="Authorization", description="Header used for the API key")
    auth_scheme: str = Field(default="Bearer", description="Prefix for the API key value; empty for raw key")

    @field_validator("extra_headers")
    @classmethod
    def _headers(cls, v: dict[str, str]) -> dict[str, str]:
        return validate_headers(v)

    @field_validator("extra_body")
    @classmethod
    def _body(cls, v: dict[str, Any]) -> dict[str, Any]:
        return validate_extra_body(v, reserved={"model", "messages", "stream", "input"})

    @field_validator("auth_header")
    @classmethod
    def _auth_header(cls, v: str) -> str:
        if v.lower() not in {"authorization", "x-api-key", "api-key"}:
            raise ValueError("auth_header must be Authorization, x-api-key or api-key")
        return v


class OpenAICompatibleProvider(ChatModelProvider, EmbeddingProvider, VisionProvider, ImageEmbeddingProvider):
    adapter = "openai_compatible"
    display_name = "OpenAI-compatible API"
    protocol = "openai_compatible"
    capabilities = frozenset({Capability.CHAT, Capability.EMBEDDING, Capability.VISION, Capability.IMAGE_EMBEDDING})
    modalities = ["text", "image"]
    description = (
        "Any server implementing /chat/completions and/or /embeddings: NVIDIA NIM, OpenRouter, ModelScope API-Inference, "
        "vLLM, SGLang, Ollama (/v1), LM Studio, OpenAI."
    )
    limitations = [
        "Optional features (JSON mode/schema, embedding dimensions, input_type) must be enabled explicitly per endpoint.",
        "Vision requires a model that accepts image_url content parts.",
        "Image embeddings are experimental: the request format differs by server.",
        "Streaming is not used; responses are awaited in full.",
    ]
    config_model = OpenAICompatibleConfig
    secret_fields = ["api_key"]
    config: OpenAICompatibleConfig

    # ---------------------------------------------------------------- helpers
    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", **self.config.extra_headers}
        key = self.secrets.get("api_key")
        if key:
            h[self.config.auth_header] = f"{self.config.auth_scheme} {key}".strip() if self.config.auth_scheme else key
        return h

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        return await self.http.request_json(
            self.provider_id, "POST", join_url(self.config.base_url, path), headers=self._headers(), json=body, timeout=self.config.timeout_s
        )

    @property
    def active_capabilities(self) -> frozenset[Capability]:
        """Only capabilities with a configured model are offered (a chat-only endpoint is not an embedder)."""
        configured = {cap for cap in self.capabilities if self.model_for(cap)}
        enabled = set(self.config.enabled_capabilities or configured)
        return frozenset(configured & enabled)

    def model_for(self, capability: Capability) -> str | None:
        c = self.config
        return {
            Capability.CHAT: c.chat_model,
            Capability.VISION: c.vision_model,
            Capability.EMBEDDING: c.embedding_model,
            Capability.IMAGE_EMBEDDING: c.image_embedding_model,
        }.get(capability) or None

    def _messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        pending_system: list[str] = []
        for m in messages:
            if m.role == "system" and not self.config.supports_system_role:
                pending_system.append(m.content)
                continue
            content: Any = m.content
            if m.role == "user" and pending_system:
                content = "\n\n".join(pending_system + [m.content])
                pending_system = []
            if m.images:
                parts: list[dict[str, Any]] = [{"type": "text", "text": content}]
                parts += [{"type": "image_url", "image_url": {"url": image_data_url(img)}} for img in m.images]
                content = parts
            out.append({"role": m.role, "content": content})
        return out

    @staticmethod
    def _extract_text(choice: dict[str, Any]) -> str:
        msg = choice.get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content or ""

    # ---------------------------------------------------------------- chat
    async def chat(self, request: ChatRequest) -> ChatResponse:
        model = require(request.model or self.config.chat_model, self.provider_id, "chat_model")
        return await self._complete(model, request.messages, request)

    async def _complete(self, model: str, messages: list[ChatMessage], request: ChatRequest) -> ChatResponse:
        body: dict[str, Any] = {**self.config.extra_body, "model": model, "messages": self._messages(messages)}
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_tokens:
            body["max_tokens"] = request.max_tokens
        if request.stop:
            body["stop"] = request.stop
        if request.json_schema and self.config.supports_json_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": request.json_schema}}
        elif (request.json_mode or request.json_schema) and self.config.supports_json_mode:
            body["response_format"] = {"type": "json_object"}
        data = await self._post("chat/completions", body)
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            raise invalid(self.provider_id, "no choices in chat completion")
        usage = data.get("usage") or {}
        return ChatResponse(
            text=self._extract_text(choices[0]),
            model=data.get("model") or model,
            finish_reason=choices[0].get("finish_reason"),
            usage=Usage(input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens")),
        )

    # ---------------------------------------------------------------- vision
    async def analyze(self, request: VisionRequest) -> VisionResponse:
        model = require(request.model or self.config.vision_model, self.provider_id, "vision_model")
        msgs = []
        if request.system:
            msgs.append(ChatMessage(role="system", content=request.system))
        msgs.append(ChatMessage(role="user", content=request.prompt, images=request.images))
        resp = await self._complete(model, msgs, ChatRequest(messages=msgs, max_tokens=request.max_tokens, temperature=0.0))
        return VisionResponse(text=resp.text, model=resp.model, usage=resp.usage)

    # ---------------------------------------------------------------- embeddings
    def embedding_fingerprint(self) -> dict[str, Any]:
        c = self.config
        return {
            "adapter": self.adapter,
            "endpoint": c.base_url.rstrip("/"),
            "model": c.embedding_model,
            "dimensions": c.embedding_dimensions,
            "normalize": c.normalize_embeddings,
            "input_type_field": c.embedding_input_type_field,
        }

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        model = require(self.config.embedding_model, self.provider_id, "embedding_model")
        vectors: list[list[float]] = []
        in_tokens = 0
        for batch in batched(request.texts, self.config.embedding_batch_size):
            body: dict[str, Any] = {"model": model, "input": batch}
            if self.config.embedding_dimensions:
                body["dimensions"] = self.config.embedding_dimensions
            if self.config.embedding_input_type_field:
                body[self.config.embedding_input_type_field] = (
                    self.config.embedding_query_value if request.input_type == "query" else self.config.embedding_document_value
                )
            data = await self._post("embeddings", body)
            items = data.get("data") if isinstance(data, dict) else None
            if not isinstance(items, list) or len(items) != len(batch):
                raise invalid(self.provider_id, "embedding count mismatch")
            items = sorted(items, key=lambda d: d.get("index", 0))
            for d in items:
                vec = d.get("embedding")
                if not isinstance(vec, list):
                    raise invalid(self.provider_id, "embedding is not a float list (base64 encoding is not supported)")
                vectors.append(l2_normalize(vec) if self.config.normalize_embeddings else vec)
            in_tokens += (data.get("usage") or {}).get("prompt_tokens") or 0
        return EmbeddingResponse(vectors=vectors, model=model, usage=Usage(input_tokens=in_tokens or None))

    async def embed_images(self, request: ImageEmbeddingRequest) -> EmbeddingResponse:
        model = require(self.config.image_embedding_model, self.provider_id, "image_embedding_model")
        if self.config.image_embedding_format == "object_image":
            inputs: list[Any] = [{"image": image_data_url(i)} for i in request.images] + [{"text": t} for t in request.texts]
        else:
            inputs = [image_data_url(i) for i in request.images] + list(request.texts)
        data = await self._post("embeddings", {"model": model, "input": inputs})
        items = sorted((data.get("data") or []), key=lambda d: d.get("index", 0))
        if len(items) != len(inputs):
            raise invalid(self.provider_id, "image embedding count mismatch")
        return EmbeddingResponse(vectors=[l2_normalize(d["embedding"]) for d in items], model=model)

    # ---------------------------------------------------------------- health
    async def health_check(self) -> HealthResult:
        t0 = time.perf_counter()
        checked: list[str] = []
        caps = self.active_capabilities
        if Capability.CHAT in caps and self.config.chat_model:
            await self.chat(ChatRequest(messages=[ChatMessage(role="user", content="Reply with the word OK.")], max_tokens=5, temperature=0))
            checked.append("chat")
        if Capability.EMBEDDING in caps and self.config.embedding_model:
            r = await self.embed(EmbeddingRequest(texts=["health check"], input_type="query"))
            checked.append(f"embedding(dims={r.dims})")
        if not checked:
            raise ProviderError(ProviderErrorKind.CONFIG, "No model configured to test", provider_id=self.provider_id)
        return HealthResult(ok=True, latency_ms=(time.perf_counter() - t0) * 1000, detail="OK", checked=checked)
