"""Mode B: native Anthropic Messages API adapter (chat + vision)."""

from __future__ import annotations

import time
from typing import Any

from pydantic import Field, field_validator

from app.providers.base import (
    Capability,
    ChatMessage,
    ChatModelProvider,
    ChatRequest,
    ChatResponse,
    HealthResult,
    ImageInput,
    ProviderConfigBase,
    Usage,
    VisionProvider,
    VisionRequest,
    VisionResponse,
)
from app.providers.common import image_b64, invalid, require, validate_headers
from app.providers.http import join_url


class AnthropicConfig(ProviderConfigBase):
    base_url: str = Field(default="https://api.anthropic.com", description="API root (without /v1)")
    chat_model: str = Field(default="claude-sonnet-5-5")
    vision_model: str = Field(default="", description="Defaults to chat_model")
    anthropic_version: str = Field(default="2023-06-01")
    extra_headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("extra_headers")
    @classmethod
    def _headers(cls, v: dict[str, str]) -> dict[str, str]:
        return validate_headers(v)


class AnthropicProvider(ChatModelProvider, VisionProvider):
    adapter = "anthropic"
    display_name = "Anthropic Messages API"
    protocol = "native"
    capabilities = frozenset({Capability.CHAT, Capability.VISION})
    modalities = ["text", "image"]
    description = "Native adapter for the Anthropic Messages API (system prompt, content blocks, base64 images)."
    limitations = ["No embeddings or reranking endpoint.", "JSON output is requested via instructions, not a response_format flag."]
    config_model = AnthropicConfig
    secret_fields = ["api_key"]
    config: AnthropicConfig

    def model_for(self, capability: Capability) -> str | None:
        if capability == Capability.VISION:
            return self.config.vision_model or self.config.chat_model
        return self.config.chat_model if capability == Capability.CHAT else None

    def _headers(self) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "anthropic-version": self.config.anthropic_version,
            "x-api-key": require(self.secrets.get("api_key"), self.provider_id, "api_key"),
            **self.config.extra_headers,
        }

    @staticmethod
    def _content(text: str, images: list[ImageInput]) -> list[dict[str, Any]] | str:
        if not images:
            return text
        blocks: list[dict[str, Any]] = [
            {"type": "image", "source": {"type": "base64", "media_type": img.mime_type, "data": image_b64(img)}} for img in images
        ]
        blocks.append({"type": "text", "text": text})
        return blocks

    async def _messages(self, model: str, messages: list[ChatMessage], max_tokens: int, temperature: float | None, stop: list[str] | None) -> ChatResponse:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        convo = [{"role": m.role, "content": self._content(m.content, m.images)} for m in messages if m.role != "system"]
        body: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": convo}
        if system:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature
        if stop:
            body["stop_sequences"] = stop
        data = await self.http.request_json(
            self.provider_id, "POST", join_url(self.config.base_url, "v1/messages"), headers=self._headers(), json=body, timeout=self.config.timeout_s
        )
        blocks = data.get("content") if isinstance(data, dict) else None
        if not isinstance(blocks, list):
            raise invalid(self.provider_id, "missing content blocks")
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage = data.get("usage") or {}
        return ChatResponse(
            text=text,
            model=data.get("model") or model,
            finish_reason=data.get("stop_reason"),
            usage=Usage(input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens")),
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        model = require(request.model or self.config.chat_model, self.provider_id, "chat_model")
        messages = list(request.messages)
        if request.json_mode or request.json_schema:
            hint = "Respond with a single valid JSON object only, no prose."
            messages = [ChatMessage(role="system", content=hint)] + messages
        return await self._messages(model, messages, request.max_tokens or 2048, request.temperature, request.stop)

    async def analyze(self, request: VisionRequest) -> VisionResponse:
        model = require(request.model or self.config.vision_model or self.config.chat_model, self.provider_id, "vision_model")
        msgs = ([ChatMessage(role="system", content=request.system)] if request.system else []) + [
            ChatMessage(role="user", content=request.prompt, images=request.images)
        ]
        r = await self._messages(model, msgs, request.max_tokens or 1024, 0.0, None)
        return VisionResponse(text=r.text, model=r.model, usage=r.usage)

    async def health_check(self) -> HealthResult:
        t0 = time.perf_counter()
        await self.chat(ChatRequest(messages=[ChatMessage(role="user", content="Reply with OK.")], max_tokens=5))
        return HealthResult(ok=True, latency_ms=(time.perf_counter() - t0) * 1000, detail="OK", checked=["chat"])
