"""OCR implemented by a vision-language model, configured explicitly as an OCR provider.

This is never used as an automatic fallback for an OCR engine: the user must assign it to the OCR
task. Output is labelled ``vlm_transcription`` so provenance shows it is model-generated text.
"""

from __future__ import annotations

from pydantic import Field

from app.providers.base import Capability, ChatMessage, ChatRequest, OCRProvider, OCRRequest, OCRResponse
from app.providers.chat.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleProvider

PROMPT = (
    "Transcribe all text on this page exactly as written, in reading order. Preserve line breaks between paragraphs. "
    "Render tables as Markdown tables. Do not summarize, translate, or add commentary. Output only the transcription."
)


class VLMOCRConfig(OpenAICompatibleConfig):
    vision_model: str = Field(..., description="Vision model used for transcription (e.g. an OCR-tuned VLM)")
    max_output_tokens: int = Field(default=4096, ge=256, le=32768)


class VLMOCRProvider(OCRProvider):
    adapter = "vlm_ocr"
    display_name = "VLM transcription OCR (OpenAI-compatible)"
    protocol = "openai_compatible"
    capabilities = frozenset({Capability.OCR})
    modalities = ["image"]
    description = "Uses an OpenAI-compatible vision model to transcribe page images. Text is marked as model-transcribed."
    limitations = [
        "Generative transcription can hallucinate or silently omit text; no per-line confidences or boxes.",
        "Page images are sent to the configured endpoint.",
    ]
    config_model = VLMOCRConfig
    secret_fields = ["api_key"]
    config: VLMOCRConfig

    def model_for(self, capability: Capability) -> str | None:
        return self.config.vision_model

    async def ocr(self, request: OCRRequest) -> OCRResponse:
        inner = OpenAICompatibleProvider(self.provider_id, self.config, self.secrets, self.http)
        msg = ChatMessage(role="user", content=PROMPT, images=[request.image])
        resp = await inner._complete(self.config.vision_model, [msg], ChatRequest(messages=[msg], temperature=0.0, max_tokens=self.config.max_output_tokens))
        return OCRResponse(text=resp.text.strip(), lines=[], engine=f"vlm:{self.config.vision_model}", method="vlm_transcription")
