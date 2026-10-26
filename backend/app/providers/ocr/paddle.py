"""OCR adapters: local PaddleOCR (PP-OCRv5, CPU) and the PaddleOCR/PaddleX serving HTTP API."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import threading
import time
from typing import Any, ClassVar

from pydantic import Field

from app.providers.base import (
    Capability,
    HealthResult,
    OCRLine,
    OCRProvider,
    OCRRequest,
    OCRResponse,
    ProviderConfigBase,
    ProviderError,
    ProviderErrorKind,
)
from app.providers.common import invalid
from app.providers.http import join_url


def _lines_from_result(texts: list, scores: list | None, boxes: Any) -> list[OCRLine]:
    lines: list[OCRLine] = []
    box_list = boxes.tolist() if hasattr(boxes, "tolist") else (boxes or [])
    for i, t in enumerate(texts or []):
        if not str(t).strip():
            continue
        bbox = None
        if i < len(box_list):
            b = box_list[i]
            if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(x, (int, float)) for x in b):
                bbox = [float(x) for x in b]
        conf = float(scores[i]) if scores is not None and i < len(scores) else None
        lines.append(OCRLine(text=str(t), confidence=conf, bbox=bbox))
    return lines


def _join_lines(lines: list[OCRLine]) -> str:
    """Join OCR lines in reading order (top-to-bottom, then left-to-right) into paragraphs."""
    if not lines or any(l.bbox is None for l in lines):
        return "\n".join(l.text for l in lines)
    ordered = sorted(lines, key=lambda l: (round(l.bbox[1] / 12), l.bbox[0]))  # type: ignore[index]
    out: list[str] = []
    prev_bottom = None
    for l in ordered:
        top, bottom = l.bbox[1], l.bbox[3]  # type: ignore[index]
        height = max(bottom - top, 1.0)
        if prev_bottom is not None and top - prev_bottom > 1.2 * height:
            out.append("")
        out.append(l.text)
        prev_bottom = bottom
    return "\n".join(out)


class PaddleLocalConfig(ProviderConfigBase):
    lang: str = Field(default="en", description="PaddleOCR language code")
    ocr_version: str = Field(default="PP-OCRv5", description="PP-OCRv5 (requires paddleocr>=3.0)")
    detection_model: str = Field(default="PP-OCRv5_mobile_det", description="Mobile models are recommended for CPU")
    recognition_model: str = Field(default="PP-OCRv5_mobile_rec")
    use_textline_orientation: bool = False
    cpu_threads: int = Field(default=4, ge=1, le=64)
    enable_mkldnn: bool = Field(
        default=False,
        description="oneDNN acceleration. Off by default: paddlepaddle 3.3 CPU builds fail with "
        "'ConvertPirAttribute2RuntimeAttribute not support' when it is on. Enable if your build works.",
    )
    det_limit_side_len: int = Field(default=1280, ge=320, le=4096, description="Max side length fed to text detection")
    max_concurrency: int = Field(default=1, ge=1, le=8)


class PaddleOCRLocalProvider(OCRProvider):
    adapter = "paddleocr_local"
    display_name = "PaddleOCR PP-OCRv5 (local CPU)"
    protocol = "local"
    capabilities = frozenset({Capability.OCR})
    modalities = ["image"]
    description = "Baidu PaddleOCR 3.x pipeline running on CPU. Models load once per process and are shared."
    limitations = [
        "Requires optional packages 'paddleocr>=3' and 'paddlepaddle>=3'.",
        "CPU inference takes seconds per page; concurrency is bounded to protect RAM.",
        "Does not reconstruct table structure; table text is returned as lines.",
    ]
    config_model = PaddleLocalConfig
    requires_network = False
    config: PaddleLocalConfig

    _engines: ClassVar[dict[tuple, Any]] = {}
    _engine_lock: ClassVar[threading.Lock] = threading.Lock()
    _run_lock: ClassVar[threading.Lock] = threading.Lock()  # PaddleOCR predictors are not thread-safe

    def model_for(self, capability: Capability) -> str | None:
        return f"{self.config.ocr_version}:{self.config.detection_model}+{self.config.recognition_model}"

    def _engine(self):
        c = self.config
        key = (c.lang, c.ocr_version, c.detection_model, c.recognition_model, c.use_textline_orientation, c.cpu_threads, c.enable_mkldnn, c.det_limit_side_len)
        with self._engine_lock:
            if key not in self._engines:
                try:
                    from paddleocr import PaddleOCR
                except ImportError as exc:
                    raise ProviderError(
                        ProviderErrorKind.CONFIG, "PaddleOCR is not installed (pip install paddleocr paddlepaddle)", provider_id=self.provider_id
                    ) from exc
                os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
                kwargs: dict[str, Any] = dict(
                    enable_mkldnn=c.enable_mkldnn,
                    text_det_limit_side_len=c.det_limit_side_len,
                    text_det_limit_type="max",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=c.use_textline_orientation,
                    device="cpu",
                    cpu_threads=c.cpu_threads,
                )
                if c.detection_model and c.recognition_model:
                    kwargs.update(text_detection_model_name=c.detection_model, text_recognition_model_name=c.recognition_model)
                else:
                    kwargs.update(ocr_version=c.ocr_version, lang=c.lang)
                self._engines[key] = PaddleOCR(**kwargs)
            return self._engines[key]

    def _run(self, data: bytes) -> list[OCRLine]:
        import numpy as np
        from PIL import Image

        img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))[:, :, ::-1]  # BGR, as PaddleOCR expects
        engine = self._engine()
        with self._run_lock:
            results = engine.predict(img)
        lines: list[OCRLine] = []
        for res in results or []:
            payload = res.get("res", res) if isinstance(res, dict) else getattr(res, "json", {}).get("res", {})
            lines += _lines_from_result(payload.get("rec_texts") or [], payload.get("rec_scores"), payload.get("rec_boxes"))
        return lines

    async def ocr(self, request: OCRRequest) -> OCRResponse:
        lines = await asyncio.to_thread(self._run, request.image.data)
        return OCRResponse(text=_join_lines(lines), lines=lines, engine=f"paddleocr-local:{self.model_for(Capability.OCR)}")

    async def health_check(self) -> HealthResult:
        t0 = time.perf_counter()
        await asyncio.to_thread(self._engine)
        return HealthResult(ok=True, latency_ms=(time.perf_counter() - t0) * 1000, detail="PaddleOCR models loaded", checked=["ocr"])


class PaddleServingConfig(ProviderConfigBase):
    base_url: str = Field(..., description="PaddleOCR/PaddleX serving root, e.g. http://ocr-host:8080")
    path: str = Field(default="ocr", description="Endpoint path (PaddleX OCR pipeline serves /ocr)")
    extra_payload: dict[str, Any] = Field(default_factory=dict, description="e.g. {useDocUnwarping: false}")


class PaddleServingOCRProvider(OCRProvider):
    adapter = "paddleocr_serving"
    display_name = "PaddleOCR serving API (remote)"
    protocol = "native"
    capabilities = frozenset({Capability.OCR})
    modalities = ["image"]
    description = "Remote PaddleOCR 3.x / PaddleX pipeline served over HTTP: POST {file: base64, fileType: 1} → result.ocrResults[].prunedResult."
    limitations = ["Page images are sent to the remote server."]
    config_model = PaddleServingConfig
    secret_fields = ["api_key"]
    config: PaddleServingConfig

    def model_for(self, capability: Capability) -> str | None:
        return "paddleocr-serving"

    async def ocr(self, request: OCRRequest) -> OCRResponse:
        headers = {"Content-Type": "application/json"}
        if key := self.secrets.get("api_key"):
            headers["Authorization"] = f"Bearer {key}"
        body = {**self.config.extra_payload, "file": base64.b64encode(request.image.data).decode(), "fileType": 1}
        data = await self.http.request_json(
            self.provider_id, "POST", join_url(self.config.base_url, self.config.path), headers=headers, json=body, timeout=self.config.timeout_s
        )
        if not isinstance(data, dict):
            raise invalid(self.provider_id, "non-object response")
        if data.get("errorCode", 0) not in (0, None):
            raise ProviderError(ProviderErrorKind.BAD_REQUEST, f"OCR server error {data.get('errorCode')}: {data.get('errorMsg')}", provider_id=self.provider_id)
        results = (data.get("result") or {}).get("ocrResults")
        if not isinstance(results, list):
            raise invalid(self.provider_id, "result.ocrResults missing")
        lines: list[OCRLine] = []
        for r in results:
            pr = r.get("prunedResult") or {}
            lines += _lines_from_result(pr.get("rec_texts") or [], pr.get("rec_scores"), pr.get("rec_boxes"))
        return OCRResponse(text=_join_lines(lines), lines=lines, engine="paddleocr-serving")
