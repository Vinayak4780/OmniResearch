"""Local (CPU) models: browse, search Hugging Face, install, use and remove — from the web UI.

Kinds
 * ``embedding``  – fastembed ONNX text embedders (built-in catalog + any ONNX sentence-embedding repo on HF)
 * ``reranking``  – fastembed ONNX cross-encoders (built-in catalog + ONNX cross-encoder repos on HF)
 * ``ocr``        – PaddleOCR PP-OCRv5 detection + recognition models (PaddlePaddle org on HF; per-language recognizers)

Installing downloads the weights once (progress is measured from bytes on disk). Using a model
creates/updates a local provider and routes the task to it; switching the embedder rebuilds the
search index automatically. Downloads go only to huggingface.co (and Paddle's own model mirrors).
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from app.providers.base import Task
from app.providers.registry import RouteConfig

if TYPE_CHECKING:  # pragma: no cover
    from app.core.container import Container

HF_API = "https://huggingface.co/api/models"
HF_FILE = "https://huggingface.co/{repo}/resolve/main/{path}"
KINDS = ("embedding", "reranking", "ocr")
CUSTOM_KEY = "local_custom_models"  # {kind: {model_id: spec}} for HF models not in fastembed's catalog
INSTALLED_KEY = "local_models_installed"  # {kind: [model ids]}

OCR_DETECTORS = {
    "PP-OCRv5_mobile_det": ("Fast text detection", 0.005),
    "PP-OCRv5_server_det": ("Most accurate text detection (slower on CPU)", 0.09),
}
OCR_RECOGNIZERS = {
    "PP-OCRv5_mobile_rec": "Chinese, English, Japanese (mixed)",
    "PP-OCRv5_server_rec": "Chinese, English, Japanese — most accurate",
    "en_PP-OCRv5_mobile_rec": "English",
    "latin_PP-OCRv5_mobile_rec": "Latin-script languages (French, German, Spanish, …)",
    "eslav_PP-OCRv5_mobile_rec": "East Slavic (Russian, Ukrainian, Belarusian)",
    "cyrillic_PP-OCRv5_mobile_rec": "Cyrillic-script languages",
    "arabic_PP-OCRv5_mobile_rec": "Arabic-script languages",
    "devanagari_PP-OCRv5_mobile_rec": "Devanagari (Hindi, Marathi, Nepali, …)",
    "korean_PP-OCRv5_mobile_rec": "Korean",
    "el_PP-OCRv5_mobile_rec": "Greek",
    "ta_PP-OCRv5_mobile_rec": "Tamil",
    "te_PP-OCRv5_mobile_rec": "Telugu",
    "th_PP-OCRv5_mobile_rec": "Thai",
}
RECOMMENDED = {
    "embedding": {"BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "intfloat/multilingual-e5-large", "nomic-ai/nomic-embed-text-v1.5"},
    "reranking": {"Xenova/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-v2-m3"},
    "ocr": {"PP-OCRv5_mobile_det+PP-OCRv5_mobile_rec", "PP-OCRv5_mobile_det+en_PP-OCRv5_mobile_rec"},
}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


class LocalModelService:
    def __init__(self, c: "Container"):
        self.c = c
        self.fastembed_cache = c.settings.data_dir / "models" / "fastembed"
        self.paddle_cache = Path(os.environ.get("PADDLE_PDX_CACHE_HOME", Path.home() / ".paddlex")) / "official_models"
        self.jobs: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self.register_custom_models()

    # ---------------------------------------------------------------- availability of libraries
    @staticmethod
    def available() -> dict[str, bool]:
        out = {}
        for kind, mod in (("embedding", "fastembed"), ("reranking", "fastembed"), ("ocr", "paddleocr")):
            try:
                __import__(mod)
                out[kind] = True
            except ImportError:
                out[kind] = False
        return out

    # ---------------------------------------------------------------- catalog
    def _installed(self) -> dict[str, list[str]]:
        return self.c.db.kv_get(INSTALLED_KEY, {}) or {}

    def _custom(self) -> dict[str, dict[str, Any]]:
        return self.c.db.kv_get(CUSTOM_KEY, {}) or {}

    def register_custom_models(self) -> None:
        """fastembed only knows custom models after add_custom_model() in this process."""
        try:
            from fastembed import TextEmbedding
            from fastembed.common.model_description import ModelSource, PoolingType
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError:
            return
        known_e = {m["model"] for m in TextEmbedding.list_supported_models()}
        known_r = {m["model"] for m in TextCrossEncoder.list_supported_models()}
        for mid, spec in (self._custom().get("embedding") or {}).items():
            if mid not in known_e:
                TextEmbedding.add_custom_model(mid, pooling=PoolingType[spec["pooling"]], normalization=True, sources=ModelSource(hf=spec["repo"]),
                                               dim=spec["dim"], model_file=spec["model_file"], description=spec.get("description", ""), size_in_gb=spec.get("size_gb", 0.0))
        for mid, spec in (self._custom().get("reranking") or {}).items():
            if mid not in known_r:
                TextCrossEncoder.add_custom_model(mid, sources=ModelSource(hf=spec["repo"]), model_file=spec["model_file"],
                                                  description=spec.get("description", ""), size_in_gb=spec.get("size_gb", 0.0))

    def _in_use(self) -> dict[str, str | None]:
        """The local model currently serving each kind (first enabled provider of the route, if local)."""
        out: dict[str, str | None] = {}
        for kind, task in (("embedding", Task.TEXT_EMBEDDING), ("reranking", Task.RERANKING), ("ocr", Task.OCR)):
            out[kind] = None
            p = self.c.router.primary(task)
            if p is None:
                continue
            if kind == "embedding" and p.adapter == "fastembed":
                out[kind] = p.config.model
            elif kind == "reranking" and p.adapter == "fastembed_rerank":
                out[kind] = p.config.model
            elif kind == "ocr" and p.adapter == "paddleocr_local":
                out[kind] = f"{p.config.detection_model}+{p.config.recognition_model}"
        return out

    def catalog(self) -> dict[str, Any]:
        avail = self.available()
        installed = self._installed()
        in_use = self._in_use()
        custom = self._custom()
        kinds: dict[str, list[dict[str, Any]]] = {k: [] for k in KINDS}
        if avail["embedding"]:
            from fastembed import TextEmbedding
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            for m in TextEmbedding.list_supported_models():
                kinds["embedding"].append({"id": m["model"], "description": m.get("description", ""), "size_gb": m.get("size_in_GB"), "dims": m.get("dim"),
                                           "license": m.get("license"), "source": "custom" if m["model"] in (custom.get("embedding") or {}) else "catalog"})
            for m in TextCrossEncoder.list_supported_models():
                kinds["reranking"].append({"id": m["model"], "description": m.get("description", ""), "size_gb": m.get("size_in_GB"),
                                           "license": m.get("license"), "source": "custom" if m["model"] in (custom.get("reranking") or {}) else "catalog"})
        if avail["ocr"]:
            for det, (ddesc, dsize) in OCR_DETECTORS.items():
                for rec, lang in OCR_RECOGNIZERS.items():
                    if det.startswith("PP-OCRv5_server") and not rec.startswith("PP-OCRv5_server"):
                        continue  # keep the list readable: server detector pairs with the server recognizer
                    kinds["ocr"].append({"id": f"{det}+{rec}", "description": f"{lang} · {ddesc}", "language": lang,
                                         "size_gb": round(dsize + (0.08 if "server" in rec else 0.01), 3), "source": "catalog"})
        for kind, items in kinds.items():
            for it in items:
                it["installed"] = it["id"] in (installed.get(kind) or []) or (kind == "ocr" and self._ocr_installed(it["id"]))
                it["in_use"] = it["id"] == in_use.get(kind)
                it["recommended"] = it["id"] in RECOMMENDED[kind]
                job = next((j for j in self.jobs.values() if j["kind"] == kind and j["model"] == it["id"] and j["status"] in ("downloading", "loading")), None)
                it["job"] = self._job_view(job) if job else None
            items.sort(key=lambda x: (not x["in_use"], not x["installed"], not x["recommended"], x["id"].lower()))
        return {"available": avail, "kinds": kinds, "in_use": in_use, "jobs": [self._job_view(j) for j in self.jobs.values()]}

    def _ocr_installed(self, model_id: str) -> bool:
        return all((self.paddle_cache / name).exists() for name in model_id.split("+"))

    # ---------------------------------------------------------------- Hugging Face search
    async def search_hf(self, kind: str, query: str, limit: int = 20) -> list[dict[str, Any]]:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        params: dict[str, Any] = {"search": query.strip()[:100], "sort": "downloads", "limit": limit, "full": "false"}
        if kind == "embedding":
            params.update({"filter": "onnx", "pipeline_tag": "sentence-similarity"})
        elif kind == "reranking":
            params.update({"filter": "onnx", "pipeline_tag": "text-classification"})
        else:
            params = {"author": "PaddlePaddle", "search": (query.strip() or "PP-OCRv5")[:100], "limit": 100}
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(HF_API, params=params)
            resp.raise_for_status()
            items = resp.json()
            if kind == "embedding" and len(items) < limit:  # many embedders are tagged feature-extraction instead
                more = await client.get(HF_API, params={**params, "pipeline_tag": "feature-extraction"})
                if more.status_code == 200:
                    seen = {i["id"] for i in items}
                    items += [i for i in more.json() if i["id"] not in seen]
        known = {m["id"] for m in self.catalog()["kinds"][kind]}
        out = []
        for it in items[:limit * 2]:
            mid = it["id"]
            if kind == "ocr":
                name = mid.split("/", 1)[1]
                if not name.endswith("_rec") or name not in OCR_RECOGNIZERS and not re.fullmatch(r"[a-z_]*PP-OCRv5_(mobile|server)_rec", name):
                    continue
                mid = f"PP-OCRv5_mobile_det+{name}"
            out.append({"id": mid, "repo": it["id"], "downloads": it.get("downloads"), "likes": it.get("likes"), "in_catalog": mid in known,
                        "url": f"https://huggingface.co/{it['id']}"})
        return out[:limit]

    async def _hf_spec(self, kind: str, repo: str) -> dict[str, Any]:
        """Inspect an HF repo and build a fastembed custom-model spec (ONNX file, dims, pooling)."""
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            info = await client.get(f"{HF_API}/{repo}", params={"blobs": "true"})
            if info.status_code != 200:
                raise ValueError(f"Hugging Face model '{repo}' not found")
            data = info.json()
            files = {s["rfilename"]: s.get("size") or 0 for s in data.get("siblings", [])}
            onnx = next((f for f in ("onnx/model.onnx", "model.onnx", "onnx/model_quantized.onnx") if f in files), None)
            if onnx is None:
                raise ValueError(f"'{repo}' has no ONNX weights (onnx/model.onnx), so it cannot run with fastembed on CPU")
            spec: dict[str, Any] = {"repo": repo, "model_file": onnx, "size_gb": round(files[onnx] / 1e9, 3),
                                    "description": f"{data.get('pipeline_tag') or 'model'} from Hugging Face ({repo})"}
            if kind == "embedding":
                cfg = await client.get(HF_FILE.format(repo=repo, path="config.json"))
                if cfg.status_code != 200:
                    raise ValueError(f"'{repo}' has no config.json; cannot determine embedding size")
                dim = cfg.json().get("hidden_size") or cfg.json().get("d_model")
                if not dim:
                    raise ValueError(f"Cannot determine the embedding size of '{repo}'")
                pooling = "MEAN"
                pool = await client.get(HF_FILE.format(repo=repo, path="1_Pooling/config.json"))
                if pool.status_code == 200 and pool.json().get("pooling_mode_cls_token"):
                    pooling = "CLS"
                spec.update({"dim": int(dim), "pooling": pooling})
        return spec

    # ---------------------------------------------------------------- install
    def _job_view(self, j: dict[str, Any]) -> dict[str, Any]:
        done_bytes = sum(_dir_size(p) for p in j.get("paths", []))
        total = j.get("expected_bytes") or 0
        return {k: j[k] for k in ("id", "kind", "model", "status", "error", "started_at")} | {
            "downloaded_mb": round(done_bytes / 1e6, 1),
            "expected_mb": round(total / 1e6, 1) if total else None,
            "progress": min(0.99, done_bytes / total) if total and j["status"] == "downloading" else (1.0 if j["status"] == "done" else None),
        }

    def _paths_for(self, kind: str, model: str, repo: str | None) -> list[Path]:
        if kind == "ocr":
            return [self.paddle_cache / n for n in model.split("+")]
        return [self.fastembed_cache / f"models--{(repo or model).replace('/', '--')}"]

    async def install(self, kind: str, model: str, repo: str | None = None) -> dict[str, Any]:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not self.available()[kind]:
            raise ValueError("The local model runtime is not installed on the server (pip install -r requirements-ml.txt)")
        running = next((j for j in self.jobs.values() if j["kind"] == kind and j["model"] == model and j["status"] in ("downloading", "loading")), None)
        if running:
            return self._job_view(running)
        expected_gb = None
        if kind == "ocr":
            if not re.fullmatch(r"PP-OCRv5_(mobile|server)_det\+[a-z_]*PP-OCRv5_(mobile|server)_rec", model):
                raise ValueError("OCR models must be a PP-OCRv5 detector + recognizer pair")
            expected_gb = next((m["size_gb"] for m in self.catalog()["kinds"]["ocr"] if m["id"] == model), 0.02)
        else:
            known = {m["id"]: m for m in self.catalog()["kinds"][kind]}
            if model not in known:
                spec = await self._hf_spec(kind, repo or model)
                custom = self._custom()
                custom.setdefault(kind, {})[model] = spec
                self.c.db.kv_set(CUSTOM_KEY, custom)
                self.register_custom_models()
                repo = spec["repo"]
                expected_gb = spec["size_gb"]
            else:
                expected_gb = known[model].get("size_gb")
                repo = None
                try:  # the HF repo fastembed downloads from (used to measure progress)
                    from fastembed import TextEmbedding
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    lst = TextEmbedding.list_supported_models() if kind == "embedding" else TextCrossEncoder.list_supported_models()
                    repo = next((m["sources"].get("hf") for m in lst if m["model"] == model), None)
                except Exception:
                    pass
        job = {"id": uuid.uuid4().hex[:10], "kind": kind, "model": model, "status": "downloading", "error": None,
               "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "paths": self._paths_for(kind, model, repo),
               "expected_bytes": int((expected_gb or 0) * 1e9)}
        self.jobs[job["id"]] = job
        self._tasks[job["id"]] = asyncio.create_task(self._run_install(job))
        return self._job_view(job)

    def _load(self, kind: str, model: str):
        """Instantiating the model downloads its files; doing it in-process also proves they load."""
        if kind == "embedding":
            from fastembed import TextEmbedding

            list(TextEmbedding(model_name=model, cache_dir=str(self.fastembed_cache)).embed(["test"]))
        elif kind == "reranking":
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            list(TextCrossEncoder(model_name=model, cache_dir=str(self.fastembed_cache)).rerank("q", ["d"]))
        else:
            from app.providers.ocr.paddle import PaddleLocalConfig, PaddleOCRLocalProvider

            det, rec = model.split("+")
            PaddleOCRLocalProvider("install", PaddleLocalConfig(detection_model=det, recognition_model=rec), {})._engine()

    async def _run_install(self, job: dict[str, Any]) -> None:
        try:
            self.fastembed_cache.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(self._load, job["kind"], job["model"])
            installed = self._installed()
            lst = installed.setdefault(job["kind"], [])
            if job["model"] not in lst:
                lst.append(job["model"])
            self.c.db.kv_set(INSTALLED_KEY, installed)
            job["status"] = "done"
        except Exception as exc:  # network, disk, unsupported architecture…
            job["status"] = "failed"
            job["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"

    async def wait(self, job_id: str, timeout: float = 1800) -> dict[str, Any]:
        t = self._tasks.get(job_id)
        if t:
            await asyncio.wait_for(asyncio.shield(t), timeout)
        return self._job_view(self.jobs[job_id])

    # ---------------------------------------------------------------- use / remove
    def use(self, kind: str, model: str) -> dict[str, Any]:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not (model in (self._installed().get(kind) or []) or (kind == "ocr" and self._ocr_installed(model))):
            raise ValueError("Install the model before using it")
        reg = self.c.registry
        if kind == "embedding":
            pid, adapter, cfg, task = f"local-emb-{_slug(model)}"[:64], "fastembed", {"model": model, "cache_dir": str(self.fastembed_cache)}, Task.TEXT_EMBEDDING
        elif kind == "reranking":
            pid, adapter, cfg, task = f"local-rr-{_slug(model)}"[:64], "fastembed_rerank", {"model": model, "cache_dir": str(self.fastembed_cache)}, Task.RERANKING
        else:
            det, rec = model.split("+")
            pid, adapter, cfg, task = f"local-ocr-{_slug(rec)}"[:64], "paddleocr_local", {"detection_model": det, "recognition_model": rec}, Task.OCR
        if self.c.db.one("SELECT 1 FROM provider_configs WHERE id=?", (pid,)):
            reg.update(pid, config=cfg, enabled=True)
        else:
            reg.create(f"{model} (this computer)", adapter, cfg, provider_id=pid)
        current = [p for p in reg.get_route(task).provider_ids if p != pid]
        reg.set_route(RouteConfig(task=task, provider_ids=[pid] + current))
        out: dict[str, Any] = {"provider_id": pid, "task": task.value, "notes": []}
        if kind == "embedding":
            active = self.c.index.active()
            if active is None or active.spec.get("provider_id") != pid:
                try:
                    self.c.index.start_reindex(pid)
                    out["notes"].append("Rebuilding the document search index with this model; search keeps working meanwhile.")
                except ValueError as exc:
                    out["notes"].append(f"Could not start rebuilding the search index: {exc}")
        if kind == "ocr":
            out["notes"].append("New uploads use this model. Use Retry on a document to re-read its scanned pages.")
        return out

    def remove(self, kind: str, model: str) -> None:
        if self._in_use().get(kind) == model:
            raise ValueError("This model is in use. Choose another model first.")
        for path in self._paths_for(kind, model, ((self._custom().get(kind) or {}).get(model) or {}).get("repo")):
            if kind == "ocr":
                # shared detectors stay if another installed OCR pair still needs them
                still = {n for m in self._installed().get("ocr", []) if m != model for n in m.split("+")}
                if path.name in still:
                    continue
            shutil.rmtree(path, ignore_errors=True)
        installed = self._installed()
        installed[kind] = [m for m in installed.get(kind, []) if m != model]
        self.c.db.kv_set(INSTALLED_KEY, installed)
