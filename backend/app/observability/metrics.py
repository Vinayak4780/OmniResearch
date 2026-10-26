"""Provider call metrics (latency, failures, tokens, cost) persisted to SQLite."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any

from app.repositories.db import Database, utcnow


@dataclass
class CallRecord:
    provider_id: str
    task: str | None
    capability: str
    ok: bool
    latency_ms: float
    model: str | None = None
    error_kind: str | None = None
    error_message: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    fallback_from: str | None = None
    attempt: int = 1


# Research jobs set their id here so every provider call made on their behalf is attributed to them.
current_job_id: ContextVar[str | None] = ContextVar("current_job_id", default=None)


class MetricsRecorder:
    def __init__(self, db: Database | None):
        self.db = db

    def record(self, rec: CallRecord) -> None:
        if self.db is None:
            return
        self.db.execute(
            """INSERT INTO provider_calls(provider_id, task, capability, ok, latency_ms, error_kind, error_message,
               input_tokens, output_tokens, cost_usd, fallback_from, created_at, job_id, model) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rec.provider_id,
                rec.task,
                rec.capability,
                int(rec.ok),
                rec.latency_ms,
                rec.error_kind,
                (rec.error_message or "")[:500] or None,
                rec.input_tokens,
                rec.output_tokens,
                rec.cost_usd,
                rec.fallback_from,
                utcnow(),
                current_job_id.get(),
                rec.model,
            ),
        )

    def summary(self, since: str | None = None) -> list[dict[str, Any]]:
        if self.db is None:
            return []
        where = "WHERE created_at >= ?" if since else ""
        params: tuple = (since,) if since else ()
        rows = self.db.all(
            f"""SELECT provider_id, capability, COUNT(*) AS calls, SUM(1-ok) AS failures,
                   AVG(latency_ms) AS avg_latency_ms, MAX(latency_ms) AS max_latency_ms,
                   SUM(COALESCE(input_tokens,0)) AS input_tokens, SUM(COALESCE(output_tokens,0)) AS output_tokens,
                   SUM(cost_usd) AS cost_usd, SUM(CASE WHEN fallback_from IS NOT NULL THEN 1 ELSE 0 END) AS fallback_calls
                FROM provider_calls {where} GROUP BY provider_id, capability ORDER BY provider_id""",
            params,
        )
        for r in rows:
            lat = [x["latency_ms"] for x in self.db.all(
                f"SELECT latency_ms FROM provider_calls WHERE provider_id=? AND capability=? AND ok=1 {'AND created_at >= ?' if since else ''} ORDER BY latency_ms",
                (r["provider_id"], r["capability"], *params),
            )]
            r["p50_latency_ms"] = lat[len(lat) // 2] if lat else None
            r["p95_latency_ms"] = lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else None
            r["failure_rate"] = (r["failures"] or 0) / r["calls"] if r["calls"] else 0.0
        return rows

    def recent_failures(self, provider_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        if self.db is None:
            return []
        if provider_id:
            return self.db.all(
                "SELECT * FROM provider_calls WHERE ok=0 AND provider_id=? ORDER BY id DESC LIMIT ?", (provider_id, limit)
            )
        return self.db.all("SELECT * FROM provider_calls WHERE ok=0 ORDER BY id DESC LIMIT ?", (limit,))


def records_to_dicts(records: list[CallRecord]) -> list[dict[str, Any]]:
    return [asdict(r) for r in records]
