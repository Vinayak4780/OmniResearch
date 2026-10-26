import type { ProviderMetricRow } from '../types/api';
import { fmtInt, fmtMs, fmtPct, fmtUsd } from '../utils/format';
import { EmptyState } from './ui';

export function ProviderMetricsTable({ rows, caption }: { rows: ProviderMetricRow[]; caption?: string }) {
  if (!rows.length) return <EmptyState title="No provider calls recorded yet" />;
  return (
    <div className="table-wrap">
      <table className="table">
        {caption && <caption className="small muted">{caption}</caption>}
        <thead>
          <tr>
            <th scope="col">Provider</th>
            <th scope="col">Capability</th>
            <th scope="col" className="num">Calls</th>
            <th scope="col" className="num">Failure rate</th>
            <th scope="col" className="num">p50</th>
            <th scope="col" className="num">p95</th>
            <th scope="col" className="num">Max</th>
            <th scope="col" className="num">Input tok.</th>
            <th scope="col" className="num">Output tok.</th>
            <th scope="col" className="num">Cost</th>
            <th scope="col" className="num">Fallback calls</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.provider_id}:${r.capability}`}>
              <td className="mono small">{r.provider_id}</td>
              <td>{r.capability}</td>
              <td className="num">{fmtInt(r.calls)}</td>
              <td className={`num ${r.failure_rate > 0.2 ? 'text-danger' : r.failure_rate > 0 ? 'text-warning' : ''}`}>
                {fmtPct(r.failure_rate)} <span className="muted small">({fmtInt(r.failures ?? 0)})</span>
              </td>
              <td className="num">{fmtMs(r.p50_latency_ms)}</td>
              <td className="num">{fmtMs(r.p95_latency_ms)}</td>
              <td className="num">{fmtMs(r.max_latency_ms)}</td>
              <td className="num">{fmtInt(r.input_tokens)}</td>
              <td className="num">{fmtInt(r.output_tokens)}</td>
              <td className="num">{fmtUsd(r.cost_usd)}</td>
              <td className="num">{fmtInt(r.fallback_calls)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
