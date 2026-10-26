import { useState } from 'react';
import { ProviderMetricsTable } from '../../components/ProviderMetricsTable';
import { Card, EmptyState, ErrorMessage, Spinner, StatusBadge, Toggle } from '../../components/ui';
import { usePolling } from '../../hooks/usePolling';
import { useResource } from '../../hooks/useResource';
import { api } from '../../services/api';
import { fmtDate, fmtInt, fmtMs, fmtSeconds } from '../../utils/format';

export function HealthPage() {
  const h = useResource(() => api.providerHealth(), []);
  const [auto, setAuto] = useState(true);
  usePolling(() => h.reload(true), 10000, auto);

  if (h.loading && !h.data) return <Spinner label="Loading provider health…" />;
  if (h.error && !h.data) return <ErrorMessage error={h.error} title="Could not load provider health" />;
  const d = h.data!;
  const circuits = Object.entries(d.circuits);

  return (
    <div className="stack">
      <Card
        title="Provider metrics"
        subtitle="All recorded provider calls since installation. Latency percentiles use successful calls only."
        actions={
          <>
            <Toggle checked={auto} onChange={setAuto} label="Auto-refresh (10 s)" />
            <button type="button" className="btn btn-sm" onClick={() => void h.reload(true)}>
              Refresh
            </button>
          </>
        }
      >
        <ErrorMessage error={h.error} />
        <ProviderMetricsTable rows={d.metrics} />
      </Card>

      <Card title="Circuit breakers" subtitle="A circuit opens after repeated failures; calls skip to fallbacks until it half-opens again. State is in memory and resets on restart.">
        {circuits.length === 0 ? (
          <EmptyState title="No circuits yet">Circuits appear after a provider has been called.</EmptyState>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th scope="col">Provider : capability</th>
                  <th scope="col">State</th>
                  <th scope="col" className="num">Consecutive failures</th>
                  <th scope="col">Re-opens in</th>
                  <th scope="col">Last error</th>
                </tr>
              </thead>
              <tbody>
                {circuits.map(([k, c]) => (
                  <tr key={k}>
                    <td className="mono small">{k}</td>
                    <td>
                      <StatusBadge status={c.state} />
                    </td>
                    <td className="num">{fmtInt(c.consecutive_failures)}</td>
                    <td>{c.reopens_in_s != null ? fmtSeconds(c.reopens_in_s) : '—'}</td>
                    <td className="small break">{c.last_error ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Card title="Recent failures">
        {d.recent_failures.length === 0 ? (
          <EmptyState title="No failed calls" />
        ) : (
          <div className="table-wrap">
            <table className="table table-compact">
              <thead>
                <tr>
                  <th scope="col">When</th>
                  <th scope="col">Provider</th>
                  <th scope="col">Task</th>
                  <th scope="col">Kind</th>
                  <th scope="col">Message</th>
                  <th scope="col" className="num">Latency</th>
                </tr>
              </thead>
              <tbody>
                {d.recent_failures.map((f) => (
                  <tr key={f.id}>
                    <td className="small nowrap">{fmtDate(f.created_at)}</td>
                    <td className="mono small">{f.provider_id}</td>
                    <td className="small">{f.task ?? f.capability}</td>
                    <td>
                      <StatusBadge status="failed" title={f.error_kind ?? undefined} /> <span className="small">{f.error_kind}</span>
                    </td>
                    <td className="small break">{f.error_message ?? '—'}</td>
                    <td className="num">{fmtMs(f.latency_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
