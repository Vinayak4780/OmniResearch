import { useCallback, useEffect, useRef, useState } from 'react';
import { usePolling } from '../../hooks/usePolling';
import { useResource } from '../../hooks/useResource';
import { api, urls } from '../../services/api';
import type { ResearchEvent, ResearchJob, ResearchReport } from '../../types/api';
import { fmtDate, fmtInt, fmtUsd, isNum } from '../../utils/format';
import { Alert, Card, Chip, ErrorMessage, KeyValue, Spinner, Stat, StatusBadge } from '../ui';
import { CompletedJob } from './CompletedJob';
import { StageTimeline } from './StageTimeline';

const ACTIVE = new Set(['queued', 'running']);

export function UsagePanel({ report }: { report: ResearchReport }) {
  const u = report.usage;
  const byTask = report.providers_by_task ?? {};
  return (
    <div>
      {u ? (
        <div className="stats-row">
          <Stat label="Provider calls" value={fmtInt(u.calls)} />
          <Stat label="Failed calls" value={fmtInt(u.failed_calls)} />
          <Stat label="Fallback calls" value={fmtInt(u.fallback_calls)} />
          <Stat label="Input tokens" value={fmtInt(u.input_tokens)} />
          <Stat label="Output tokens" value={fmtInt(u.output_tokens)} />
          <Stat label="Estimated cost" value={fmtUsd(u.cost_usd)} hint={u.cost_usd == null ? 'no pricing configured' : undefined} />
        </div>
      ) : (
        <p className="muted small">No usage was recorded.</p>
      )}
      {Object.keys(byTask).length > 0 && (
        <div className="table-wrap">
          <table className="table table-compact">
            <caption className="small muted">Providers used per task</caption>
            <thead>
              <tr>
                <th scope="col">Task</th>
                <th scope="col">Provider (model)</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(byTask).map(([t, ps]) => (
                <tr key={t}>
                  <td>{t.replace(/_/g, ' ')}</td>
                  <td className="small">{ps.join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <KeyValue
        items={[
          ['Web research used', report.web_used === undefined ? null : report.web_used ? 'yes' : 'no'],
          ['Web decision', report.web_reason ?? null],
          ['Follow-up rounds', isNum(report.iterations) ? String(report.iterations) : null],
          ['Queries issued', report.queries ? String(report.queries.length) : null],
        ]}
      />
      {report.queries && report.queries.length > 0 && (
        <details>
          <summary>Queries ({report.queries.length})</summary>
          <ul className="tight-list small">
            {report.queries.map((q, i) => (
              <li key={i}>
                {q.query} {q.source && <span className="muted">({q.source})</span>}
              </li>
            ))}
          </ul>
        </details>
      )}
      {report.warnings && report.warnings.length > 0 && (
        <details>
          <summary>Warnings ({report.warnings.length})</summary>
          <ul className="tight-list small">
            {report.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

export function JobView({ jobId, onChanged }: { jobId: string; onChanged: () => void }) {
  const job = useResource(() => api.researchJob(jobId), [jobId]);
  const [events, setEvents] = useState<ResearchEvent[]>([]);
  const [eventsError, setEventsError] = useState<unknown>(null);
  const lastId = useRef(0);
  const [cancelling, setCancelling] = useState(false);
  const [cancelError, setCancelError] = useState<unknown>(null);
  const prevStatus = useRef<string | undefined>(undefined);

  const status = job.data?.status;
  const active = !!status && ACTIVE.has(status);

  const fetchEvents = useCallback(async () => {
    try {
      const evs = await api.researchEvents(jobId, lastId.current);
      if (evs.length) {
        lastId.current = evs[evs.length - 1].id;
        setEvents((prev) => {
          const known = new Set(prev.map((p) => p.id));
          return [...prev, ...evs.filter((e) => !known.has(e.id))];
        });
      }
      setEventsError(null);
    } catch (e) {
      setEventsError(e);
    }
  }, [jobId]);

  useEffect(() => {
    lastId.current = 0;
    setEvents([]);
    prevStatus.current = undefined;
    void fetchEvents();
  }, [jobId, fetchEvents]);

  usePolling(
    async () => {
      await fetchEvents();
      await job.reload(true);
    },
    1500,
    active,
  );

  useEffect(() => {
    if (!status) return;
    if (prevStatus.current && prevStatus.current !== status) {
      onChanged();
      void fetchEvents();
    }
    prevStatus.current = status;
  }, [status, onChanged, fetchEvents]);

  const cancel = async () => {
    if (!window.confirm('Cancel this investigation? Work done so far is discarded.')) return;
    setCancelError(null);
    setCancelling(true);
    try {
      await api.cancelResearch(jobId);
      await job.reload(true);
      onChanged();
    } catch (e) {
      setCancelError(e);
    } finally {
      setCancelling(false);
    }
  };

  if (job.loading && !job.data) return <Spinner label="Loading investigation…" />;
  if (job.error && !job.data) return <ErrorMessage error={job.error} title="Could not load investigation" />;
  const j: ResearchJob = job.data!;
  const report = j.report;

  return (
    <div className="job-view">
      <Card>
        <div className="job-head">
          <div className="job-head-main">
            <div className="chip-row small">
              <StatusBadge status={j.status} />
              <span className="muted">Started {fmtDate(j.created_at)}</span>
              {j.status === 'completed' && <span className="muted">· finished {fmtDate(j.updated_at)}</span>}
            </div>
            <h2 className="job-question">{j.question}</h2>
            <div className="chip-row small">
              <span className="muted">Scope:</span>
              {j.options.collections?.length ? j.options.collections.map((c) => <Chip key={c}>{c}</Chip>) : <Chip>all collections</Chip>}
              {j.options.document_ids?.length ? <Chip>{j.options.document_ids.length} selected documents</Chip> : null}
              <Chip>web: {j.options.web_mode ?? 'default'}</Chip>
            </div>
          </div>
          <div className="btn-row">
            {active && (
              <button type="button" className="btn btn-danger" onClick={cancel} disabled={cancelling}>
                {cancelling ? 'Cancelling…' : 'Cancel'}
              </button>
            )}
            {j.status === 'completed' && j.report_md && (
              <a className="btn btn-primary" href={urls.reportMarkdown(j.id)} download={`omniresearch-${j.id}.md`}>
                Download Markdown
              </a>
            )}
          </div>
        </div>
        <ErrorMessage error={cancelError} title="Could not cancel" />
        {j.status === 'failed' && (
          <Alert tone="danger" title="The investigation failed">
            <p className="pre-wrap">{j.error ?? 'No error message was recorded.'}</p>
          </Alert>
        )}
        {j.status === 'cancelled' && <Alert tone="neutral" title="Cancelled">This investigation was cancelled before it finished.</Alert>}
        {j.status === 'interrupted' && (
          <Alert tone="warning" title="Interrupted">
            {j.error ?? 'The server restarted while this investigation was running.'}
          </Alert>
        )}
      </Card>

      {active ? (
        <Card title="Progress" subtitle="Updates every 1.5 seconds">
          <ErrorMessage error={eventsError} title="Could not fetch progress" />
          <StageTimeline events={events} status={j.status} createdAt={j.created_at} />
        </Card>
      ) : j.status !== 'completed' ? (
        <Card>
          <details open>
            <summary className="card-title-summary">Stage timeline ({events.length} events)</summary>
            <StageTimeline events={events} status={j.status} />
          </details>
        </Card>
      ) : null}

      {j.status === 'completed' && (
        <>
          <CompletedJob job={j} />
          <Card>
            <details>
              <summary className="card-title-summary">Stage timeline ({events.length} events)</summary>
              <StageTimeline events={events} status={j.status} />
            </details>
          </Card>
          <Card title="Providers, usage and cost">{report ? <UsagePanel report={report} /> : <p className="muted">No structured report data.</p>}</Card>
        </>
      )}
    </div>
  );
}
