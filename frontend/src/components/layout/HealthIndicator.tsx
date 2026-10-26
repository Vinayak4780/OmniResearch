import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import type { Task } from '../../types/api';
import { useHealth } from './HealthContext';

const TASK_ORDER: Task[] = [
  'synthesis',
  'planning',
  'decomposition',
  'rewriting',
  'verification',
  'text_embedding',
  'image_embedding',
  'reranking',
  'ocr',
  'vision',
  'web_search',
  'web_extraction',
];

const STATE_LABEL = { down: 'Offline', unknown: 'Checking…', ok: 'Healthy', degraded: 'Degraded' } as const;

function Yes({ ok }: { ok: boolean }) {
  return <span className={ok ? 'text-success' : 'muted'}>{ok ? 'yes' : 'no'}</span>;
}

export function HealthIndicator() {
  const { health, error, reload } = useHealth();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDoc);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const state: keyof typeof STATE_LABEL = error ? 'down' : !health ? 'unknown' : health.status === 'ok' ? 'ok' : 'degraded';

  return (
    <div className="health" ref={ref}>
      <button
        type="button"
        className={`health-button health-${state}`}
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-label={`System health: ${STATE_LABEL[state]}`}
        onClick={() => setOpen(!open)}
      >
        <span className="health-dot" aria-hidden="true" />
        <span className="health-label">{STATE_LABEL[state]}</span>
      </button>
      {open && (
        <div className="health-popover" role="dialog" aria-label="System health">
          {error ? (
            <p>
              The API at <code>/api/health</code> is unreachable. Is the backend running?
            </p>
          ) : !health ? (
            <p className="muted">Loading…</p>
          ) : (
            <>
              <dl className="kv kv-compact">
                <div className="kv-row">
                  <dt>Status</dt>
                  <dd>{health.status}</dd>
                </div>
                <div className="kv-row">
                  <dt>Database</dt>
                  <dd>
                    <Yes ok={health.database} />
                  </dd>
                </div>
                <div className="kv-row">
                  <dt>Vector store</dt>
                  <dd>
                    {health.vector_store.mode} · <Yes ok={health.vector_store.ok} />
                  </dd>
                </div>
                <div className="kv-row">
                  <dt>Parser</dt>
                  <dd>{health.parser.configured}</dd>
                </div>
                <div className="kv-row">
                  <dt>Docling installed</dt>
                  <dd>
                    <Yes ok={health.parser.docling_installed} />
                  </dd>
                </div>
                <div className="kv-row">
                  <dt>PaddleOCR installed</dt>
                  <dd>
                    <Yes ok={health.paddleocr_installed} />
                  </dd>
                </div>
                <div className="kv-row">
                  <dt>Admin token</dt>
                  <dd>{health.admin_token_configured ? 'required' : 'not configured (localhost only)'}</dd>
                </div>
              </dl>
              <div className="small muted health-tasks-title">Configured tasks</div>
              <ul className="task-grid">
                {TASK_ORDER.map((t) => {
                  const on = !!health.configured_tasks[t];
                  return (
                    <li key={t} className={on ? 'task-on' : 'task-off'}>
                      <span aria-hidden="true">{on ? '●' : '○'}</span> {t.replace(/_/g, ' ')}
                      <span className="sr-only">{on ? ' configured' : ' not configured'}</span>
                    </li>
                  );
                })}
              </ul>
            </>
          )}
          <div className="popover-actions">
            <button type="button" className="btn btn-sm" onClick={() => void reload()}>
              Refresh
            </button>
            <Link className="btn btn-sm" to="/settings/connect" onClick={() => setOpen(false)}>
              Connect AI
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
