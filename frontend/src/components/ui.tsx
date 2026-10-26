import { useId, useRef, type KeyboardEvent, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { ApiError, errorMessage } from '../services/api';
import { DASH, humanize } from '../utils/format';

// ------------------------------------------------------------------ badges

export type Tone = 'neutral' | 'info' | 'success' | 'warning' | 'danger' | 'accent';

export function Badge({ tone = 'neutral', children, title }: { tone?: Tone; children: ReactNode; title?: string }) {
  return (
    <span className={`badge badge-${tone}`} title={title}>
      {children}
    </span>
  );
}

const STATUS_TONES: Record<string, Tone> = {
  queued: 'neutral',
  processing: 'info',
  running: 'info',
  building: 'info',
  validating: 'info',
  ready: 'success',
  completed: 'success',
  succeeded: 'success',
  active: 'success',
  done: 'success',
  ok: 'success',
  closed: 'success',
  ready_lexical_only: 'warning',
  ready_with_errors: 'warning',
  interrupted: 'warning',
  cancelled: 'neutral',
  skipped: 'warning',
  previous: 'neutral',
  half_open: 'warning',
  degraded: 'warning',
  failed: 'danger',
  open: 'danger',
  error: 'danger',
};

const STATUS_LABELS: Record<string, string> = {
  ready_lexical_only: 'Ready (lexical only)',
  ready_with_errors: 'Ready with errors',
  half_open: 'Half-open',
};

export function StatusBadge({ status, title }: { status: string | null | undefined; title?: string }) {
  if (!status) return <span className="muted">{DASH}</span>;
  const tone = STATUS_TONES[status] ?? 'neutral';
  const live = tone === 'info';
  return (
    <Badge tone={tone} title={title}>
      {live && <span className="pulse-dot" aria-hidden="true" />}
      {STATUS_LABELS[status] ?? humanize(status)}
    </Badge>
  );
}

// ------------------------------------------------------------------ feedback

export function Spinner({ label = 'Loading…', inline = false }: { label?: string; inline?: boolean }) {
  return (
    <span className={inline ? 'spinner-inline' : 'spinner-block'} role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <span className={inline ? 'sr-only' : 'muted'}>{label}</span>
    </span>
  );
}

export function Alert({ tone = 'info', title, children, role }: { tone?: Tone; title?: ReactNode; children?: ReactNode; role?: 'alert' | 'status' }) {
  return (
    <div className={`alert alert-${tone}`} role={role ?? (tone === 'danger' ? 'alert' : undefined)}>
      {title && <div className="alert-title">{title}</div>}
      {children && <div className="alert-body">{children}</div>}
    </div>
  );
}

/** Renders an API error; 401/403 get a pointer to the admin-token setting. */
export function ErrorMessage({ error, title }: { error: unknown; title?: string }) {
  if (!error) return null;
  if (error instanceof ApiError && error.isAuthError) {
    return (
      <Alert tone="danger" title={title ?? 'Admin access required'}>
        <p>{error.detail}</p>
        <p>
          {error.status === 401
            ? 'This server requires an admin token. '
            : 'This server only allows admin actions from the local machine unless OMNI_ADMIN_TOKEN is configured. '}
          Enter the admin token under <Link to="/settings/access">Settings → Admin access</Link>.
        </p>
      </Alert>
    );
  }
  return (
    <Alert tone="danger" title={title ?? 'Something went wrong'}>
      <p className="pre-wrap">{errorMessage(error)}</p>
    </Alert>
  );
}

export function EmptyState({ title, children, icon }: { title: string; children?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="empty-state">
      {icon && <div className="empty-icon" aria-hidden="true">{icon}</div>}
      <div className="empty-title">{title}</div>
      {children && <div className="empty-body">{children}</div>}
    </div>
  );
}

export function ProgressBar({ value, label, indeterminate = false }: { value: number | null; label?: string; indeterminate?: boolean }) {
  const pct = value == null ? 0 : Math.max(0, Math.min(1, value)) * 100;
  return (
    <div
      className={`progress ${indeterminate ? 'progress-indeterminate' : ''}`}
      role="progressbar"
      aria-label={label ?? 'Progress'}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={indeterminate ? undefined : Math.round(pct)}
    >
      <div className="progress-fill" style={{ width: indeterminate ? undefined : `${pct}%` }} />
    </div>
  );
}

// ------------------------------------------------------------------ layout helpers

export function Card({ title, actions, children, className = '', subtitle }: { title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <header className="card-header">
          <div>
            {title && <h2 className="card-title">{title}</h2>}
            {subtitle && <div className="card-subtitle">{subtitle}</div>}
          </div>
          {actions && <div className="card-actions">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

export function PageHeader({ title, description, actions }: { title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="page-header">
      <div>
        <h1>{title}</h1>
        {description && <p className="page-description">{description}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </div>
  );
}

export function KeyValue({ items }: { items: [ReactNode, ReactNode][] }) {
  return (
    <dl className="kv">
      {items.map(([k, v], i) => (
        <div className="kv-row" key={i}>
          <dt>{k}</dt>
          <dd>{v === null || v === undefined || v === '' ? <span className="muted">{DASH}</span> : v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}

// ------------------------------------------------------------------ tabs

export interface TabDef {
  id: string;
  label: ReactNode;
}

export function Tabs({ tabs, active, onChange, label }: { tabs: TabDef[]; active: string; onChange: (id: string) => void; label: string }) {
  const base = useId();
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const onKey = (e: KeyboardEvent<HTMLButtonElement>, i: number) => {
    let next = -1;
    if (e.key === 'ArrowRight') next = (i + 1) % tabs.length;
    if (e.key === 'ArrowLeft') next = (i - 1 + tabs.length) % tabs.length;
    if (e.key === 'Home') next = 0;
    if (e.key === 'End') next = tabs.length - 1;
    if (next >= 0) {
      e.preventDefault();
      onChange(tabs[next].id);
      refs.current[next]?.focus();
    }
  };
  return (
    <div className="tabs" role="tablist" aria-label={label}>
      {tabs.map((t, i) => (
        <button
          key={t.id}
          ref={(el) => {
            refs.current[i] = el;
          }}
          type="button"
          role="tab"
          id={`${base}-tab-${t.id}`}
          aria-selected={active === t.id}
          tabIndex={active === t.id ? 0 : -1}
          className={`tab ${active === t.id ? 'tab-active' : ''}`}
          onClick={() => onChange(t.id)}
          onKeyDown={(e) => onKey(e, i)}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

export function Toggle({ checked, onChange, label, disabled, id }: { checked: boolean; onChange: (v: boolean) => void; label: string; disabled?: boolean; id?: string }) {
  return (
    <label className={`toggle ${disabled ? 'toggle-disabled' : ''}`}>
      <input id={id} type="checkbox" role="switch" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <span className="toggle-track" aria-hidden="true">
        <span className="toggle-thumb" />
      </span>
      <span className="toggle-label">{label}</span>
    </label>
  );
}

export function Chip({ children, tone = 'neutral' }: { children: ReactNode; tone?: Tone }) {
  return <span className={`chip chip-${tone}`}>{children}</span>;
}

export function Num({ children }: { children: ReactNode }) {
  return <span className="num">{children}</span>;
}
