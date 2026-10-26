import { useState, type FormEvent } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { Alert, Badge, Card, EmptyState, ErrorMessage, ProgressBar, Spinner, Tabs } from '../../components/ui';
import { usePolling } from '../../hooks/usePolling';
import { useResource } from '../../hooks/useResource';
import { api } from '../../services/api';
import type { HfResult, LocalCatalog, LocalJob, LocalKind, LocalModel } from '../../types/localmodels';
import { fmtInt } from '../../utils/format';

export const KIND_LABEL: Record<LocalKind, string> = {
  ocr: 'Reads scanned pages (OCR)',
  embedding: 'Document search (embeddings)',
  reranking: 'Ranks search results',
};

const KIND_INTRO: Record<LocalKind, string> = {
  ocr: 'Turns images of scanned pages into text. Pick a model for the language of your documents.',
  embedding: 'Turns your documents into vectors so search finds passages by meaning, not just matching words.',
  reranking: 'Re-orders search results so the most relevant passages come first.',
};

const KINDS: LocalKind[] = ['ocr', 'embedding', 'reranking'];
const COLLAPSED_ROWS = 8;

export function fmtSize(gb: number | null | undefined): string {
  if (gb == null || !Number.isFinite(gb)) return 'size unknown';
  return gb < 1 ? `${Math.max(1, Math.round(gb * 1000))} MB` : `${gb.toFixed(1)} GB`;
}

export function jobProgressText(j: LocalJob): string {
  const done = j.downloaded_mb != null ? Math.round(j.downloaded_mb) : null;
  const total = j.expected_mb != null ? Math.round(j.expected_mb) : null;
  if (done != null && total) return `${done} / ${total} MB`;
  if (done != null) return `${done} MB downloaded`;
  return 'Starting download…';
}

function JobProgress({ job }: { job: LocalJob }) {
  if (job.status === 'failed') return <div className="small text-danger">Install failed: {job.error ?? 'unknown error'}</div>;
  if (job.status !== 'downloading') return null;
  return (
    <div className="local-job" role="status">
      <ProgressBar value={job.progress} indeterminate={job.progress == null} label={`Downloading ${job.model}`} />
      <span className="small muted num">{jobProgressText(job)}</span>
    </div>
  );
}

interface RowProps {
  kind: LocalKind;
  m: LocalModel;
  busy: boolean;
  onInstall: () => void;
  onUse: () => void;
  onRemove: () => void;
}

function ModelRowView({ kind, m, busy, onInstall, onUse, onRemove }: RowProps) {
  const downloading = m.job?.status === 'downloading';
  return (
    <li className={`local-model ${m.in_use ? 'local-model-inuse' : ''}`} data-model={m.id}>
      <div className="local-model-main">
        <div className="local-model-head">
          <span className="local-model-name">{m.id}</span>
          {m.in_use && <Badge tone="success">In use</Badge>}
          {!m.in_use && m.installed && <Badge tone="info">Installed</Badge>}
          {m.recommended && <Badge tone="accent">Recommended</Badge>}
          {m.source === 'custom' && <Badge>From Hugging Face</Badge>}
        </div>
        <div className="small local-model-desc">
          {kind === 'ocr' && m.language ? <strong>{m.language}. </strong> : null}
          {kind === 'ocr' && m.language ? m.description.replace(new RegExp(`^${m.language}\\s*·\\s*`), '') : m.description}
        </div>
        <div className="small muted local-model-meta">
          <span>{fmtSize(m.size_gb)}</span>
          {m.dims ? <span>{m.dims} dimensions</span> : null}
          {m.license ? <span>license: {m.license}</span> : null}
        </div>
        {m.job && <JobProgress job={m.job} />}
      </div>
      <div className="local-model-actions">
        {m.in_use ? null : m.installed ? (
          <>
            <button type="button" className="btn btn-sm btn-primary" disabled={busy} onClick={onUse} aria-label={`Use ${m.id}`}>
              Use
            </button>
            <button type="button" className="btn btn-sm btn-danger-ghost" disabled={busy} onClick={onRemove} aria-label={`Remove ${m.id}`}>
              Remove
            </button>
          </>
        ) : (
          <button type="button" className="btn btn-sm" disabled={busy || downloading} onClick={onInstall} aria-label={`Install ${m.id}`}>
            {downloading ? 'Installing…' : m.job?.status === 'failed' ? 'Retry install' : 'Install'}
          </button>
        )}
      </div>
    </li>
  );
}

function HfSearch({ kind, busy, onInstall }: { kind: LocalKind; busy: boolean; onInstall: (r: HfResult) => void }) {
  const [q, setQ] = useState('');
  const [results, setResults] = useState<HfResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      setResults(await api.searchLocalModels(kind, q.trim()));
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };
  return (
    <div className="hf-search">
      <form onSubmit={submit} className="hf-search-form" role="search">
        <label htmlFor={`hf-${kind}`} className="sr-only">
          Search Hugging Face for {KIND_LABEL[kind]} models
        </label>
        <input id={`hf-${kind}`} type="search" placeholder="Search Hugging Face…" value={q} onChange={(e) => setQ(e.target.value)} />
        <button type="submit" className="btn" disabled={loading}>
          {loading ? 'Searching…' : 'Search'}
        </button>
      </form>
      <p className="small muted">
        {kind === 'ocr'
          ? 'Finds PaddleOCR recognition models (e.g. for other languages).'
          : 'Only models with ONNX weights are listed, because those run fast on a CPU.'}
      </p>
      <ErrorMessage error={error} title="Search failed" />
      {results && results.length === 0 && <p className="small muted">No matching models found.</p>}
      {results && results.length > 0 && (
        <ul className="hf-results" aria-label="Hugging Face results">
          {results.map((r) => (
            <li key={r.id} className="hf-result" data-hf={r.id}>
              <div className="hf-result-main">
                <a href={r.url} target="_blank" rel="noopener noreferrer" className="mono small">
                  {r.repo}
                  <span className="sr-only"> (opens Hugging Face in a new tab)</span>
                </a>
                {r.id !== r.repo && <div className="small muted mono">{r.id}</div>}
                <div className="small muted">
                  {fmtInt(r.downloads)} downloads · {fmtInt(r.likes)} likes {r.in_catalog && <Badge>in the list above</Badge>}
                </div>
              </div>
              <button type="button" className="btn btn-sm" disabled={busy} onClick={() => onInstall(r)} aria-label={`Install ${r.id} from Hugging Face`}>
                Install
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function KindSection({ kind, catalog, reload }: { kind: LocalKind; catalog: LocalCatalog; reload: () => Promise<unknown> }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [notes, setNotes] = useState<string[]>([]);
  const [filter, setFilter] = useState('');
  const [showAll, setShowAll] = useState(false);
  const models = catalog.kinds[kind] ?? [];
  const inUse = catalog.in_use[kind];
  const available = catalog.available[kind];

  const run = async (fn: () => Promise<string[] | void>) => {
    setBusy(true);
    setError(null);
    setNotes([]);
    try {
      const n = await fn();
      if (n) setNotes(n);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
      await reload();
    }
  };

  const install = (model: string, repo?: string) => run(async () => void (await api.installLocalModel({ kind, model, ...(repo ? { repo } : {}) })));
  const use = (m: LocalModel) => {
    if (kind === 'embedding' && !window.confirm(`Use ${m.id} for document search? The search index is rebuilt automatically in the background.`)) return;
    void run(async () => {
      const r = await api.useLocalModel({ kind, model: m.id });
      return r.notes?.length ? r.notes : [`${m.id} is now used for “${KIND_LABEL[kind]}”.`];
    });
  };
  const remove = (m: LocalModel) => {
    if (!window.confirm(`Remove ${m.id} from this computer? You can install it again later.`)) return;
    void run(async () => void (await api.removeLocalModel({ kind, model: m.id })));
  };

  const q = filter.trim().toLowerCase();
  const filtered = models.filter((m) => !q || m.id.toLowerCase().includes(q) || m.description.toLowerCase().includes(q) || (m.language ?? '').toLowerCase().includes(q));
  const visible = showAll || q ? filtered : filtered.slice(0, COLLAPSED_ROWS);

  return (
    <section className="local-kind" aria-label={KIND_LABEL[kind]}>
      <p className="small">{KIND_INTRO[kind]}</p>
      <div className="local-inuse">
        <span className="muted">In use now:</span>{' '}
        {inUse ? <strong>{inUse}</strong> : <span>No model on this computer — this job uses the connected cloud service (or is off).</span>}
      </div>
      <p className="small muted">
        Using a local model replaces the connected cloud model for this job. You can switch back in <Link to="/settings/connect">Connect AI</Link>.
        {kind === 'embedding' && ' Changing the embedding model rebuilds the search index automatically.'}
      </p>
      {!available ? (
        <Alert tone="warning" title="Local models aren't available on this server">
          The server doesn't have the local model runtime installed, so models can't be installed here. (Server admins: <code>pip install -r requirements-ml.txt</code>.)
        </Alert>
      ) : (
        <>
          <ErrorMessage error={error} title="That didn't work" />
          {notes.map((n) => (
            <Alert key={n} tone="info" role="status">
              {n}
            </Alert>
          ))}
          {models.length > COLLAPSED_ROWS && (
            <input
              type="search"
              className="local-filter"
              placeholder={`Filter ${models.length} models${kind === 'ocr' ? ' (e.g. a language)' : ''}…`}
              aria-label={`Filter ${KIND_LABEL[kind]} models`}
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
          )}
          {filtered.length === 0 ? (
            <EmptyState title="No models match" />
          ) : (
            <ul className="local-model-list">
              {visible.map((m) => (
                <ModelRowView key={m.id} kind={kind} m={m} busy={busy} onInstall={() => void install(m.id)} onUse={() => use(m)} onRemove={() => remove(m)} />
              ))}
            </ul>
          )}
          {!q && filtered.length > COLLAPSED_ROWS && (
            <button type="button" className="btn btn-link btn-sm" onClick={() => setShowAll(!showAll)}>
              {showAll ? 'Show fewer' : `Show all ${filtered.length} models`}
            </button>
          )}
          <h3 className="local-subtitle">More models from Hugging Face</h3>
          <HfSearch kind={kind} busy={busy} onInstall={(r) => void install(r.id, r.repo)} />
        </>
      )}
    </section>
  );
}

export function LocalModelsPage() {
  const catalog = useResource(() => api.localModels(), []);
  const [params, setParams] = useSearchParams();
  const k = params.get('kind') as LocalKind | null;
  const kind: LocalKind = k && KINDS.includes(k) ? k : 'ocr';
  const downloading = (catalog.data?.jobs ?? []).some((j) => j.status === 'downloading') || KINDS.some((x) => (catalog.data?.kinds[x] ?? []).some((m) => m.job?.status === 'downloading'));
  usePolling(() => catalog.reload(true), 1500, downloading);

  return (
    <Card title="Models on this computer" subtitle="Free, private, run on your CPU. Install from the list or from Hugging Face.">
      {catalog.error && !catalog.data ? (
        <ErrorMessage error={catalog.error} title="Could not load local models" />
      ) : !catalog.data ? (
        <Spinner label="Loading models…" />
      ) : (
        <>
          <Tabs
            label="Kinds of local models"
            active={kind}
            onChange={(t) => setParams(t === 'ocr' ? {} : { kind: t }, { replace: true })}
            tabs={KINDS.map((x) => ({
              id: x,
              label: (
                <>
                  {KIND_LABEL[x]}
                  {catalog.data!.in_use[x] && <span className="tab-dot" aria-label=" (a local model is in use)" />}
                </>
              ),
            }))}
          />
          <div role="tabpanel" aria-label={KIND_LABEL[kind]}>
            <KindSection key={kind} kind={kind} catalog={catalog.data} reload={() => catalog.reload(true)} />
          </div>
        </>
      )}
    </Card>
  );
}
