import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { useResource } from '../../hooks/useResource';
import { api, urls } from '../../services/api';
import type { NotebookSource } from '../../types/notebooks';
import { fmtBytes, humanize } from '../../utils/format';
import { ExtractionBadge } from '../ExtractionBadge';
import { Badge, ErrorMessage, Spinner } from '../ui';
import { FileIcon, GlobeIcon, domainOf } from './webHints';

export { AddSourcesDialog } from './AddSourcesDialog';

export const isReady = (s: NotebookSource) => s.status.startsWith('ready');
export const isProcessing = (s: NotebookSource) => s.status === 'queued' || s.status === 'processing' || s.job?.status === 'queued' || s.job?.status === 'running';

function StatusPill({ s, onRetry }: { s: NotebookSource; onRetry: () => void }) {
  if (isProcessing(s)) {
    const pct = s.job?.progress != null ? ` ${Math.round(s.job.progress * 100)}%` : '';
    return (
      <span className="nb-pill nb-pill-busy" role="status">
        <span className="pulse-dot" aria-hidden="true" /> {humanize(s.job?.stage ?? s.status)}
        {pct}
      </span>
    );
  }
  if (s.status === 'failed' || s.status === 'ready_with_errors') {
    return (
      <span className="nb-pill-row">
        <span className={`nb-pill ${s.status === 'failed' ? 'nb-pill-bad' : 'nb-pill-warn'}`} title={s.error ?? undefined}>
          {s.status === 'failed' ? 'Failed' : 'Some pages unreadable'}
        </span>
        <button type="button" className="btn btn-link btn-sm" onClick={onRetry} aria-label={`Retry ${s.title || s.filename}`}>
          Retry
        </button>
      </span>
    );
  }
  return null;
}

// ------------------------------------------------------------------ source preview (pages and passages)

export function SourcePreview({ source, onBack }: { source: NotebookSource; onBack: () => void }) {
  const [page, setPage] = useState(1);
  const chunks = useResource(() => api.chunks(source.id, page), [source.id, page]);
  const pages = source.page_count ?? 1;
  return (
    <div className="nb-source-preview">
      <div className="nb-preview-head">
        <button type="button" className="btn btn-ghost btn-sm" onClick={onBack} aria-label="Back to sources">
          ← Sources
        </button>
      </div>
      <h3 className="nb-preview-title">{source.title || source.filename}</h3>
      <div className="small muted">
        {source.source_url ? `Saved from ${domainOf(source.source_url)}` : source.filename} · {source.page_count ?? '?'} pages · {fmtBytes(source.size_bytes)}
      </div>
      {source.source_url && (
        <a className="nb-original-link small" href={source.source_url} target="_blank" rel="noopener noreferrer">
          Open original ↗
        </a>
      )}
      {isReady(source) ? (
        <>
          <div className="nb-preview-nav">
            <button type="button" className="btn btn-sm" disabled={page <= 1} onClick={() => setPage(page - 1)} aria-label="Previous page">
              ←
            </button>
            <span className="small">
              Page {page} / {pages}
            </span>
            <button type="button" className="btn btn-sm" disabled={page >= pages} onClick={() => setPage(page + 1)} aria-label="Next page">
              →
            </button>
            <Link className="btn btn-sm btn-ghost" to={`/documents/${source.id}/view?page=${page}`}>
              Full viewer
            </Link>
          </div>
          <img className="nb-preview-img" src={urls.pageImage(source.id, page)} alt={`Page ${page} of ${source.title || source.filename}`} />
          <h4 className="nb-preview-sub">Passages on this page</h4>
          {chunks.loading && !chunks.data ? (
            <Spinner inline label="Loading passages…" />
          ) : chunks.error ? (
            <ErrorMessage error={chunks.error} />
          ) : !chunks.data?.length ? (
            <p className="small muted">No text was found on this page.</p>
          ) : (
            <ol className="nb-passages">
              {chunks.data.map((c) => (
                <li key={c.id}>
                  <div className="chip-row small">
                    <Badge>{c.element_type}</Badge>
                    <ExtractionBadge method={c.extraction_method} />
                    {c.section_path && <span className="muted">{c.section_path}</span>}
                  </div>
                  <p className="pre-wrap small">{c.text}</p>
                </li>
              ))}
            </ol>
          )}
        </>
      ) : (
        <p className="small muted">{isProcessing(source) ? 'This source is still being processed.' : source.error ?? 'This source could not be read.'}</p>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ panel

interface Props {
  sources: NotebookSource[] | undefined;
  loading: boolean;
  error: unknown;
  selected: Set<string>;
  onToggle: (id: string, on: boolean) => void;
  onSelectAll: (on: boolean) => void;
  onAdd: () => void;
  onRetry: (id: string) => void;
  onDelete: (s: NotebookSource) => void;
}

export function SourcesPanel({ sources, loading, error, selected, onToggle, onSelectAll, onAdd, onRetry, onDelete }: Props) {
  const [preview, setPreview] = useState<string | null>(null);
  const list = sources ?? [];
  const ready = list.filter(isReady);
  const allOn = ready.length > 0 && ready.every((s) => selected.has(s.id));
  const someOn = ready.some((s) => selected.has(s.id));
  const selectAllRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (selectAllRef.current) selectAllRef.current.indeterminate = someOn && !allOn;
  }, [someOn, allOn]);

  const previewed = preview ? list.find((s) => s.id === preview) : undefined;
  if (previewed) return <SourcePreview source={previewed} onBack={() => setPreview(null)} />;

  return (
    <div className="nb-sources">
      <button type="button" className="btn nb-add-btn" onClick={onAdd}>
        + Add sources
      </button>
      <ErrorMessage error={error} title="Could not load sources" />
      {loading && !sources ? (
        <Spinner label="Loading sources…" />
      ) : list.length === 0 ? (
        <div className="nb-sources-empty">
          <div className="nb-empty-title">Saved sources will appear here</div>
          <p className="small muted">Click “Add sources” to upload PDFs, paste links or search the web. Answers in the chat cite the exact pages they come from.</p>
        </div>
      ) : (
        <>
          <label className="nb-source-row nb-select-all">
            <span className="nb-source-label">Select all sources</span>
            <input ref={selectAllRef} type="checkbox" checked={allOn} disabled={!ready.length} onChange={(e) => onSelectAll(e.target.checked)} />
          </label>
          <ul className="nb-source-list" aria-label="Sources">
            {list.map((s) => (
              <li key={s.id} className={`nb-source-row ${selected.has(s.id) ? 'nb-source-on' : ''}`} data-source={s.id}>
                {s.source_url ? <GlobeIcon /> : <FileIcon />}
                <div className="nb-source-main">
                  <button type="button" className="nb-source-name" onClick={() => setPreview(s.id)} title={s.source_url ?? s.filename}>
                    {s.title || s.filename}
                  </button>
                  {s.source_url && <span className="nb-source-domain small muted">{domainOf(s.source_url)}</span>}
                  <StatusPill s={s} onRetry={() => onRetry(s.id)} />
                </div>
                <button type="button" className="btn btn-ghost btn-icon btn-sm nb-source-del" aria-label={`Remove ${s.title || s.filename}`} onClick={() => onDelete(s)}>
                  ✕
                </button>
                <input
                  type="checkbox"
                  aria-label={`Use ${s.title || s.filename}`}
                  checked={selected.has(s.id)}
                  disabled={!isReady(s)}
                  onChange={(e) => onToggle(s.id, e.target.checked)}
                />
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
