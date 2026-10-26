import { useEffect, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { ExtractionBadge } from '../components/ExtractionBadge';
import { Badge, Chip, EmptyState, ErrorMessage, Spinner, StatusBadge } from '../components/ui';
import { useResource } from '../hooks/useResource';
import { api, urls } from '../services/api';
import { DASH, fmtInt, fmtMs } from '../utils/format';

export function DocumentViewerPage() {
  const { id = '' } = useParams();
  const [params, setParams] = useSearchParams();
  const doc = useResource(() => api.document(id), [id]);
  const pageCount = doc.data?.page_count ?? doc.data?.pages.length ?? null;
  const requested = Number.parseInt(params.get('page') ?? '1', 10);
  const page = Number.isFinite(requested) && requested >= 1 ? (pageCount ? Math.min(requested, pageCount) : requested) : 1;
  const chunks = useResource(() => api.chunks(id, page), [id, page]);
  const [imgState, setImgState] = useState<'loading' | 'ok' | 'error'>('loading');
  const [jump, setJump] = useState(String(page));
  const [zoom, setZoom] = useState<'fit' | 'full'>('fit');

  useEffect(() => {
    setImgState('loading');
    setJump(String(page));
  }, [id, page]);

  const go = (n: number) => {
    if (!pageCount || n < 1 || n > pageCount) return;
    setParams({ page: String(n) });
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
      if (e.key === 'ArrowLeft') go(page - 1);
      if (e.key === 'ArrowRight') go(page + 1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  if (doc.loading && !doc.data) return <Spinner label="Loading document…" />;
  if (doc.error && !doc.data)
    return (
      <div>
        <ErrorMessage error={doc.error} title="Could not open document" />
        <Link to="/">Back to notebooks</Link>
      </div>
    );
  const d = doc.data!;
  const pinfo = d.pages.find((p) => p.page_no === page);

  return (
    <div className="viewer">
      <div className="viewer-header">
        <div>
          <nav aria-label="Breadcrumb" className="small">
            <Link to="/">Notebooks</Link> / <Link to={`/notebooks/${d.collection}`}>Notebook</Link> / {d.filename}
          </nav>
          <h1 className="viewer-title">{d.title || d.filename}</h1>
          <div className="chip-row small">
            <StatusBadge status={d.status} />
            <Chip>{d.collection}</Chip>
            {pinfo && <ExtractionBadge method={pinfo.extraction_method} />}
            {pinfo?.ocr_status && (
              <span>
                OCR: <StatusBadge status={pinfo.ocr_status} /> {pinfo.ocr_provider && <span className="muted">{pinfo.ocr_provider}</span>}{' '}
                {pinfo.ocr_latency_ms ? <span className="muted">{fmtMs(pinfo.ocr_latency_ms)}</span> : null}
              </span>
            )}
          </div>
          {pinfo?.ocr_error && <div className="small text-danger">OCR error: {pinfo.ocr_error}</div>}
        </div>
        <div className="viewer-controls">
          <button type="button" className="btn" onClick={() => go(page - 1)} disabled={page <= 1} aria-label="Previous page">
            ← Prev
          </button>
          <form
            className="page-jump"
            onSubmit={(e) => {
              e.preventDefault();
              const n = Number.parseInt(jump, 10);
              if (Number.isFinite(n)) go(n);
            }}
          >
            <label htmlFor="page-jump" className="sr-only">
              Page number
            </label>
            <input id="page-jump" type="number" min={1} max={pageCount ?? undefined} value={jump} onChange={(e) => setJump(e.target.value)} />
            <span className="muted">/ {pageCount ?? DASH}</span>
          </form>
          <button type="button" className="btn" onClick={() => go(page + 1)} disabled={!pageCount || page >= pageCount} aria-label="Next page">
            Next →
          </button>
          <button type="button" className="btn btn-ghost" onClick={() => setZoom(zoom === 'fit' ? 'full' : 'fit')} aria-pressed={zoom === 'full'}>
            {zoom === 'fit' ? 'Actual size' : 'Fit width'}
          </button>
          <a className="btn" href={urls.pdf(d.id, page)} target="_blank" rel="noopener noreferrer">
            Open PDF at page {page}
          </a>
        </div>
      </div>

      <div className="viewer-body">
        <div className={`page-image-wrap ${zoom === 'full' ? 'page-image-full' : ''}`}>
          {imgState === 'loading' && <Spinner label={`Rendering page ${page}…`} />}
          {imgState === 'error' && (
            <EmptyState title="Page image unavailable">
              The page could not be rendered. <a href={urls.pdf(d.id, page)} target="_blank" rel="noopener noreferrer">Open the PDF</a> instead.
            </EmptyState>
          )}
          <img
            key={`${d.id}-${page}`}
            src={urls.pageImage(d.id, page, zoom === 'full' ? 160 : 110)}
            alt={`Page ${page} of ${d.filename}`}
            className="page-image"
            hidden={imgState !== 'ok'}
            onLoad={() => setImgState('ok')}
            onError={() => setImgState('error')}
          />
        </div>

        <aside className="viewer-chunks" aria-label={`Chunks on page ${page}`}>
          <h2>
            Chunks on page {page} {chunks.data && <span className="muted small">({chunks.data.length})</span>}
          </h2>
          {chunks.error ? (
            <ErrorMessage error={chunks.error} />
          ) : chunks.loading && !chunks.data ? (
            <Spinner label="Loading chunks…" />
          ) : !chunks.data?.length ? (
            <EmptyState title="No chunks on this page">
              {pinfo && pinfo.text_chars === 0 ? 'The page has no extractable text.' : 'Nothing was indexed for this page.'}
            </EmptyState>
          ) : (
            <ol className="chunk-list">
              {chunks.data.map((c) => (
                <li key={c.id} className="chunk">
                  <div className="chip-row small">
                    <span className="mono muted">#{c.chunk_index}</span>
                    <Chip>{c.element_type}</Chip>
                    <ExtractionBadge method={c.extraction_method} />
                    {c.page_end !== c.page_no && <Badge>pp. {c.page_no}–{c.page_end}</Badge>}
                  </div>
                  {c.section_path && <div className="small muted section-path">{c.section_path}</div>}
                  <p className="chunk-text pre-wrap">{c.text}</p>
                  <div className="small muted">{fmtInt(c.text.length)} chars</div>
                </li>
              ))}
            </ol>
          )}
        </aside>
      </div>
    </div>
  );
}
