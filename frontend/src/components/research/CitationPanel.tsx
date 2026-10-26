import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { useResource } from '../../hooks/useResource';
import { api, urls } from '../../services/api';
import type { Evidence, PageInfo } from '../../types/api';
import type { CitationTarget } from '../../utils/citations';
import { DASH } from '../../utils/format';
import { ExtractionBadge } from '../ExtractionBadge';
import { Badge, Chip, ErrorMessage, Spinner } from '../ui';
import { CopyCitationButton } from './CopyCitationButton';

/**
 * Highlight rectangle for a bbox given in PDF points (top-left origin) on a page of the given size.
 * Returns percentages so it can be laid over the page image at any rendered size; null when unusable.
 */
export function bboxToPercent(bbox: number[] | null | undefined, page: Pick<PageInfo, 'width' | 'height'> | undefined) {
  if (!bbox || bbox.length !== 4 || !page?.width || !page?.height) return null;
  const [x0, y0, x1, y1] = bbox;
  if (![x0, y0, x1, y1].every(Number.isFinite) || x1 <= x0 || y1 <= y0) return null;
  const clamp = (v: number) => Math.max(0, Math.min(100, v));
  return {
    left: clamp((x0 / page.width) * 100),
    top: clamp((y0 / page.height) * 100),
    width: clamp(((x1 - x0) / page.width) * 100),
    height: clamp(((y1 - y0) / page.height) * 100),
  };
}

interface Props {
  target: CitationTarget;
  evidence?: Evidence;
  onClose: () => void;
}

export function CitationPanel({ target, evidence, onClose }: Props) {
  const doc = useResource(() => api.document(target.documentId), [target.documentId]);
  const [page, setPage] = useState(target.page);
  const [img, setImg] = useState<'loading' | 'ok' | 'error'>('loading');
  const headingRef = useRef<HTMLHeadingElement>(null);
  const highlightRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setPage(target.page);
    headingRef.current?.focus();
  }, [target.documentId, target.page, target.evidenceId]);
  useEffect(() => setImg('loading'), [target.documentId, page]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const d = doc.data;
  const pageCount = d?.page_count ?? d?.pages.length ?? null;
  const pinfo = d?.pages.find((p) => p.page_no === page);
  const onEvidencePage = !!evidence && evidence.page_no === page;
  const box = onEvidencePage ? bboxToPercent(evidence?.bbox, pinfo) : null;
  const title = evidence?.doc_title || evidence?.filename || d?.title || d?.filename || 'Document';

  return (
    <section className="citation-panel" aria-labelledby="citation-panel-title">
      <header className="citation-panel-head">
        <div className="citation-panel-title-wrap">
          {evidence?.id && <span className="evidence-id">{evidence.id}</span>}
          <h2 id="citation-panel-title" ref={headingRef} tabIndex={-1} className="citation-panel-title">
            {title}
          </h2>
        </div>
        <button type="button" className="btn btn-ghost btn-icon" aria-label="Close cited source" onClick={onClose}>
          ✕
        </button>
      </header>

      <div className="citation-panel-nav">
        <button type="button" className="btn btn-sm" onClick={() => setPage(page - 1)} disabled={page <= 1} aria-label="Previous page">
          ←
        </button>
        <span className="small">
          Page {page} / {pageCount ?? DASH}
        </span>
        <button type="button" className="btn btn-sm" onClick={() => setPage(page + 1)} disabled={!pageCount || page >= pageCount} aria-label="Next page">
          →
        </button>
        {evidence && evidence.page_no !== page && evidence.page_no && (
          <button type="button" className="btn btn-sm btn-link" onClick={() => setPage(evidence.page_no!)}>
            Back to cited page {evidence.page_no}
          </button>
        )}
        <span className="spacer" />
        <Link className="btn btn-sm" to={`/documents/${target.documentId}/view?page=${page}`}>
          Open full viewer
        </Link>
      </div>

      <ErrorMessage error={doc.error} title="Could not load the document" />
      <div className="citation-page">
        {img === 'loading' && <Spinner label={`Rendering page ${page}…`} />}
        {img === 'error' && <p className="small text-danger">The page image could not be rendered.</p>}
        <div className="citation-page-frame" hidden={img !== 'ok'}>
          <img
            key={`${target.documentId}-${page}`}
            src={urls.pageImage(target.documentId, page, 130)}
            alt={`Page ${page} of ${title}`}
            onLoad={() => {
              setImg('ok');
              requestAnimationFrame(() => highlightRef.current?.scrollIntoView?.({ block: 'center', behavior: 'smooth' }));
            }}
            onError={() => setImg('error')}
          />
          {box && (
            <div
              ref={highlightRef}
              className="bbox-highlight"
              data-testid="bbox-highlight"
              style={{ left: `${box.left}%`, top: `${box.top}%`, width: `${box.width}%`, height: `${box.height}%` }}
              title={`Cited passage ${evidence?.id ?? ''}`}
            />
          )}
        </div>
        {onEvidencePage && !box && img === 'ok' && (
          <p className="small muted">No region was recorded for this passage{evidence?.extraction_method?.startsWith('ocr') ? ' (OCR text)' : ''}; the whole page is shown.</p>
        )}
      </div>

      {evidence ? (
        <div className="citation-provenance">
          <div className="chip-row small">
            <span>{evidence.page_end && evidence.page_end !== evidence.page_no ? `pp. ${evidence.page_no}–${evidence.page_end}` : `p. ${evidence.page_no ?? DASH}`}</span>
            <Chip>{evidence.element_type ?? 'unknown element'}</Chip>
            <ExtractionBadge method={evidence.extraction_method} />
            {evidence.inspected_visually && <Badge tone="warning">inspected visually</Badge>}
          </div>
          {evidence.section_path && <div className="small muted section-path">{evidence.section_path}</div>}
          {evidence.filename && evidence.doc_title && <div className="small muted mono break">{evidence.filename}</div>}
          <blockquote className="citation-excerpt pre-wrap">{evidence.text}</blockquote>
          {evidence.visual_observation && (
            <div className="visual-observation" role="note">
              <div className="visual-observation-label">
                <Badge tone="warning">Model interpretation</Badge> by <code>{evidence.visual_model ?? 'vision model'}</code> — not text from the source
              </div>
              <p className="pre-wrap">{evidence.visual_observation}</p>
            </div>
          )}
          <div className="btn-row">
            <CopyCitationButton evidence={evidence} />
            <a className="btn btn-sm btn-ghost" href={urls.pdf(target.documentId, page)} target="_blank" rel="noopener noreferrer">
              Open PDF
            </a>
          </div>
        </div>
      ) : (
        <p className="small muted">This link is not tied to a known evidence record; showing the page only.</p>
      )}
    </section>
  );
}
