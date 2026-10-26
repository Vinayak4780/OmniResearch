import { useState } from 'react';
import { Link } from 'react-router-dom';
import type { Evidence } from '../../types/api';
import { DASH, fmtDate, isNum } from '../../utils/format';
import { Badge, Chip, EmptyState, Tabs } from '../ui';
import { CopyCitationButton } from './CopyCitationButton';

export function splitEvidence(items: Evidence[]): { local: Evidence[]; web: Evidence[] } {
  return {
    local: items.filter((e) => e.source_type === 'local'),
    web: items.filter((e) => e.source_type === 'web'),
  };
}

/** Flatten `retrieval` ({query, scores: {bm25, dense, rrf, rerank}, ranks: {...}, question_rerank_score}) for display. */
export function scoreEntries(retrieval: Record<string, unknown> | undefined): [string, number][] {
  const out: [string, number][] = [];
  for (const [k, v] of Object.entries(retrieval ?? {})) {
    if (isNum(v)) out.push([k.replace(/_/g, ' '), v]);
    else if (v && typeof v === 'object' && !Array.isArray(v)) {
      for (const [k2, v2] of Object.entries(v as Record<string, unknown>)) {
        if (isNum(v2)) out.push([k === 'ranks' ? `${k2} rank` : k === 'scores' ? `${k2} score` : `${k} ${k2}`, v2]);
      }
    }
  }
  return out;
}

function Scores({ e }: { e: Evidence }) {
  const entries = scoreEntries(e.retrieval);
  if (!entries.length && !isNum(e.relevance)) return null;
  return (
    <div className="scores small" aria-label="Retrieval scores">
      {isNum(e.relevance) && (
        <span className="score">
          relevance <span className="num">{e.relevance.toFixed(3)}</span>
        </span>
      )}
      {entries.map(([k, v]) => (
        <span className="score" key={k}>
          {k} <span className="num">{Number.isInteger(v) ? v : v.toFixed(3)}</span>
        </span>
      ))}
    </div>
  );
}

function Excerpt({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const long = text.length > 420;
  return (
    <div className="excerpt">
      <blockquote className="pre-wrap">{long && !open ? `${text.slice(0, 420)}…` : text}</blockquote>
      {long && (
        <button type="button" className="btn btn-link btn-sm" aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? 'Show less' : 'Show full excerpt'}
        </button>
      )}
    </div>
  );
}

function VisualObservation({ e }: { e: Evidence }) {
  if (!e.visual_observation) return null;
  return (
    <div className="visual-observation" role="note">
      <div className="visual-observation-label">
        <Badge tone="warning">Model interpretation</Badge> Visual inspection by <code>{e.visual_model ?? 'vision model'}</code> — not text from the source
      </div>
      <p className="pre-wrap">{e.visual_observation}</p>
    </div>
  );
}

function LocalCard({ e, onOpen }: { e: Evidence; onOpen?: (id: string) => void }) {
  const pages = e.page_end && e.page_end !== e.page_no ? `pp. ${e.page_no}–${e.page_end}` : `p. ${e.page_no ?? DASH}`;
  return (
    <li className="evidence-card" id={e.id ? `evidence-${e.id}` : undefined}>
      <div className="evidence-head">
        <span className="evidence-id">{e.id ?? '—'}</span>
        <div className="evidence-title">
          {e.document_id && e.page_no ? (
            <Link to={`/documents/${e.document_id}/view?page=${e.page_no}`}>{e.doc_title || e.filename || e.document_id}</Link>
          ) : (
            <span>{e.doc_title || e.filename || 'Unknown document'}</span>
          )}
          <span className="muted"> · {pages}</span>
        </div>
      </div>
      {e.filename && e.doc_title && e.doc_title !== e.filename && <div className="small muted mono">{e.filename}</div>}
      <div className="chip-row small">
        <span>
          <span className="muted">Section:</span> {e.section_path || DASH}
        </span>
        <Chip>{e.element_type ?? 'unknown element'}</Chip>
        <Chip tone={e.extraction_method?.startsWith('ocr') ? 'warning' : 'neutral'}>{e.extraction_method ?? 'unknown extraction'}</Chip>
        {e.inspected_visually && <Badge tone="warning">inspected visually</Badge>}
      </div>
      <Scores e={e} />
      <FoundBy e={e} />
      <Excerpt text={e.text} />
      <VisualObservation e={e} />
      <div className="btn-row evidence-actions">
        {onOpen && e.id && e.document_id && (
          <button type="button" className="btn btn-sm" onClick={() => onOpen(e.id!)}>
            Show on page
          </button>
        )}
        <CopyCitationButton evidence={e} className="btn btn-sm btn-ghost" />
      </div>
    </li>
  );
}

function FoundBy({ e }: { e: Evidence }) {
  const q = e.subquery || (typeof e.retrieval?.query === 'string' ? e.retrieval.query : null);
  if (!q) return null;
  return (
    <div className="small muted">
      Found by query: <q>{q}</q>
    </div>
  );
}

function WebCard({ e }: { e: Evidence }) {
  const href = e.canonical_url || e.final_url || e.url || undefined;
  return (
    <li className="evidence-card" id={e.id ? `evidence-${e.id}` : undefined}>
      <div className="evidence-head">
        <span className="evidence-id">{e.id ?? '—'}</span>
        <div className="evidence-title">
          {href ? (
            <a href={href} target="_blank" rel="noopener noreferrer">
              {e.web_title || e.domain || href}
              <span className="sr-only"> (opens in a new tab)</span>
            </a>
          ) : (
            <span>{e.web_title || e.domain}</span>
          )}
        </div>
      </div>
      <div className="chip-row small">
        <Chip>{e.domain ?? 'unknown domain'}</Chip>
        <span>
          <span className="muted">Published:</span> {e.published_at ? fmtDate(e.published_at) : 'unknown'}
        </span>
        <span>
          <span className="muted">Retrieved:</span> {fmtDate(e.retrieved_at)}
        </span>
        {e.search_engine && (
          <span>
            <span className="muted">via</span> {e.search_engine}
          </span>
        )}
        {e.snippet_only && (
          <Badge tone="warning" title="The page could not be fetched; this text is the search engine's snippet">
            Snippet only — page not fetched
          </Badge>
        )}
      </div>
      <Scores e={e} />
      <FoundBy e={e} />
      <Excerpt text={e.text} />
      <VisualObservation e={e} />
      <div className="btn-row evidence-actions">
        <CopyCitationButton evidence={e} className="btn btn-sm btn-ghost" />
      </div>
    </li>
  );
}

export function EvidencePanel({ evidence, onOpen }: { evidence: Evidence[]; /** Open a local evidence item in the side panel. */ onOpen?: (evidenceId: string) => void }) {
  const { local, web } = splitEvidence(evidence);
  const [tab, setTab] = useState<'local' | 'web'>(local.length || !web.length ? 'local' : 'web');
  const items = tab === 'local' ? local : web;
  return (
    <div className="evidence-panel">
      <Tabs
        label="Evidence sources"
        active={tab}
        onChange={(t) => setTab(t as 'local' | 'web')}
        tabs={[
          { id: 'local', label: `Uploaded documents (${local.length})` },
          { id: 'web', label: `Web sources (${web.length})` },
        ]}
      />
      <div role="tabpanel" aria-label={tab === 'local' ? 'Uploaded documents' : 'Web sources'}>
        {items.length === 0 ? (
          <EmptyState title={tab === 'local' ? 'No evidence from uploaded documents' : 'No web evidence'}>
            {tab === 'web' ? 'Web research was not used or found nothing relevant.' : 'Nothing relevant was selected from your library.'}
          </EmptyState>
        ) : (
          <ul className="evidence-list">{items.map((e) => (tab === 'local' ? <LocalCard key={e.key} e={e} onOpen={onOpen} /> : <WebCard key={e.key} e={e} />))}</ul>
        )}
      </div>
    </div>
  );
}
