import { useId, useMemo, useState, type ComponentPropsWithoutRef, type ReactNode } from 'react';
import Markdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { NotebookCitation } from '../../types/notebooks';

const CITE_RE = /\[(E\d+(?:\s*[,;]\s*E\d+)*)\]/g;

export interface NumberedCitations {
  /** Markdown with each [E#] replaced by a chip link `[[n]](#nbcite-E#)` (unknown ids get `[?]`). */
  markdown: string;
  /** E-ids in display order: order[0] is [1]. */
  order: string[];
}

/** Number a message's citations 1..n by first appearance (ids are per message). */
export function numberCitations(content: string, citations: Record<string, NotebookCitation>): NumberedCitations {
  const order: string[] = [];
  const markdown = content.replace(CITE_RE, (_m, ids: string) =>
    ids
      .split(/\s*[,;]\s*/)
      .map((id) => {
        if (!citations[id]) return `[[?]](#nbcite-${id})`;
        if (!order.includes(id)) order.push(id);
        return `[[${order.indexOf(id) + 1}]](#nbcite-${id})`;
      })
      .join(''),
  );
  return { markdown, order };
}

/** Plain text for the clipboard: [E3] → [1], plus a numbered source list. */
export function answerToPlainText(content: string, citations: Record<string, NotebookCitation>): string {
  const { order } = numberCitations(content, citations);
  const text = content.replace(CITE_RE, (_m, ids: string) =>
    ids
      .split(/\s*[,;]\s*/)
      .map((id) => (citations[id] ? `[${order.indexOf(id) + 1}]` : ''))
      .join(''),
  );
  if (!order.length) return text;
  const refs = order.map((id, i) => {
    const c = citations[id];
    const where = c.source_type === 'local' ? `${c.doc_title || c.filename}, p. ${c.page_no}` : `${c.web_title || c.domain} (${c.url})`;
    return `[${i + 1}] ${where}`;
  });
  return `${text}\n\nSources:\n${refs.join('\n')}`;
}

export function citationSourceLabel(c: NotebookCitation): string {
  if (c.source_type === 'web') return c.web_title || c.domain || 'Web page';
  const pages = c.page_end && c.page_end !== c.page_no ? `pp. ${c.page_no}–${c.page_end}` : `p. ${c.page_no ?? '?'}`;
  return `${c.doc_title || c.filename || 'Source'} · ${pages}`;
}

/** The page address of a web citation (`link` is the external URL for web evidence). */
export function webCitationUrl(c: NotebookCitation): string | null {
  for (const u of [c.link, c.canonical_url, c.final_url, c.url]) if (u && /^https?:\/\//i.test(u)) return u;
  return null;
}

/**
 * A numbered citation chip with a hover/focus preview. Local passages call `onOpen` (the page drawer);
 * web passages are a plain link that opens the page in a new tab.
 */
export function CitationChip({ n, id, c, onOpen }: { n: string; id: string; c?: NotebookCitation; onOpen: (id: string, c: NotebookCitation) => void }) {
  const [open, setOpen] = useState(false);
  const tipId = useId();
  if (!c) {
    return (
      <span className="nb-chip nb-chip-unknown" title={`${id} does not match any passage`}>
        ?
      </span>
    );
  }
  const excerpt = c.text.length > 260 ? `${c.text.slice(0, 260)}…` : c.text;
  const web = c.source_type === 'web';
  const href = web ? webCitationUrl(c) : null;
  const common = {
    'aria-describedby': open ? tipId : undefined,
    onFocus: () => setOpen(true),
    onBlur: () => setOpen(false),
  };
  return (
    <span className="nb-chip-wrap" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      {web && href ? (
        <a className="nb-chip nb-chip-web" href={href} target="_blank" rel="noopener noreferrer" aria-label={`Citation ${n} (web): ${citationSourceLabel(c)}`} {...common}>
          {n}
        </a>
      ) : (
        <button type="button" className={`nb-chip ${web ? 'nb-chip-web' : ''}`} aria-label={`Citation ${n}${web ? ' (web)' : ''}: ${citationSourceLabel(c)}`} {...common} onClick={() => onOpen(id, c)}>
          {n}
        </button>
      )}
      {open && (
        <span className="nb-chip-tip" role="tooltip" id={tipId}>
          {web ? (
            <>
              <span className="nb-chip-tip-head">
                <span className="nb-web-tag">Web</span>
                {c.domain && <span className="nb-chip-tip-domain">{c.domain}</span>}
              </span>
              <span className="nb-chip-tip-src">{c.web_title || c.domain || 'Web page'}</span>
            </>
          ) : (
            <span className="nb-chip-tip-src">{citationSourceLabel(c)}</span>
          )}
          <span className="nb-chip-tip-text">{excerpt}</span>
          {web && c.snippet_only && <span className="nb-chip-tip-note">Search snippet only — the full page was not read.</span>}
        </span>
      )}
    </span>
  );
}

function textOf(children: ReactNode): string {
  if (typeof children === 'string' || typeof children === 'number') return String(children);
  if (Array.isArray(children)) return children.map(textOf).join('');
  return '';
}

interface Props {
  content: string;
  citations: Record<string, NotebookCitation>;
  onOpen: (id: string, c: NotebookCitation) => void;
}

/** Assistant answer / note: Markdown with numbered, hoverable citation chips. */
export function NotebookAnswer({ content, citations, onOpen }: Props) {
  const { markdown } = numberCitations(content, citations);
  // Memoised so a parent re-render does not remount the chips (which would drop an open preview).
  const components = useMemo<Components>(() => {
    const A = ({ href, children }: ComponentPropsWithoutRef<'a'>) => {
      if (href?.startsWith('#nbcite-')) {
        const id = href.slice('#nbcite-'.length);
        const n = textOf(children).replace(/^\[|\]$/g, '');
        return <CitationChip n={n} id={id} c={citations[id]} onOpen={onOpen} />;
      }
      return (
        <a href={href} target="_blank" rel="noopener noreferrer">
          {children}
        </a>
      );
    };
    return {
      a: A,
      table: ({ children }) => (
        <div className="table-wrap">
          <table>{children}</table>
        </div>
      ),
    };
  }, [citations, onOpen]);
  return (
    <div className="nb-answer markdown-body">
      <Markdown remarkPlugins={[remarkGfm]} components={components}>
        {markdown}
      </Markdown>
    </div>
  );
}
