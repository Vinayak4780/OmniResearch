import type { ComponentPropsWithoutRef, MouseEvent, ReactNode } from 'react';
import Markdown, { type Components } from 'react-markdown';
import { Link } from 'react-router-dom';
import remarkGfm from 'remark-gfm';
import { CITATION_LABEL_RE, isWebUrl, parseDocumentLink, uniqueSlug } from '../../utils/citations';

/** App routes (e.g. citation links `/documents/<id>/view?page=2`) navigate in-app; `/api/...` does not. */
export function isInternalHref(href: string | undefined): boolean {
  if (!href) return false;
  return href.startsWith('/') && !href.startsWith('//') && !href.startsWith('/api/');
}

function textOf(children: ReactNode): string {
  if (typeof children === 'string' || typeof children === 'number') return String(children);
  if (Array.isArray(children)) return children.map(textOf).join('');
  if (children && typeof children === 'object' && 'props' in children) return textOf((children as { props: { children?: ReactNode } }).props.children);
  return '';
}

function plainClick(e: MouseEvent): boolean {
  return e.button === 0 && !e.metaKey && !e.ctrlKey && !e.shiftKey && !e.altKey;
}

export interface MarkdownReportProps {
  markdown: string;
  /**
   * Called when a local citation (`[E3]` linking to `/documents/...`) or another document-page link is
   * clicked (evidenceId is '' when the link has no [E#] label). When provided, the click does not
   * navigate; modifier-clicks still open the full viewer in a new tab.
   */
  onCitation?: (evidenceId: string, href: string) => void;
  /** Links for `#cite-E3` placeholders (follow-up chat answers). Missing ids render as "unknown". */
  citationLinks?: Record<string, string>;
  /** Give h1–h3 stable ids for a table of contents. */
  headingIds?: boolean;
  className?: string;
}

function MdTable({ children }: ComponentPropsWithoutRef<'table'>) {
  return (
    <div className="table-wrap">
      <table>{children}</table>
    </div>
  );
}

export function MarkdownReport({ markdown, onCitation, citationLinks, headingIds = false, className = '' }: MarkdownReportProps) {
  const seen = new Map<string, number>();

  const heading =
    (Tag: 'h1' | 'h2' | 'h3') =>
    ({ children }: ComponentPropsWithoutRef<'h2'>) => {
      const id = headingIds ? uniqueSlug(textOf(children), seen) : undefined;
      return <Tag id={id}>{children}</Tag>;
    };

  const MdLink = ({ href, children, title }: ComponentPropsWithoutRef<'a'>) => {
    const label = textOf(children);
    const cite = CITATION_LABEL_RE.exec(label);
    let target = href;
    if (href?.startsWith('#cite-')) {
      const id = href.slice('#cite-'.length);
      target = citationLinks?.[id];
      if (!target) {
        return (
          <span className="citation citation-unknown" title={`${id} is not a known evidence id for this investigation`}>
            {children}
          </span>
        );
      }
    }
    const cls = cite ? 'citation' : undefined;

    if (isInternalHref(target)) {
      const to = target as string;
      return (
        <Link
          to={to}
          className={cls}
          title={title ?? (cite ? `Show the cited passage for ${cite[1]}` : undefined)}
          onClick={(e) => {
            if (onCitation && plainClick(e) && (cite || parseDocumentLink(to))) {
              e.preventDefault();
              onCitation(cite ? cite[1] : '', to);
            }
          }}
        >
          {children}
        </Link>
      );
    }
    if (target?.startsWith('#')) {
      return (
        <a href={target} className={cls} title={title}>
          {children}
        </a>
      );
    }
    return (
      <a href={target} className={cls ? `${cls} citation-web` : undefined} title={title ?? (isWebUrl(target) ? target : undefined)} target="_blank" rel="noopener noreferrer">
        {children}
        <span className="sr-only"> (opens in a new tab)</span>
      </a>
    );
  };

  const components: Components = { a: MdLink, table: MdTable, h1: heading('h1'), h2: heading('h2'), h3: heading('h3') };

  return (
    <article className={`markdown-body ${className}`}>
      <Markdown remarkPlugins={[remarkGfm]} components={components}>
        {markdown}
      </Markdown>
    </article>
  );
}
