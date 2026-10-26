import { useEffect, useMemo, useState } from 'react';
import type { ResearchReport, VerificationSummary } from '../../types/api';
import { extractToc, type TocEntry } from '../../utils/citations';
import { fmtInt, isNum } from '../../utils/format';
import { MarkdownReport } from './MarkdownReport';

/** Verification summary as coloured counts plus a proportional bar. */
export function VerificationCounts({ summary }: { summary: VerificationSummary | undefined }) {
  if (!summary || !isNum(summary.total)) return <p className="muted small">No verification summary was recorded.</p>;
  const parts = [
    { key: 'verified', label: 'verified', n: summary.verified ?? 0, cls: 'st-verified', glyph: '✓' },
    { key: 'partial', label: 'partially verified', n: summary.partially_verified ?? 0, cls: 'st-partial', glyph: '◐' },
    { key: 'unverified', label: 'unverified / uncited', n: summary.unverified ?? 0, cls: 'st-unverified', glyph: '!' },
    { key: 'interp', label: 'interpretation', n: summary.interpretation ?? 0, cls: 'st-interp', glyph: '~' },
  ];
  const total = summary.total || 0;
  const evChecks = summary.evidence_checks ? (Object.values(summary.evidence_checks) as { source_exists?: boolean; excerpt_match?: boolean }[]) : [];
  const evOk = evChecks.filter((c) => c && c.source_exists && c.excerpt_match).length;
  const webChecks = Array.isArray(summary.web_citation_checks) ? (summary.web_citation_checks as { ok?: boolean }[]) : [];
  return (
    <div className="verification">
      <div className="verification-counts">
        <span className="verification-total">
          <strong className="num">{fmtInt(total)}</strong> claims checked
        </span>
        {parts.map((p) => (
          <span key={p.key} className={`status-pill ${p.cls}`}>
            <span aria-hidden="true">{p.glyph}</span> <strong className="num">{fmtInt(p.n)}</strong> {p.label}
          </span>
        ))}
        {isNum(summary.repaired_by_rewrite) && summary.repaired_by_rewrite > 0 && (
          <span className="small muted">{summary.repaired_by_rewrite} repaired by rewriting</span>
        )}
      </div>
      {total > 0 && (
        <div className="stacked-bar" role="img" aria-label={parts.map((p) => `${p.label}: ${p.n}`).join(', ')}>
          {parts.map((p) => (p.n > 0 ? <span key={p.key} className={`seg ${p.cls}`} style={{ flexGrow: p.n }} title={`${p.label}: ${p.n}`} /> : null))}
        </div>
      )}
      <p className="small muted">
        {evChecks.length > 0 && `Evidence excerpts matched their source: ${evOk}/${evChecks.length}. `}
        {webChecks.length > 0 && `Web excerpts found in fetched pages: ${webChecks.filter((w) => w.ok).length}/${webChecks.length}. `}
        Verification combines deterministic checks (source exists, excerpt matches, numbers present) with a model-based support judgment, which is not proof.
      </p>
    </div>
  );
}

function useActiveHeading(ids: string[]): string | null {
  const [active, setActive] = useState<string | null>(null);
  useEffect(() => {
    if (typeof IntersectionObserver === 'undefined' || !ids.length) return;
    const els = ids.map((id) => document.getElementById(id)).filter((e): e is HTMLElement => !!e);
    const visible = new Map<string, number>();
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) visible.set(e.target.id, e.boundingClientRect.top);
          else visible.delete(e.target.id);
        }
        const first = ids.find((id) => visible.has(id));
        if (first) setActive(first);
      },
      { rootMargin: '-64px 0px -65% 0px' },
    );
    els.forEach((e) => io.observe(e));
    return () => io.disconnect();
  }, [ids]);
  return active;
}

export function TableOfContents({ toc, active, label = 'Report sections' }: { toc: TocEntry[]; active: string | null; label?: string }) {
  if (toc.length < 2) return null;
  return (
    <nav className="toc" aria-label={label}>
      <div className="toc-title">Contents</div>
      <ol>
        {toc.map((t) => (
          <li key={t.id} className={`toc-l${t.level} ${active === t.id ? 'toc-active' : ''}`}>
            <a
              href={`#${t.id}`}
              aria-current={active === t.id ? 'location' : undefined}
              onClick={(e) => {
                const el = document.getElementById(t.id);
                if (!el) return;
                e.preventDefault();
                el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                el.focus?.({ preventScroll: true });
              }}
            >
              {t.text}
            </a>
          </li>
        ))}
      </ol>
    </nav>
  );
}

interface Props {
  markdown: string;
  report: ResearchReport | null;
  onCitation: (evidenceId: string, href: string) => void;
}

export function ReportReader({ markdown, report, onCitation }: Props) {
  const toc = useMemo(() => extractToc(markdown), [markdown]);
  const ids = useMemo(() => toc.map((t) => t.id), [toc]);
  const active = useActiveHeading(ids);
  return (
    <div className="report-reader">
      <div className="report-verification">
        <VerificationCounts summary={report?.verification?.summary} />
      </div>
      <div className={`report-layout ${toc.length >= 2 ? 'has-toc' : ''}`}>
        {toc.length >= 2 && (
          <>
            <div className="report-toc-col">
              <TableOfContents toc={toc} active={active} />
            </div>
            <details className="toc-mobile">
              <summary>Contents ({toc.length} sections)</summary>
              <TableOfContents toc={toc} active={active} label="Report sections (compact)" />
            </details>
          </>
        )}
        <MarkdownReport markdown={markdown} onCitation={onCitation} headingIds className="report-article" />
      </div>
    </div>
  );
}
