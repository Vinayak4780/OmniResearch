import type { Evidence } from '../types/api';

/** Where a local citation points: a document page, optionally tied to a known evidence record. */
export interface CitationTarget {
  documentId: string;
  page: number;
  evidenceId?: string;
}

export const CITATION_LABEL_RE = /^\[(E\d+)\]$/;
const DOC_LINK_RE = /^\/documents\/([^/?#]+)\/view(?:\?(.*))?$/;

/** Parse an in-app viewer link (`/documents/<id>/view?page=N&evidence=E3`) into document id, page and evidence id. */
export function parseDocumentLink(href: string | undefined | null): { documentId: string; page: number; evidenceId?: string } | null {
  if (!href) return null;
  const m = DOC_LINK_RE.exec(href);
  if (!m) return null;
  const params = new URLSearchParams(m[2] ?? '');
  const page = Number.parseInt(params.get('page') ?? '1', 10);
  const ev = params.get('evidence');
  return {
    documentId: decodeURIComponent(m[1]),
    page: Number.isFinite(page) && page > 0 ? page : 1,
    ...(ev && /^E\d+$/.test(ev) ? { evidenceId: ev } : {}),
  };
}

export function isWebUrl(href: string | undefined | null): href is string {
  return !!href && /^https?:\/\//i.test(href);
}

/** The link the backend would produce for an evidence record (mirrors Evidence.link() on the server). */
export function evidenceLink(e: Evidence): string {
  if (e.source_type === 'local') return `/documents/${e.document_id}/view?page=${e.page_no ?? 1}${e.id ? `&evidence=${e.id}` : ''}`;
  return e.canonical_url || e.final_url || e.url || '';
}

/** Resolve a citation (by evidence id and/or link) to a panel target for local sources. */
export function citationTarget(evidenceId: string | undefined, href: string | undefined, evidence: Map<string, Evidence>): CitationTarget | null {
  const e = evidenceId ? evidence.get(evidenceId) : undefined;
  if (e && e.source_type === 'local' && e.document_id) return { documentId: e.document_id, page: e.page_no ?? 1, evidenceId: e.id ?? evidenceId };
  const parsed = parseDocumentLink(href);
  if (!parsed) return null;
  if (!evidenceId && parsed.evidenceId) {
    const linked = evidence.get(parsed.evidenceId);
    if (linked && linked.source_type === 'local' && linked.document_id) return { documentId: linked.document_id, page: linked.page_no ?? parsed.page, evidenceId: parsed.evidenceId };
    return { documentId: parsed.documentId, page: parsed.page, evidenceId: parsed.evidenceId };
  }
  if (!evidenceId) {
    // A plain page link (e.g. the report's source list): attach the evidence only when it is unambiguous.
    const same = [...evidence.values()].filter((x) => x.source_type === 'local' && x.document_id === parsed.documentId && x.page_no === parsed.page);
    if (same.length === 1 && same[0].id) return { ...parsed, evidenceId: same[0].id };
  }
  return { ...parsed, evidenceId: evidenceId || undefined };
}

/** Plain-text citation for the clipboard. */
export function formatCitation(e: Evidence, origin = typeof window !== 'undefined' ? window.location.origin : ''): string {
  if (e.source_type === 'local') {
    const pages = e.page_end && e.page_end !== e.page_no ? `pp. ${e.page_no}–${e.page_end}` : `p. ${e.page_no ?? '?'}`;
    const section = e.section_path ? `, “${e.section_path}”` : '';
    return `${e.doc_title || e.filename || 'Document'}${section}, ${pages} [${e.id}]. ${origin}${evidenceLink(e)}`;
  }
  const date = e.published_at ? ` (${e.published_at.slice(0, 10)})` : '';
  const retrieved = e.retrieved_at ? ` Retrieved ${e.retrieved_at.slice(0, 10)}.` : '';
  return `${e.web_title || e.domain || 'Web page'}${date}. ${e.domain ?? ''}. ${evidenceLink(e)} [${e.id}].${retrieved}`.replace(/\s+\./g, '.');
}

/** Copy text to the clipboard; falls back to a hidden textarea where the async API is unavailable. */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall through */
  }
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    document.body.removeChild(ta);
    return ok;
  } catch {
    return false;
  }
}

/**
 * Turn inline `[E3]` / `[E3, E4]` markers in a chat answer into Markdown links (`[[E3]](#cite-E3)`) so the
 * renderer can show them as citation chips. Already-linked citations (`[E3](…)`) are left alone.
 */
export function linkifyCitations(markdown: string): string {
  return markdown.replace(/(?<!\[)\[(E\d+(?:\s*[,;]\s*E\d+)*)\](?![(\]])/g, (_m, ids: string) =>
    ids
      .split(/\s*[,;]\s*/)
      .map((id) => `[[${id}]](#cite-${id})`)
      .join(' '),
  );
}

/** GitHub-style heading slug used for both the TOC and the rendered heading ids. */
export function slugify(text: string): string {
  const s = text
    .toLowerCase()
    .replace(/[`*_~[\]()<>]/g, '')
    .replace(/[^\p{L}\p{N}]+/gu, '-')
    .replace(/^-+|-+$/g, '');
  return s || 'section';
}

export interface TocEntry {
  level: number;
  text: string;
  id: string;
}

/** Headings (levels 2–3) outside fenced code, with ids deduplicated the same way the renderer does. */
export function extractToc(markdown: string): TocEntry[] {
  const out: TocEntry[] = [];
  const seen = new Map<string, number>();
  let fenced = false;
  for (const line of markdown.split('\n')) {
    if (/^\s*(```|~~~)/.test(line)) fenced = !fenced;
    if (fenced) continue;
    const m = /^(#{1,6})\s+(.+?)\s*#*\s*$/.exec(line);
    if (!m) continue;
    const level = m[1].length;
    const text = m[2].replace(/\[([^\]]*)\]\([^)]*\)/g, '$1').replace(/[*_`]/g, '');
    const id = uniqueSlug(text, seen);
    if (level === 2 || level === 3) out.push({ level, text, id });
  }
  return out;
}

export function uniqueSlug(text: string, seen: Map<string, number>): string {
  const base = slugify(text);
  const n = seen.get(base) ?? 0;
  seen.set(base, n + 1);
  return n === 0 ? base : `${base}-${n}`;
}
