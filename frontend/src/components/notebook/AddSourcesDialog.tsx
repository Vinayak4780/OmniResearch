import { useEffect, useRef, useState, type DragEvent, type FormEvent } from 'react';
import { fmtDuration, useNow } from '../../hooks/useNow';
import { api, errorMessage } from '../../services/api';
import type { DiscoverHit, DiscoverMode, DiscoverResponse, LinkResult } from '../../types/notebooks';
import { rejectReason } from '../../utils/upload';
import { useHealth } from '../layout/HealthContext';
import { ErrorMessage, Tabs } from '../ui';
import { FileIcon, GlobeIcon, NoSearchHint, domainOf, isNoSearchError } from './webHints';

export const MAX_LINKS = 10;

type Way = 'upload' | 'links' | 'search';

/** Split pasted text into unique links (one per line; a missing https:// is added). */
export function parseLinks(text: string): string[] {
  const out: string[] = [];
  for (const raw of text.split(/[\r\n]+/)) {
    const line = raw.trim();
    if (!line) continue;
    const url = /^[a-z][a-z0-9+.-]*:\/\//i.test(line) ? line : `https://${line}`;
    if (!out.includes(url)) out.push(url);
  }
  return out;
}

// ------------------------------------------------------------------ upload PDFs

interface UploadRow {
  key: string;
  name: string;
  status: 'waiting' | 'uploading' | 'done' | 'duplicate' | 'error';
  message?: string;
}

function UploadTab({ notebookId, onUploaded, setBusy }: { notebookId: string; onUploaded: () => void; setBusy: (b: boolean) => void }) {
  const { health } = useHealth();
  const maxMb = health?.max_upload_mb;
  const [rows, setRows] = useState<UploadRow[]>([]);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const update = (key: string, patch: Partial<UploadRow>) => setRows((xs) => xs.map((x) => (x.key === key ? { ...x, ...patch } : x)));

  const handle = async (files: File[]) => {
    if (!files.length) return;
    const batch = files.map((f, i) => {
      const reason = rejectReason(f, maxMb);
      return { file: f, row: { key: `${Date.now()}-${i}`, name: f.name, status: reason ? 'error' : 'waiting', message: reason } as UploadRow };
    });
    setRows((xs) => [...xs, ...batch.map((b) => b.row)]);
    setBusy(true);
    for (const { file, row } of batch) {
      if (row.status === 'error') continue;
      update(row.key, { status: 'uploading' });
      try {
        const res = await api.uploadDocument(file, notebookId);
        update(row.key, res.duplicate ? { status: 'duplicate', message: 'Already uploaded' } : { status: 'done' });
        onUploaded();
      } catch (e) {
        update(row.key, { status: 'error', message: errorMessage(e) });
      }
    }
    setBusy(false);
  };

  return (
    <>
      <div
        className={`nb-drop ${dragging ? 'nb-drop-active' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e: DragEvent) => {
          e.preventDefault();
          setDragging(false);
          void handle(Array.from(e.dataTransfer.files));
        }}
      >
        <div className="nb-drop-icon" aria-hidden="true">
          ⇪
        </div>
        <p>
          <strong>Drop PDF files here</strong> or{' '}
          <button type="button" className="btn btn-link" onClick={() => inputRef.current?.click()}>
            choose files
          </button>
        </p>
        <p className="small muted">PDF only{maxMb ? `, up to ${maxMb} MB each` : ''}. Scanned pages are read automatically.</p>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          multiple
          className="sr-only"
          aria-label="Choose PDF files"
          tabIndex={-1}
          onChange={(e) => {
            const files = Array.from(e.target.files ?? []);
            e.target.value = '';
            void handle(files);
          }}
        />
      </div>
      {rows.length > 0 && (
        <ul className="nb-upload-list" aria-live="polite">
          {rows.map((r) => (
            <li key={r.key} className={`nb-upload nb-upload-${r.status}`}>
              <FileIcon />
              <span className="nb-upload-name">{r.name}</span>
              <span className={`small ${r.status === 'error' ? 'text-danger' : 'muted'}`}>
                {r.status === 'uploading' ? 'Uploading…' : r.status === 'waiting' ? 'Waiting' : r.status === 'done' ? 'Added' : r.message}
              </span>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

// ------------------------------------------------------------------ link results (shared by links + search)

function linkStatus(r: LinkResult): { text: string; tone: 'ok' | 'dup' | 'bad' } {
  if (!r.ok) return { text: r.error || 'Could not add this link.', tone: 'bad' };
  if (r.duplicate) return { text: 'Already in this notebook', tone: 'dup' };
  return { text: r.kind === 'web' ? 'Added (page saved as a PDF)' : 'Added', tone: 'ok' };
}

function LinkResults({ results }: { results: LinkResult[] }) {
  if (!results.length) return null;
  return (
    <ul className="nb-upload-list nb-link-results" aria-label="Link results" aria-live="polite">
      {results.map((r) => {
        const st = linkStatus(r);
        return (
          <li key={r.url} className={`nb-upload nb-link-${st.tone}`}>
            <span className="nb-link-mark" aria-hidden="true">
              {st.tone === 'ok' ? '✓' : st.tone === 'dup' ? '•' : '!'}
            </span>
            <span className="nb-upload-name" title={r.url}>
              {r.url}
            </span>
            <span className={`small nb-link-msg ${st.tone === 'bad' ? 'text-danger' : 'muted'}`}>{st.text}</span>
          </li>
        );
      })}
    </ul>
  );
}

// ------------------------------------------------------------------ paste links

function LinksTab({ notebookId, onUploaded, busy, setBusy }: { notebookId: string; onUploaded: () => void; busy: boolean; setBusy: (b: boolean) => void }) {
  const [text, setText] = useState('');
  const [results, setResults] = useState<LinkResult[]>([]);
  const [error, setError] = useState<unknown>(null);
  const links = parseLinks(text);
  const tooMany = links.length > MAX_LINKS;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!links.length || tooMany || busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.addNotebookLinks(notebookId, links);
      setResults(res.results);
      if (res.results.some((r) => r.ok)) onUploaded();
      // keep only the links that failed, so they can be fixed and tried again
      setText(res.results.filter((r) => !r.ok).map((r) => r.url).join('\n'));
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="nb-links-form" onSubmit={submit}>
      <label htmlFor={`links-${notebookId}`} className="nb-field-label">
        Links, one per line
      </label>
      <textarea
        id={`links-${notebookId}`}
        rows={5}
        placeholder={'https://example.com/report.pdf\nhttps://example.com/article'}
        value={text}
        onChange={(e) => setText(e.target.value)}
        disabled={busy}
      />
      <p className="small muted">
        Up to {MAX_LINKS} links. PDF links are downloaded; for a webpage, its main text is saved as a PDF so answers can cite it.
      </p>
      {tooMany && (
        <p className="small text-danger" role="alert">
          That's {links.length} links — please add {MAX_LINKS} or fewer at a time.
        </p>
      )}
      <ErrorMessage error={error} title="Could not add the links" />
      <div className="form-actions">
        <button type="submit" className="btn btn-primary" disabled={busy || !links.length || tooMany}>
          {busy ? 'Adding…' : links.length > 1 ? `Add ${links.length} links` : 'Add link'}
        </button>
      </div>
      <LinkResults results={results} />
    </form>
  );
}

// ------------------------------------------------------------------ search the web

const MODE_HINT: Record<DiscoverMode, string> = {
  fast: 'Fast: one search, a few seconds',
  deep: 'Deep: AI plans several searches and picks the best sources, up to a minute',
};

function SearchTab({ notebookId, onUploaded, busy, setBusy }: { notebookId: string; onUploaded: () => void; busy: boolean; setBusy: (b: boolean) => void }) {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<DiscoverMode>('fast');
  const [searching, setSearching] = useState<{ mode: DiscoverMode; startedAt: number } | null>(null);
  const [found, setFound] = useState<DiscoverResponse | null>(null);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [results, setResults] = useState<LinkResult[]>([]);
  const [error, setError] = useState<unknown>(null);
  const abortRef = useRef<AbortController | null>(null);
  const now = useNow(1000, !!searching);

  useEffect(() => () => abortRef.current?.abort(), []);

  const search = async (e: FormEvent) => {
    e.preventDefault();
    const q = query.trim();
    if (q.length < 2 || searching) return;
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setSearching({ mode, startedAt: Date.now() });
    setError(null);
    setResults([]);
    try {
      const res = await api.discoverSources(notebookId, q, mode, ctrl.signal);
      setFound(res);
      // The AI's deep picks start selected; fast results are picked by hand.
      setPicked(new Set(res.mode === 'deep' ? res.results.filter((r) => !r.already_added).slice(0, MAX_LINKS).map((r) => r.url) : []));
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      setError(err);
    } finally {
      if (abortRef.current === ctrl) setSearching(null);
    }
  };

  const toggle = (url: string, on: boolean) =>
    setPicked((p) => {
      const n = new Set(p);
      if (on) n.add(url);
      else n.delete(url);
      return n;
    });

  const add = async () => {
    if (!found || !picked.size || busy) return;
    const chosen = found.results.filter((r) => picked.has(r.url)).map((r) => r.url);
    setBusy(true);
    setError(null);
    try {
      const res = await api.addNotebookLinks(notebookId, chosen);
      setResults(res.results);
      const done = new Set(res.results.filter((r) => r.ok).map((r) => r.url));
      if (done.size) onUploaded();
      setFound((f) => (f ? { ...f, results: f.results.map((r) => (done.has(r.url) ? { ...r, already_added: true } : r)) } : f));
      setPicked((p) => new Set([...p].filter((u) => !done.has(u))));
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  const overLimit = picked.size > MAX_LINKS;
  return (
    <div className="nb-search-tab">
      <form className="nb-search-form" onSubmit={search}>
        <label htmlFor={`discover-${notebookId}`} className="nb-field-label">
          What are you looking for?
        </label>
        <div className="nb-search-row">
          <input
            id={`discover-${notebookId}`}
            type="search"
            maxLength={400}
            placeholder="e.g. KV cache compression for long context"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button type="submit" className="btn btn-primary" disabled={!!searching || query.trim().length < 2}>
            {searching ? 'Searching…' : 'Search'}
          </button>
        </div>
        <div className="nb-segmented" role="radiogroup" aria-label="Search depth">
          {(['fast', 'deep'] as const).map((m) => (
            <label key={m} className={`nb-segment ${mode === m ? 'nb-segment-on' : ''}`}>
              <input type="radio" name={`mode-${notebookId}`} value={m} checked={mode === m} onChange={() => setMode(m)} className="sr-only" />
              {m === 'fast' ? 'Fast' : 'Deep'}
            </label>
          ))}
        </div>
        <p className="small muted nb-mode-hint">{MODE_HINT[mode]}</p>
      </form>

      {searching && (
        <div className="nb-typing nb-search-busy" role="status">
          <span className="nb-dots" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
          {searching.mode === 'deep' ? 'Planning searches and picking the best sources…' : 'Searching the web…'}{' '}
          <span className="muted num">{fmtDuration(now - searching.startedAt)}</span>
        </div>
      )}
      {isNoSearchError(error) ? <NoSearchHint /> : <ErrorMessage error={error} title="Something went wrong" />}

      {found && !searching && (
        <>
          {found.warnings.length > 0 && (
            <ul className="nb-warnings" aria-label="Search notes">
              {found.warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          )}
          {found.mode === 'deep' && found.queries.length > 0 && <p className="small muted nb-queries">Searched for: {found.queries.join(' · ')}</p>}
          {found.results.length === 0 ? (
            <p className="small muted">Nothing found. Try different words.</p>
          ) : (
            <ul className="nb-hits" aria-label="Search results">
              {found.results.map((r: DiscoverHit, i) => (
                <li key={r.url} className={`nb-hit ${r.already_added ? 'nb-hit-added' : ''} ${picked.has(r.url) ? 'nb-hit-on' : ''}`}>
                  <input
                    type="checkbox"
                    id={`hit-${notebookId}-${i}`}
                    aria-label={`Select ${r.title || r.url}`}
                    checked={r.already_added || picked.has(r.url)}
                    disabled={r.already_added || busy}
                    onChange={(e) => toggle(r.url, e.target.checked)}
                  />
                  <div className="nb-hit-main">
                    <label htmlFor={`hit-${notebookId}-${i}`} className="nb-hit-title">
                      {r.title || r.url}
                    </label>
                    <div className="nb-hit-meta small">
                      <GlobeIcon className="nb-hit-globe" />
                      <span className="muted">{r.domain || domainOf(r.url)}</span>
                      {r.is_pdf && <span className="nb-pdf-badge">PDF</span>}
                      {r.published_at && <span className="muted">· {r.published_at.slice(0, 10)}</span>}
                      {r.already_added && <span className="nb-added-badge">Already added</span>}
                      <a href={r.url} target="_blank" rel="noopener noreferrer" className="nb-hit-open" aria-label={`Open ${r.title || r.url} in a new tab`}>
                        ↗
                      </a>
                    </div>
                    {r.snippet && <p className="nb-hit-snippet small">{r.snippet}</p>}
                    {r.reason && (
                      <p className="nb-hit-reason small">
                        <span className="nb-hit-reason-tag">Why</span> {r.reason}
                      </p>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}
          {overLimit && <p className="small text-danger">Please pick {MAX_LINKS} or fewer at a time.</p>}
          <div className="form-actions">
            <button type="button" className="btn btn-primary" disabled={busy || !picked.size || overLimit} onClick={() => void add()}>
              {busy ? 'Adding…' : `Add ${picked.size} selected`}
            </button>
          </div>
          <LinkResults results={results} />
        </>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ dialog

export function AddSourcesDialog({ notebookId, onClose, onUploaded }: { notebookId: string; onClose: () => void; onUploaded: () => void }) {
  const [way, setWay] = useState<Way>('upload');
  const [busy, setBusy] = useState(false);
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    dialogRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && !busy && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [busy, onClose]);

  return (
    <div className="nb-modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && !busy && onClose()}>
      <div className="nb-modal" role="dialog" aria-modal="true" aria-labelledby="add-sources-title" tabIndex={-1} ref={dialogRef}>
        <div className="nb-modal-head">
          <h2 id="add-sources-title">Add sources</h2>
          <button type="button" className="btn btn-ghost btn-icon" aria-label="Close" onClick={onClose} disabled={busy}>
            ✕
          </button>
        </div>
        <div className="nb-ways">
          <Tabs
            label="Ways to add sources"
            active={way}
            onChange={(w) => !busy && setWay(w as Way)}
            tabs={[
              { id: 'upload', label: 'Upload PDFs' },
              { id: 'links', label: 'Paste links' },
              { id: 'search', label: 'Search the web' },
            ]}
          />
        </div>
        {/* All three stay mounted so results survive switching between them. */}
        <div role="tabpanel" aria-label="Upload PDFs" className="nb-way-panel" hidden={way !== 'upload'}>
          <UploadTab notebookId={notebookId} onUploaded={onUploaded} setBusy={setBusy} />
        </div>
        <div role="tabpanel" aria-label="Paste links" className="nb-way-panel" hidden={way !== 'links'}>
          <LinksTab notebookId={notebookId} onUploaded={onUploaded} busy={busy} setBusy={setBusy} />
        </div>
        <div role="tabpanel" aria-label="Search the web" className="nb-way-panel" hidden={way !== 'search'}>
          <SearchTab notebookId={notebookId} onUploaded={onUploaded} busy={busy} setBusy={setBusy} />
        </div>
        <div className="form-actions">
          <button type="button" className="btn btn-primary" onClick={onClose} disabled={busy}>
            {busy ? 'Adding…' : 'Done'}
          </button>
        </div>
      </div>
    </div>
  );
}
