import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { ChatPanel } from '../components/notebook/ChatPanel';
import { notebookStyle } from '../components/notebook/notebookStyle';
import { AddSourcesDialog, SourcesPanel, isProcessing, isReady } from '../components/notebook/SourcesPanel';
import { ACTIVE_REPORT } from '../components/notebook/ReportProgress';
import { StudioPanel } from '../components/notebook/StudioPanel';
import { CitationPanel } from '../components/research/CitationPanel';
import { JobView } from '../components/research/JobView';
import { ErrorMessage, Spinner, Tabs } from '../components/ui';
import { useMediaQuery } from '../hooks/useMediaQuery';
import { usePolling } from '../hooks/usePolling';
import { useResource } from '../hooks/useResource';
import { api } from '../services/api';
import type { NotebookCitation, NotebookSource } from '../types/notebooks';
import type { CitationTarget } from '../utils/citations';

export const NARROW_QUERY = '(max-width: 999px)';
type Pane = 'sources' | 'chat' | 'studio';

function EditableTitle({ title, onSave }: { title: string; onSave: (t: string) => Promise<void> }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(title);
  const saving = useRef(false);
  useEffect(() => setValue(title), [title]);
  const save = async (e?: FormEvent) => {
    e?.preventDefault();
    if (saving.current) return;
    saving.current = true;
    setEditing(false);
    try {
      if (value.trim() && value.trim() !== title) await onSave(value.trim());
      else setValue(title);
    } finally {
      saving.current = false;
    }
  };
  if (editing) {
    return (
      <form onSubmit={save} className="nb-title-form">
        <label htmlFor="nb-title-input" className="sr-only">
          Notebook title
        </label>
        <input
          id="nb-title-input"
          autoFocus
          maxLength={120}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onBlur={() => void save()}
          onKeyDown={(e) => {
            if (e.key === 'Escape') {
              setValue(title);
              setEditing(false);
            }
          }}
        />
      </form>
    );
  }
  return (
    <h1 className="nb-title">
      <button type="button" className="nb-title-btn" onClick={() => setEditing(true)} title="Rename notebook">
        {title}
      </button>
    </h1>
  );
}

function PanelHeader({ title, collapsed, onToggle, side }: { title: string; collapsed?: boolean; onToggle?: () => void; side?: 'left' | 'right' }) {
  return (
    <div className="nb-panel-head">
      {!collapsed && <h2 className="nb-panel-title">{title}</h2>}
      {onToggle && (
        <button
          type="button"
          className="btn btn-ghost btn-icon btn-sm nb-collapse"
          aria-label={`${collapsed ? 'Expand' : 'Collapse'} ${title}`}
          aria-expanded={!collapsed}
          onClick={onToggle}
        >
          {side === 'left' ? (collapsed ? '»' : '«') : collapsed ? '«' : '»'}
        </button>
      )}
    </div>
  );
}

export function NotebookPage() {
  const { id = '' } = useParams();
  const [params, setParams] = useSearchParams();
  const reportId = params.get('report');
  const notebook = useResource(() => api.notebook(id), [id]);
  const sources = useResource(() => api.notebookSources(id), [id]);
  const notes = useResource(() => api.notebookNotes(id), [id]);
  const reports = useResource(() => api.notebookReports(id), [id]);
  usePolling(() => reports.reload(true), 5000, (reports.data ?? []).some((r) => ACTIVE_REPORT.has(r.status)));
  const narrow = useMediaQuery(NARROW_QUERY);
  const [pane, setPane] = useState<Pane>('chat');
  const [collapsed, setCollapsed] = useState<{ sources: boolean; studio: boolean }>({ sources: false, studio: false });
  const [excluded, setExcluded] = useState<Set<string>>(new Set());
  const [adding, setAdding] = useState(false);
  const [drawer, setDrawer] = useState<{ target: CitationTarget; evidence: NotebookCitation } | null>(null);
  const [actionError, setActionError] = useState<unknown>(null);

  const list = sources.data ?? notebook.data?.sources;
  usePolling(() => sources.reload(true), 2000, (list ?? []).some(isProcessing));

  // Default selection = every ready source; the user's unchecks are remembered.
  const selectedIds = useMemo(() => (list ?? []).filter((s) => isReady(s) && !excluded.has(s.id)).map((s) => s.id), [list, excluded]);
  const selected = useMemo(() => new Set(selectedIds), [selectedIds]);
  const readyCount = (list ?? []).filter(isReady).length;

  const toggle = (sid: string, on: boolean) =>
    setExcluded((ex) => {
      const n = new Set(ex);
      if (on) n.delete(sid);
      else n.add(sid);
      return n;
    });
  const selectAll = (on: boolean) => setExcluded(on ? new Set() : new Set((list ?? []).map((s) => s.id)));

  const openCitation = useCallback((evId: string, c: NotebookCitation) => {
    if (c.source_type === 'web') {
      const url = c.canonical_url || c.final_url || c.url || c.link;
      if (url) window.open(url, '_blank', 'noopener,noreferrer');
      return;
    }
    if (!c.document_id) return;
    setDrawer({ target: { documentId: c.document_id, page: c.page_no ?? 1, evidenceId: evId }, evidence: c });
  }, []);
  const closeDrawer = useCallback(() => setDrawer(null), []);

  const rename = async (title: string) => {
    try {
      notebook.setData(await api.renameNotebook(id, title));
    } catch (e) {
      setActionError(e);
    }
  };
  const retry = async (sid: string) => {
    setActionError(null);
    try {
      await api.retryDocument(sid);
    } catch (e) {
      setActionError(e);
    }
    void sources.reload(true);
  };
  const removeSource = async (s: NotebookSource) => {
    if (!window.confirm(`Remove “${s.title || s.filename}” from this notebook? Its pages and passages are deleted.`)) return;
    setActionError(null);
    try {
      await api.deleteDocument(s.id);
    } catch (e) {
      setActionError(e);
    }
    void sources.reload(true);
  };

  const openReport = (rid: string) => {
    const next = new URLSearchParams(params);
    next.set('report', rid);
    next.delete('tab');
    setParams(next);
  };
  const closeReport = () => {
    const next = new URLSearchParams(params);
    next.delete('report');
    next.delete('tab');
    setParams(next);
  };

  if (notebook.loading && !notebook.data) return <Spinner label="Opening notebook…" />;
  if (notebook.error && !notebook.data)
    return (
      <div className="nb-missing">
        <ErrorMessage error={notebook.error} title="Could not open this notebook" />
        <Link to="/">Back to notebooks</Link>
      </div>
    );
  const nb = notebook.data!;
  const { emoji } = notebookStyle(nb.id);

  const sourcesPanel = (
    <SourcesPanel
      sources={list}
      loading={sources.loading}
      error={sources.error}
      selected={selected}
      onToggle={toggle}
      onSelectAll={selectAll}
      onAdd={() => setAdding(true)}
      onRetry={(sid) => void retry(sid)}
      onDelete={(s) => void removeSource(s)}
    />
  );
  const chatPanel = (
    <ChatPanel
      notebookId={id}
      title={nb.title}
      readyCount={readyCount}
      selectedIds={selectedIds}
      reports={reports.data}
      onReportStarted={() => void reports.reload(true)}
      onOpenReport={openReport}
      onOpenCitation={openCitation}
      onNoteSaved={() => void notes.reload(true)}
    />
  );
  const studioPanel = (
    <StudioPanel
      notebookId={id}
      selectedIds={selectedIds}
      notes={notes.data}
      reports={reports}
      onNotesChanged={() => void notes.reload(true)}
      onOpenReport={openReport}
      onOpenCitation={openCitation}
    />
  );

  return (
    <div className="nb-page">
      <header className="nb-header">
        <Link to="/" className="nb-home-link" aria-label="All notebooks">
          <span aria-hidden="true">{emoji}</span>
        </Link>
        <EditableTitle title={nb.title} onSave={rename} />
      </header>
      <ErrorMessage error={actionError} />

      {reportId ? (
        <div className="nb-report-view">
          <button type="button" className="btn btn-ghost nb-back" onClick={closeReport}>
            ← Back to notebook
          </button>
          <JobView key={reportId} jobId={reportId} onChanged={() => undefined} />
        </div>
      ) : narrow ? (
        <div className="nb-tabs-layout">
          <Tabs
            label="Notebook sections"
            active={pane}
            onChange={(t) => setPane(t as Pane)}
            tabs={[
              { id: 'sources', label: `Sources${list ? ` (${list.length})` : ''}` },
              { id: 'chat', label: 'Chat' },
              { id: 'studio', label: 'Studio' },
            ]}
          />
          <div role="tabpanel" aria-label={pane} className={`nb-pane nb-pane-${pane}`}>
            {pane === 'sources' ? sourcesPanel : pane === 'chat' ? chatPanel : studioPanel}
          </div>
        </div>
      ) : (
        <div className={`nb-columns ${collapsed.sources ? 'nb-sources-collapsed' : ''} ${collapsed.studio ? 'nb-studio-collapsed' : ''}`}>
          <section className="nb-panel nb-panel-sources" aria-label="Sources">
            <PanelHeader title="Sources" side="left" collapsed={collapsed.sources} onToggle={() => setCollapsed((c) => ({ ...c, sources: !c.sources }))} />
            {!collapsed.sources && sourcesPanel}
          </section>
          <section className="nb-panel nb-panel-chat" aria-label="Chat">
            <PanelHeader title="Chat" />
            {chatPanel}
          </section>
          <section className="nb-panel nb-panel-studio" aria-label="Studio">
            <PanelHeader title="Studio" side="right" collapsed={collapsed.studio} onToggle={() => setCollapsed((c) => ({ ...c, studio: !c.studio }))} />
            {!collapsed.studio && studioPanel}
          </section>
        </div>
      )}

      {drawer && (
        <aside className="nb-drawer" aria-label="Cited source">
          <CitationPanel target={drawer.target} evidence={drawer.evidence} onClose={closeDrawer} />
        </aside>
      )}
      {adding && <AddSourcesDialog notebookId={id} onClose={() => setAdding(false)} onUploaded={() => void sources.reload(true)} />}
    </div>
  );
}
