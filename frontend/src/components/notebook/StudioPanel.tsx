import { useState, type FormEvent } from 'react';
import { usePolling } from '../../hooks/usePolling';
import { useResource, type Resource } from '../../hooks/useResource';
import { api } from '../../services/api';
import type { NotebookCitation, NotebookNote, NotebookOutput, NotebookReport, OutputKind } from '../../types/notebooks';
import { fmtRelative } from '../../utils/format';
import { ErrorMessage, Spinner, StatusBadge } from '../ui';
import { NotebookAnswer } from './NotebookAnswer';
import { ACTIVE_REPORT as ACTIVE, RunningProgress } from './ReportProgress';
import { OutputButtons, OutputList, OutputReader } from './StudioOutputs';

/** How often the Outputs list is refreshed while something is being written. */
export const OUTPUT_POLL_MS = 3000;

function NoteCard({ note, onOpenCitation, onDelete }: { note: NotebookNote; onOpenCitation: (id: string, c: NotebookCitation) => void; onDelete: () => void }) {
  const [open, setOpen] = useState(false);
  const n = Object.keys(note.citations ?? {}).length;
  return (
    <li className={`nb-note ${open ? 'nb-note-open' : ''}`} data-note={note.id}>
      <button type="button" className="nb-note-head" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span className="nb-note-title">{note.title}</span>
        {!open && <span className="nb-note-preview small muted">{note.content.replace(/\[E\d+\]/g, '').slice(0, 140)}</span>}
        <span className="small muted">
          {n ? `${n} citation${n === 1 ? '' : 's'} · ` : ''}
          {fmtRelative(note.created_at)}
        </span>
      </button>
      {open && (
        <div className="nb-note-body">
          <NotebookAnswer content={note.content} citations={note.citations ?? {}} onOpen={onOpenCitation} />
          <button type="button" className="btn btn-sm btn-danger-ghost" onClick={onDelete} aria-label={`Delete note ${note.title}`}>
            Delete note
          </button>
        </div>
      )}
    </li>
  );
}

function Notes({ notebookId, notes, onChanged, onOpenCitation }: { notebookId: string; notes: NotebookNote[] | undefined; onChanged: () => void; onOpenCitation: (id: string, c: NotebookCitation) => void }) {
  const [adding, setAdding] = useState(false);
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [error, setError] = useState<unknown>(null);

  const add = async (e: FormEvent) => {
    e.preventDefault();
    if (!content.trim()) return;
    setError(null);
    try {
      await api.addNotebookNote(notebookId, { content: content.trim(), ...(title.trim() ? { title: title.trim() } : {}) });
      setTitle('');
      setContent('');
      setAdding(false);
      onChanged();
    } catch (err) {
      setError(err);
    }
  };
  const remove = async (n: NotebookNote) => {
    if (!window.confirm(`Delete the note “${n.title}”?`)) return;
    try {
      await api.deleteNotebookNote(notebookId, n.id);
      onChanged();
    } catch (err) {
      setError(err);
    }
  };

  return (
    <section className="nb-notes" aria-labelledby="notes-title">
      <div className="nb-section-head">
        <h3 id="notes-title">Notes</h3>
        <button type="button" className="btn btn-sm" onClick={() => setAdding(!adding)} aria-expanded={adding}>
          + Add note
        </button>
      </div>
      {adding && (
        <form className="nb-note-form" onSubmit={add}>
          <label htmlFor={`note-title-${notebookId}`} className="sr-only">
            Note title
          </label>
          <input id={`note-title-${notebookId}`} placeholder="Title (optional)" maxLength={120} value={title} onChange={(e) => setTitle(e.target.value)} />
          <label htmlFor={`note-body-${notebookId}`} className="sr-only">
            Note text
          </label>
          <textarea id={`note-body-${notebookId}`} rows={4} placeholder="Write a note…" value={content} onChange={(e) => setContent(e.target.value)} />
          <div className="form-actions">
            <button type="submit" className="btn btn-primary btn-sm" disabled={!content.trim()}>
              Save note
            </button>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setAdding(false)}>
              Cancel
            </button>
          </div>
        </form>
      )}
      <ErrorMessage error={error} />
      {!notes ? (
        <Spinner inline label="Loading notes…" />
      ) : notes.length === 0 ? (
        <p className="small muted">Save a chat answer or write your own note — it appears here.</p>
      ) : (
        <ul className="nb-note-list">
          {notes.map((n) => (
            <NoteCard key={n.id} note={n} onOpenCitation={onOpenCitation} onDelete={() => void remove(n)} />
          ))}
        </ul>
      )}
    </section>
  );
}

interface Props {
  notebookId: string;
  selectedIds: string[];
  notes: NotebookNote[] | undefined;
  onNotesChanged: () => void;
  /** Deep-research reports (loaded and polled by the page; started from the chat composer). */
  reports: Resource<NotebookReport[]>;
  onOpenReport: (id: string) => void;
  onOpenCitation: (id: string, c: NotebookCitation) => void;
}

export function StudioPanel({ notebookId, selectedIds, notes, reports, onNotesChanged, onOpenReport, onOpenCitation }: Props) {

  const outputs = useResource(() => api.notebookOutputs(notebookId), [notebookId]);
  const outputsRunning = (outputs.data ?? []).some((o) => o.status === 'running');
  usePolling(() => outputs.reload(true), OUTPUT_POLL_MS, outputsRunning);
  const [reading, setReading] = useState<string | null>(null);
  const [outputError, setOutputError] = useState<unknown>(null);
  const readingOutput = reading ? outputs.data?.find((o) => o.id === reading) : undefined;

  const createOutput = async (kind: OutputKind, sourceIds: string[]) => {
    const created = await api.createNotebookOutput(notebookId, kind, sourceIds);
    outputs.setData((prev) => [created, ...(prev ?? []).filter((o) => o.id !== created.id)]);
  };
  const retryOutput = async (o: NotebookOutput) => {
    setOutputError(null);
    try {
      await createOutput(o.kind, o.source_ids?.length ? o.source_ids : selectedIds);
    } catch (e) {
      setOutputError(e);
    }
  };
  const deleteOutput = async (o: NotebookOutput) => {
    if (!window.confirm(`Delete “${o.title}”?`)) return;
    setOutputError(null);
    try {
      await api.deleteNotebookOutput(notebookId, o.id);
      outputs.setData((prev) => (prev ?? []).filter((x) => x.id !== o.id));
      if (reading === o.id) setReading(null);
    } catch (e) {
      setOutputError(e);
    }
  };

  if (readingOutput) {
    return (
      <div className="nb-studio">
        <ErrorMessage error={outputError} />
        <OutputReader
          notebookId={notebookId}
          output={readingOutput}
          onBack={() => setReading(null)}
          onDelete={(o) => void deleteOutput(o)}
          onNoteSaved={onNotesChanged}
          onOpenCitation={onOpenCitation}
        />
      </div>
    );
  }

  return (
    <div className="nb-studio">
      <OutputButtons selectedIds={selectedIds} onCreate={(kind) => createOutput(kind, selectedIds)} />
      <OutputList
        outputs={outputs.data}
        error={outputError || outputs.error}
        onOpen={(o) => setReading(o.id)}
        onRetry={(o) => void retryOutput(o)}
        onDelete={(o) => void deleteOutput(o)}
      />
      <section className="nb-reports" aria-labelledby="reports-title">
        <div className="nb-section-head">
          <h3 id="reports-title">Reports</h3>
        </div>
        <ErrorMessage error={reports.error} />
        {!reports.data ? (
          <Spinner inline label="Loading reports…" />
        ) : reports.data.length === 0 ? (
          <p className="small muted">Pick “Deep research” in the chat box to write a cited report — it appears here.</p>
        ) : (
          <ul className="nb-report-list">
            {reports.data.map((r) => (
              <li key={r.id} className="nb-report" data-report={r.id}>
                <button type="button" className="nb-report-btn" onClick={() => onOpenReport(r.id)}>
                  <span className="nb-report-q">{r.question}</span>
                  <span className="nb-report-meta small">
                    <StatusBadge status={r.status} /> <span className="muted">{fmtRelative(r.created_at)}</span>
                  </span>
                </button>
                {ACTIVE.has(r.status) && <RunningProgress report={r} />}
              </li>
            ))}
          </ul>
        )}
      </section>
      <Notes notebookId={notebookId} notes={notes} onChanged={onNotesChanged} onOpenCitation={onOpenCitation} />
    </div>
  );
}
