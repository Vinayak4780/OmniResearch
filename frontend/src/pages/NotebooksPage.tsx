import { useEffect, useRef, useState, type CSSProperties, type FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useHealth } from '../components/layout/HealthContext';
import { notebookStyle } from '../components/notebook/notebookStyle';
import { ErrorMessage, Spinner } from '../components/ui';
import { useResource } from '../hooks/useResource';
import { api } from '../services/api';
import type { NotebookSummary } from '../types/notebooks';
import { fmtRelative } from '../utils/format';

function NotebookCard({ nb, onRename, onDelete }: { nb: NotebookSummary; onRename: (title: string) => Promise<void>; onDelete: () => void }) {
  const [menu, setMenu] = useState(false);
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState(nb.title);
  const menuRef = useRef<HTMLDivElement>(null);
  const { emoji, hue } = notebookStyle(nb.id);

  useEffect(() => {
    if (!menu) return;
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent ? e.key === 'Escape' : !menuRef.current?.contains(e.target as Node)) setMenu(false);
    };
    document.addEventListener('mousedown', close);
    document.addEventListener('keydown', close);
    return () => {
      document.removeEventListener('mousedown', close);
      document.removeEventListener('keydown', close);
    };
  }, [menu]);

  const saving = useRef(false);
  const save = async (e: FormEvent) => {
    e.preventDefault();
    if (saving.current) return;
    saving.current = true;
    setEditing(false);
    try {
      if (title.trim() && title.trim() !== nb.title) await onRename(title.trim());
    } finally {
      saving.current = false;
    }
  };

  return (
    <li className="nb-card" style={{ '--nb-hue': hue } as CSSProperties} data-notebook={nb.id}>
      <div className="nb-card-top">
        <span className="nb-emoji" aria-hidden="true">
          {emoji}
        </span>
        <div className="nb-menu" ref={menuRef}>
          <button type="button" className="btn btn-ghost btn-icon nb-menu-btn" aria-haspopup="menu" aria-expanded={menu} aria-label={`More actions for ${nb.title}`} onClick={() => setMenu(!menu)}>
            ⋯
          </button>
          {menu && (
            <div className="nb-menu-list" role="menu">
              <button
                type="button"
                role="menuitem"
                onClick={() => {
                  setMenu(false);
                  setTitle(nb.title);
                  setEditing(true);
                }}
              >
                Rename
              </button>
              <button
                type="button"
                role="menuitem"
                className="text-danger"
                onClick={() => {
                  setMenu(false);
                  onDelete();
                }}
              >
                Delete
              </button>
            </div>
          )}
        </div>
      </div>
      {editing ? (
        <form onSubmit={save} className="nb-rename">
          <label htmlFor={`rename-${nb.id}`} className="sr-only">
            Notebook title
          </label>
          <input id={`rename-${nb.id}`} autoFocus maxLength={120} value={title} onChange={(e) => setTitle(e.target.value)} onBlur={save} onKeyDown={(e) => e.key === 'Escape' && setEditing(false)} />
        </form>
      ) : (
        <Link to={`/notebooks/${nb.id}`} className="nb-card-link">
          <span className="nb-card-title">{nb.title}</span>
        </Link>
      )}
      <div className="nb-card-meta small">
        {fmtRelative(nb.updated_at)} · {nb.source_count} source{nb.source_count === 1 ? '' : 's'}
        {nb.report_count > 0 && ` · ${nb.report_count} report${nb.report_count === 1 ? '' : 's'}`}
      </div>
    </li>
  );
}

export function NotebooksPage() {
  const list = useResource(() => api.notebooks(), []);
  const navigate = useNavigate();
  const { health } = useHealth();
  const [error, setError] = useState<unknown>(null);
  const [creating, setCreating] = useState(false);

  const create = async () => {
    setCreating(true);
    setError(null);
    try {
      const nb = await api.createNotebook('Untitled notebook');
      navigate(`/notebooks/${nb.id}`);
    } catch (e) {
      setError(e);
      setCreating(false);
    }
  };

  const rename = async (id: string, title: string) => {
    setError(null);
    try {
      await api.renameNotebook(id, title);
      await list.reload(true);
    } catch (e) {
      setError(e);
    }
  };

  const remove = async (nb: NotebookSummary) => {
    if (
      !window.confirm(
        `Delete “${nb.title}”?\n\nThis permanently removes the notebook with its ${nb.source_count} source${nb.source_count === 1 ? '' : 's'}, chat, notes and ${nb.report_count} report${nb.report_count === 1 ? '' : 's'}.`,
      )
    )
      return;
    setError(null);
    try {
      await api.deleteNotebook(nb.id);
      await list.reload(true);
    } catch (e) {
      setError(e);
    }
  };

  const items = list.data ?? [];
  return (
    <div className="notebooks-home">
      <div className="nb-home-head">
        <h1>Notebooks</h1>
        <p className="page-description">Collect sources, ask questions with cited answers, and run deep research — each topic in its own notebook.</p>
      </div>
      {health && !health.configured_tasks.synthesis && (
        <div className="nb-hint" role="status">
          <span aria-hidden="true">✨</span> To chat with your sources, first <Link to="/settings/connect">connect an AI service</Link> — paste a key and it's ready in about a minute.
        </div>
      )}
      <ErrorMessage error={error} />
      {list.error ? (
        <ErrorMessage error={list.error} title="Could not load notebooks" />
      ) : list.loading && !list.data ? (
        <Spinner label="Loading notebooks…" />
      ) : (
        <>
          <ul className="nb-grid" aria-label="Notebooks">
            <li className="nb-card nb-card-new">
              <button type="button" className="nb-new" onClick={create} disabled={creating}>
                <span className="nb-new-plus" aria-hidden="true">
                  +
                </span>
                <span>{creating ? 'Creating…' : 'Create new notebook'}</span>
              </button>
            </li>
            {items.map((nb) => (
              <NotebookCard key={nb.id} nb={nb} onRename={(t) => rename(nb.id, t)} onDelete={() => void remove(nb)} />
            ))}
          </ul>
          {items.length === 0 && (
            <div className="nb-empty">
              <div className="nb-empty-title">Create your first notebook</div>
              <p>Add a few PDFs on a topic, then ask questions — every answer cites the exact page it came from.</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
