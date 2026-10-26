import { useState } from 'react';
import { fmtDuration, useNow } from '../../hooks/useNow';
import { api } from '../../services/api';
import type { NotebookCitation, NotebookOutput, OutputCheck, OutputKind } from '../../types/notebooks';
import { copyText } from '../../utils/citations';
import { fmtRelative } from '../../utils/format';
import { ErrorMessage } from '../ui';
import { ConnectAiHint, isNoModelError } from './ChatPanel';
import { NotebookAnswer, answerToPlainText } from './NotebookAnswer';
import { OutputMindMap, mindMapToText } from './OutputMindMap';

export const OUTPUT_KINDS: { kind: OutputKind; title: string; icon: string; hint: string }[] = [
  { kind: 'summary', title: 'Summary', icon: '📝', hint: 'The key points in a few paragraphs' },
  { kind: 'briefing', title: 'Briefing doc', icon: '📄', hint: 'Main themes, facts and takeaways' },
  { kind: 'study_guide', title: 'Study guide', icon: '🎓', hint: 'Key terms, quiz questions and answers' },
  { kind: 'faq', title: 'FAQ', icon: '❓', hint: 'Common questions answered from your sources' },
  { kind: 'timeline', title: 'Timeline', icon: '🕒', hint: 'Events in date order' },
  { kind: 'mindmap', title: 'Mind map', icon: '🧠', hint: 'The main ideas and how they connect' },
];

export const outputIcon = (kind: OutputKind) => OUTPUT_KINDS.find((k) => k.kind === kind)?.icon ?? '📄';
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** The gentle "what was checked" line under an output; only non-zero parts are shown. */
export function trustParts(check: OutputCheck | undefined): string[] {
  if (!check) return [];
  const out: string[] = [];
  if (check.passages_used) out.push(`Checked: ${plural(check.passages_used, 'passage')}`);
  if (check.uncited_statements) out.push(`${plural(check.uncited_statements, 'statement')} without a citation`);
  if (check.numbers_not_in_cited_evidence?.length) out.push(`numbers not found in cited passages: ${check.numbers_not_in_cited_evidence.join(', ')}`);
  if (check.unknown_citations?.length) out.push(`unknown citation${check.unknown_citations.length === 1 ? '' : 's'}: ${check.unknown_citations.join(', ')}`);
  return out;
}

export function sortOutputs(list: NotebookOutput[]): NotebookOutput[] {
  return [...list].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
}

// ------------------------------------------------------------------ create buttons

export function OutputButtons({ selectedIds, onCreate }: { selectedIds: string[]; onCreate: (kind: OutputKind) => Promise<void> }) {
  const [busy, setBusy] = useState<OutputKind | null>(null);
  const [error, setError] = useState<unknown>(null);
  const none = selectedIds.length === 0;
  const create = async (kind: OutputKind) => {
    setBusy(kind);
    setError(null);
    try {
      await onCreate(kind);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(null);
    }
  };
  return (
    <section className="nb-studio-card" aria-labelledby="create-outputs-title">
      <div className="nb-studio-card-head">
        <span aria-hidden="true">✨</span>
        <h3 id="create-outputs-title">Create from your sources</h3>
      </div>
      <p className="small muted">Uses the {plural(selectedIds.length, 'checked source')}. Takes about a minute and appears under Outputs.</p>
      <div className="nb-output-grid">
        {OUTPUT_KINDS.map((k) => (
          <button
            key={k.kind}
            type="button"
            className="nb-output-btn"
            aria-label={`Create ${k.title}`}
            title={k.hint}
            disabled={none || busy !== null}
            onClick={() => void create(k.kind)}
          >
            <span className="nb-output-icon" aria-hidden="true">
              {k.icon}
            </span>
            <span>{busy === k.kind ? 'Starting…' : k.title}</span>
          </button>
        ))}
      </div>
      {none && <span className="small muted">Select at least one source.</span>}
      {isNoModelError(error) ? <ConnectAiHint what="Creating summaries, guides and mind maps" /> : <ErrorMessage error={error} title="Could not start" />}
    </section>
  );
}

// ------------------------------------------------------------------ list

function Elapsed({ since }: { since: string }) {
  const now = useNow(1000, true);
  const t = Date.parse(since);
  return <span className="muted num">{Number.isFinite(t) ? fmtDuration(now - t) : ''}</span>;
}

interface ListProps {
  outputs: NotebookOutput[] | undefined;
  error: unknown;
  onOpen: (o: NotebookOutput) => void;
  onRetry: (o: NotebookOutput) => void;
  onDelete: (o: NotebookOutput) => void;
}

export function OutputList({ outputs, error, onOpen, onRetry, onDelete }: ListProps) {
  return (
    <section className="nb-reports nb-outputs" aria-labelledby="outputs-title">
      <div className="nb-section-head">
        <h3 id="outputs-title">Outputs</h3>
      </div>
      <ErrorMessage error={error} />
      {!outputs ? null : outputs.length === 0 ? (
        <p className="small muted">Summaries, guides and mind maps you create appear here.</p>
      ) : (
        <ul className="nb-report-list">
          {sortOutputs(outputs).map((o) => (
            <li key={o.id} className={`nb-report nb-output nb-output-${o.status}`} data-output={o.id}>
              {o.status === 'completed' ? (
                <button type="button" className="nb-report-btn" onClick={() => onOpen(o)}>
                  <span className="nb-report-q">
                    <span aria-hidden="true">{outputIcon(o.kind)}</span> {o.title}
                  </span>
                  <span className="nb-report-meta small muted">{fmtRelative(o.created_at)}</span>
                </button>
              ) : (
                <div className="nb-output-row">
                  <span className="nb-report-q">
                    <span aria-hidden="true">{outputIcon(o.kind)}</span> {o.title}
                  </span>
                  {o.status === 'running' ? (
                    <span className="nb-report-meta small" role="status">
                      <span className="spinner spinner-sm" aria-hidden="true" /> Writing… <Elapsed since={o.created_at} />
                    </span>
                  ) : (
                    <>
                      <span className="small text-danger nb-output-error">{o.error || 'This could not be created.'}</span>
                      <span className="nb-output-actions">
                        <button type="button" className="btn btn-sm" onClick={() => onRetry(o)} aria-label={`Retry ${o.title}`}>
                          Retry
                        </button>
                        <button type="button" className="btn btn-sm btn-ghost" onClick={() => onDelete(o)} aria-label={`Delete ${o.title}`}>
                          Delete
                        </button>
                      </span>
                    </>
                  )}
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

// ------------------------------------------------------------------ reader

interface ReaderProps {
  notebookId: string;
  output: NotebookOutput;
  onBack: () => void;
  onDelete: (o: NotebookOutput) => void;
  onNoteSaved: () => void;
  onOpenCitation: (id: string, c: NotebookCitation) => void;
}

export function OutputReader({ notebookId, output, onBack, onDelete, onNoteSaved, onOpenCitation }: ReaderProps) {
  const [copied, setCopied] = useState(false);
  const [saved, setSaved] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const isMap = output.kind === 'mindmap' && !!output.data?.branches?.length;
  const trust = trustParts(output.check);
  const plain = isMap ? mindMapToText(output.data) : answerToPlainText(output.content, output.citations ?? {});

  const save = async () => {
    setSaved('saving');
    try {
      await api.addNotebookNote(notebookId, { title: output.title, content: isMap ? output.content || plain : output.content });
      setSaved('saved');
      onNoteSaved();
    } catch {
      setSaved('error');
    }
  };

  return (
    <article className="nb-output-reader" aria-labelledby={`output-title-${output.id}`}>
      <div className="nb-preview-head">
        <button type="button" className="btn btn-ghost btn-sm" onClick={onBack}>
          ← Back
        </button>
      </div>
      <h3 className="nb-preview-title" id={`output-title-${output.id}`}>
        <span aria-hidden="true">{outputIcon(output.kind)}</span> {output.title}
      </h3>
      <div className="small muted">
        {plural(output.source_ids?.length ?? 0, 'source')} · {fmtRelative(output.created_at)}
      </div>
      {trust.length > 0 && (
        <p className="nb-trust small" aria-label="What was checked">
          {trust.join(' · ')}
        </p>
      )}
      <div className="nb-answer-actions">
        <button
          type="button"
          className="btn btn-sm btn-ghost"
          onClick={async () => {
            setCopied(await copyText(plain));
            setTimeout(() => setCopied(false), 1800);
          }}
        >
          {copied ? 'Copied ✓' : 'Copy'}
        </button>
        <button type="button" className="btn btn-sm btn-ghost" disabled={saved === 'saving' || saved === 'saved'} onClick={() => void save()}>
          {saved === 'saved' ? 'Saved to notes ✓' : saved === 'saving' ? 'Saving…' : saved === 'error' ? 'Could not save — retry' : 'Save to note'}
        </button>
        <button type="button" className="btn btn-sm btn-danger-ghost" onClick={() => onDelete(output)}>
          Delete
        </button>
      </div>
      <div className="nb-output-body">
        {isMap ? (
          <OutputMindMap data={output.data} citations={output.citations ?? {}} onOpen={onOpenCitation} />
        ) : (
          <NotebookAnswer content={output.content} citations={output.citations ?? {}} onOpen={onOpenCitation} />
        )}
      </div>
    </article>
  );
}
