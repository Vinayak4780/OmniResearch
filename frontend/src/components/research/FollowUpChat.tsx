import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react';
import { useNow, fmtDuration } from '../../hooks/useNow';
import { useResource } from '../../hooks/useResource';
import { api } from '../../services/api';
import type { ChatCheck, ChatMessage, Evidence } from '../../types/api';
import { evidenceLink, linkifyCitations } from '../../utils/citations';
import { fmtDate } from '../../utils/format';
import { Alert, ErrorMessage, Spinner } from '../ui';
import { MarkdownReport } from './MarkdownReport';

/** Human-readable warnings for an assistant answer's mechanical check. Empty when the answer is clean. */
export function checkWarnings(check: ChatCheck | null | undefined): string[] {
  if (!check) return [];
  const out: string[] = [];
  if (check.answered_from_sources === false) out.push('The model says the sources do not answer this question; anything stated beyond the citations is not supported.');
  if (check.unknown_citations?.length) out.push(`Cites evidence that does not exist: ${check.unknown_citations.join(', ')}.`);
  if (check.numbers_not_in_cited_evidence?.length)
    out.push(`Numbers not found in the cited evidence: ${check.numbers_not_in_cited_evidence.join(', ')}. Verify them before relying on them.`);
  return out;
}

interface Props {
  jobId: string;
  evidence: Map<string, Evidence>;
  onCitation: (evidenceId: string, href: string) => void;
  /** Called after an answer that added new evidence ids to the job. */
  onEvidenceAdded: (added: Evidence[]) => void;
}

interface Pending {
  message: string;
  startedAt: number;
}

function AssistantMessage({ m, evidence, onCitation }: { m: ChatMessage; evidence: Map<string, Evidence>; onCitation: Props['onCitation'] }) {
  // Links from the server win; fall back to evidence we know about (e.g. after a refresh).
  const links: Record<string, string> = { ...m.links };
  for (const id of m.citations) if (!links[id] && evidence.has(id)) links[id] = evidenceLink(evidence.get(id)!);
  for (const [id, e] of evidence) if (!links[id]) links[id] = evidenceLink(e);
  for (const id of m.check?.unknown_citations ?? []) delete links[id];
  const warnings = checkWarnings(m.check);
  const added = m.check?.new_evidence ?? [];
  return (
    <div className="chat-msg chat-assistant">
      <div className="chat-meta small muted">Answer · {fmtDate(m.created_at)}</div>
      <MarkdownReport markdown={linkifyCitations(m.content)} citationLinks={links} onCitation={onCitation} className="chat-markdown" />
      {warnings.length > 0 && (
        <Alert tone="warning" title="Check this answer" role="status">
          <ul className="tight-list">
            {warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </Alert>
      )}
      {added.length > 0 && <div className="small muted">New passages found for this question: {added.join(', ')} (added to the evidence list).</div>}
    </div>
  );
}

export function FollowUpChat({ jobId, evidence, onCitation, onEvidenceAdded }: Props) {
  const history = useResource(() => api.chatHistory(jobId), [jobId]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [pending, setPending] = useState<Pending | null>(null);
  const [error, setError] = useState<{ err: unknown; message: string } | null>(null);
  const now = useNow(1000, !!pending);
  const endRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (history.data) setMessages(history.data);
  }, [history.data]);
  useEffect(() => () => abortRef.current?.abort(), []);
  useEffect(() => {
    if (pending || messages.length) endRef.current?.scrollIntoView?.({ block: 'nearest' });
  }, [messages.length, pending]);

  const send = async (text: string) => {
    const message = text.trim();
    if (message.length < 2 || pending) return;
    setError(null);
    setPending({ message, startedAt: Date.now() });
    setInput('');
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const res = await api.sendChat(jobId, message, ctrl.signal);
      const t = new Date().toISOString();
      setMessages((ms) => [
        ...ms,
        { id: -Date.now() - 1, role: 'user', content: message, citations: [], links: {}, check: {}, created_at: t },
        { id: -Date.now(), role: 'assistant', content: res.answer, citations: res.citations, links: res.links, check: res.check, created_at: t },
      ]);
      if (res.evidence?.length || res.check?.new_evidence?.length) onEvidenceAdded(res.evidence ?? []);
    } catch (e) {
      if (!(e instanceof DOMException && e.name === 'AbortError')) {
        setError({ err: e, message });
        setInput(message);
      }
    } finally {
      setPending(null);
      abortRef.current = null;
    }
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void send(input);
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void send(input);
    }
  };

  const len = input.trim().length;
  return (
    <section className="chat" aria-labelledby="chat-title">
      <h2 id="chat-title" className="chat-title">
        Ask a follow-up question
      </h2>
      <p className="small muted">
        Answers use this investigation's evidence plus a fresh search of the same documents. Citations open the cited page; numbers are checked against the
        cited passages.
      </p>
      {history.loading && !history.data && <Spinner label="Loading conversation…" inline />}
      <ErrorMessage error={history.error} title="Could not load the conversation" />
      <div className="chat-log" aria-live="polite">
        {messages.map((m) =>
          m.role === 'user' ? (
            <div key={m.id} className="chat-msg chat-user">
              <div className="chat-meta small muted">You · {fmtDate(m.created_at)}</div>
              <p className="pre-wrap">{m.content}</p>
            </div>
          ) : (
            <AssistantMessage key={m.id} m={m} evidence={evidence} onCitation={onCitation} />
          ),
        )}
        {pending && (
          <>
            <div className="chat-msg chat-user chat-pending-user">
              <div className="chat-meta small muted">You</div>
              <p className="pre-wrap">{pending.message}</p>
            </div>
            <div className="chat-msg chat-assistant chat-thinking" role="status">
              <span className="spinner" aria-hidden="true" /> Searching the evidence and drafting an answer… {fmtDuration(now - pending.startedAt)}
              <span className="small muted"> (usually 10–60 s)</span>
              <button type="button" className="btn btn-sm btn-ghost" onClick={() => abortRef.current?.abort()}>
                Stop waiting
              </button>
            </div>
          </>
        )}
        <div ref={endRef} />
      </div>
      {error && (
        <div className="chat-error">
          <ErrorMessage error={error.err} title="The question could not be answered" />
          <button type="button" className="btn btn-sm" onClick={() => void send(error.message)} disabled={!!pending}>
            Retry
          </button>
        </div>
      )}
      <form className="chat-form" onSubmit={onSubmit}>
        <label htmlFor={`chat-input-${jobId}`} className="sr-only">
          Follow-up question
        </label>
        <textarea
          id={`chat-input-${jobId}`}
          rows={2}
          maxLength={2000}
          value={input}
          disabled={!!pending}
          placeholder="e.g. How much memory do 4-bit weights need on a 24 GB GPU?"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKey}
        />
        <div className="chat-form-actions">
          <span className="small muted">Enter to send · Shift+Enter for a new line · {len}/2000</span>
          <button type="submit" className="btn btn-primary" disabled={!!pending || len < 2}>
            {pending ? 'Answering…' : 'Ask'}
          </button>
        </div>
      </form>
    </section>
  );
}
