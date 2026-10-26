import { useEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent } from 'react';
import { Link } from 'react-router-dom';
import { fmtDuration, useNow } from '../../hooks/useNow';
import { useResource } from '../../hooks/useResource';
import { ApiError, api } from '../../services/api';
import { safeGet, safeSet } from '../../services/storage';
import type { PrivacyDestination } from '../../types/api';
import type { NotebookCheck, NotebookCitation, NotebookGuide, NotebookMessage, NotebookReport } from '../../types/notebooks';
import { copyText } from '../../utils/citations';
import { ErrorMessage, Spinner, StatusBadge } from '../ui';
import { NotebookAnswer, answerToPlainText } from './NotebookAnswer';
import { ACTIVE_REPORT, RunningProgress } from './ReportProgress';
import { NoSearchHint, isNoSearchError } from './webHints';

/** True when the server says no AI model is connected (503 "No provider configured…"). */
export function isNoModelError(e: unknown): boolean {
  return e instanceof ApiError && (e.status === 503 || e.status === 400) && /no provider|not configured/i.test(e.detail);
}

export function ConnectAiHint({ what }: { what: string }) {
  return (
    <div className="nb-connect-hint" role="status">
      {what} needs an AI model. <Link to="/settings/connect">Connect an AI service</Link> — paste a key and it's ready in about a minute.
    </div>
  );
}

/** Gentle, plain-language notes about an answer's automatic checks. */
export function answerWarnings(check: NotebookCheck | undefined): string[] {
  if (!check) return [];
  const out: string[] = [];
  if (check.answered_from_sources === false) out.push("The selected sources don't fully answer this question.");
  if (check.numbers_not_in_cited_evidence?.length)
    out.push(`Some numbers are calculated or not found in the cited passages: ${check.numbers_not_in_cited_evidence.join(', ')}`);
  if (check.unknown_citations?.length) out.push('Some citation marks did not match a passage (shown as ?).');
  for (const w of check.retrieval_warnings ?? []) out.push(`Search note: ${w}`);
  return out;
}

function GuideCard({ notebookId, title, readyCount, onAsk, disabled }: { notebookId: string; title: string; readyCount: number; onAsk: (q: string) => void; disabled: boolean }) {
  const [guide, setGuide] = useState<NotebookGuide | null>(null);
  const [error, setError] = useState<unknown>(null);
  useEffect(() => {
    if (!readyCount) return;
    const ctrl = new AbortController();
    setGuide(null);
    setError(null);
    api
      .notebookGuide(notebookId, ctrl.signal)
      .then(setGuide)
      .catch((e) => {
        if (!(e instanceof DOMException && e.name === 'AbortError')) setError(e);
      });
    return () => ctrl.abort();
  }, [notebookId, readyCount]);

  return (
    <section className="nb-guide" aria-label="Notebook guide">
      <div className="nb-guide-title">{title}</div>
      <div className="small muted">
        {readyCount} source{readyCount === 1 ? '' : 's'}
      </div>
      {!readyCount ? (
        <p className="nb-guide-summary muted">Add a source to get a summary and suggested questions.</p>
      ) : error ? (
        isNoModelError(error) ? (
          <ConnectAiHint what="A summary of your sources" />
        ) : (
          <ErrorMessage error={error} title="The summary isn't available right now" />
        )
      ) : !guide ? (
        <div className="nb-skeleton" aria-busy="true" aria-label="Writing a summary of your sources">
          <span />
          <span />
          <span className="short" />
          <div className="small muted">Reading your sources to write a summary… (about half a minute the first time)</div>
        </div>
      ) : guide.summary ? (
        <>
          <p className="nb-guide-summary">{guide.summary}</p>
          {guide.questions.length > 0 && (
            <div className="nb-suggestions" aria-label="Suggested questions">
              {guide.questions.map((q) => (
                <button key={q} type="button" className="nb-suggestion" onClick={() => onAsk(q)} disabled={disabled}>
                  {q}
                </button>
              ))}
            </div>
          )}
        </>
      ) : (
        <p className="nb-guide-summary muted">{guide.reason ?? 'No summary yet.'}</p>
      )}
    </section>
  );
}

function AnswerActions({ notebookId, m, onNoteSaved }: { notebookId: string; m: NotebookMessage; onNoteSaved: () => void }) {
  const [saved, setSaved] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [copied, setCopied] = useState(false);
  return (
    <div className="nb-answer-actions">
      <button
        type="button"
        className="btn btn-sm btn-ghost"
        disabled={saved === 'saving' || saved === 'saved'}
        onClick={async () => {
          setSaved('saving');
          try {
            await api.addNotebookNote(notebookId, { message_id: m.id });
            setSaved('saved');
            onNoteSaved();
          } catch {
            setSaved('error');
          }
        }}
      >
        {saved === 'saved' ? 'Saved to notes ✓' : saved === 'saving' ? 'Saving…' : saved === 'error' ? 'Could not save — retry' : 'Save to note'}
      </button>
      <button
        type="button"
        className="btn btn-sm btn-ghost"
        onClick={async () => {
          setCopied(await copyText(answerToPlainText(m.content, m.citations)));
          setTimeout(() => setCopied(false), 1800);
        }}
      >
        {copied ? 'Copied ✓' : 'Copy'}
      </button>
    </div>
  );
}

// ------------------------------------------------------------------ privacy line

const PROVIDER_NAMES: Record<string, string> = {
  nvidia: 'NVIDIA',
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  google: 'Google',
  gemini: 'Google',
  vertex: 'Google',
  mistral: 'Mistral',
  groq: 'Groq',
  openrouter: 'OpenRouter',
  together: 'Together AI',
  deepseek: 'DeepSeek',
  cohere: 'Cohere',
  azure: 'Azure',
  aws: 'AWS',
  bedrock: 'AWS',
  fireworks: 'Fireworks',
  perplexity: 'Perplexity',
  xai: 'xAI',
  huggingface: 'Hugging Face',
  hf: 'Hugging Face',
  jina: 'Jina',
  voyage: 'Voyage AI',
  tavily: 'Tavily',
  brave: 'Brave',
  exa: 'Exa',
};

/** "nvidia-nim-chat" → "NVIDIA"; unknown ids are grouped by their first segment and title-cased. */
export function friendlyProviderName(providerId: string): string {
  const first = providerId.trim().split(/[-_:/.\s]+/).find(Boolean) ?? '';
  if (!first) return providerId;
  const known = PROVIDER_NAMES[first.toLowerCase()];
  if (known) return known;
  return first.charAt(0).toUpperCase() + first.slice(1);
}

function joinNames(names: string[]): string {
  if (names.length <= 1) return names.join('');
  return `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
}

/** One short line: "🔒 Passages are sent to NVIDIA" (null when nothing leaves this computer). */
export function privacyLine(dest: PrivacyDestination[] | undefined): { text: string; title: string } | null {
  if (!dest?.length) return null;
  const names = [...new Set(dest.map((d) => friendlyProviderName(d.provider_id)))];
  const ids = [...new Set(dest.map((d) => d.provider_id))];
  return { text: `Passages are sent to ${joinNames(names)}`, title: `Services that receive passages from your sources: ${ids.join(', ')}` };
}

// ------------------------------------------------------------------ composer preferences

export type ChatMode = 'fast' | 'deep';
interface ComposerPrefs {
  mode: ChatMode;
  web: boolean;
}

const prefsKey = (notebookId: string) => `omni.notebook.${notebookId}.composer`;
const clearedKey = (notebookId: string) => `omni.notebook.${notebookId}.chatClearedAt`;

function loadPrefs(notebookId: string): ComposerPrefs {
  try {
    const raw = JSON.parse(safeGet(prefsKey(notebookId)) ?? '{}') as Partial<ComposerPrefs>;
    return { mode: raw.mode === 'deep' ? 'deep' : 'fast', web: raw.web === true };
  } catch {
    return { mode: 'fast', web: false };
  }
}

// ------------------------------------------------------------------ deep research card in the chat

function ReportCard({ report, onOpen }: { report: NotebookReport; onOpen: () => void }) {
  const active = ACTIVE_REPORT.has(report.status);
  return (
    <li className="nb-msg nb-msg-report" data-report-card={report.id}>
      <div className="nb-report-card">
        <div className="nb-report-card-head small">
          <span aria-hidden="true">🔬</span> <span className="nb-report-card-kind">Deep research</span> <StatusBadge status={report.status} />
        </div>
        <div className="nb-report-card-q">{report.question}</div>
        {active && <RunningProgress report={report} />}
        {report.status === 'failed' && <p className="small text-danger nb-report-card-err">{report.error || 'The research could not be finished.'}</p>}
        {report.status === 'completed' && (
          <button type="button" className="btn btn-primary btn-sm nb-report-card-open" onClick={onOpen}>
            Open report
          </button>
        )}
        {active && <p className="small muted nb-report-card-note">Usually takes 5–10 minutes. You can keep chatting meanwhile.</p>}
      </div>
    </li>
  );
}

// ------------------------------------------------------------------ panel

interface Props {
  notebookId: string;
  title: string;
  readyCount: number;
  selectedIds: string[];
  /** Deep-research reports of this notebook; shown in the chat timeline. */
  reports?: NotebookReport[];
  onReportStarted?: () => void;
  onOpenReport?: (id: string) => void;
  onOpenCitation: (id: string, c: NotebookCitation) => void;
  onNoteSaved: () => void;
}

type TimelineItem = { kind: 'msg'; at: number; m: NotebookMessage } | { kind: 'report'; at: number; r: NotebookReport };

export function ChatPanel({ notebookId, title, readyCount, selectedIds, reports, onReportStarted, onOpenReport, onOpenCitation, onNoteSaved }: Props) {
  const messages = useResource(() => api.notebookMessages(notebookId), [notebookId]);
  const preflight = useResource(() => api.preflight(), []);
  const [input, setInput] = useState('');
  const [pending, setPending] = useState<{ text: string; startedAt: number; web: boolean; mode: ChatMode } | null>(null);
  const [prefs, setPrefs] = useState<ComposerPrefs>(() => loadPrefs(notebookId));
  const [clearedAt, setClearedAt] = useState<number>(() => Number(safeGet(clearedKey(notebookId))) || 0);
  const [showWebHint, setShowWebHint] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const now = useNow(1000, !!pending);
  const endRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const list = messages.data ?? [];

  useEffect(() => {
    setPrefs(loadPrefs(notebookId));
    setClearedAt(Number(safeGet(clearedKey(notebookId))) || 0);
  }, [notebookId]);
  const updatePrefs = (patch: Partial<ComposerPrefs>) =>
    setPrefs((p) => {
      const next = { ...p, ...patch };
      safeSet(prefsKey(notebookId), JSON.stringify(next));
      return next;
    });

  /** False only once we know no search service is connected. */
  const webAvailable = preflight.data ? preflight.data.web_available : true;
  const web = prefs.web && webAvailable;
  const deep = prefs.mode === 'deep';

  const timeline = useMemo<TimelineItem[]>(() => {
    const items: TimelineItem[] = list.map((m) => ({ kind: 'msg', at: Date.parse(m.created_at) || 0, m }));
    for (const r of reports ?? []) {
      const at = Date.parse(r.created_at) || 0;
      if (at > clearedAt) items.push({ kind: 'report', at, r });
    }
    return items.sort((a, b) => a.at - b.at);
  }, [list, reports, clearedAt]);

  useEffect(() => () => abortRef.current?.abort(), []);
  useEffect(() => {
    endRef.current?.scrollIntoView?.({ block: 'end' });
  }, [timeline.length, pending]);

  const noSelection = selectedIds.length === 0;
  /** With web search on, a question can be asked even with no sources selected. */
  const blocked = noSelection && !web;
  const minLen = deep ? 5 : 2;

  const send = async (text: string) => {
    const msg = text.trim();
    if (msg.length < minLen || pending || blocked) return;
    setError(null);
    setPending({ text: msg, startedAt: Date.now(), web, mode: prefs.mode });
    setInput('');
    if (deep) {
      try {
        await api.startNotebookReport(notebookId, { question: msg, source_ids: selectedIds, web_mode: web ? 'auto' : 'never' });
        onReportStarted?.();
      } catch (e) {
        setError(e);
        setInput(msg);
      } finally {
        setPending(null);
      }
      return;
    }
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      if (web) await api.askNotebook(notebookId, msg, selectedIds, ctrl.signal, { web: true });
      else await api.askNotebook(notebookId, msg, selectedIds, ctrl.signal);
      await messages.reload(true);
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return;
      setError(e);
      setInput(msg);
    } finally {
      setPending(null);
    }
  };

  const clear = async () => {
    if (!window.confirm('Clear the chat? Saved notes and reports are kept (reports stay in Studio).')) return;
    try {
      await api.clearNotebookChat(notebookId);
      const t = Date.now();
      safeSet(clearedKey(notebookId), String(t));
      setClearedAt(t);
      await messages.reload(true);
    } catch (e) {
      setError(e);
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
  const onWebChip = () => {
    if (!webAvailable) {
      setShowWebHint(true);
      return;
    }
    setShowWebHint(false);
    updatePrefs({ web: !prefs.web });
  };

  const placeholder = deep
    ? blocked
      ? 'Select a source or turn on Web to start a report'
      : 'What should the report answer?'
    : blocked
      ? 'Select at least one source to start chatting'
      : noSelection
        ? 'Ask anything — the answer comes from the web…'
        : 'Ask about your sources…';
  const privacy = privacyLine(preflight.data?.external_destinations);

  return (
    <div className="nb-chat">
      <div className="nb-chat-scroll">
        {messages.loading && !messages.data ? (
          <Spinner label="Loading chat…" />
        ) : (
          <>
            {timeline.length === 0 && !pending && <GuideCard notebookId={notebookId} title={title} readyCount={readyCount} onAsk={(q) => void send(q)} disabled={blocked} />}
            <ErrorMessage error={messages.error} title="Could not load the chat" />
            <ol className="nb-messages" aria-label="Chat">
              {timeline.map((item) => {
                if (item.kind === 'report') return <ReportCard key={`r-${item.r.id}`} report={item.r} onOpen={() => onOpenReport?.(item.r.id)} />;
                const m = item.m;
                return m.role === 'user' ? (
                  <li key={m.id} className="nb-msg nb-msg-user">
                    <div className="nb-bubble">{m.content}</div>
                  </li>
                ) : (
                  <li key={m.id} className="nb-msg nb-msg-assistant" data-message={m.id}>
                    {m.check?.web && (
                      <span className="nb-web-used small" title="This answer also used a quick web search">
                        <span className="nb-web-tag">Web</span> Includes web results
                      </span>
                    )}
                    <NotebookAnswer content={m.content} citations={m.citations} onOpen={onOpenCitation} />
                    {answerWarnings(m.check).length > 0 && (
                      <ul className="nb-warnings" aria-label="Notes about this answer">
                        {answerWarnings(m.check).map((w) => (
                          <li key={w}>{w}</li>
                        ))}
                      </ul>
                    )}
                    <AnswerActions notebookId={notebookId} m={m} onNoteSaved={onNoteSaved} />
                  </li>
                );
              })}
              {pending && (
                <>
                  <li className="nb-msg nb-msg-user">
                    <div className="nb-bubble">{pending.text}</div>
                  </li>
                  <li className="nb-msg nb-msg-assistant nb-typing" role="status">
                    <span className="nb-dots" aria-hidden="true">
                      <span />
                      <span />
                      <span />
                    </span>
                    {pending.mode === 'deep'
                      ? 'Starting deep research…'
                      : pending.web
                        ? noSelection
                          ? 'Searching the web…'
                          : 'Searching the web and reading your sources…'
                        : 'Reading your sources…'}{' '}
                    <span className="muted num">{fmtDuration(now - pending.startedAt)}</span>
                  </li>
                </>
              )}
            </ol>
            <div ref={endRef} />
          </>
        )}
      </div>
      {isNoModelError(error) ? (
        <ConnectAiHint what={deep ? 'Deep research' : 'Chatting with your sources'} />
      ) : isNoSearchError(error) ? (
        <NoSearchHint />
      ) : (
        <ErrorMessage error={error} title={deep ? 'Could not start the research' : 'The question could not be answered'} />
      )}
      {showWebHint && !webAvailable && <NoSearchHint />}
      <form className="nb-composer nb-composer-v2" onSubmit={onSubmit}>
        <div className="nb-composer-row">
          <label htmlFor={`nb-input-${notebookId}`} className="sr-only">
            Ask about your sources
          </label>
          <textarea
            id={`nb-input-${notebookId}`}
            rows={1}
            maxLength={4000}
            placeholder={placeholder}
            value={input}
            disabled={!!pending || blocked}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
          />
          <button
            type="submit"
            className="btn btn-primary nb-send"
            disabled={!!pending || blocked || input.trim().length < minLen}
            aria-label={deep ? 'Start deep research' : 'Send'}
          >
            ➤
          </button>
        </div>
        <div className="nb-composer-bar">
          <div className="nb-segmented nb-mode-switch" role="radiogroup" aria-label="Answer mode">
            {(['fast', 'deep'] as const).map((m) => (
              <label key={m} className={`nb-segment ${prefs.mode === m ? 'nb-segment-on' : ''}`} title={m === 'fast' ? 'A quick cited answer' : 'A multi-step cited report, usually 5–10 min'}>
                <input type="radio" name={`chat-mode-${notebookId}`} value={m} checked={prefs.mode === m} onChange={() => updatePrefs({ mode: m })} className="sr-only" disabled={!!pending} />
                <span aria-hidden="true">{m === 'fast' ? '⚡' : '🔬'}</span> {m === 'fast' ? 'Fast' : 'Deep research'}
              </label>
            ))}
          </div>
          <button
            type="button"
            className={`nb-web-chip ${web ? 'nb-web-chip-on' : ''} ${webAvailable ? '' : 'nb-web-chip-off'}`}
            aria-pressed={web}
            aria-disabled={!webAvailable || undefined}
            title={webAvailable ? (web ? 'Web search is on' : 'Also search the web') : 'Connect a search service to use the web'}
            onClick={onWebChip}
            disabled={!!pending}
          >
            <span aria-hidden="true">🌐</span> Web
          </button>
          <span className="nb-composer-count small muted">
            {selectedIds.length} source{selectedIds.length === 1 ? '' : 's'} selected
          </span>
        </div>
      </form>
      {(privacy || timeline.length > 0) && (
        <div className="nb-chat-foot">
          {privacy && (
            <span className="small muted nb-privacy" data-testid="privacy-note" title={privacy.title}>
              <span aria-hidden="true">🔒</span> {privacy.text}
            </span>
          )}
          {list.length > 0 && (
            <button type="button" className="btn btn-link btn-sm" onClick={() => void clear()} disabled={!!pending}>
              Clear chat
            </button>
          )}
        </div>
      )}
    </div>
  );
}
