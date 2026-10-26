import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { Link } from 'react-router-dom';
import { fmtDuration, useNow } from '../../hooks/useNow';
import { ApiError, api } from '../../services/api';
import { streamDiscover } from '../../services/discoverStream';
import type { Assignments, DiscoveredModel, ModelRole, QuickApplyResult, QuickPreset } from '../../types/quickconnect';
import { Alert, ErrorMessage, ProgressBar } from '../ui';
import { ModelPicker } from './ModelPicker';
import { ROLES, buildApplyPayload, summaryLines, supportsModelTools } from './connectLogic';
import { EMPTY_DISCOVERY, TOOL_ROLES, applyEvent, mergeRecommendations, testingIds, type LiveDiscovery } from './liveDiscovery';

type Step = 'key' | 'live' | 'applying' | 'done';

interface Props {
  preset: QuickPreset;
  /** 'reuse' = already connected: discover with the saved key (no key field). */
  mode: 'new' | 'reuse';
  /** Whether an embedding model is active now; false = document search is keyword-only. */
  embeddingsActive?: boolean;
  onClose: () => void;
  onConnected: () => void;
}

const NO_ASSIGN: Assignments = { synthesis: null, planning: null, vision: null, reranking: null, ocr: null, fallback: null, embedding: null };

function StatusIcon({ m, testing }: { m: DiscoveredModel; testing: boolean }) {
  if (testing) return <span className="spinner spinner-sm" role="img" aria-label="testing" />;
  if (m.available === true) return <span className="mstat mstat-ok" role="img" aria-label="works">✓</span>;
  if (m.available === false) return <span className="mstat mstat-bad" role="img" aria-label="unavailable">✗</span>;
  return <span className="mstat mstat-none" role="img" aria-label="not tested">○</span>;
}

function ModelRow({ m, testing }: { m: DiscoveredModel; testing: boolean }) {
  let note: string | undefined;
  if (testing) note = 'testing…';
  else if (m.available === false) note = m.detail || 'not available with this key';
  else if (m.available === true && m.capabilities.includes('chat') && m.json_ok === false) note = 'works, but not in the structured format research needs';
  else if (m.available === true && m.dims) note = `${m.dims} dimensions`;
  else if (m.available === true && m.detail) note = m.detail;
  else if (m.available == null) note = 'not tested';
  const caps = m.capabilities.filter((c) => c !== 'chat');
  return (
    <li className={`model-row model-${testing ? 'testing' : m.available === true ? 'ok' : m.available === false ? 'bad' : 'none'}`} data-model={m.id}>
      <StatusIcon m={m} testing={testing} />
      <span className="model-id">{m.id}</span>
      {caps.map((c) => (
        <span key={c} className="chip">
          {c === 'vision' ? 'sees images' : c}
        </span>
      ))}
      {note && <span className="small muted model-note">{note}</span>}
    </li>
  );
}

function LiveModels({ live, testing }: { live: LiveDiscovery; testing: Set<string> }) {
  const [showAll, setShowAll] = useState(false);
  const tested = new Set(live.toTest);
  const first = live.toTest.map((id) => live.models.find((m) => m.id === id)).filter((m): m is DiscoveredModel => !!m);
  const rest = live.models.filter((m) => !tested.has(m.id));
  return (
    <div className="live-models">
      <ul className="model-list" aria-label="Models being tested">
        {first.map((m) => (
          <ModelRow key={m.id} m={m} testing={testing.has(m.id)} />
        ))}
      </ul>
      {rest.length > 0 && (
        <details open={showAll} onToggle={(e) => setShowAll((e.target as HTMLDetailsElement).open)}>
          <summary className="small">{rest.length} more models listed by the service (not tested)</summary>
          {showAll && (
            <ul className="model-list model-list-rest">
              {rest.map((m) => (
                <ModelRow key={m.id} m={m} testing={false} />
              ))}
            </ul>
          )}
        </details>
      )}
    </div>
  );
}

function Progress({ live, startedAt, ready, onStop }: { live: LiveDiscovery; startedAt: number; ready: boolean; onStop: () => void }) {
  const now = useNow(1000, !live.finished);
  const remaining = Math.max(0, live.total - live.done);
  const works = live.models.filter((m) => m.available === true).length;
  if (live.finished) {
    return (
      <p className="small live-summary" role="status">
        {live.stopped ? 'Stopped testing. ' : ''}Tested {live.done} of {live.total} models: {works} work with your key.
      </p>
    );
  }
  return (
    <div className="live-progress" role="status" aria-live="polite">
      <div className="discover-head">
        <span className="spinner" aria-hidden="true" />
        <strong>
          {live.total === 0 ? 'Fetching the list of models…' : `${live.done} of ${live.total} tested`}
        </strong>
        <span className="muted num">{fmtDuration(now - startedAt)}</span>
        <button type="button" className="btn btn-sm btn-ghost" onClick={onStop}>
          Stop testing
        </button>
      </div>
      <ProgressBar value={live.total ? live.done / live.total : null} indeterminate={!live.total} label="Model testing progress" />
      {ready && remaining > 0 ? (
        <p className="small text-success">
          Still checking {remaining} slower model{remaining === 1 ? '' : 's'} — you don't need to wait. You can connect now.
        </p>
      ) : (
        live.total > 0 && <p className="small muted">Sending each promising model a tiny test question to see whether it works with your key…</p>
      )}
    </div>
  );
}

export function QuickConnectFlow({ preset, mode, embeddingsActive = true, onClose, onConnected }: Props) {
  const isSearch = preset.kind === 'search';
  const isCustom = preset.id === 'custom';
  const modelTools = supportsModelTools(preset);
  const askKey = (preset.needs_key && mode === 'new') || isCustom;
  const keyRequired = preset.needs_key && mode === 'new';
  const askUrl = !preset.needs_key;
  const [step, setStep] = useState<Step>('key');
  const [apiKey, setApiKey] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [live, setLive] = useState<LiveDiscovery>(EMPTY_DISCOVERY);
  const [startedAt, setStartedAt] = useState(0);
  const [assign, setAssign] = useState<Assignments>(NO_ASSIGN);
  const touched = useRef(new Set<ModelRole>());
  const [summary, setSummary] = useState<{ lines: string[]; notes: string[] } | null>(null);
  const [error, setError] = useState<unknown>(null);
  const abortRef = useRef<AbortController | null>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    headingRef.current?.focus();
    return () => abortRef.current?.abort(); // leaving the page cancels the remaining tests
  }, []);

  const testing = useMemo(() => testingIds(live), [live]);
  const ready = !!assign.synthesis;

  const stopTesting = (markStopped = true) => {
    abortRef.current?.abort();
    abortRef.current = null;
    setLive((l) => (l.finished ? l : { ...l, finished: true, stopped: markStopped }));
  };

  const discover = async () => {
    setError(null);
    if (keyRequired && !apiKey.trim()) return setError(new Error(`Paste your ${preset.name} API key first.`));
    if (isCustom && !baseUrl.trim()) return setError(new Error('Enter the address of your server, e.g. http://localhost:8000/v1'));
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    touched.current = new Set();
    setAssign(NO_ASSIGN);
    setLive(EMPTY_DISCOVERY);
    setStartedAt(Date.now());
    setStep('live');
    const payload = { preset: preset.id, ...(apiKey.trim() ? { api_key: apiKey.trim() } : {}), ...(baseUrl.trim() ? { base_url: baseUrl.trim() } : {}) };
    let state = EMPTY_DISCOVERY;
    let searchResult: LiveDiscovery['search'];
    try {
      await streamDiscover(payload, {
        signal: ctrl.signal,
        onEvent: (ev) => {
          state = applyEvent(state, ev);
          if (ev.type === 'done' && ev.result.kind === 'search') searchResult = ev.result;
          setLive(state);
          if (ev.type === 'probe' || ev.type === 'done') {
            const rec = state.recommended;
            const models = state.models;
            setAssign((a) => mergeRecommendations(a, touched.current, rec, models, modelTools));
          }
        },
      });
      if (ctrl.signal.aborted) return;
      setLive((l) => ({ ...l, finished: true }));
      if (state.error && state.models.length === 0) {
        setError(new Error(state.error));
        setStep('key');
        return;
      }
      if (isSearch) {
        if (!searchResult?.ok) {
          setError(new Error(`The test search failed: ${searchResult?.detail || state.error || 'no details'}. Check the key and try again.`));
          setStep('key');
          return;
        }
        await apply();
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return;
      setError(e);
      setStep('key');
    } finally {
      if (abortRef.current === ctrl) abortRef.current = null;
    }
  };

  const apply = async () => {
    setError(null);
    stopTesting(false); // connecting early: no need to finish the slow tests
    setStep('applying');
    try {
      const payload = isSearch
        ? { preset: preset.id, assignments: {}, ...(apiKey.trim() ? { api_key: apiKey.trim() } : {}) }
        : buildApplyPayload(preset, assign, apiKey, baseUrl || (isCustom ? live.baseUrl ?? '' : ''));
      const res: QuickApplyResult = await api.quickApply(payload);
      setApiKey('');
      setSummary({ lines: isSearch ? [`Web search: ${preset.name}`] : summaryLines(assign), notes: res.notes ?? [] });
      setStep('done');
      onConnected();
    } catch (e) {
      setError(e);
      setStep(isSearch ? 'key' : 'live');
    }
  };

  const setRole = (role: ModelRole, v: string | null) => {
    touched.current.add(role);
    setAssign((a) => ({ ...a, [role]: v }));
  };

  const onSubmitKey = (e: FormEvent) => {
    e.preventDefault();
    void discover();
  };

  const title = step === 'done' ? `${isCustom ? live.baseUrl || 'Your server' : preset.name} is connected` : mode === 'reuse' ? `Change ${preset.name} models` : `Connect ${preset.name}`;

  return (
    <section className="connect-flow card" aria-labelledby={`flow-${preset.id}`}>
      <div className="connect-flow-head">
        <h3 id={`flow-${preset.id}`} ref={headingRef} tabIndex={-1}>
          {title}
        </h3>
        <button type="button" className="btn btn-ghost btn-icon" aria-label="Close" onClick={onClose}>
          ✕
        </button>
      </div>

      {step === 'key' && (
        <form onSubmit={onSubmitKey} className="connect-key-form" noValidate>
          {askUrl && (
            <div className="field">
              <label htmlFor={`url-${preset.id}`} className="field-label">
                {isCustom ? 'Address' : 'Server address'} {!isCustom && <span className="muted">(optional)</span>}
              </label>
              <input
                id={`url-${preset.id}`}
                type="url"
                spellCheck={false}
                required={isCustom}
                placeholder={preset.base_url || 'http://localhost:8000/v1'}
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
              />
              <div className="field-hint">
                {isCustom ? (
                  <>
                    The server's OpenAI-compatible address, usually ending in <code>/v1</code>. Addresses on your network other than localhost / 127.0.0.1 must be
                    allowed by the server's <code>OMNI_OUTBOUND_ALLOW_PRIVATE_HOSTS</code> setting.
                  </>
                ) : (
                  <>Leave empty if {preset.name.replace(/ \(local\)/, '')} runs on this computer with default settings.</>
                )}
              </div>
            </div>
          )}
          {askKey && (
            <div className="field">
              <label htmlFor={`key-${preset.id}`} className="field-label">
                {isCustom ? 'API key' : `${preset.name} API key`} {isCustom && <span className="muted">(optional)</span>}
              </label>
              <input
                id={`key-${preset.id}`}
                type="password"
                autoComplete="off"
                spellCheck={false}
                placeholder={isCustom ? 'Only if your server needs one' : preset.key_hint ? `Starts with ${preset.key_hint}` : 'Paste your key'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
              />
              <div className="field-hint">
                {preset.key_url ? (
                  <>
                    Don't have one?{' '}
                    <a href={preset.key_url} target="_blank" rel="noopener noreferrer">
                      Get a key from {preset.name}
                      <span className="sr-only"> (opens in a new tab)</span>
                    </a>
                    .{' '}
                  </>
                ) : null}
                The key is stored encrypted on this computer and never shown again.
              </div>
            </div>
          )}
          {mode === 'reuse' && preset.needs_key && <p className="small muted">Your saved key will be used.</p>}
          {preset.note && !isCustom && <p className="small muted">{preset.note}</p>}
          <ErrorMessage error={error} title={error instanceof ApiError ? 'Could not connect' : 'Check this'} />
          <div className="form-actions">
            <button type="submit" className="btn btn-primary">
              {isSearch ? 'Test & connect' : 'Find models'}
            </button>
            <button type="button" className="btn btn-ghost" onClick={onClose}>
              Cancel
            </button>
          </div>
        </form>
      )}

      {(step === 'live' || step === 'applying') && isSearch && (
        <div className="discover-head" role="status">
          <span className="spinner" aria-hidden="true" /> Running a test search…
        </div>
      )}

      {(step === 'live' || step === 'applying') && !isSearch && (
        <div className="live">
          <Progress live={live} startedAt={startedAt} ready={ready} onStop={() => stopTesting(true)} />
          {live.error && live.models.length > 0 && (
            <Alert tone="danger" title="No working model found yet">
              {live.error}
            </Alert>
          )}
          {live.models.length > 0 && (
            <div className="live-grid">
              <div className="assign">
                <h4 className="assign-title">Assign models</h4>
                {!ready && !live.finished && <p className="small muted">Waiting for the first model that works with your key…</p>}
                {ROLES.map((def) => (
                  <ModelPicker
                    key={def.role}
                    def={def}
                    models={!modelTools && TOOL_ROLES.includes(def.role) ? [] : live.models}
                    recommended={def.role === 'embedding' ? null : live.recommended[def.role]}
                    value={assign[def.role]}
                    testing={testing}
                    disabled={!ready || step === 'applying'}
                    onChange={(v) => setRole(def.role, v)}
                  />
                ))}
                {assign.embedding ? (
                  <Alert tone="warning" title="Cloud embeddings">
                    Your document text will be sent to {isCustom ? 'your server' : preset.name} to build the search index, and existing documents need a reindex
                    (Settings → Document search index) before they use the new model.
                  </Alert>
                ) : (
                  <p className="small muted embed-note">
                    Keeping embeddings local means your documents never leave this computer for search, and nothing needs to be re-processed.
                    {!embeddingsActive && (
                      <span className="text-warning">
                        {' '}
                        Note: no embedding model is active right now, so document search is keyword-only. Turn on the local embedder in “Currently using” below,
                        or choose a cloud embedder above.
                      </span>
                    )}
                  </p>
                )}
                <p className="small muted backup-note">
                  {modelTools
                    ? 'The tools on this computer (result ranking and PaddleOCR) stay available as backups: if you turn them on under “Tools on this computer”, they take over whenever the connected model is unavailable.'
                    : `${preset.name} can't rank results or read scanned pages, so the tools on this computer are used for those (turn them on under “Tools on this computer”).`}
                </p>
                <ErrorMessage error={error} title="Could not connect" />
                <div className="form-actions">
                  <button type="button" className="btn btn-primary btn-lg" disabled={!ready || step === 'applying'} onClick={() => void apply()}>
                    {step === 'applying' ? 'Connecting…' : 'Connect'}
                  </button>
                  <button type="button" className="btn btn-ghost" onClick={onClose} disabled={step === 'applying'}>
                    Cancel
                  </button>
                </div>
              </div>
              <div className="live-list">
                <h4 className="assign-title">Models</h4>
                <LiveModels live={live} testing={testing} />
              </div>
            </div>
          )}
        </div>
      )}

      {step === 'done' && summary && (
        <div className="connect-done" role="status">
          <Alert tone="success" title="All set">
            <ul className="tight-list">
              {summary.lines.map((l) => (
                <li key={l}>{l}</li>
              ))}
            </ul>
          </Alert>
          {summary.notes.map((n) => (
            <Alert key={n} tone="info">
              {n}
            </Alert>
          ))}
          <div className="form-actions">
            {!isSearch && (
              <Link className="btn btn-primary" to="/">
                Open notebooks
              </Link>
            )}
            <button type="button" className="btn" onClick={onClose}>
              Done
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
