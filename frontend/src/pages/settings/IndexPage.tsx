import { useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Alert, Badge, Card, EmptyState, ErrorMessage, KeyValue, ProgressBar, Spinner, Stat, StatusBadge } from '../../components/ui';
import { usePolling } from '../../hooks/usePolling';
import { useResource } from '../../hooks/useResource';
import { api } from '../../services/api';
import type { IndexCompatibility, IndexInfo } from '../../types/api';
import { fmtDate, fmtInt } from '../../utils/format';

const REASONS: Record<string, string> = {
  no_active_index: 'There is no active vector index yet. One is created automatically when the first document is ingested with a text-embedding provider routed.',
  pinned_provider_unavailable: 'The provider this index was built with is disabled or deleted, so queries cannot be embedded into the same vector space.',
  provider_config_changed: "The pinned provider's configuration changed since the index was built (e.g. endpoint or model). Its vectors may no longer be comparable.",
  chunking_version_changed: 'The chunking algorithm changed since this index was built. Reindex to rebuild chunks and vectors.',
};

function SpecView({ idx }: { idx: IndexInfo }) {
  const s = idx.spec;
  return (
    <KeyValue
      items={[
        ['Provider', <code key="p">{s.provider_id}</code>],
        ['Model', s.model ?? null],
        ['Dimensions', s.dimensions != null ? fmtInt(s.dimensions) : null],
        ['Distance', s.distance],
        ['Chunking version', <code key="c">{s.chunking_version}</code>],
        ['Preprocessing', <code key="pp">{s.preprocessing}</code>],
        ['Collection', <code key="col">{idx.collection_name}</code>],
        ['Index ID', <code key="id">{idx.id}</code>],
        ['Created', fmtDate(idx.created_at)],
        ['Activated', idx.activated_at ? fmtDate(idx.activated_at) : null],
        [
          'Compatibility confirmations',
          s.compatibility_confirmations?.length ? `${s.compatibility_confirmations.length} (last ${fmtDate(s.compatibility_confirmations[s.compatibility_confirmations.length - 1].at)})` : null,
        ],
      ]}
    />
  );
}

function Compat({ c }: { c: IndexCompatibility }) {
  if (c.ok) return <Badge tone="success">Compatible</Badge>;
  return (
    <div>
      <Badge tone="danger">Not usable: {(c.reason ?? 'unknown').replace(/_/g, ' ')}</Badge>
      <p className="small">{REASONS[c.reason ?? ''] ?? 'Dense retrieval is skipped until this is resolved.'}</p>
      {c.indexed_with && c.current && (
        <details>
          <summary className="small">Compare fingerprints</summary>
          <div className="fingerprint-compare">
            <div>
              <div className="small muted">Indexed with</div>
              <pre className="code-block">{JSON.stringify(c.indexed_with, null, 2)}</pre>
            </div>
            <div>
              <div className="small muted">Current configuration</div>
              <pre className="code-block">{JSON.stringify(c.current, null, 2)}</pre>
            </div>
          </div>
        </details>
      )}
    </div>
  );
}

export function IndexPage() {
  const st = useResource(() => api.indexStatus(), []);
  const providers = useResource(() => api.providers().catch(() => []), []);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [reindexProvider, setReindexProvider] = useState('');

  const building = st.data?.building ?? [];
  // After starting a reindex the building index appears only once the provider has been probed,
  // so keep polling for a while even if `building` is still empty.
  const watchUntil = useRef(0);
  const [watching, setWatching] = useState(false);
  usePolling(
    async () => {
      const r = await st.reload(true);
      if (watching && Date.now() > watchUntil.current && !(r?.building.length)) setWatching(false);
    },
    2000,
    building.length > 0 || watching,
  );

  const run = async (key: string, fn: () => Promise<string | void>, confirmText?: string) => {
    if (confirmText && !window.confirm(confirmText)) return;
    setBusy(key);
    setError(null);
    setNotice(null);
    try {
      const msg = await fn();
      if (msg) setNotice(msg);
      await st.reload(true);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(null);
    }
  };

  if (st.loading && !st.data) return <Spinner label="Loading index status…" />;
  if (st.error && !st.data) return <ErrorMessage error={st.error} title="Could not load index status" />;
  const s = st.data!;
  const embedProviders = (providers.data ?? []).filter((p) => p.capabilities.includes('embedding'));

  return (
    <div className="stack">
      <Card title="Embedding index" subtitle="Dense retrieval uses one active vector index, always queried with the provider it was built with.">
        <Alert tone="info">
          Changing the <strong>text_embedding</strong> route does not take effect until you reindex: the active index stays pinned to its own provider so
          vectors from different embedding spaces are never mixed. A reindex builds a new index in the background, validates it, then switches; the
          previous index is kept for rollback until you confirm the migration.
        </Alert>
        {s.reindex_recommended && (
          <Alert tone="warning" title="Reindex recommended">
            {!s.compatibility.ok
              ? 'The active index is not usable with its pinned provider.'
              : `The text-embedding route's primary provider (${s.route_primary_provider ?? 'none'}) differs from the provider the active index was built with.`}
          </Alert>
        )}
        <div className="stats-row">
          <Stat label="Total chunks" value={fmtInt(s.total_chunks)} />
          <Stat label="Indexed chunks" value={fmtInt(s.indexed_chunks)} />
          <Stat label="Vectors stored" value={fmtInt(s.vector_count)} />
          <Stat label="Missing vectors" value={fmtInt(s.missing_vectors)} />
          <Stat
            label="Route matches active"
            value={s.route_matches_active === undefined ? '—' : s.route_matches_active ? 'yes' : 'no'}
            hint={`route primary: ${s.route_primary_provider ?? 'none'}`}
          />
        </div>
        <ErrorMessage error={error} />
        {notice && (
          <Alert tone="success" role="status">
            {notice}
          </Alert>
        )}
      </Card>

      <Card title="Active index" actions={s.active ? <StatusBadge status={s.active.status} /> : undefined}>
        {s.active ? (
          <>
            <Compat c={s.compatibility} />
            <SpecView idx={s.active} />
            <div className="btn-row">
              {(s.missing_vectors ?? 0) > 0 && (
                <button
                  type="button"
                  className="btn"
                  disabled={!!busy || !s.compatibility.ok}
                  onClick={() =>
                    void run('backfill', async () => {
                      const r = await api.backfill();
                      return r.reason ? `Backfill skipped: ${r.reason}` : `Backfilled ${r.backfilled} chunk vector(s).`;
                    })
                  }
                >
                  {busy === 'backfill' ? 'Backfilling…' : `Backfill ${fmtInt(s.missing_vectors)} missing vectors`}
                </button>
              )}
              {!s.compatibility.ok && s.compatibility.reason === 'provider_config_changed' && (
                <button
                  type="button"
                  className="btn btn-danger"
                  disabled={!!busy}
                  onClick={() =>
                    void run(
                      'compat',
                      async () => {
                        await api.confirmCompatibility();
                        return 'Compatibility confirmed; the index is pinned to the current configuration.';
                      },
                      'Confirm compatibility?\n\nYou are asserting that the changed endpoint/configuration produces exactly the same vector space as before (same model, same dimensions, same normalization). If this is wrong, dense retrieval will silently return poor results. Reindexing is the safe option.',
                    )
                  }
                >
                  {busy === 'compat' ? 'Confirming…' : 'Confirm compatibility…'}
                </button>
              )}
            </div>
            {!s.compatibility.ok && s.compatibility.reason === 'provider_config_changed' && (
              <p className="small text-warning">
                Confirming compatibility asserts that the changed endpoint produces the <em>same</em> vector space. Only do this if you know the model and
                its output are identical (e.g. the same model moved to a new host). Otherwise reindex.
              </p>
            )}
          </>
        ) : (
          <EmptyState title="No active index">{REASONS.no_active_index}</EmptyState>
        )}
      </Card>

      {building.length > 0 && (
        <Card title="Reindex in progress" subtitle="Refreshing every 2 seconds">
          {building.map((b) => {
            const pct = b.progress_total ? b.progress_done / b.progress_total : null;
            return (
              <div key={b.id} className="building">
                <div className="chip-row small">
                  <StatusBadge status={b.status} /> <code>{b.id}</code> · provider <code>{b.spec.provider_id}</code>
                  {b.spec.model && <span>· {b.spec.model}</span>}
                  {b.spec.dimensions && <span>· {b.spec.dimensions} dims</span>}
                </div>
                <ProgressBar value={pct} indeterminate={pct === null} label="Reindex progress" />
                <div className="small muted">
                  {fmtInt(b.progress_done)} / {fmtInt(b.progress_total)} chunks
                  {b.failed_batches > 0 && <span className="text-danger"> · {b.failed_batches} failed batch(es)</span>}
                </div>
                {b.error && <div className="small text-danger">{b.error}</div>}
              </div>
            );
          })}
        </Card>
      )}

      {s.previous && (
        <Card title="Previous index (kept for rollback)" actions={<StatusBadge status={s.previous.status} />}>
          <SpecView idx={s.previous} />
          <div className="btn-row">
            <button
              type="button"
              className="btn"
              disabled={!!busy}
              onClick={() =>
                void run(
                  'rollback',
                  async () => {
                    await api.rollbackIndex();
                    return 'Rolled back to the previous index.';
                  },
                  'Roll back to the previous index? The current index will be marked rolled back.',
                )
              }
            >
              {busy === 'rollback' ? 'Rolling back…' : 'Roll back'}
            </button>
            <button
              type="button"
              className="btn btn-primary"
              disabled={!!busy}
              onClick={() =>
                void run(
                  'confirm',
                  async () => {
                    await api.confirmMigration();
                    return 'Migration confirmed; the previous index was deleted.';
                  },
                  'Confirm the migration? The previous index and its vectors are deleted and rollback will no longer be possible.',
                )
              }
            >
              {busy === 'confirm' ? 'Confirming…' : 'Confirm migration'}
            </button>
          </div>
        </Card>
      )}

      <Card title="Reindex">
        <p className="small">
          Builds a new index for all {fmtInt(s.total_chunks)} chunks with the chosen provider (default: the primary of the{' '}
          <Link to="/settings/connect">document search setting in Connect AI</Link>). Search keeps using the current index until the new one is validated.
        </p>
        <div className="form-inline">
          <div className="field">
            <label htmlFor="reindex-provider" className="field-label">
              Embedding provider
            </label>
            <select id="reindex-provider" value={reindexProvider} onChange={(e) => setReindexProvider(e.target.value)}>
              <option value="">Route primary ({s.route_primary_provider ?? 'none configured'})</option>
              {embedProviders.map((p) => (
                <option key={p.id} value={p.id} disabled={!p.enabled}>
                  {p.name} ({p.id}){p.enabled ? '' : ' — disabled'}
                </option>
              ))}
            </select>
          </div>
          <button
            type="button"
            className="btn btn-primary"
            disabled={!!busy || building.length > 0}
            onClick={() =>
              void run(
                'reindex',
                async () => {
                  const r = await api.reindex(reindexProvider || null);
                  watchUntil.current = Date.now() + 20000;
                  setWatching(true);
                  return `Reindex started (job ${r.job}).`;
                },
                'Start reindexing? Every chunk is re-embedded; with a remote provider this sends all document text to it and may incur cost.',
              )
            }
          >
            {busy === 'reindex' ? 'Starting…' : building.length > 0 ? 'Reindex running…' : 'Reindex'}
          </button>
        </div>
        <p className="small muted">If a reindex fails validation it is discarded and the current index stays active.</p>
      </Card>

      {s.recent_failed && s.recent_failed.length > 0 && (
        <Card title="Recent failed reindexes" subtitle="Builds that failed or did not pass validation. The active index was not changed.">
          <div className="table-wrap">
            <table className="table table-compact">
              <thead>
                <tr>
                  <th scope="col">Index</th>
                  <th scope="col">Provider / model</th>
                  <th scope="col" className="num">Progress</th>
                  <th scope="col">Error</th>
                  <th scope="col">Created</th>
                </tr>
              </thead>
              <tbody>
                {s.recent_failed.map((f) => (
                  <tr key={f.id}>
                    <td className="mono small">{f.id}</td>
                    <td className="small">
                      <code>{f.spec.provider_id}</code>
                      {f.spec.model ? ` · ${f.spec.model}` : ''}
                      {f.spec.dimensions ? ` · ${f.spec.dimensions} dims` : ''}
                    </td>
                    <td className="num small">
                      {fmtInt(f.progress_done)} / {fmtInt(f.progress_total)}
                      {f.failed_batches > 0 && <div className="text-danger">{f.failed_batches} failed batch(es)</div>}
                    </td>
                    <td className="small break text-danger">{f.error ?? '—'}</td>
                    <td className="small nowrap">{fmtDate(f.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
