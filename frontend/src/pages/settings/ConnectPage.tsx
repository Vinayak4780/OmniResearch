import { useEffect, useRef, useState } from 'react';
import { CurrentlyUsing, currentlyUsing, type ToggleAction } from '../../components/connect/CurrentlyUsing';
import { PRESET_BLURB } from '../../components/connect/connectLogic';
import { QuickConnectFlow } from '../../components/connect/QuickConnectFlow';
import { useHealth } from '../../components/layout/HealthContext';
import { Badge, Card, ErrorMessage, Spinner } from '../../components/ui';
import { useResource } from '../../hooks/useResource';
import { api } from '../../services/api';
import type { QuickPreset } from '../../types/quickconnect';

interface Active {
  preset: QuickPreset;
  mode: 'new' | 'reuse';
}

function ServiceCard({ p, active, onStart }: { p: QuickPreset; active: boolean; onStart: (mode: 'new' | 'reuse') => void }) {
  return (
    <li className={`service-card ${active ? 'service-card-active' : ''} ${p.connected ? 'service-card-connected' : ''}`}>
      <div className="service-head">
        <span className="service-name">{p.name}</span>
        {p.connected && <Badge tone="success">✓ Connected</Badge>}
        {p.id === 'custom' ? <Badge>Your own server</Badge> : !p.needs_key && <Badge>No key needed</Badge>}
      </div>
      <p className="small service-blurb">{PRESET_BLURB[p.id] ?? p.note}</p>
      <div className="service-actions">
        {p.connected ? (
          <>
            <button type="button" className="btn btn-sm btn-primary" onClick={() => onStart('reuse')} aria-label={`${p.kind === 'search' ? 'Re-test' : 'Change models for'} ${p.name}`}>
              {p.kind === 'search' ? 'Re-test' : 'Change models'}
            </button>
            {p.needs_key && (
              <button type="button" className="btn btn-sm" onClick={() => onStart('new')} aria-label={`Replace key for ${p.name}`}>
                Replace key
              </button>
            )}
          </>
        ) : (
          <button type="button" className="btn btn-sm btn-primary" onClick={() => onStart('new')} aria-label={`Connect ${p.name}`}>
            Connect
          </button>
        )}
        {p.key_url && !p.connected && (
          <a className="small" href={p.key_url} target="_blank" rel="noopener noreferrer">
            Get a key ↗<span className="sr-only"> from {p.name} (opens in a new tab)</span>
          </a>
        )}
      </div>
    </li>
  );
}

function ServiceGroup({ title, intro, presets, active, embeddingsActive, onStart, onClose, onConnected }: {
  title: string;
  intro: string;
  presets: QuickPreset[];
  active: Active | null;
  embeddingsActive: boolean;
  onStart: (p: QuickPreset, mode: 'new' | 'reuse') => void;
  onClose: () => void;
  onConnected: () => void;
}) {
  const flowRef = useRef<HTMLDivElement>(null);
  const mine = active && presets.some((p) => p.id === active.preset.id) ? active : null;
  useEffect(() => {
    if (mine) flowRef.current?.scrollIntoView?.({ behavior: 'smooth', block: 'nearest' });
  }, [mine?.preset.id, mine?.mode]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <section className="service-group" aria-label={title}>
      <h2 className="service-group-title">{title}</h2>
      <p className="small muted">{intro}</p>
      <ul className="service-grid">
        {presets.map((p) => (
          <ServiceCard key={p.id} p={p} active={mine?.preset.id === p.id} onStart={(mode) => onStart(p, mode)} />
        ))}
      </ul>
      <div ref={flowRef}>
        {mine && <QuickConnectFlow key={`${mine.preset.id}-${mine.mode}`} preset={mine.preset} mode={mine.mode} embeddingsActive={embeddingsActive} onClose={onClose} onConnected={onConnected} />}
      </div>
    </section>
  );
}

export function ConnectPage() {
  const presets = useResource(() => api.quickPresets(), []);
  const routes = useResource(() => api.routes(), []);
  const providers = useResource(() => api.providers(), []);
  const catalog = useResource(() => api.catalog(), []);
  const { reload: reloadHealth } = useHealth();
  const [active, setActive] = useState<Active | null>(null);

  const refresh = () => {
    void presets.reload(true);
    void routes.reload(true);
    void providers.reload(true);
    void reloadHealth();
  };

  const using = routes.data && providers.data ? currentlyUsing(routes.data, providers.data, catalog.data ?? [], presets.data ?? []) : undefined;
  const embeddingsActive = using ? !!using.find((l) => l.task === 'text_embedding')?.value : true;
  const localAdapters = new Set((catalog.data ?? []).filter((d) => !d.requires_network).map((d) => d.adapter));
  const localTools = (providers.data ?? []).filter((p) => localAdapters.has(p.adapter));
  const [toggling, setToggling] = useState(false);
  const [toggleError, setToggleError] = useState<unknown>(null);
  /** Turn providers on/off only when the user asks; nothing is ever enabled automatically. */
  const toggle = async (a: ToggleAction) => {
    setToggling(true);
    setToggleError(null);
    try {
      for (const id of a.ids) await api.updateProvider(id, { enabled: a.enable });
    } catch (e) {
      setToggleError(e);
    } finally {
      setToggling(false);
      refresh();
    }
  };
  // The custom endpoint card goes last among the AI services.
  const llm = (presets.data ?? []).filter((p) => p.kind === 'llm').sort((a, b) => Number(a.id === 'custom') - Number(b.id === 'custom'));
  const search = (presets.data ?? []).filter((p) => p.kind === 'search');
  const group = {
    active,
    embeddingsActive,
    onStart: (p: QuickPreset, mode: 'new' | 'reuse') => setActive({ preset: p, mode }),
    onClose: () => setActive(null),
    onConnected: refresh,
  };

  return (
    <div className="stack connect-page">
      <Card title="Connect AI" subtitle="Pick a service, paste its API key, and OmniResearch finds the models your key can use and sets everything up.">
        {presets.error ? (
          <ErrorMessage error={presets.error} title="Could not load the list of services" />
        ) : presets.loading && !presets.data ? (
          <Spinner label="Loading services…" />
        ) : (
          <>
            <ServiceGroup title="AI models" intro="Needed for research. One service is enough; you can connect more later." presets={llm} {...group} />
            <ServiceGroup title="Web search" intro="Optional. Web research turns on automatically once a search service is connected." presets={search} {...group} />
          </>
        )}
      </Card>

      <Card title="Currently using" subtitle="What each part of the research pipeline runs on right now.">
        {routes.error || providers.error ? (
          <ErrorMessage error={routes.error || providers.error} title="Could not load the current setup" />
        ) : !routes.data || !providers.data ? (
          <Spinner />
        ) : (
          <>
            <ErrorMessage error={toggleError} title="Could not change that setting" />
            <CurrentlyUsing lines={using ?? []} localTools={localTools} busy={toggling} onToggle={(a) => void toggle(a)} />
          </>
        )}
      </Card>
    </div>
  );
}
