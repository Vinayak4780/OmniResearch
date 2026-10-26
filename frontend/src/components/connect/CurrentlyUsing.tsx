import { Link } from 'react-router-dom';
import type { ProviderDescriptor, ProviderRecord, RouteRow, Task } from '../../types/api';
import type { QuickPreset } from '../../types/quickconnect';
import { Toggle } from '../ui';
import { prettyModel } from './connectLogic';

interface RoleLine {
  task: Task;
  label: string;
  capability: string;
  missing: string;
  /** What happens while the assigned provider is turned off. */
  effect: string;
}

const LINES: RoleLine[] = [
  { task: 'synthesis', label: 'Writes reports & checks claims', capability: 'chat', missing: 'Not connected — research cannot run yet', effect: 'research cannot run' },
  { task: 'planning', label: 'Plans research', capability: 'chat', missing: 'Not connected', effect: 'research cannot run' },
  { task: 'vision', label: 'Reads charts & figures', capability: 'vision', missing: 'Not used — figures are not inspected', effect: 'figures are not inspected' },
  { task: 'text_embedding', label: 'Document search', capability: 'embedding', missing: 'Keyword search only', effect: 'keyword search only' },
  { task: 'reranking', label: 'Ranks search results', capability: 'reranking', missing: 'Not used', effect: 'results are not re-ranked' },
  { task: 'ocr', label: 'Reads scanned pages', capability: 'ocr', missing: 'Not set up — scanned pages are skipped', effect: 'scanned pages are skipped' },
  { task: 'web_search', label: 'Web search', capability: 'web_search', missing: 'Not set up — connect Tavily, Brave or Exa', effect: 'no web results' },
];

/** Plain descriptions of the built-in local tools. */
export const LOCAL_TOOL_LABEL: Record<string, string> = {
  fastembed: 'Meaning-based document search (embeddings)',
  local_hashing: 'Simple keyword-style document search (embeddings)',
  fastembed_rerank: 'Result ranking (cross-encoder)',
  local_lexical_rerank: 'Result ranking (keyword-based)',
  paddleocr_local: 'Reads scanned pages (PaddleOCR)',
  builtin_web_extractor: 'Reads web pages found by web search',
};

export interface ToggleAction {
  ids: string[];
  enable: boolean;
}

export interface UsingLine {
  task: Task;
  label: string;
  value: string | null;
  where: string | null;
  backups: number;
  missing: string;
  /** Providers are assigned but all of them are turned off. */
  disabled: boolean;
  effect: string;
  /** Turn on (assigned but off) or turn off (running locally). Cloud models are changed in Connect AI instead. */
  toggle?: ToggleAction;
}

function normUrl(u: unknown): string {
  return String(u ?? '').trim().replace(/\/+$/, '').toLowerCase();
}

/** Effective model per role, from routes (effective_provider_ids) and provider records. */
export function currentlyUsing(routes: RouteRow[], providers: ProviderRecord[], catalog: ProviderDescriptor[], presets: QuickPreset[] = []): UsingLine[] {
  const byId = new Map(providers.map((p) => [p.id, p]));
  const local = new Set(catalog.filter((d) => !d.requires_network).map((d) => d.adapter));
  const service = (p: ProviderRecord): string => {
    const url = normUrl(p.config?.base_url);
    const preset = url ? presets.find((x) => x.base_url && normUrl(x.base_url) === url) : presets.find((x) => x.kind === 'search' && x.id === p.adapter);
    if (preset) return preset.name;
    if (url) {
      try {
        return new URL(String(p.config?.base_url)).host;
      } catch {
        /* fall through */
      }
    }
    return p.name.includes(' · ') ? p.name.split(' · ')[0] : p.name;
  };
  return LINES.map((l) => {
    const r = routes.find((x) => x.task === l.task);
    const assigned = (r?.effective_provider_ids ?? []).filter((id) => byId.has(id));
    const ids = assigned.filter((id) => byId.get(id)?.enabled);
    const p = ids.length ? byId.get(ids[0]) : undefined;
    if (!p) {
      return {
        task: l.task,
        label: l.label,
        value: null,
        where: null,
        backups: 0,
        missing: l.missing,
        disabled: assigned.length > 0,
        effect: l.effect,
        toggle: assigned.length ? { ids: assigned, enable: true } : undefined,
      };
    }
    const model = p.models[l.capability] ?? null;
    const isLocal = local.has(p.adapter);
    const localIds = ids.filter((id) => local.has(byId.get(id)!.adapter));
    return {
      task: l.task,
      label: l.label,
      value: model ? prettyModel(model) : p.name,
      where: isLocal ? 'on this computer' : service(p),
      backups: ids.length - 1,
      missing: l.missing,
      disabled: false,
      effect: l.effect,
      toggle: isLocal && localIds.length ? { ids: localIds, enable: false } : undefined,
    };
  });
}

interface Props {
  lines: UsingLine[];
  /** Local tools (adapters that run on this computer), shown with on/off switches. */
  localTools: ProviderRecord[];
  busy: boolean;
  onToggle: (action: ToggleAction) => void;
}

export function CurrentlyUsing({ lines, localTools, busy, onToggle }: Props) {
  return (
    <>
      <dl className="using">
        {lines.map((l) => (
          <div key={l.task} className={`using-row ${l.value ? '' : 'using-missing'}`} data-task={l.task}>
            <dt>{l.label}</dt>
            <dd>
              <span className="using-value">
                {l.value ? (
                  <>
                    <strong>{l.value}</strong>
                    {l.where && <span className="muted"> · {l.where}</span>}
                    {l.backups > 0 && <span className="muted small"> (+{l.backups} backup{l.backups > 1 ? 's' : ''})</span>}
                  </>
                ) : (
                  <span className={l.task === 'synthesis' ? 'text-danger' : 'muted'}>{l.disabled ? `Turned off — ${l.effect}` : l.missing}</span>
                )}
              </span>
              {l.toggle && (
                <button
                  type="button"
                  className="btn btn-sm btn-ghost using-toggle"
                  disabled={busy}
                  aria-label={`${l.toggle.enable ? 'Turn on' : 'Turn off'}: ${l.label}`}
                  onClick={() => onToggle(l.toggle!)}
                >
                  {l.toggle.enable ? 'Turn on' : 'Turn off'}
                </button>
              )}
            </dd>
          </div>
        ))}
      </dl>
      {localTools.length > 0 && (
        <div className="local-tools">
          <h3 className="local-tools-title">Tools on this computer</h3>
          <p className="small muted">
            These run locally and never send your documents anywhere. Turn them off to save memory or disk space.{" "}
            <Link to="/settings/local-models">Install or change local models</Link>
          </p>
          <ul className="local-tool-list">
            {localTools.map((p) => (
              <li key={p.id} className="local-tool">
                <Toggle
                  checked={p.enabled}
                  disabled={busy}
                  onChange={(v) => onToggle({ ids: [p.id], enable: v })}
                  label={`${LOCAL_TOOL_LABEL[p.adapter] ?? p.name}${p.enabled ? '' : ' (off)'}`}
                />
                <span className="small muted mono">{Object.values(p.models).find(Boolean) ?? p.id}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      <p className="using-foot small muted">
        Call statistics and recent errors are under <Link to="/settings/health">Advanced → Provider health</Link>.
      </p>
    </>
  );
}
