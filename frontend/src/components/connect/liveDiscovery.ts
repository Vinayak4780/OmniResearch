/** State of a streaming discovery, built from NDJSON events (pure, so it is easy to test). */
import type { Assignments, DiscoverEvent, DiscoveredModel, ModelRole, SearchDiscovery } from '../../types/quickconnect';

export interface LiveDiscovery {
  models: DiscoveredModel[];
  baseUrl?: string;
  /** Model ids the server is testing (in its order). */
  toTest: string[];
  /** Ids whose test has finished. */
  tested: string[];
  done: number;
  total: number;
  recommended: Partial<Record<ModelRole, string | null>>;
  /** The stream ended (done, error, or stopped by the user). */
  finished: boolean;
  /** The user stopped testing early. */
  stopped: boolean;
  error?: string;
  search?: SearchDiscovery;
}

export const EMPTY_DISCOVERY: LiveDiscovery = { models: [], toTest: [], tested: [], done: 0, total: 0, recommended: {}, finished: false, stopped: false };

export function applyEvent(s: LiveDiscovery, e: DiscoverEvent): LiveDiscovery {
  switch (e.type) {
    case 'models':
      return { ...s, models: e.models.map((m) => ({ ...m })), baseUrl: e.base_url, toTest: e.to_test, total: e.to_test.length, done: 0, tested: [] };
    case 'probe':
      return {
        ...s,
        models: s.models.map((m) => (m.id === e.id ? { ...m, ...e.result } : m)),
        tested: s.tested.includes(e.id) ? s.tested : [...s.tested, e.id],
        done: e.done,
        total: e.total,
        recommended: e.recommended ?? s.recommended,
      };
    case 'done':
      if (e.result.kind === 'search') return { ...s, finished: true, search: e.result };
      return {
        ...s,
        models: e.result.models,
        baseUrl: e.result.base_url ?? s.baseUrl,
        recommended: e.result.recommended,
        tested: s.tested.length ? s.tested : s.toTest,
        done: s.total || e.result.tested,
        total: s.total || e.result.tested,
        finished: true,
      };
    case 'error':
      return { ...s, finished: true, error: e.detail };
    default:
      return s;
  }
}

/** Ids still being tested (empty once finished or stopped). */
export function testingIds(s: LiveDiscovery): Set<string> {
  if (s.finished) return new Set();
  const done = new Set(s.tested);
  return new Set(s.toTest.filter((id) => !done.has(id)));
}

export const CHAT_ROLES: ModelRole[] = ['synthesis', 'planning', 'vision', 'fallback'];

/**
 * Fold new recommendations into the current choices without overriding anything the user picked.
 * Embeddings are never auto-filled (local by default).
 */
/** Roles that can use the connected model, the tools on this computer ('local'), or be 'off'. */
export const TOOL_ROLES: ModelRole[] = ['reranking', 'ocr'];

export function mergeRecommendations(
  current: Assignments,
  touched: ReadonlySet<ModelRole>,
  rec: Partial<Record<ModelRole, string | null>>,
  models: DiscoveredModel[],
  /** false for services whose models can't rank or read scans (they always use the local tools). */
  modelTools = true,
): Assignments {
  const usable = (id: string | null | undefined) => (id && models.some((m) => m.id === id && m.available !== false) ? id : null);
  const next = { ...current };
  for (const role of CHAT_ROLES) {
    if (touched.has(role)) continue;
    next[role] = usable(rec[role]);
  }
  for (const role of TOOL_ROLES) {
    if (touched.has(role)) continue;
    // automatic: the recommended connected model, else the tools on this computer
    next[role] = (modelTools && usable(rec[role])) || 'local';
  }
  // a model the user picked may have just failed its test
  for (const role of [...CHAT_ROLES, ...TOOL_ROLES, 'embedding'] as ModelRole[]) {
    if (next[role] && models.some((m) => m.id === next[role] && m.available === false)) {
      next[role] = role === 'synthesis' ? usable(rec.synthesis) : TOOL_ROLES.includes(role) ? 'local' : null;
    }
  }
  return next;
}
