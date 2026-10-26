/** Quick connect API (backend/app/services/quickconnect.py, /api/quick-connect/*). */

export interface QuickPreset {
  id: string;
  name: string;
  kind: 'llm' | 'search';
  needs_key: boolean;
  key_url: string;
  key_hint: string;
  base_url: string;
  note: string;
  connected: boolean;
}

export interface DiscoveredModel {
  id: string;
  capabilities: string[];
  /** true = responded, false = not usable with this key, null = not tested. */
  available: boolean | null;
  /** Returned valid structured output (needed by the research agent); null = not tested. */
  json_ok: boolean | null;
  dims?: number;
  context_length?: number | null;
  detail?: string;
}

export type ModelRole = 'synthesis' | 'planning' | 'vision' | 'reranking' | 'ocr' | 'fallback' | 'embedding';
/** Model id per role; 'reranking' and 'ocr' also accept 'local' (tools on this computer) or 'off'. */
export type Assignments = Record<ModelRole, string | null>;

export interface LlmDiscovery {
  preset: string;
  kind: 'llm';
  base_url?: string;
  models: DiscoveredModel[];
  tested: number;
  recommended: Partial<Record<ModelRole, string | null>>;
}

export interface SearchDiscovery {
  preset: string;
  kind: 'search';
  ok: boolean;
  detail: string;
  models: [];
  recommended: { web_search?: string };
}

export type DiscoverResult = LlmDiscovery | SearchDiscovery;

export interface ProbeResult {
  available: boolean | null;
  json_ok?: boolean | null;
  dims?: number;
  detail?: string;
}

/** NDJSON events from POST /api/quick-connect/discover/stream. */
export type DiscoverEvent =
  | { type: 'models'; preset: string; base_url?: string; models: DiscoveredModel[]; to_test: string[] }
  | { type: 'probe'; id: string; result: ProbeResult; done: number; total: number; recommended: Partial<Record<ModelRole, string | null>> }
  | { type: 'done'; result: DiscoverResult }
  | { type: 'error'; detail: string };

export interface QuickDiscoverPayload {
  preset: string;
  /** Omit to reuse the key of an already-connected service. */
  api_key?: string;
  base_url?: string;
}

export interface QuickApplyPayload extends QuickDiscoverPayload {
  assignments: Partial<Assignments>;
}

export interface QuickApplyResult {
  routes: Record<string, string[]>;
  notes?: string[];
}
