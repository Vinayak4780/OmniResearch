import type { Assignments, DiscoveredModel, ModelRole, QuickApplyPayload, QuickPreset } from '../../types/quickconnect';

/** One-line, non-technical descriptions for the service cards. */
export const PRESET_BLURB: Record<string, string> = {
  nvidia: 'Free starter credits and many open models (Nemotron, Llama, Qwen, DeepSeek).',
  openrouter: 'One key for hundreds of models from many AI labs.',
  openai: 'GPT models from OpenAI.',
  anthropic: 'Claude models from Anthropic.',
  groq: 'Very fast responses from popular open models.',
  together: 'Popular open models hosted by Together AI.',
  deepseek: "DeepSeek's own reasoning and chat models.",
  mistral: "Mistral's models, hosted in Europe.",
  modelscope: 'Qwen and other models hosted on ModelScope.',
  ollama: 'Models running on this computer. Nothing leaves your machine.',
  custom: 'Your own server: vLLM, SGLang, LM Studio, LiteLLM or a company gateway (any OpenAI-compatible address).',
  tavily: 'Web search built for AI research. Also reads web pages.',
  brave: 'Independent web search from Brave.',
  exa: 'Web search that finds pages by meaning, not just keywords.',
};

export interface RoleDef {
  role: ModelRole;
  label: string;
  hint: string;
  /** Which discovered models can fill the role. */
  capability: 'chat' | 'vision' | 'embedding';
  /** Label of the "empty" choice, or undefined when a model is required. */
  noneLabel?: string;
  /** Non-model choices (e.g. 'local', 'off') shown before the models. */
  specials?: { value: string; label: string }[];
  /** Choice used when no model is recommended (e.g. 'local'). */
  defaultSpecial?: string;
}

export const LOCAL = 'local';
export const OFF = 'off';

export const ROLES: RoleDef[] = [
  { role: 'synthesis', label: 'Writes reports & checks claims', hint: 'Your strongest model. It writes the final report and double-checks each claim against the sources.', capability: 'chat' },
  { role: 'planning', label: 'Plans research', hint: 'Breaks your question into searches. A faster model works well here.', capability: 'chat', noneLabel: 'Same as the report writer' },
  { role: 'vision', label: 'Reads charts & figures', hint: 'Looks at figures in your documents. Only models that can see images are listed.', capability: 'vision', noneLabel: 'None (skip figures)' },
  {
    role: 'reranking',
    label: 'Ranks search results',
    hint: 'Uses your connected model to put the most relevant passages first. About 1–2 s per search.',
    capability: 'chat',
    specials: [
      { value: LOCAL, label: 'Use tools on this computer' },
      { value: OFF, label: 'Off' },
    ],
    defaultSpecial: LOCAL,
  },
  {
    role: 'ocr',
    label: 'Reads scanned pages',
    hint: 'Reads pages that have no text layer. With a connected model, images of those scanned pages are sent to the service.',
    capability: 'vision',
    specials: [
      { value: LOCAL, label: 'Use PaddleOCR on this computer' },
      { value: OFF, label: 'Off' },
    ],
    defaultSpecial: LOCAL,
  },
  { role: 'fallback', label: 'Backup model', hint: 'Optional. Used automatically if the main models are busy or fail.', capability: 'chat', noneLabel: 'None' },
  {
    role: 'embedding',
    label: 'Document search embeddings',
    hint: 'Turns your documents into searchable vectors.',
    capability: 'embedding',
    noneLabel: 'Keep local (private, recommended)',
  },
];

const ACRONYMS = new Set(['vl', 'vlm', 'gpt', 'glm', 'moe', 'ai', 'ocr']);

/** "meta/llama-3.3-70b-instruct" → "Llama 3.3 70B Instruct"; "…-super-vl-120b-a12b" → "Super VL 120B A12B". */
export function prettyModel(id: string | null | undefined): string {
  if (!id) return '—';
  const last = id.split('/').pop() || id;
  return last
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((w) =>
      /^a?\d+(\.\d+)?b$/i.test(w) || ACRONYMS.has(w.toLowerCase()) ? w.toUpperCase() : /^[a-z]/.test(w) ? w[0].toUpperCase() + w.slice(1) : w,
    )
    .join(' ');
}

export function modelsFor(models: DiscoveredModel[], capability: RoleDef['capability']): DiscoveredModel[] {
  return models.filter((m) => m.capabilities.includes(capability));
}

export type OptionState = 'tested' | 'untested' | 'unavailable';

export function optionState(m: DiscoveredModel): OptionState {
  if (m.available === false) return 'unavailable';
  return m.available === true ? 'tested' : 'untested';
}

export interface ModelOption {
  id: string;
  label: string;
  state: OptionState;
  disabled: boolean;
  recommended: boolean;
  /** Why it is unavailable / caveats (e.g. "did not return valid JSON"). */
  note?: string;
}

/** Options for a role: usable (tested) first, then untested, then unavailable (disabled); filtered by `query`. */
export function roleOptions(
  models: DiscoveredModel[],
  def: RoleDef,
  recommended: string | null | undefined,
  query = '',
  keep?: string | null,
  testing: ReadonlySet<string> = new Set(),
): ModelOption[] {
  const q = query.trim().toLowerCase();
  const rank: Record<OptionState, number> = { tested: 0, untested: 1, unavailable: 2 };
  return modelsFor(models, def.capability)
    .map((m, i) => ({ m, i }))
    .filter(({ m }) => !q || m.id.toLowerCase().includes(q) || m.id === keep)
    .sort((a, b) => rank[optionState(a.m)] - rank[optionState(b.m)] || a.i - b.i)
    .map(({ m }) => {
      const state = optionState(m);
      let note = m.detail;
      if (state === 'tested' && def.capability === 'chat' && m.json_ok === false) note = 'answers, but not in the structured format research needs';
      if (state === 'tested' && def.capability === 'embedding' && m.dims) note = `${m.dims} dimensions`;
      const mark = state === 'tested' ? '✓' : state === 'untested' ? '○' : '✗';
      const suffix = state === 'untested' ? (testing.has(m.id) ? ' (testing…)' : ' (not tested)') : note ? ` — ${note}` : '';
      return {
        id: m.id,
        label: `${mark} ${m.id}${m.id === recommended ? ' (recommended)' : ''}${suffix}`,
        state,
        disabled: state === 'unavailable',
        recommended: m.id === recommended,
        note,
      };
    });
}

/** Initial choices: the recommendations, except embeddings which stay local unless the user opts in. */
/** Services whose models can rank results / read scans (OpenAI-compatible); others always use the local tools. */
export function supportsModelTools(preset: Pick<QuickPreset, 'id'>): boolean {
  return preset.id !== 'anthropic';
}

export function buildApplyPayload(preset: QuickPreset, a: Assignments, apiKey: string, baseUrl: string): QuickApplyPayload {
  const p: QuickApplyPayload = {
    preset: preset.id,
    assignments: {
      synthesis: a.synthesis,
      planning: a.planning || a.synthesis,
      vision: a.vision || null,
      reranking: a.reranking || LOCAL,
      ocr: a.ocr || LOCAL,
      fallback: a.fallback || null,
      embedding: a.embedding || null,
    },
  };
  if (apiKey.trim()) p.api_key = apiKey.trim();
  if (baseUrl.trim()) p.base_url = baseUrl.trim();
  return p;
}

/** Plain-language summary lines after connecting. */
export function summaryLines(a: Assignments): string[] {
  const out = [`Reports and claim checks: ${prettyModel(a.synthesis)}`, `Planning: ${prettyModel(a.planning || a.synthesis)}`];
  out.push(a.vision ? `Charts & figures: ${prettyModel(a.vision)}` : 'Charts & figures: not used');
  const tool = (v: string | null, local: string) => (!v || v === LOCAL ? local : v === OFF ? 'off' : prettyModel(v));
  out.push(`Result ranking: ${tool(a.reranking, 'tools on this computer')}`);
  out.push(`Scanned pages: ${tool(a.ocr, 'PaddleOCR on this computer')}`);
  if (a.fallback) out.push(`Backup: ${prettyModel(a.fallback)}`);
  out.push(a.embedding ? `Document search: ${prettyModel(a.embedding)} (cloud)` : 'Document search: stays on this computer');
  return out;
}
