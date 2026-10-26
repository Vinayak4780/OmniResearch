/**
 * Typed client for the OmniResearch API. Every request goes to `/api/...`, carries the admin token
 * (when one is stored) and throws an {@link ApiError} with the HTTP status and the server's `detail`.
 */
import type {
  ChatAnswer,
  ChatMessage,
  Chunk,
  CollectionInfo,
  DocumentDetail,
  DocumentListItem,
  Evidence,
  HealthResponse,
  IndexInfo,
  IndexStatus,
  IngestionJob,
  MindMap,
  Preflight,
  PrivacyDestination,
  ProviderDescriptor,
  ProviderHealth,
  ProviderPatchPayload,
  ProviderRecord,
  ResearchEvent,
  ResearchJob,
  ResearchJobSummary,
  ResearchStartPayload,
  RetryResponse,
  RouteRow,
  UploadResponse,
} from '../types/api';
import type { DiscoverMode, DiscoverResponse, LinksResponse, NoteCreate, Notebook, NotebookOutput, OutputKind,NotebookGuide, NotebookMessage, NotebookNote, NotebookReport, NotebookSource, NotebookSummary, ReportStart } from '../types/notebooks';
import type { HfResult, LocalCatalog, LocalJob, LocalKind, LocalModelRequest, LocalUseResult } from '../types/localmodels';
import type { DiscoverResult, QuickApplyPayload, QuickApplyResult, QuickDiscoverPayload, QuickPreset } from '../types/quickconnect';
import { getAdminToken } from './storage';

export const API_BASE = '/api';

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;
  /** The parsed JSON error body (FastAPI `detail` plus any extra fields). */
  readonly body: Record<string, unknown> | null;

  constructor(status: number, detail: string, body: Record<string, unknown> | null = null) {
    super(detail);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
    this.body = body;
  }

  /** 401/403: the admin token is missing or wrong. */
  get isAuthError(): boolean {
    return this.status === 401 || this.status === 403;
  }
}

/** FastAPI `detail` may be a string or a list of validation errors ({loc, msg}). */
export function formatDetail(detail: unknown): string {
  if (detail == null) return '';
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        if (d && typeof d === 'object') {
          const o = d as { loc?: unknown[]; msg?: string };
          const loc = Array.isArray(o.loc) ? o.loc.filter((x) => x !== 'body').join('.') : '';
          return loc ? `${loc}: ${o.msg ?? ''}` : String(o.msg ?? JSON.stringify(d));
        }
        return String(d);
      })
      .join('; ');
  }
  if (typeof detail === 'object') return JSON.stringify(detail);
  return String(detail);
}

export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.detail || `HTTP ${err.status}`;
  if (err instanceof Error) return err.message;
  return String(err);
}

type Body = BodyInit | Record<string, unknown> | unknown[] | null | undefined;

export interface RequestOptions {
  method?: string;
  body?: Body;
  signal?: AbortSignal;
  /** Return the response body as text instead of JSON. */
  text?: boolean;
  query?: Record<string, string | number | boolean | null | undefined>;
}

function isRawBody(b: unknown): b is BodyInit {
  return (
    typeof b === 'string' ||
    (typeof FormData !== 'undefined' && b instanceof FormData) ||
    (typeof Blob !== 'undefined' && b instanceof Blob) ||
    (typeof URLSearchParams !== 'undefined' && b instanceof URLSearchParams) ||
    b instanceof ArrayBuffer
  );
}

export function buildUrl(path: string, query?: RequestOptions['query']): string {
  let url = path.startsWith('/api') ? path : `${API_BASE}${path.startsWith('/') ? '' : '/'}${path}`;
  if (query) {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(query)) {
      if (v !== undefined && v !== null && v !== '') qs.set(k, String(v));
    }
    const s = qs.toString();
    if (s) url += (url.includes('?') ? '&' : '?') + s;
  }
  return url;
}

/** Accept header plus the admin token when one is stored. */
export function baseHeaders(accept = 'application/json'): Record<string, string> {
  const headers: Record<string, string> = { Accept: accept };
  const token = getAdminToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  return headers;
}

/** Turn a non-OK response into an ApiError carrying the server's `detail`. */
export async function errorFromResponse(res: Response): Promise<ApiError> {
  let parsed: Record<string, unknown> | null = null;
  let detail = '';
  const raw = await res.text().catch(() => '');
  if (raw) {
    try {
      const j = JSON.parse(raw) as unknown;
      if (j && typeof j === 'object' && !Array.isArray(j)) {
        parsed = j as Record<string, unknown>;
        detail = formatDetail(parsed.detail);
      }
    } catch {
      detail = raw.length > 300 ? `${raw.slice(0, 300)}…` : raw;
    }
  }
  if (!detail) detail = res.statusText || `HTTP ${res.status}`;
  return new ApiError(res.status, detail, parsed);
}

export function networkError(e: unknown): ApiError {
  return new ApiError(0, `Network error: could not reach the OmniResearch server (${e instanceof Error ? e.message : String(e)})`);
}

export async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const headers = baseHeaders();

  let body: BodyInit | undefined;
  if (opts.body !== undefined && opts.body !== null) {
    if (isRawBody(opts.body)) {
      body = opts.body;
    } else {
      body = JSON.stringify(opts.body);
      headers['Content-Type'] = 'application/json';
    }
  }

  let res: Response;
  try {
    res = await fetch(buildUrl(path, opts.query), { method: opts.method ?? 'GET', headers, body, signal: opts.signal });
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e;
    throw networkError(e);
  }

  if (!res.ok) throw await errorFromResponse(res);

  if (res.status === 204) return undefined as T;
  if (opts.text) return (await res.text()) as T;
  const ct = res.headers.get('content-type') ?? '';
  if (!ct.includes('json')) {
    const t = await res.text();
    return (t ? (JSON.parse(t) as T) : (undefined as T));
  }
  return (await res.json()) as T;
}

const enc = encodeURIComponent;

// ------------------------------------------------------------------ static URLs (used in <img>/<a>)

export const urls = {
  pdf: (docId: string, page?: number) => `${API_BASE}/documents/${enc(docId)}/pdf${page ? `#page=${page}` : ''}`,
  pageImage: (docId: string, page: number, dpi?: number) =>
    `${API_BASE}/documents/${enc(docId)}/pages/${page}.png${dpi ? `?dpi=${dpi}` : ''}`,
  reportMarkdown: (jobId: string) => `${API_BASE}/research/${enc(jobId)}/report.md`,
};

// ------------------------------------------------------------------ endpoints

export const api = {
  health: () => request<HealthResponse>('/health'),

  // providers
  catalog: () => request<ProviderDescriptor[]>('/providers/catalog'),
  providers: () => request<ProviderRecord[]>('/providers'),
  updateProvider: (id: string, p: ProviderPatchPayload) =>
    request<ProviderRecord>(`/providers/${enc(id)}`, { method: 'PATCH', body: p as Record<string, unknown> }),
  providerHealth: () => request<ProviderHealth>('/provider-health'),
  routes: () => request<RouteRow[]>('/routes'),
  // notebooks
  notebooks: () => request<NotebookSummary[]>('/notebooks'),
  createNotebook: (title: string) => request<Notebook>('/notebooks', { method: 'POST', body: { title } }),
  notebook: (id: string) => request<Notebook>(`/notebooks/${enc(id)}`),
  renameNotebook: (id: string, title: string) => request<Notebook>(`/notebooks/${enc(id)}`, { method: 'PATCH', body: { title } }),
  deleteNotebook: (id: string) => request<void>(`/notebooks/${enc(id)}`, { method: 'DELETE' }),
  notebookSources: (id: string) => request<NotebookSource[]>(`/notebooks/${enc(id)}/sources`),
  /** First call runs one model call (~40 s); cached until the sources change. */
  notebookGuide: (id: string, signal?: AbortSignal) => request<NotebookGuide>(`/notebooks/${enc(id)}/guide`, { signal }),
  notebookMessages: (id: string) => request<NotebookMessage[]>(`/notebooks/${enc(id)}/messages`),
  /** `opts.web` also runs a quick web search (works with zero sources selected). */
  askNotebook: (id: string, message: string, sourceIds?: string[], signal?: AbortSignal, opts?: { web?: boolean }) =>
    request<NotebookMessage>(`/notebooks/${enc(id)}/messages`, {
      method: 'POST',
      body: { message, ...(sourceIds ? { source_ids: sourceIds } : {}), ...(opts?.web ? { web: true } : {}) },
      signal,
    }),
  /** Add 1–10 links: PDFs are downloaded, webpages are captured as a PDF snapshot. */
  addNotebookLinks: (id: string, links: string[]) => request<LinksResponse>(`/notebooks/${enc(id)}/sources/links`, { method: 'POST', body: { urls: links } }),
  /** Fast: one web search (~2 s). Deep: the AI plans several searches and picks the best (20–60 s). */
  discoverSources: (id: string, query: string, mode: DiscoverMode, signal?: AbortSignal) =>
    request<DiscoverResponse>(`/notebooks/${enc(id)}/discover`, { method: 'POST', body: { query, mode }, signal }),
  notebookOutputs: (id: string) => request<NotebookOutput[]>(`/notebooks/${enc(id)}/outputs`),
  /** Starts generation in the background; poll notebookOutputs() while any output is running. */
  createNotebookOutput: (id: string, kind: OutputKind, sourceIds?: string[]) =>
    request<NotebookOutput>(`/notebooks/${enc(id)}/outputs`, { method: 'POST', body: { kind, ...(sourceIds ? { source_ids: sourceIds } : {}) } }),
  notebookOutput: (id: string, outputId: string) => request<NotebookOutput>(`/notebooks/${enc(id)}/outputs/${enc(outputId)}`),
  deleteNotebookOutput: (id: string, outputId: string) => request<void>(`/notebooks/${enc(id)}/outputs/${enc(outputId)}`, { method: 'DELETE' }),
  clearNotebookChat: (id: string) => request<void>(`/notebooks/${enc(id)}/messages`, { method: 'DELETE' }),
  notebookNotes: (id: string) => request<NotebookNote[]>(`/notebooks/${enc(id)}/notes`),
  addNotebookNote: (id: string, note: NoteCreate) => request<NotebookNote>(`/notebooks/${enc(id)}/notes`, { method: 'POST', body: note as Record<string, unknown> }),
  deleteNotebookNote: (id: string, noteId: number) => request<void>(`/notebooks/${enc(id)}/notes/${noteId}`, { method: 'DELETE' }),
  notebookReports: (id: string) => request<NotebookReport[]>(`/notebooks/${enc(id)}/reports`),
  startNotebookReport: (id: string, body: ReportStart) => request<ResearchJob>(`/notebooks/${enc(id)}/reports`, { method: 'POST', body: body as unknown as Record<string, unknown> }),

  localModels: () => request<LocalCatalog>('/local-models'),
  searchLocalModels: (kind: LocalKind, q: string, signal?: AbortSignal) => request<HfResult[]>('/local-models/search', { query: { kind, q }, signal }),
  /** Starts a background download; poll localModels() while the job is downloading. */
  installLocalModel: (p: LocalModelRequest) => request<LocalJob>('/local-models/install', { method: 'POST', body: p as unknown as Record<string, unknown> }),
  useLocalModel: (p: LocalModelRequest) => request<LocalUseResult>('/local-models/use', { method: 'POST', body: p as unknown as Record<string, unknown> }),
  removeLocalModel: (p: LocalModelRequest) => request<{ removed: string }>('/local-models/remove', { method: 'POST', body: p as unknown as Record<string, unknown> }),
  quickPresets: () => request<QuickPreset[]>('/quick-connect/presets'),
  /** Lists and test-calls the service's models; takes ~30–100 s. */
  quickDiscover: (p: QuickDiscoverPayload, signal?: AbortSignal) =>
    request<DiscoverResult>('/quick-connect/discover', { method: 'POST', body: p as unknown as Record<string, unknown>, signal }),
  quickApply: (p: QuickApplyPayload) => request<QuickApplyResult>('/quick-connect/apply', { method: 'POST', body: p as unknown as Record<string, unknown> }),
  privacyDestinations: () => request<PrivacyDestination[]>('/privacy/destinations'),

  // documents
  documents: (q?: { q?: string; collection?: string }) => request<DocumentListItem[]>('/documents', { query: q }),
  document: (id: string) => request<DocumentDetail>(`/documents/${enc(id)}`),
  uploadDocument: (file: File, collection: string) => {
    const fd = new FormData();
    fd.append('file', file, file.name);
    fd.append('collection', collection);
    return request<UploadResponse>('/documents', { method: 'POST', body: fd });
  },
  deleteDocument: (id: string) => request<void>(`/documents/${enc(id)}`, { method: 'DELETE' }),
  retryDocument: (id: string) => request<RetryResponse>(`/documents/${enc(id)}/retry`, { method: 'POST' }),
  chunks: (id: string, page?: number) => request<Chunk[]>(`/documents/${enc(id)}/chunks`, { query: { page } }),
  collections: () => request<CollectionInfo[]>('/collections'),
  ingestionJobs: () => request<IngestionJob[]>('/ingestion/jobs'),

  // vector index
  indexStatus: () => request<IndexStatus>('/index/status'),
  reindex: (providerId?: string | null) => request<{ job: string }>('/index/reindex', { method: 'POST', body: { provider_id: providerId || null } }),
  rollbackIndex: () => request<IndexInfo>('/index/rollback', { method: 'POST' }),
  confirmMigration: () => request<IndexStatus>('/index/confirm-migration', { method: 'POST' }),
  confirmCompatibility: () => request<IndexInfo>('/index/confirm-compatibility', { method: 'POST' }),
  backfill: () => request<{ backfilled: number; reason?: string }>('/index/backfill', { method: 'POST' }),

  // research
  preflight: () => request<Preflight>('/research/preflight'),
  startResearch: (p: ResearchStartPayload) => request<ResearchJob>('/research', { method: 'POST', body: p as unknown as Record<string, unknown> }),
  researchJobs: () => request<ResearchJobSummary[]>('/research'),
  researchJob: (id: string) => request<ResearchJob>(`/research/${enc(id)}`),
  researchEvents: (id: string, after = 0) => request<ResearchEvent[]>(`/research/${enc(id)}/events`, { query: { after } }),
  researchEvidence: (id: string) => request<Evidence[]>(`/research/${enc(id)}/evidence`),
  cancelResearch: (id: string) => request<{ cancelled: boolean }>(`/research/${enc(id)}/cancel`, { method: 'POST' }),
  deleteResearch: (id: string) => request<void>(`/research/${enc(id)}`, { method: 'DELETE' }),
  mindmap: (id: string) => request<MindMap>(`/research/${enc(id)}/mindmap`),
  chatHistory: (id: string) => request<ChatMessage[]>(`/research/${enc(id)}/chat`),
  sendChat: (id: string, message: string, signal?: AbortSignal) =>
    request<ChatAnswer>(`/research/${enc(id)}/chat`, { method: 'POST', body: { message }, signal }),

};
