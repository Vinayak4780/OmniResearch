/**
 * Types mirroring the OmniResearch FastAPI backend (backend/app/api/*).
 * Optional / nullable fields follow the server code; anything open-ended is typed as `unknown`.
 */

export type Capability =
  | 'chat'
  | 'embedding'
  | 'image_embedding'
  | 'reranking'
  | 'ocr'
  | 'vision'
  | 'web_search'
  | 'web_extraction';

export type Task =
  | 'planning'
  | 'decomposition'
  | 'rewriting'
  | 'synthesis'
  | 'verification'
  | 'text_embedding'
  | 'image_embedding'
  | 'reranking'
  | 'ocr'
  | 'vision'
  | 'web_search'
  | 'web_extraction';

// ------------------------------------------------------------------ JSON schema (Pydantic output)

export interface JsonSchema {
  type?: string | string[];
  title?: string;
  description?: string;
  default?: unknown;
  enum?: unknown[];
  anyOf?: JsonSchema[];
  oneOf?: JsonSchema[];
  allOf?: JsonSchema[];
  $ref?: string;
  items?: JsonSchema;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  additionalProperties?: boolean | JsonSchema;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  minLength?: number;
  maxLength?: number;
  $defs?: Record<string, JsonSchema>;
  definitions?: Record<string, JsonSchema>;
}

// ------------------------------------------------------------------ providers

export interface ProviderDescriptor {
  adapter: string;
  display_name: string;
  protocol: 'openai_compatible' | 'native' | 'custom' | 'local';
  capabilities: Capability[];
  modalities: string[];
  description: string;
  limitations: string[];
  config_schema: JsonSchema;
  secret_fields: string[];
  requires_network: boolean;
}

export interface ProviderRecord {
  id: string;
  name: string;
  adapter: string;
  enabled: boolean;
  config: Record<string, unknown>;
  secrets_set: Record<string, boolean>;
  capabilities: Capability[];
  models: Record<string, string | null>;
  created_at: string;
  updated_at: string;
}

/** PATCH /api/providers/{id} */
export interface ProviderPatchPayload {
  name?: string;
  config?: Record<string, unknown>;
  secrets?: Record<string, string | null>;
  enabled?: boolean;
}

export interface RouteConfig {
  task: Task;
  provider_ids: string[];
  fallback_enabled: boolean;
  params: RouteParams;
}

export interface RouteParams {
  model?: string;
  temperature?: number;
  max_tokens?: number;
}

export interface RouteRow extends RouteConfig {
  capability: Capability;
  description: string;
  inherits_from: string | null;
  effective_provider_ids: string[];
}

export interface ProviderMetricRow {
  provider_id: string;
  capability: string;
  calls: number;
  failures: number | null;
  avg_latency_ms: number | null;
  max_latency_ms: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  cost_usd: number | null;
  fallback_calls: number | null;
  p50_latency_ms: number | null;
  p95_latency_ms: number | null;
  failure_rate: number;
}

export interface CircuitState {
  state: 'closed' | 'open' | 'half_open' | string;
  consecutive_failures: number;
  last_error: string | null;
  reopens_in_s: number | null;
}

export interface ProviderCall {
  id: number;
  provider_id: string;
  task: string | null;
  capability: string;
  ok: number;
  latency_ms: number;
  error_kind: string | null;
  error_message: string | null;
  input_tokens: number | null;
  output_tokens: number | null;
  cost_usd: number | null;
  fallback_from: string | null;
  created_at: string;
}

export interface ProviderHealth {
  metrics: ProviderMetricRow[];
  circuits: Record<string, CircuitState>;
  recent_failures: ProviderCall[];
}

export interface PrivacyDestination {
  task: Task;
  provider_id: string;
  adapter: string;
}

// ------------------------------------------------------------------ health

export interface HealthResponse {
  status: 'ok' | 'degraded';
  database: boolean;
  vector_store: { ok: boolean; mode: 'server' | 'embedded' };
  parser: { docling_installed: boolean; configured: string };
  paddleocr_installed: boolean;
  configured_tasks: Record<Task, boolean>;
  admin_token_configured: boolean;
  /** Upload size limit (added in API round 2). */
  max_upload_mb?: number;
}

// ------------------------------------------------------------------ documents

export type DocumentStatus =
  | 'queued'
  | 'processing'
  | 'ready'
  | 'ready_lexical_only'
  | 'ready_with_errors'
  | 'failed';

export interface IngestionJob {
  id: string;
  document_id: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed' | string;
  stage: string | null;
  progress: number;
  error: string | null;
  attempts: number;
  created_at: string;
  updated_at: string;
  filename?: string;
}

export interface OcrPageReport {
  page: number;
  engine: string;
  chars: number;
  latency_ms: number;
  cached: boolean;
}

export interface OcrReport {
  pages_needing_ocr?: number[];
  ocr_pages?: OcrPageReport[];
  failed_pages?: number[];
  skipped_reason?: string | null;
}

export interface IndexReport {
  indexed: boolean;
  chunks?: number;
  index_id?: string;
  reason?: string;
}

export interface DocumentMetadata {
  ocr?: OcrReport;
  index?: IndexReport;
  parse_seconds?: number;
  element_counts?: Record<string, number>;
  chunk_count?: number;
  parser_fallback_reason?: string;
  [key: string]: unknown;
}

export interface DocumentRow {
  id: string;
  filename: string;
  sha256: string;
  size_bytes: number;
  collection: string;
  page_count: number | null;
  title: string | null;
  status: DocumentStatus;
  error: string | null;
  parser: string | null;
  metadata: DocumentMetadata;
  created_at: string;
  updated_at: string;
}

export interface DocumentListItem extends DocumentRow {
  job: IngestionJob | null;
  chunk_count: number;
}

export interface PageInfo {
  page_no: number;
  width: number | null;
  height: number | null;
  text_chars: number;
  extraction_method: string;
  ocr_status: 'done' | 'failed' | 'skipped' | null | string;
  ocr_error: string | null;
  ocr_provider: string | null;
  ocr_latency_ms: number | null;
}

export interface ElementTypeCount {
  element_type: string;
  extraction_method: string;
  n: number;
}

export interface DocumentDetail extends DocumentRow {
  pages: PageInfo[];
  jobs: IngestionJob[];
  element_types: ElementTypeCount[];
}

export interface UploadResponse {
  document: DocumentRow;
  duplicate: boolean;
  job_id: string | null;
}

export interface RetryResponse {
  document: DocumentRow;
  job_id: string;
}

export interface Chunk {
  id: string;
  document_id: string;
  chunk_index: number;
  page_no: number;
  page_end: number;
  section_path: string;
  element_type: string;
  extraction_method: string;
  text: string;
  bbox_json: string | null;
  source_element_id: string | null;
  chunk_version: string;
  content_hash: string;
}

export interface CollectionInfo {
  collection: string;
  documents: number;
}

// ------------------------------------------------------------------ vector index

export interface IndexSpec {
  provider_id: string;
  provider_fingerprint: Record<string, unknown>;
  distance: string;
  chunking_version: string;
  preprocessing: string;
  dimensions?: number;
  model?: string | null;
  compatibility_confirmations?: { at: string; from: unknown; to: unknown }[];
}

export interface IndexInfo {
  id: string;
  collection_name: string;
  fingerprint: string;
  spec: IndexSpec;
  status: string;
  progress_done: number;
  progress_total: number;
  failed_batches: number;
  error: string | null;
  created_at: string;
  activated_at: string | null;
}

export interface IndexCompatibility {
  ok: boolean;
  reason?: 'no_active_index' | 'pinned_provider_unavailable' | 'provider_config_changed' | 'chunking_version_changed' | string;
  provider_id?: string;
  indexed_with?: Record<string, unknown>;
  current?: Record<string, unknown>;
}

export interface IndexStatus {
  active: IndexInfo | null;
  previous: IndexInfo | null;
  building: IndexInfo[];
  /** Last failed builds (newest first). */
  recent_failed?: IndexInfo[];
  route_primary_provider: string | null;
  compatibility: IndexCompatibility;
  total_chunks: number;
  indexed_chunks?: number;
  missing_vectors?: number;
  vector_count?: number;
  route_matches_active?: boolean;
  reindex_recommended?: boolean;
}

// ------------------------------------------------------------------ research

export type ResearchStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted';
export type WebMode = 'auto' | 'always' | 'never';

export interface ResearchOptions {
  collections?: string[];
  document_ids?: string[];
  web_mode?: WebMode;
  /** Set for deep research started from a notebook. */
  notebook_id?: string;
}

export interface ResearchJobSummary {
  id: string;
  question: string;
  status: ResearchStatus;
  stage: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface VerificationSummary {
  total?: number;
  verified?: number;
  partially_verified?: number;
  unverified?: number;
  interpretation?: number;
  repaired_by_rewrite?: number;
  evidence_checks?: Record<string, unknown>;
  web_citation_checks?: unknown;
}

export interface ReportUsage {
  calls: number;
  failed_calls?: number;
  fallback_calls?: number;
  input_tokens: number | null;
  output_tokens: number | null;
  cost_usd: number | null;
}

export interface ResearchReport {
  draft?: Record<string, unknown>;
  verification?: { statements?: Record<string, unknown>; summary?: VerificationSummary };
  web_used?: boolean;
  web_reason?: string | null;
  iterations?: number;
  queries?: { query: string; focus?: string; source?: string; purpose?: string }[];
  comparison?: unknown;
  warnings?: string[];
  usage?: ReportUsage;
  providers_by_task?: Record<string, string[]>;
}

export interface ResearchJob extends ResearchJobSummary {
  options: ResearchOptions;
  report_md: string | null;
  report: ResearchReport | null;
}

export interface ResearchEvent {
  id: number;
  job_id: string;
  ts: string;
  stage: string;
  message: string;
  data: Record<string, unknown> & { seconds?: number };
}

export interface Preflight {
  problems: string[];
  external_destinations: PrivacyDestination[];
  web_available: boolean;
  web_reason: string;
}

export interface ResearchStartPayload {
  question: string;
  collections?: string[];
  document_ids?: string[];
  web_mode?: WebMode;
}

export interface Evidence {
  key: string;
  id: string | null;
  source_type: 'local' | 'web';
  text: string;
  subquery?: string | null;
  retrieval?: Record<string, unknown>;
  relevance?: number;

  document_id?: string | null;
  filename?: string | null;
  doc_title?: string | null;
  page_no?: number | null;
  page_end?: number | null;
  section_path?: string | null;
  element_type?: string | null;
  extraction_method?: string | null;
  chunk_id?: string | null;
  bbox?: number[] | null;
  source_element_id?: string | null;

  url?: string | null;
  final_url?: string | null;
  canonical_url?: string | null;
  domain?: string | null;
  web_title?: string | null;
  published_at?: string | null;
  retrieved_at?: string | null;
  search_engine?: string | null;
  snippet_only?: boolean;

  visual_observation?: string | null;
  visual_model?: string | null;
  inspected_visually?: boolean;
}

// ------------------------------------------------------------------ mind map & follow-up chat

export type MindMapNodeType = 'question' | 'group' | 'query' | 'finding' | 'claim' | 'document' | 'website' | 'evidence' | 'contradiction' | 'gap';
export type ClaimStatus = 'verified' | 'partially_verified' | 'mechanically_verified' | 'unverified' | 'uncited' | 'interpretation';

export interface MindMapNode {
  id: string;
  label: string;
  type: MindMapNodeType | string;
  data: {
    // claim
    statement_id?: string;
    text?: string;
    kind?: 'source' | 'interpretation';
    status?: ClaimStatus | null;
    citations?: string[];
    // query
    focus?: string | null;
    source?: string | null;
    // evidence
    evidence_id?: string;
    source_type?: 'local' | 'web';
    page_no?: number | null;
    document_id?: string | null;
    url?: string | null;
    element_type?: string | null;
    extraction_method?: string | null;
    excerpt?: string;
    cited?: boolean;
    inspected_visually?: boolean;
    // contradiction
    description?: string;
    explanation?: string;
    [key: string]: unknown;
  };
}

export interface MindMapEdge {
  source: string;
  target: string;
  type: 'child' | 'cites' | 'contradicts' | string;
}

export interface MindMap {
  title: string;
  nodes: MindMapNode[];
  edges: MindMapEdge[];
}

export interface ChatCheck {
  answered_from_sources?: boolean;
  unknown_citations?: string[];
  numbers_not_in_cited_evidence?: string[];
  new_evidence?: string[];
}

export interface ChatMessage {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  citations: string[];
  links: Record<string, string>;
  check: ChatCheck;
  created_at: string;
}

export interface ChatAnswer {
  answer: string;
  citations: string[];
  links: Record<string, string>;
  check: ChatCheck;
  evidence: Evidence[];
}
