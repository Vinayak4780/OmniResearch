/** Notebooks API (backend/app/api/routes_notebooks.py). */
import type { DocumentStatus, Evidence, ResearchJobSummary, WebMode } from './api';

export interface NotebookSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  source_count: number;
  report_count: number;
}

export interface SourceJob {
  status: string;
  stage: string | null;
  progress: number | null;
}

export interface NotebookSource {
  id: string;
  filename: string;
  title: string | null;
  status: DocumentStatus;
  error: string | null;
  page_count: number | null;
  size_bytes: number;
  parser?: string | null;
  created_at?: string;
  updated_at?: string;
  job: SourceJob | null;
  /** The original web address when the source was added from a link (PDF or captured webpage). */
  source_url?: string | null;
}

export interface Notebook {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  sources: NotebookSource[];
}

export interface NotebookGuide {
  summary: string | null;
  questions: string[];
  sources_hash: string | null;
  /** Why there is no summary (e.g. no sources yet). */
  reason?: string;
}

/** A cited passage: the full evidence record plus its viewer link. */
export type NotebookCitation = Evidence & { link: string };

export interface NotebookCheck {
  answered_from_sources?: boolean;
  unknown_citations?: string[];
  numbers_not_in_cited_evidence?: string[];
  retrieval_warnings?: string[];
  /** True when the answer also used a quick web search. */
  web?: boolean;
}

export interface NotebookMessage {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  /** Per-message ids (E1, E2…) → passage. Only cited ids are present. */
  citations: Record<string, NotebookCitation>;
  check: NotebookCheck;
  source_ids: string[];
  created_at: string;
}

export interface NotebookNote {
  id: number;
  notebook_id?: string;
  title: string;
  content: string;
  citations: Record<string, NotebookCitation>;
  created_at: string;
}

export interface NoteCreate {
  content?: string;
  title?: string;
  message_id?: number;
}

export type NotebookReport = ResearchJobSummary;

export interface ReportStart {
  question: string;
  source_ids?: string[];
  web_mode?: WebMode;
}

// ------------------------------------------------------------------ add sources from links / the web

export interface LinkResult {
  url: string;
  ok: boolean;
  kind?: 'pdf' | 'web';
  duplicate?: boolean;
  document?: Record<string, unknown>;
  error?: string;
}

export interface LinksResponse {
  results: LinkResult[];
}

export type DiscoverMode = 'fast' | 'deep';

export interface DiscoverHit {
  url: string;
  title: string;
  snippet: string;
  domain: string;
  published_at: string | null;
  is_pdf: boolean;
  /** Why the AI picked it (deep mode only). */
  reason: string | null;
  already_added: boolean;
}

export interface DiscoverResponse {
  mode: DiscoverMode;
  query: string;
  queries: string[];
  results: DiscoverHit[];
  warnings: string[];
}

// ------------------------------------------------------------------ studio outputs

export type OutputKind = 'summary' | 'briefing' | 'study_guide' | 'faq' | 'timeline' | 'mindmap';
export type OutputStatus = 'running' | 'completed' | 'failed';

export interface OutputMapNode {
  label: string;
  citations?: string[];
  children?: OutputMapNode[];
}

export interface OutputMindMap {
  central?: string;
  branches?: OutputMapNode[];
}

export interface OutputCheck {
  unknown_citations?: string[];
  numbers_not_in_cited_evidence?: string[];
  uncited_statements?: number;
  passages_used?: number;
}

export interface NotebookOutput {
  id: string;
  notebook_id: string;
  kind: OutputKind;
  title: string;
  status: OutputStatus;
  content: string;
  /** `{}` except for mind maps. */
  data: OutputMindMap;
  citations: Record<string, NotebookCitation>;
  check: OutputCheck;
  source_ids: string[];
  error: string | null;
  created_at: string;
  updated_at: string;
}
