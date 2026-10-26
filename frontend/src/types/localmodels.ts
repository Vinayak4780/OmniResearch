/** Local CPU models (backend/app/services/localmodels.py, /api/local-models/*). */

export type LocalKind = 'ocr' | 'embedding' | 'reranking';

export interface LocalJob {
  id: string;
  kind: LocalKind;
  model: string;
  status: 'downloading' | 'done' | 'failed' | string;
  error: string | null;
  downloaded_mb: number | null;
  expected_mb: number | null;
  /** 0–1, or null when the size is unknown. */
  progress: number | null;
}

export interface LocalModel {
  id: string;
  description: string;
  size_gb: number | null;
  dims?: number;
  /** OCR models only. */
  language?: string;
  license?: string;
  source: 'catalog' | 'custom';
  installed: boolean;
  in_use: boolean;
  recommended: boolean;
  job: LocalJob | null;
}

export interface LocalCatalog {
  available: Record<LocalKind, boolean>;
  kinds: Record<LocalKind, LocalModel[]>;
  in_use: Record<LocalKind, string | null>;
  jobs: LocalJob[];
}

export interface HfResult {
  /** Ready-to-install model id (for OCR a detector+recognizer pair). */
  id: string;
  repo: string;
  downloads: number;
  likes: number;
  in_catalog: boolean;
  url: string;
}

export interface LocalModelRequest {
  kind: LocalKind;
  model: string;
  /** Hugging Face repo, for models that are not in the built-in list. */
  repo?: string;
}

export interface LocalUseResult {
  provider_id: string;
  task: string;
  notes: string[];
}
