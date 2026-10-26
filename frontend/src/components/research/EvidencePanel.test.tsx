import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import type { Evidence, ResearchEvent } from '../../types/api';
import { EvidencePanel, scoreEntries, splitEvidence } from './EvidencePanel';
import { buildSteps } from './StageTimeline';

const evidence: Evidence[] = [
  {
    key: 'chunk-1',
    id: 'E1',
    source_type: 'local',
    text: 'PagedAttention keeps KV-cache waste under 4%.',
    retrieval: { query: 'kv cache waste', scores: { bm25: 7.25, rrf: 0.0325 }, ranks: { bm25: 1 } },
    relevance: 0.91,
    document_id: 'doc1',
    filename: 'vllm.pdf',
    doc_title: 'Efficient Memory Management for LLM Serving',
    page_no: 4,
    page_end: 4,
    section_path: '3 Method > 3.2 Paging',
    element_type: 'paragraph',
    extraction_method: 'text_layer',
  },
  {
    key: 'chunk-2',
    id: 'E2',
    source_type: 'local',
    text: 'Figure 2: throughput vs batch size.',
    document_id: 'doc1',
    filename: 'vllm.pdf',
    page_no: 6,
    element_type: 'figure',
    extraction_method: 'ocr:ocr_engine:paddle',
    visual_observation: 'The chart shows throughput rising roughly linearly up to batch 64.',
    visual_model: 'llama-3.2-vision',
    inspected_visually: true,
  },
  {
    key: 'https://blog.example.com/post#abc',
    id: 'E3',
    source_type: 'web',
    text: 'Continuous batching improved throughput by 23x in our tests.',
    url: 'https://blog.example.com/post',
    domain: 'blog.example.com',
    web_title: 'How continuous batching works',
    published_at: '2025-06-01',
    retrieved_at: '2026-10-20T10:00:00Z',
    search_engine: 'tavily',
    snippet_only: true,
  },
];

function renderPanel(items = evidence) {
  return render(
    <MemoryRouter>
      <EvidencePanel evidence={items} />
    </MemoryRouter>,
  );
}

describe('EvidencePanel', () => {
  it('splits evidence by source type', () => {
    const { local, web } = splitEvidence(evidence);
    expect(local.map((e) => e.id)).toEqual(['E1', 'E2']);
    expect(web.map((e) => e.id)).toEqual(['E3']);
  });

  it('flattens nested retrieval scores', () => {
    expect(scoreEntries(evidence[0].retrieval)).toEqual([
      ['bm25 score', 7.25],
      ['rrf score', 0.0325],
      ['bm25 rank', 1],
    ]);
  });

  it('shows uploaded-document provenance on the first tab', () => {
    renderPanel();
    expect(screen.getByRole('tab', { name: 'Uploaded documents (2)' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tab', { name: 'Web sources (1)' })).toHaveAttribute('aria-selected', 'false');
    const panel = screen.getByRole('tabpanel');
    const link = within(panel).getByRole('link', { name: 'Efficient Memory Management for LLM Serving' });
    expect(link).toHaveAttribute('href', '/documents/doc1/view?page=4');
    expect(within(panel).getByText('3 Method > 3.2 Paging')).toBeInTheDocument();
    expect(within(panel).getByText('text_layer')).toBeInTheDocument();
    expect(within(panel).getByText('ocr:ocr_engine:paddle')).toBeInTheDocument();
    expect(within(panel).getByText('0.910')).toBeInTheDocument();
    // visual observation is labelled as model interpretation, separate from source text
    expect(within(panel).getByText('Model interpretation')).toBeInTheDocument();
    expect(within(panel).getByText(/not text from the source/)).toBeInTheDocument();
    expect(within(panel).getByText('llama-3.2-vision')).toBeInTheDocument();
    expect(within(panel).queryByText('blog.example.com')).not.toBeInTheDocument();
  });

  it('shows web provenance on the web tab', async () => {
    const user = userEvent.setup();
    renderPanel();
    await user.click(screen.getByRole('tab', { name: 'Web sources (1)' }));
    const panel = screen.getByRole('tabpanel');
    const link = within(panel).getByRole('link', { name: /How continuous batching works/ });
    expect(link).toHaveAttribute('href', 'https://blog.example.com/post');
    expect(link).toHaveAttribute('target', '_blank');
    expect(link).toHaveAttribute('rel', 'noopener noreferrer');
    expect(within(panel).getByText('blog.example.com')).toBeInTheDocument();
    expect(within(panel).getByText(/Snippet only/)).toBeInTheDocument();
    expect(within(panel).getByText('tavily')).toBeInTheDocument();
    expect(within(panel).queryByText('vllm.pdf')).not.toBeInTheDocument();
  });

  it('opens on the web tab when there is only web evidence, and shows empty states', async () => {
    const user = userEvent.setup();
    renderPanel([evidence[2]]);
    expect(screen.getByRole('tab', { name: 'Web sources (1)' })).toHaveAttribute('aria-selected', 'true');
    await user.click(screen.getByRole('tab', { name: 'Uploaded documents (0)' }));
    expect(screen.getByText('No evidence from uploaded documents')).toBeInTheDocument();
  });
});

describe('buildSteps', () => {
  const ev = (id: number, stage: string, message: string, data: Record<string, unknown> = {}): ResearchEvent => ({ id, job_id: 'j', ts: '2026-01-01T00:00:00Z', stage, message, data });

  it('pairs started/completed events, handling repeated and parallel stages', () => {
    const steps = buildSteps(
      [
        ev(1, 'analyze_question', 'started'),
        ev(2, 'analyze_question', 'compare things', { seconds: 1.5 }),
        ev(3, 'retrieve_local', 'started'),
        ev(4, 'web_research', 'started'),
        ev(5, 'web_research', '3 web passages', { seconds: 4 }),
        ev(6, 'retrieve_local', '12 local passages', { seconds: 0.4 }),
        ev(7, 'retrieve_local', 'started'),
      ],
      true,
    );
    expect(steps.map((s) => [s.stage, s.status, s.seconds])).toEqual([
      ['analyze_question', 'done', 1.5],
      ['retrieve_local', 'done', 0.4],
      ['web_research', 'done', 4],
      ['retrieve_local', 'running', undefined],
    ]);
    const stopped = buildSteps([ev(1, 'draft_report', 'started'), ev(2, 'system', 'failed: boom')], false);
    expect(stopped.map((s) => s.status)).toEqual(['stopped', 'system']);
  });
});
