import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import { api } from '../../services/api';
import type { DocumentDetail, Evidence } from '../../types/api';
import { citationTarget, extractToc, formatCitation, linkifyCitations, parseDocumentLink } from '../../utils/citations';
import { CitationPanel, bboxToPercent } from './CitationPanel';
import { MarkdownReport } from './MarkdownReport';

const doc: DocumentDetail = {
  id: 'doc1',
  filename: 'kv.pdf',
  sha256: 'x',
  size_bytes: 1000,
  collection: 'papers',
  page_count: 2,
  title: 'Paged KV-Cache Management',
  status: 'ready',
  error: null,
  parser: 'docling',
  metadata: {},
  created_at: '2026-10-24T00:00:00Z',
  updated_at: '2026-10-24T00:00:00Z',
  pages: [
    { page_no: 1, width: 595, height: 842, text_chars: 400, extraction_method: 'text_layer', ocr_status: null, ocr_error: null, ocr_provider: null, ocr_latency_ms: null },
    { page_no: 2, width: 595, height: 842, text_chars: 0, extraction_method: 'ocr:paddle', ocr_status: 'done', ocr_error: null, ocr_provider: 'paddle', ocr_latency_ms: 900 },
  ],
  jobs: [],
  element_types: [],
};

const e3: Evidence = {
  key: 'c3',
  id: 'E3',
  source_type: 'local',
  text: 'Paged KV cache keeps waste under 4%.',
  document_id: 'doc1',
  filename: 'kv.pdf',
  doc_title: 'Paged KV-Cache Management',
  page_no: 1,
  page_end: 1,
  section_path: '2 Method',
  element_type: 'text',
  extraction_method: 'text_layer',
  bbox: [59.5, 84.2, 535.5, 168.4],
};

function Loc() {
  const l = useLocation();
  return <div data-testid="loc">{l.pathname + l.search}</div>;
}

describe('citation utilities', () => {
  it('parses viewer links and resolves panel targets', () => {
    expect(parseDocumentLink('/documents/abc/view?page=3')).toEqual({ documentId: 'abc', page: 3 });
    expect(parseDocumentLink('/documents/abc/view')).toEqual({ documentId: 'abc', page: 1 });
    expect(parseDocumentLink('https://x.org/documents/a/view')).toBeNull();
    expect(parseDocumentLink('/documents/abc/view?page=2&evidence=E7')).toEqual({ documentId: 'abc', page: 2, evidenceId: 'E7' });
    expect(parseDocumentLink('/documents/abc/view?page=2&evidence=bogus')).toEqual({ documentId: 'abc', page: 2 });
    const map = new Map([['E3', e3]]);
    expect(citationTarget('E3', undefined, map)).toEqual({ documentId: 'doc1', page: 1, evidenceId: 'E3' });
    // plain page link → attaches the evidence when it is the only one on that page
    expect(citationTarget('', '/documents/doc1/view?page=1', map)).toEqual({ documentId: 'doc1', page: 1, evidenceId: 'E3' });
    // an explicit evidence id in the link wins, even when several evidence items share the page
    expect(citationTarget('', '/documents/doc1/view?page=1&evidence=E3', map)).toEqual({ documentId: 'doc1', page: 1, evidenceId: 'E3' });
    expect(citationTarget('E99', '/documents/zzz/view?page=2', map)).toEqual({ documentId: 'zzz', page: 2, evidenceId: 'E99' });
    expect(citationTarget('E5', 'https://example.org', map)).toBeNull();
  });

  it('linkifies inline citations without touching existing links', () => {
    expect(linkifyCitations('Waste is 3.8% [E3]. See [E1, E2] and [[E4]](/documents/a/view?page=1).')).toBe(
      'Waste is 3.8% [[E3]](#cite-E3). See [[E1]](#cite-E1) [[E2]](#cite-E2) and [[E4]](/documents/a/view?page=1).',
    );
  });

  it('builds a TOC with deduplicated ids matching rendered headings', () => {
    const md = '# Title\n\n## Findings\n\n### Paged **KV**\n\n```\n## not a heading\n```\n\n## Findings\n';
    const toc = extractToc(md);
    expect(toc).toEqual([
      { level: 2, text: 'Findings', id: 'findings' },
      { level: 3, text: 'Paged KV', id: 'paged-kv' },
      { level: 2, text: 'Findings', id: 'findings-1' },
    ]);
    const { container } = render(
      <MemoryRouter>
        <MarkdownReport markdown={md} headingIds />
      </MemoryRouter>,
    );
    expect([...container.querySelectorAll('h2,h3')].map((h) => h.id)).toEqual(['findings', 'paged-kv', 'findings-1']);
  });

  it('formats a copyable citation', () => {
    expect(formatCitation(e3, 'http://localhost:5173')).toBe(
      'Paged KV-Cache Management, “2 Method”, p. 1 [E3]. http://localhost:5173/documents/doc1/view?page=1&evidence=E3',
    );
  });
});

describe('bboxToPercent', () => {
  it('converts PDF points (top-left origin) to percentages of the page', () => {
    expect(bboxToPercent([59.5, 84.2, 535.5, 168.4], { width: 595, height: 842 })).toEqual({ left: 10, top: 10, width: 80, height: 10 });
  });
  it('rejects missing or degenerate boxes', () => {
    expect(bboxToPercent(null, { width: 595, height: 842 })).toBeNull();
    expect(bboxToPercent([10, 10, 5, 20], { width: 595, height: 842 })).toBeNull();
    expect(bboxToPercent([1, 2, 3, 4], { width: null, height: 842 })).toBeNull();
  });
});

describe('MarkdownReport citation clicks', () => {
  it('opens local citations through onCitation instead of navigating; web citations stay external', async () => {
    const user = userEvent.setup();
    const onCitation = vi.fn();
    render(
      <MemoryRouter initialEntries={['/research/j']}>
        <Routes>
          <Route
            path="*"
            element={
              <>
                <MarkdownReport
                  markdown={'Waste fell [[E3]](/documents/doc1/view?page=1) per [[E7]](https://blog.example/x).\n\n- **[E3]** [Paged KV, p. 1](/documents/doc1/view?page=1)'}
                  onCitation={onCitation}
                />
                <Loc />
              </>
            }
          />
        </Routes>
      </MemoryRouter>,
    );
    await user.click(screen.getByRole('link', { name: '[E3]' }));
    expect(onCitation).toHaveBeenCalledWith('E3', '/documents/doc1/view?page=1');
    expect(screen.getByTestId('loc')).toHaveTextContent('/research/j');
    await user.click(screen.getByRole('link', { name: 'Paged KV, p. 1' }));
    expect(onCitation).toHaveBeenLastCalledWith('', '/documents/doc1/view?page=1');
    const web = screen.getByRole('link', { name: /\[E7\]/ });
    expect(web).toHaveAttribute('target', '_blank');
    expect(web).toHaveAttribute('rel', 'noopener noreferrer');
    expect(onCitation).toHaveBeenCalledTimes(2);
  });

  it('renders chat citation placeholders as chips, and unknown ids as non-links', () => {
    render(
      <MemoryRouter>
        <MarkdownReport markdown={linkifyCitations('A [E3] B [E9] C [E7]')} citationLinks={{ E3: '/documents/doc1/view?page=1', E7: 'https://blog.example/x' }} onCitation={() => {}} />
      </MemoryRouter>,
    );
    expect(screen.getByRole('link', { name: '[E3]' })).toHaveAttribute('href', '/documents/doc1/view?page=1');
    expect(screen.getByRole('link', { name: /\[E7\]/ })).toHaveAttribute('href', 'https://blog.example/x');
    const unknown = screen.getByText('[E9]');
    expect(unknown.closest('a')).toBeNull();
    expect(unknown).toHaveClass('citation-unknown');
  });
});

describe('CitationPanel', () => {
  function renderPanel(evidence: Evidence | undefined = e3, page = 1) {
    vi.spyOn(api, 'document').mockResolvedValue(doc);
    const onClose = vi.fn();
    render(
      <MemoryRouter>
        <CitationPanel target={{ documentId: 'doc1', page, evidenceId: evidence?.id ?? undefined }} evidence={evidence} onClose={onClose} />
      </MemoryRouter>,
    );
    return { onClose };
  }

  it('shows the page with the evidence bbox highlighted, provenance and links', async () => {
    renderPanel();
    expect(screen.getByRole('heading', { name: 'Paged KV-Cache Management' })).toBeInTheDocument();
    const img = screen.getByAltText(/Page 1 of/) as HTMLImageElement;
    expect(img.src).toContain('/api/documents/doc1/pages/1.png');
    fireEvent.load(img);
    const hl = await screen.findByTestId('bbox-highlight');
    expect(hl.style.left).toBe('10%');
    expect(hl.style.top).toBe('10%');
    expect(hl.style.width).toBe('80%');
    expect(hl.style.height).toBe('10%');
    expect(screen.getByText('Paged KV cache keeps waste under 4%.')).toBeInTheDocument();
    expect(screen.getByText('2 Method')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open full viewer' })).toHaveAttribute('href', '/documents/doc1/view?page=1');
    expect(screen.getByRole('button', { name: /Copy citation/ })).toBeInTheDocument();
  });

  it('pages through the document; the highlight only shows on the cited page', async () => {
    const user = userEvent.setup();
    renderPanel();
    await waitFor(() => expect(screen.getByText('Page 1 / 2')).toBeInTheDocument());
    await user.click(screen.getByRole('button', { name: 'Next page' }));
    expect(screen.getByText('Page 2 / 2')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Next page' })).toBeDisabled();
    const img = screen.getByAltText(/Page 2 of/);
    fireEvent.load(img);
    expect(screen.queryByTestId('bbox-highlight')).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open full viewer' })).toHaveAttribute('href', '/documents/doc1/view?page=2');
    await user.click(screen.getByRole('button', { name: 'Back to cited page 1' }));
    expect(screen.getByText('Page 1 / 2')).toBeInTheDocument();
  });

  it('explains when no region was recorded, and closes on Escape', async () => {
    const { onClose } = renderPanel({ ...e3, bbox: null, extraction_method: 'ocr:paddle' });
    fireEvent.load(screen.getByAltText(/Page 1 of/));
    expect(await screen.findByText(/No region was recorded for this passage \(OCR text\)/)).toBeInTheDocument();
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalled();
  });
});
