import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import { MarkdownReport, isInternalHref } from './MarkdownReport';

const md = `# Findings

- Paged attention reduces KV-cache waste to under 4%. [[E3]](/documents/doc123/view?page=2)
- Throughput improved 2–4× [[E7]](https://example.org/blog/vllm)

| System | Waste | Sources |
|---|---|---|
| vLLM | <4% | [[E3]](/documents/doc123/view?page=2) |

See [the raw file](/api/research/j1/report.md).
`;

function Viewer() {
  const loc = useLocation();
  return <div data-testid="viewer">{`${loc.pathname}${loc.search}`}</div>;
}

describe('MarkdownReport', () => {
  it('classifies hrefs', () => {
    expect(isInternalHref('/documents/a/view?page=1')).toBe(true);
    expect(isInternalHref('https://x.org')).toBe(false);
    expect(isInternalHref('//evil.example')).toBe(false);
    expect(isInternalHref('/api/documents/a/pdf')).toBe(false);
    expect(isInternalHref(undefined)).toBe(false);
  });

  it('renders GFM tables and routes internal citation links in-app', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={['/research/j1']}>
        <Routes>
          <Route path="/research/:id" element={<MarkdownReport markdown={md} />} />
          <Route path="/documents/:id/view" element={<Viewer />} />
        </Routes>
      </MemoryRouter>,
    );
    expect(screen.getByRole('table')).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'Waste' })).toBeInTheDocument();

    const internal = screen.getAllByRole('link', { name: '[E3]' });
    expect(internal).toHaveLength(2);
    for (const a of internal) {
      expect(a).toHaveAttribute('href', '/documents/doc123/view?page=2');
      expect(a).not.toHaveAttribute('target');
      expect(a).toHaveClass('citation');
    }
    await user.click(internal[0]);
    expect(screen.getByTestId('viewer')).toHaveTextContent('/documents/doc123/view?page=2');
  });

  it('opens external links in a new tab with noopener noreferrer', () => {
    render(
      <MemoryRouter>
        <MarkdownReport markdown={md} />
      </MemoryRouter>,
    );
    const ext = screen.getByRole('link', { name: /\[E7\]/ });
    expect(ext).toHaveAttribute('href', 'https://example.org/blog/vllm');
    expect(ext).toHaveAttribute('target', '_blank');
    expect(ext).toHaveAttribute('rel', 'noopener noreferrer');

    const api = screen.getByRole('link', { name: /the raw file/ });
    expect(api).toHaveAttribute('href', '/api/research/j1/report.md');
    expect(api).toHaveAttribute('target', '_blank');
  });

  it('strips javascript: URLs', () => {
    render(
      <MemoryRouter>
        <MarkdownReport markdown={'[click](javascript:alert(1))'} />
      </MemoryRouter>,
    );
    const a = screen.getByText('click').closest('a');
    expect(a?.getAttribute('href') ?? '').not.toMatch(/javascript/i);
  });
});
