import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import type { MindMap } from '../../../types/api';
import { MindMapView } from './MindMapView';
import { buildTree, crossLinksFor, defaultCollapsed, indexMindMap, nearestVisible, visibleIds, visibleTree } from './mindmapLogic';

/** Same shape as backend/app/services/research/mindmap.py output, trimmed. */
const mm: MindMap = {
  title: 'KV cache',
  nodes: [
    { id: 'root', label: 'How do KV-cache techniques compare?', type: 'question', data: {} },
    { id: 'queries', label: 'Searches (2)', type: 'group', data: {} },
    { id: 'q:Q1', label: 'paged attention waste', type: 'query', data: { source: 'local' } },
    { id: 'q:Q2', label: 'continuous batching throughput', type: 'query', data: { source: 'local' } },
    { id: 'findings', label: 'Findings', type: 'group', data: {} },
    { id: 'h:0', label: 'Summary', type: 'finding', data: {} },
    { id: 's:S1', label: 'Paging cuts waste to 3.8%', type: 'claim', data: { text: 'Paging cuts KV waste from 71% to 3.8%.', status: 'verified', kind: 'source', citations: ['E1', 'E2'] } },
    { id: 's:S2', label: 'Batching gives 23x', type: 'claim', data: { text: 'Batching gives 23x.', status: 'unverified', kind: 'source', citations: ['E2'] } },
    { id: 'sources', label: 'Sources', type: 'group', data: {} },
    { id: 'd:doc1', label: 'kv.pdf', type: 'document', data: {} },
    { id: 'e:E1', label: 'E1 · p.1', type: 'evidence', data: { evidence_id: 'E1', source_type: 'local', document_id: 'doc1', page_no: 1, excerpt: 'waste 3.8%', cited: true } },
    { id: 'e:E2', label: 'E2 · p.2', type: 'evidence', data: { evidence_id: 'E2', source_type: 'local', document_id: 'doc1', page_no: 2, excerpt: 'throughput 23x', cited: true } },
    { id: 'contradictions', label: 'Contradictions (1)', type: 'group', data: {} },
    { id: 'c:0', label: 'Throughput gain', type: 'contradiction', data: { description: '23x vs 8.6x', explanation: 'paging', citations: ['E2'] } },
  ],
  edges: [
    { source: 'root', target: 'queries', type: 'child' },
    { source: 'queries', target: 'q:Q1', type: 'child' },
    { source: 'queries', target: 'q:Q2', type: 'child' },
    { source: 'root', target: 'findings', type: 'child' },
    { source: 'findings', target: 'h:0', type: 'child' },
    { source: 'h:0', target: 's:S1', type: 'child' },
    { source: 'h:0', target: 's:S2', type: 'child' },
    { source: 'root', target: 'sources', type: 'child' },
    { source: 'sources', target: 'd:doc1', type: 'child' },
    { source: 'd:doc1', target: 'e:E1', type: 'child' },
    { source: 'd:doc1', target: 'e:E2', type: 'child' },
    { source: 'root', target: 'contradictions', type: 'child' },
    { source: 'contradictions', target: 'c:0', type: 'child' },
    { source: 's:S1', target: 'e:E1', type: 'cites' },
    { source: 's:S1', target: 'e:E2', type: 'cites' },
    { source: 's:S2', target: 'e:E2', type: 'cites' },
    { source: 'c:0', target: 'e:E2', type: 'contradicts' },
  ],
};

describe('mind map logic', () => {
  const idx = indexMindMap(mm);
  const tree = buildTree(mm, idx)!;

  it('builds the child tree over all nodes', () => {
    const count = (t: typeof tree): number => 1 + t.children.reduce((a, c) => a + count(c), 0);
    expect(tree.id).toBe('root');
    expect(count(tree)).toBe(mm.nodes.length);
  });

  it('starts with searches collapsed and findings/sources expanded', () => {
    const c = defaultCollapsed(mm, idx);
    expect([...c]).toEqual(['queries']);
    const v = visibleTree(tree, c);
    const ids = visibleIds(v);
    expect(ids.has('q:Q1')).toBe(false);
    expect(ids.has('s:S1')).toBe(true);
    expect(ids.has('e:E2')).toBe(true);
    expect(v.children.find((x) => x.id === 'queries')).toMatchObject({ collapsed: true, hiddenCount: 2, children: [] });
  });

  it('maps cross-links onto the nearest visible ancestor when a branch is collapsed', () => {
    const open = visibleIds(visibleTree(tree, new Set()));
    expect(crossLinksFor(mm, 's:S1', idx, open).map((l) => [l.source, l.target, l.type])).toEqual([
      ['s:S1', 'e:E1', 'cites'],
      ['s:S1', 'e:E2', 'cites'],
    ]);
    // incoming links are included for evidence nodes
    expect(crossLinksFor(mm, 'e:E2', idx, open).map((l) => l.source).sort()).toEqual(['c:0', 's:S1', 's:S2']);
    expect(crossLinksFor(mm, 'e:E2', idx, open).find((l) => l.source === 'c:0')?.type).toBe('contradicts');

    const collapsedDoc = visibleIds(visibleTree(tree, new Set(['d:doc1'])));
    expect(nearestVisible('e:E1', idx, collapsedDoc)).toBe('d:doc1');
    // both citations collapse onto the document node and are deduplicated
    expect(crossLinksFor(mm, 's:S1', idx, collapsedDoc)).toEqual([{ source: 's:S1', target: 'd:doc1', type: 'cites', from: 's:S1', to: 'e:E1' }]);
    // a link whose two ends collapse into the same node is dropped
    const collapsedRootKids = visibleIds(visibleTree(tree, new Set(['findings', 'sources'])));
    expect(crossLinksFor(mm, null, idx, collapsedRootKids)).toEqual([]);
  });
});

describe('MindMapView', () => {
  function renderMap() {
    const onOpenEvidence = vi.fn();
    const r = render(
      <MemoryRouter>
        <MindMapView mindmap={mm} onOpenEvidence={onOpenEvidence} />
      </MemoryRouter>,
    );
    const node = (id: string) => r.container.querySelector(`[data-node-id="${id}"]`) as SVGGElement | null;
    const xlinks = () => r.container.querySelectorAll('[data-testid="mm-cross-links"] path');
    return { ...r, node, xlinks, onOpenEvidence };
  }

  it('collapses and expands branches via the toggle', async () => {
    const user = userEvent.setup();
    const { node } = renderMap();
    expect(node('q:Q1')).toBeNull();
    expect(node('queries')).toHaveAttribute('aria-expanded', 'false');
    await user.click(node('queries')!.querySelector('.mm-toggle')!);
    expect(node('q:Q1')).not.toBeNull();
    expect(node('queries')).toHaveAttribute('aria-expanded', 'true');
    await user.click(node('findings')!.querySelector('.mm-toggle')!);
    expect(node('s:S1')).toBeNull();
    await user.click(screen.getByRole('button', { name: 'Expand all' }));
    expect(node('s:S1')).not.toBeNull();
    expect(node('q:Q2')).not.toBeNull();
  });

  it('draws cross-links only for the hovered or selected node and shows details', async () => {
    const user = userEvent.setup();
    const { node, xlinks, onOpenEvidence } = renderMap();
    expect(xlinks()).toHaveLength(0);
    fireEvent.mouseEnter(node('s:S1')!);
    expect(xlinks()).toHaveLength(2);
    fireEvent.mouseLeave(node('s:S1')!);
    expect(xlinks()).toHaveLength(0);

    await user.click(node('s:S2')!);
    expect(node('s:S2')).toHaveAttribute('aria-selected', 'true');
    expect(xlinks()).toHaveLength(1);
    expect(screen.getByText('Batching gives 23x.')).toBeInTheDocument();
    expect(screen.getAllByText('Unverified').length).toBeGreaterThan(0);

    await user.click(node('c:0')!);
    expect(xlinks()[0]).toHaveClass('mm-xlink-contradicts');
    expect(screen.getByText('23x vs 8.6x')).toBeInTheDocument();

    await user.click(node('e:E1')!);
    expect(screen.getByText('waste 3.8%')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Open in side panel' }));
    expect(onOpenEvidence).toHaveBeenCalledWith('E1');
  });

  it('selecting a cited evidence chip reveals it inside a collapsed branch', async () => {
    const user = userEvent.setup();
    const { node } = renderMap();
    await user.click(node('sources')!.querySelector('.mm-toggle')!);
    expect(node('e:E1')).toBeNull();
    await user.click(node('s:S1')!);
    await user.click(screen.getByRole('button', { name: 'E1' }));
    expect(node('e:E1')).not.toBeNull();
    expect(node('e:E1')).toHaveAttribute('aria-selected', 'true');
  });

  it('colours claim nodes by status and supports keyboard toggling', async () => {
    const { node } = renderMap();
    expect(node('s:S1')).toHaveClass('st-verified');
    expect(node('s:S2')).toHaveClass('st-unverified');
    const q = node('queries')!;
    q.focus();
    fireEvent.keyDown(q, { key: 'ArrowRight' });
    expect(node('q:Q1')).not.toBeNull();
    fireEvent.keyDown(node('queries')!, { key: 'ArrowLeft' });
    expect(node('q:Q1')).toBeNull();
  });
});
