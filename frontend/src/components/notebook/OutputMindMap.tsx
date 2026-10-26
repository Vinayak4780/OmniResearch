import { useMemo, useState } from 'react';
import type { NotebookCitation, OutputMapNode, OutputMindMap as MapData } from '../../types/notebooks';
import { CitationChip } from './NotebookAnswer';

interface Keyed {
  key: string;
  label: string;
  citations: string[];
  children: Keyed[];
}

function keyed(nodes: OutputMapNode[] | undefined, prefix: string): Keyed[] {
  return (nodes ?? [])
    .filter((n) => n && typeof n.label === 'string')
    .map((n, i) => {
      const key = `${prefix}${i}`;
      return { key, label: n.label, citations: Array.isArray(n.citations) ? n.citations : [], children: keyed(n.children, `${key}.`) };
    });
}

/** Citation ids in reading order (central → branches, depth first), for 1..n numbering. */
export function mapCitationOrder(branches: Keyed[]): string[] {
  const out: string[] = [];
  const walk = (ns: Keyed[]) => {
    for (const n of ns) {
      for (const c of n.citations) if (!out.includes(c)) out.push(c);
      walk(n.children);
    }
  };
  walk(branches);
  return out;
}

/** The map as an indented plain-text outline (for Copy). */
export function mindMapToText(data: MapData): string {
  const lines: string[] = [data.central ?? 'Mind map'];
  const walk = (ns: OutputMapNode[] | undefined, depth: number) => {
    for (const n of ns ?? []) {
      lines.push(`${'  '.repeat(depth)}- ${n.label}`);
      walk(n.children, depth + 1);
    }
  };
  walk(data.branches, 0);
  return lines.join('\n');
}

const HUES = [222, 158, 28, 280, 340, 195, 95, 250];

interface Props {
  data: MapData;
  citations: Record<string, NotebookCitation>;
  onOpen: (id: string, c: NotebookCitation) => void;
}

/** Horizontal, collapsible mind map: central idea → branches → details. Scrolls sideways inside its own box. */
export function OutputMindMap({ data, citations, onOpen }: Props) {
  const branches = useMemo(() => keyed(data.branches, 'b'), [data.branches]);
  const order = useMemo(() => mapCitationOrder(branches), [branches]);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const parents = useMemo(() => {
    const out: string[] = [];
    const walk = (ns: Keyed[]) => ns.forEach((n) => n.children.length && (out.push(n.key), walk(n.children)));
    walk(branches);
    return out;
  }, [branches]);

  if (!branches.length) return <p className="small muted">This mind map is empty.</p>;

  const toggle = (k: string) =>
    setCollapsed((c) => {
      const n = new Set(c);
      if (n.has(k)) n.delete(k);
      else n.add(k);
      return n;
    });

  const renderNode = (n: Keyed, depth: number, hue: number) => {
    const isCollapsed = collapsed.has(n.key);
    const hasKids = n.children.length > 0;
    return (
      <li
        key={n.key}
        className={`om-item om-depth-${depth}`}
        role="treeitem"
        aria-expanded={hasKids ? !isCollapsed : undefined}
        aria-level={depth + 1}
        aria-label={n.label}
        style={{ ['--om-hue' as string]: hue }}
      >
        <div className="om-node">
          <span className="om-label">{n.label}</span>
          {n.citations.length > 0 && (
            <span className="om-cites">
              {n.citations.map((id) => (
                <CitationChip key={id} n={citations[id] ? String(order.indexOf(id) + 1) : '?'} id={id} c={citations[id]} onOpen={onOpen} />
              ))}
            </span>
          )}
          {hasKids && (
            <button
              type="button"
              className="om-toggle"
              aria-label={`${isCollapsed ? 'Expand' : 'Collapse'} ${n.label}`}
              aria-expanded={!isCollapsed}
              onClick={() => toggle(n.key)}
            >
              {isCollapsed ? `+${n.children.length}` : '−'}
            </button>
          )}
        </div>
        {hasKids && !isCollapsed && (
          <ul className="om-kids" role="group">
            {n.children.map((c) => renderNode(c, depth + 1, hue))}
          </ul>
        )}
      </li>
    );
  };

  return (
    <div className="om-wrap">
      <div className="om-toolbar">
        <span className="small muted">Click − or + to fold a branch.</span>
        <span className="om-toolbar-btns">
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => setCollapsed(new Set())} disabled={!collapsed.size}>
            Expand all
          </button>
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => setCollapsed(new Set(parents))} disabled={!parents.length}>
            Collapse all
          </button>
        </span>
      </div>
      <div className="om-scroll" tabIndex={0} aria-label="Mind map (scroll sideways to see more)">
        <div className="om-tree" role="tree" aria-label={data.central || 'Mind map'}>
          <div className="om-central">{data.central || 'Mind map'}</div>
          <ul className="om-kids om-branches" role="group">
            {branches.map((b, i) => renderNode(b, 0, HUES[i % HUES.length]))}
          </ul>
        </div>
      </div>
    </div>
  );
}
