/** Pure helpers for the investigation mind map: tree building, collapsing and cross-link resolution. */
import type { ClaimStatus, MindMap, MindMapEdge, MindMapNode } from '../../../types/api';

export interface TreeNode {
  id: string;
  node: MindMapNode;
  children: TreeNode[];
}

/** A node as rendered: children are omitted when its branch is collapsed. */
export interface VisibleNode {
  id: string;
  node: MindMapNode;
  children: VisibleNode[];
  /** Number of descendants hidden by collapsing (0 when expanded). */
  hiddenCount: number;
  hasChildren: boolean;
  collapsed: boolean;
}

export interface Index {
  byId: Map<string, MindMapNode>;
  parentOf: Map<string, string>;
  childrenOf: Map<string, string[]>;
}

export function indexMindMap(mm: MindMap): Index {
  const byId = new Map(mm.nodes.map((n) => [n.id, n]));
  const parentOf = new Map<string, string>();
  const childrenOf = new Map<string, string[]>();
  for (const e of mm.edges) {
    if (e.type !== 'child' || !byId.has(e.source) || !byId.has(e.target) || parentOf.has(e.target)) continue;
    parentOf.set(e.target, e.source);
    const list = childrenOf.get(e.source) ?? [];
    list.push(e.target);
    childrenOf.set(e.source, list);
  }
  return { byId, parentOf, childrenOf };
}

export function rootId(mm: MindMap, idx: Index = indexMindMap(mm)): string | null {
  if (idx.byId.has('root')) return 'root';
  return mm.nodes.find((n) => !idx.parentOf.has(n.id))?.id ?? null;
}

export function buildTree(mm: MindMap, idx: Index = indexMindMap(mm)): TreeNode | null {
  const rid = rootId(mm, idx);
  if (!rid) return null;
  const seen = new Set<string>();
  const make = (id: string): TreeNode => {
    seen.add(id);
    const kids = (idx.childrenOf.get(id) ?? []).filter((c) => !seen.has(c));
    return { id, node: idx.byId.get(id)!, children: kids.map(make) };
  };
  return make(rid);
}

/** Searches start collapsed; findings, sources, contradictions and gaps start expanded. */
export function defaultCollapsed(mm: MindMap, idx: Index = indexMindMap(mm)): Set<string> {
  const out = new Set<string>();
  for (const n of mm.nodes) {
    const kids = idx.childrenOf.get(n.id) ?? [];
    if (n.id === 'queries') out.add(n.id);
    else if (n.type === 'group' && kids.length > 0 && kids.every((k) => idx.byId.get(k)?.type === 'query')) out.add(n.id);
  }
  return out;
}

function countDescendants(t: TreeNode): number {
  return t.children.reduce((a, c) => a + 1 + countDescendants(c), 0);
}

export function visibleTree(t: TreeNode, collapsed: Set<string>): VisibleNode {
  const isCollapsed = collapsed.has(t.id) && t.children.length > 0;
  return {
    id: t.id,
    node: t.node,
    hasChildren: t.children.length > 0,
    collapsed: isCollapsed,
    hiddenCount: isCollapsed ? countDescendants(t) : 0,
    children: isCollapsed ? [] : t.children.map((c) => visibleTree(c, collapsed)),
  };
}

export function visibleIds(v: VisibleNode, out = new Set<string>()): Set<string> {
  out.add(v.id);
  for (const c of v.children) visibleIds(c, out);
  return out;
}

/** The node itself if visible, else its closest visible ancestor (a collapsed branch). */
export function nearestVisible(id: string, idx: Index, visible: Set<string>): string | null {
  let cur: string | undefined = id;
  while (cur !== undefined) {
    if (visible.has(cur)) return cur;
    cur = idx.parentOf.get(cur);
  }
  return null;
}

export interface CrossLink {
  source: string;
  target: string;
  type: 'cites' | 'contradicts' | string;
  /** Original edge endpoints (before mapping onto collapsed ancestors). */
  from: string;
  to: string;
}

/**
 * Non-tree edges touching `focusId`, with hidden endpoints mapped to their nearest visible ancestor.
 * Links that collapse onto a single node, and duplicates, are dropped.
 */
export function crossLinksFor(mm: MindMap, focusId: string | null, idx: Index, visible: Set<string>): CrossLink[] {
  if (!focusId) return [];
  const out: CrossLink[] = [];
  const keys = new Set<string>();
  for (const e of mm.edges as MindMapEdge[]) {
    if (e.type === 'child' || (e.source !== focusId && e.target !== focusId)) continue;
    const s = nearestVisible(e.source, idx, visible);
    const t = nearestVisible(e.target, idx, visible);
    if (!s || !t || s === t) continue;
    const key = `${s}|${t}|${e.type}`;
    if (keys.has(key)) continue;
    keys.add(key);
    out.push({ source: s, target: t, type: e.type, from: e.source, to: e.target });
  }
  return out;
}

/** Ids of every node with children (for "collapse all"), excluding the root. */
export function branchIds(mm: MindMap, idx: Index = indexMindMap(mm)): string[] {
  const rid = rootId(mm, idx);
  return [...idx.childrenOf.keys()].filter((id) => id !== rid);
}

export const STATUS_INFO: Record<ClaimStatus | 'none', { label: string; cls: string; glyph: string }> = {
  verified: { label: 'Verified', cls: 'st-verified', glyph: '✓' },
  partially_verified: { label: 'Partially verified', cls: 'st-partial', glyph: '◐' },
  mechanically_verified: { label: 'Mechanically verified', cls: 'st-partial', glyph: '◐' },
  unverified: { label: 'Unverified', cls: 'st-unverified', glyph: '!' },
  uncited: { label: 'No citation', cls: 'st-unverified', glyph: '?' },
  interpretation: { label: 'Interpretation', cls: 'st-interp', glyph: '~' },
  none: { label: 'Not checked', cls: 'st-none', glyph: '·' },
};

export function statusInfo(status: string | null | undefined) {
  return STATUS_INFO[(status ?? 'none') as ClaimStatus] ?? STATUS_INFO.none;
}

/** Legend rows (merged where two statuses share a colour). */
export const STATUS_LEGEND: { key: string; label: string; cls: string; glyph: string }[] = [
  { key: 'verified', ...STATUS_INFO.verified },
  { key: 'partial', label: 'Partially / mechanically verified', cls: 'st-partial', glyph: '◐' },
  { key: 'unverified', label: 'Unverified or uncited', cls: 'st-unverified', glyph: '!' },
  { key: 'interpretation', ...STATUS_INFO.interpretation },
];

export const TYPE_LABEL: Record<string, string> = {
  question: 'Question',
  group: 'Group',
  query: 'Search',
  finding: 'Finding',
  claim: 'Claim',
  document: 'Document',
  website: 'Website',
  evidence: 'Evidence',
  contradiction: 'Contradiction',
  gap: 'Open question',
};

export function truncate(s: string, n: number): string {
  return s.length <= n ? s : `${s.slice(0, n - 1).trimEnd()}…`;
}
