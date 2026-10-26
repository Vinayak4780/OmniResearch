import { hierarchy, tree, type HierarchyPointNode } from 'd3-hierarchy';
import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import { Link } from 'react-router-dom';
import type { MindMap, MindMapNode } from '../../../types/api';
import { Badge, Chip, EmptyState } from '../../ui';
import {
  STATUS_LEGEND,
  TYPE_LABEL,
  branchIds,
  buildTree,
  crossLinksFor,
  defaultCollapsed,
  indexMindMap,
  statusInfo,
  truncate,
  visibleIds,
  visibleTree,
  type CrossLink,
  type VisibleNode,
} from './mindmapLogic';

const NODE_W = 210;
const NODE_H = 28;
const ROW = 36;
const COL = 260;
const CANVAS_H = 580;

interface Transform {
  x: number;
  y: number;
  k: number;
}

interface Props {
  mindmap: MindMap;
  /** Open a local evidence item in the citation side panel. */
  onOpenEvidence: (evidenceId: string) => void;
}

type PNode = HierarchyPointNode<VisibleNode>;

function crossPath(a: PNode, b: PNode): string {
  const ax = a.y, ay = a.x, bx = b.y, by = b.x;
  if (bx > ax + NODE_W) {
    const mx = (ax + NODE_W + bx) / 2;
    return `M${ax + NODE_W},${ay} C${mx},${ay} ${mx},${by} ${bx},${by}`;
  }
  if (ax > bx + NODE_W) {
    const mx = (bx + NODE_W + ax) / 2;
    return `M${ax},${ay} C${mx},${ay} ${mx},${by} ${bx + NODE_W},${by}`;
  }
  const off = 40 + Math.min(220, Math.abs(ay - by) * 0.25);
  const sx = Math.max(ax, bx) + NODE_W;
  return `M${ax + NODE_W},${ay} C${sx + off},${ay} ${sx + off},${by} ${bx + NODE_W},${by}`;
}

function childPath(p: PNode, c: PNode): string {
  const sx = p.y + NODE_W, sy = p.x, tx = c.y, ty = c.x;
  const mx = (sx + tx) / 2;
  return `M${sx},${sy} C${mx},${sy} ${mx},${ty} ${tx},${ty}`;
}

function NodeDetails({ n, onOpenEvidence, byId, onSelect }: { n: MindMapNode; onOpenEvidence: Props['onOpenEvidence']; byId: Map<string, MindMapNode>; onSelect: (id: string) => void }) {
  const d = n.data ?? {};
  const cites = (d.citations ?? []) as string[];
  const citeChips = cites.length > 0 && (
    <div className="chip-row small">
      <span className="muted">Cites:</span>
      {cites.map((c) =>
        byId.has(`e:${c}`) ? (
          <button key={c} type="button" className="citation-chip" onClick={() => onSelect(`e:${c}`)} title={`Show ${c} in the map`}>
            {c}
          </button>
        ) : (
          <span key={c} className="citation citation-unknown">
            {c}
          </span>
        ),
      )}
    </div>
  );
  return (
    <div className="mm-details" aria-live="polite">
      <div className="small muted">{TYPE_LABEL[n.type] ?? n.type}</div>
      {n.type === 'claim' ? (
        <>
          <p className="mm-details-text">{d.text ?? n.label}</p>
          <div className="chip-row small">
            <span className={`status-pill ${statusInfo(d.status).cls}`}>
              <span aria-hidden="true">{statusInfo(d.status).glyph}</span> {statusInfo(d.status).label}
            </span>
            {d.kind === 'interpretation' && <Badge>interpretation</Badge>}
            {d.statement_id && <span className="mono muted">{d.statement_id}</span>}
          </div>
          {citeChips}
        </>
      ) : n.type === 'evidence' ? (
        <>
          <p className="mm-details-title">{n.label}</p>
          <div className="chip-row small">
            {d.element_type && <Chip>{d.element_type}</Chip>}
            {d.extraction_method && <Chip tone={d.extraction_method.startsWith('ocr') ? 'warning' : 'neutral'}>{d.extraction_method}</Chip>}
            {d.cited ? <Badge tone="success">cited in report</Badge> : <Badge>not cited</Badge>}
            {d.inspected_visually && <Badge tone="warning">inspected visually</Badge>}
          </div>
          {d.excerpt && <blockquote className="citation-excerpt pre-wrap">{d.excerpt}</blockquote>}
          <div className="btn-row">
            {d.source_type === 'local' && d.evidence_id ? (
              <button type="button" className="btn btn-sm btn-primary" onClick={() => onOpenEvidence(d.evidence_id!)}>
                Open in side panel
              </button>
            ) : d.url ? (
              <a className="btn btn-sm" href={d.url} target="_blank" rel="noopener noreferrer">
                Open web page
              </a>
            ) : null}
          </div>
        </>
      ) : n.type === 'contradiction' ? (
        <>
          <p className="mm-details-title">{(d.full_text as string | undefined) ?? n.label}</p>
          {d.description && <p>{d.description}</p>}
          {d.explanation && (
            <p className="small">
              <span className="muted">Possible explanation:</span> {d.explanation}
            </p>
          )}
          {citeChips}
        </>
      ) : (
        <>
          <p className="mm-details-title">{(d.full_text as string | undefined) ?? (d.text as string | undefined) ?? n.label}</p>
          {n.type === 'query' && (d.source || d.focus || typeof d.round === 'number') && (
            <p className="small muted">
              {typeof d.round === 'number' ? (d.round === 0 ? 'initial search' : `follow-up round ${d.round}`) : ''}
              {d.source ? ` · source: ${d.source}` : ''} {d.focus ? `· focus: ${d.focus}` : ''}
            </p>
          )}
          {n.type === 'document' && n.id.startsWith('d:') && (
            <Link className="btn btn-sm" to={`/documents/${(d.document_id as string | undefined) ?? n.id.slice(2)}/view?page=1`}>
              Open document
            </Link>
          )}
        </>
      )}
    </div>
  );
}

export function MindMapView({ mindmap, onOpenEvidence }: Props) {
  const idx = useMemo(() => indexMindMap(mindmap), [mindmap]);
  const fullTree = useMemo(() => buildTree(mindmap, idx), [mindmap, idx]);
  const [collapsed, setCollapsed] = useState<Set<string>>(() => defaultCollapsed(mindmap, idx));
  const [selected, setSelected] = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [width, setWidth] = useState(900);
  const [t, setT] = useState<Transform>({ x: 24, y: CANVAS_H / 2, k: 1 });
  const wrapRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const nodeRefs = useRef(new Map<string, SVGGElement>());
  const drag = useRef<{ px: number; py: number; x: number; y: number } | null>(null);

  useEffect(() => {
    setCollapsed(defaultCollapsed(mindmap, idx));
    setSelected(null);
  }, [mindmap, idx]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver((entries) => setWidth(Math.max(320, Math.round(entries[0].contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const visRoot = useMemo(() => (fullTree ? visibleTree(fullTree, collapsed) : null), [fullTree, collapsed]);
  const layout = useMemo(() => {
    if (!visRoot) return null;
    const h = hierarchy(visRoot, (d) => d.children);
    return tree<VisibleNode>()
      .nodeSize([ROW, COL])
      .separation((a, b) => (a.parent === b.parent ? 1 : 1.3))(h);
  }, [visRoot]);
  const nodes = useMemo(() => (layout ? layout.descendants() : []), [layout]);
  const pos = useMemo(() => new Map(nodes.map((n) => [n.data.id, n])), [nodes]);
  const visible = useMemo(() => (visRoot ? visibleIds(visRoot) : new Set<string>()), [visRoot]);
  const focus = hovered ?? selected;
  const links: CrossLink[] = useMemo(() => crossLinksFor(mindmap, focus, idx, visible), [mindmap, focus, idx, visible]);
  const linked = useMemo(() => new Set(links.flatMap((l) => [l.source, l.target])), [links]);

  const fit = useCallback(() => {
    if (!nodes.length) return;
    const xs = nodes.map((n) => n.y), ys = nodes.map((n) => n.x);
    const minX = Math.min(...xs), maxX = Math.max(...xs) + NODE_W, minY = Math.min(...ys) - NODE_H, maxY = Math.max(...ys) + NODE_H;
    const k = Math.max(0.25, Math.min(1.1, (width - 48) / (maxX - minX), (CANVAS_H - 48) / (maxY - minY)));
    setT({ k, x: (width - (maxX - minX) * k) / 2 - minX * k, y: (CANVAS_H - (maxY - minY) * k) / 2 - minY * k });
  }, [nodes, width]);

  // Initial view: fit, but never smaller than 0.6 (then the root stays on screen at the left).
  const fitted = useRef(false);
  useEffect(() => {
    if (fitted.current || !nodes.length) return;
    fitted.current = true;
    const ys = nodes.map((n) => n.x);
    const span = Math.max(...ys) - Math.min(...ys) + 2 * NODE_H;
    const k = Math.max(0.85, Math.min(1, (CANVAS_H - 48) / span));
    setT({ k, x: 24, y: CANVAS_H / 2 });
  }, [nodes]);

  // Wheel zoom around the cursor (non-passive so the page does not scroll).
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = svg.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      setT((cur) => {
        const k = Math.max(0.2, Math.min(3, cur.k * Math.exp(-e.deltaY * 0.0015)));
        return { k, x: px - ((px - cur.x) * k) / cur.k, y: py - ((py - cur.y) * k) / cur.k };
      });
    };
    svg.addEventListener('wheel', onWheel, { passive: false });
    return () => svg.removeEventListener('wheel', onWheel);
  }, []);

  const zoom = (f: number) =>
    setT((cur) => {
      const k = Math.max(0.2, Math.min(3, cur.k * f));
      const cx = width / 2, cy = CANVAS_H / 2;
      return { k, x: cx - ((cx - cur.x) * k) / cur.k, y: cy - ((cy - cur.y) * k) / cur.k };
    });

  const toggle = (id: string) =>
    setCollapsed((c) => {
      const n = new Set(c);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });

  /** Expand every collapsed ancestor of the node, then select and focus it. */
  const reveal = (id: string) => {
    setCollapsed((c) => {
      const n = new Set(c);
      for (let p = idx.parentOf.get(id); p !== undefined; p = idx.parentOf.get(p)) n.delete(p);
      return n;
    });
    focusNode(id);
  };

  const order = useMemo(() => nodes.map((n) => n.data.id), [nodes]);
  const [centerOn, setCenterOn] = useState<string | null>(null);
  const focusNode = (id: string) => {
    setSelected(id);
    setCenterOn(id);
  };
  // Once the (possibly re-laid-out) node exists: pan it into view if it is off-canvas, then focus it.
  useEffect(() => {
    if (!centerOn) return;
    const n = pos.get(centerOn);
    if (!n) return;
    setCenterOn(null);
    setT((cur) => {
      const sx = n.y * cur.k + cur.x;
      const sy = n.x * cur.k + cur.y;
      const inside = sx >= 8 && sx + NODE_W * cur.k <= width - 8 && sy >= NODE_H && sy <= CANVAS_H - NODE_H;
      return inside ? cur : { ...cur, x: width / 2 - (n.y + NODE_W / 2) * cur.k, y: CANVAS_H / 2 - n.x * cur.k };
    });
    requestAnimationFrame(() => nodeRefs.current.get(n.data.id)?.focus({ preventScroll: true }));
  }, [centerOn, pos, width]);
  const onNodeKey = (e: KeyboardEvent, n: PNode) => {
    const i = order.indexOf(n.data.id);
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      if (selected === n.data.id && n.data.hasChildren) toggle(n.data.id);
      else setSelected(n.data.id);
    } else if (e.key === 'ArrowDown' && i < order.length - 1) {
      e.preventDefault();
      focusNode(order[i + 1]);
    } else if (e.key === 'ArrowUp' && i > 0) {
      e.preventDefault();
      focusNode(order[i - 1]);
    } else if (e.key === 'ArrowRight' && n.data.hasChildren) {
      e.preventDefault();
      if (n.data.collapsed) toggle(n.data.id);
      else if (n.children?.length) focusNode(n.children[0].data.id);
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      if (n.data.hasChildren && !n.data.collapsed) toggle(n.data.id);
      else if (n.parent) focusNode(n.parent.data.id);
    }
  };

  const onPointerDown = (e: PointerEvent<SVGSVGElement>) => {
    if ((e.target as Element).closest('.mm-node')) return;
    drag.current = { px: e.clientX, py: e.clientY, x: t.x, y: t.y };
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent<SVGSVGElement>) => {
    const d = drag.current;
    if (!d) return;
    setT((cur) => ({ ...cur, x: d.x + e.clientX - d.px, y: d.y + e.clientY - d.py }));
  };
  const endDrag = () => {
    drag.current = null;
  };

  if (!layout || !fullTree) return <EmptyState title="The mind map is empty" />;
  const selNode = selected ? idx.byId.get(selected) : undefined;
  const tabStop = selected && visible.has(selected) ? selected : layout.data.id;

  return (
    <div className="mindmap">
      <div className="mm-toolbar" role="toolbar" aria-label="Mind map controls">
        <button type="button" className="btn btn-sm" onClick={() => zoom(1.25)} aria-label="Zoom in">
          +
        </button>
        <button type="button" className="btn btn-sm" onClick={() => zoom(0.8)} aria-label="Zoom out">
          −
        </button>
        <button type="button" className="btn btn-sm" onClick={fit}>
          Fit
        </button>
        <button type="button" className="btn btn-sm" onClick={() => setCollapsed(new Set())}>
          Expand all
        </button>
        <button type="button" className="btn btn-sm" onClick={() => setCollapsed(new Set(branchIds(mindmap, idx)))}>
          Collapse all
        </button>
        <button type="button" className="btn btn-sm btn-ghost" onClick={() => setCollapsed(defaultCollapsed(mindmap, idx))}>
          Reset
        </button>
        <span className="small muted mm-hint">Drag to pan · scroll to zoom · click a node for details · arrow keys to move</span>
      </div>
      <div className="mm-body">
        <div className="mm-canvas" ref={wrapRef}>
          <svg
            ref={svgRef}
            className="mm-svg"
            width="100%"
            height={CANVAS_H}
            viewBox={`0 0 ${width} ${CANVAS_H}`}
            role="tree"
            aria-label={`Mind map: ${mindmap.title || 'investigation'}`}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            onClick={(e) => {
              if (!(e.target as Element).closest('.mm-node')) setSelected(null);
            }}
          >
            <defs>
              <marker id="mm-arrow-cites" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" className="mm-arrow-cites" />
              </marker>
              <marker id="mm-arrow-contradicts" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" className="mm-arrow-contradicts" />
              </marker>
            </defs>
            <g transform={`translate(${t.x},${t.y}) scale(${t.k})`}>
              <g className="mm-links">
                {layout.links().map((l) => (
                  <path key={`${l.source.data.id}->${l.target.data.id}`} d={childPath(l.source, l.target)} className="mm-link" />
                ))}
              </g>
              <g className="mm-cross" data-testid="mm-cross-links">
                {links.map((l) => {
                  const a = pos.get(l.source), b = pos.get(l.target);
                  if (!a || !b) return null;
                  return <path key={`${l.source}-${l.target}-${l.type}`} d={crossPath(a, b)} className={`mm-xlink mm-xlink-${l.type}`} markerEnd={`url(#mm-arrow-${l.type === 'contradicts' ? 'contradicts' : 'cites'})`} />;
                })}
              </g>
              {nodes.map((n) => {
                const v = n.data;
                const st = v.node.type === 'claim' ? statusInfo(v.node.data?.status) : null;
                const isSel = selected === v.id;
                const label = truncate(v.node.label, st ? 27 : 30);
                return (
                  <g
                    key={v.id}
                    ref={(el) => {
                      if (el) nodeRefs.current.set(v.id, el);
                      else nodeRefs.current.delete(v.id);
                    }}
                    className={`mm-node mm-${v.node.type} ${isSel ? 'mm-selected' : ''} ${linked.has(v.id) && focus !== v.id ? 'mm-linked' : ''} ${st ? st.cls : ''} ${v.node.type === 'evidence' && !v.node.data?.cited ? 'mm-uncited' : ''}`}
                    transform={`translate(${n.y},${n.x - NODE_H / 2})`}
                    role="treeitem"
                    aria-level={n.depth + 1}
                    aria-selected={isSel}
                    aria-expanded={v.hasChildren ? !v.collapsed : undefined}
                    aria-label={`${TYPE_LABEL[v.node.type] ?? v.node.type}: ${v.node.label}${st ? `, ${st.label}` : ''}${v.collapsed ? `, collapsed, ${v.hiddenCount} hidden` : ''}`}
                    tabIndex={v.id === tabStop ? 0 : -1}
                    data-node-id={v.id}
                    onClick={() => setSelected(v.id)}
                    onDoubleClick={() => v.hasChildren && toggle(v.id)}
                    onMouseEnter={() => setHovered(v.id)}
                    onMouseLeave={() => setHovered((h) => (h === v.id ? null : h))}
                    onKeyDown={(e) => onNodeKey(e, n)}
                  >
                    <title>{v.node.type === 'claim' ? `${v.node.data?.text ?? v.node.label} (${st?.label})` : v.node.label}</title>
                    <rect className="mm-box" width={NODE_W} height={NODE_H} rx={NODE_H / 2} />
                    {st ? (
                      <>
                        <circle className="mm-status" cx={14} cy={NODE_H / 2} r={8} />
                        <text className="mm-glyph" x={14} y={NODE_H / 2 + 4} textAnchor="middle" aria-hidden="true">
                          {st.glyph}
                        </text>
                      </>
                    ) : (
                      <circle className="mm-dot" cx={12} cy={NODE_H / 2} r={4} />
                    )}
                    <text className="mm-label" x={st ? 28 : 22} y={NODE_H / 2 + 4}>
                      {label}
                    </text>
                    {v.hasChildren && (
                      <g
                        className="mm-toggle"
                        transform={`translate(${NODE_W},${NODE_H / 2})`}
                        onClick={(e) => {
                          e.stopPropagation();
                          toggle(v.id);
                        }}
                        aria-hidden="true"
                      >
                        <circle r={9} />
                        <text y={4} textAnchor="middle">
                          {v.collapsed ? (v.hiddenCount > 99 ? '99+' : `+${v.hiddenCount}`) : '−'}
                        </text>
                      </g>
                    )}
                  </g>
                );
              })}
            </g>
          </svg>
          <div className="mm-legend small" aria-label="Legend">
            {STATUS_LEGEND.map((s) => (
              <span key={s.key} className={`status-pill ${s.cls}`}>
                <span aria-hidden="true">{s.glyph}</span> {s.label}
              </span>
            ))}
            <span className="mm-legend-line mm-legend-cites">cites</span>
            <span className="mm-legend-line mm-legend-contradicts">contradicts</span>
            <span className="muted">Cross-links show for the hovered or selected node.</span>
          </div>
        </div>
        <aside className="mm-side" aria-label="Node details">
          {selNode ? (
            <NodeDetails n={selNode} onOpenEvidence={onOpenEvidence} byId={idx.byId} onSelect={reveal} />
          ) : (
            <p className="small muted">Select a node to see its details. Claims are coloured by verification status; select one to see the evidence it cites.</p>
          )}
          {selNode && links.length > 0 && focus === selected && (
            <p className="small muted">
              {links.filter((l) => l.type === 'cites').length} citation link(s), {links.filter((l) => l.type === 'contradicts').length} contradiction link(s) shown.
            </p>
          )}
        </aside>
      </div>
    </div>
  );
}
