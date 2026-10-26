import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useResource } from '../../hooks/useResource';
import { api } from '../../services/api';
import type { Evidence, ResearchJob } from '../../types/api';
import { citationTarget, isWebUrl, type CitationTarget } from '../../utils/citations';
import { Card, ErrorMessage, Spinner, Tabs } from '../ui';
import { CitationPanel } from './CitationPanel';
import { EvidencePanel } from './EvidencePanel';
import { FollowUpChat } from './FollowUpChat';
import { ReportReader } from './ReportReader';

// The mind map (and d3-hierarchy) is only downloaded when the tab is first opened.
const MindMapView = lazy(() => import('./mindmap/MindMapView').then((m) => ({ default: m.MindMapView })));

type TabId = 'report' | 'mindmap' | 'evidence';
const TABS: TabId[] = ['report', 'mindmap', 'evidence'];

function MindMapTab({ jobId, onOpenEvidence }: { jobId: string; onOpenEvidence: (id: string) => void }) {
  const mm = useResource(() => api.mindmap(jobId), [jobId]);
  if (mm.loading && !mm.data) return <Spinner label="Building the mind map…" />;
  if (mm.error) return <ErrorMessage error={mm.error} title="Could not load the mind map" />;
  return mm.data ? (
    <Suspense fallback={<Spinner label="Loading the mind map…" />}>
      <MindMapView mindmap={mm.data} onOpenEvidence={onOpenEvidence} />
    </Suspense>
  ) : null;
}

/**
 * Completed investigation: Report | Mind map | Evidence, a citation side panel (page image with the cited
 * region highlighted) and the follow-up chat. Local citations open the panel instead of navigating.
 */
export function CompletedJob({ job }: { job: ResearchJob }) {
  const [params, setParams] = useSearchParams();
  const tabParam = params.get('tab') as TabId | null;
  const tab: TabId = tabParam && TABS.includes(tabParam) ? tabParam : 'report';
  const [visited, setVisited] = useState<Set<TabId>>(() => new Set([tab]));
  const ev = useResource(() => api.researchEvidence(job.id), [job.id]);
  const evidence = useMemo(() => ev.data ?? [], [ev.data]);
  const evMap = useMemo(() => new Map(evidence.filter((e) => e.id).map((e) => [e.id as string, e])), [evidence]);
  const [panel, setPanel] = useState<CitationTarget | null>(null);
  const panelRef = useRef<HTMLElement>(null);

  const setTab = (t: string) => {
    const next = new URLSearchParams(params);
    if (t === 'report') next.delete('tab');
    else next.set('tab', t);
    setParams(next, { replace: true });
  };
  useEffect(() => setVisited((v) => (v.has(tab) ? v : new Set([...v, tab]))), [tab]);

  const openCitation = useCallback(
    (evidenceId: string | undefined, href?: string) => {
      const target = citationTarget(evidenceId, href, evMap);
      if (target) {
        setPanel(target);
        // On narrow screens the panel is stacked above the content: bring it into view.
        requestAnimationFrame(() => {
          if (window.matchMedia?.('(max-width: 1099px)').matches) panelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
        return;
      }
      const e = evidenceId ? evMap.get(evidenceId) : undefined;
      const url = e?.canonical_url || e?.final_url || e?.url || href;
      if (isWebUrl(url)) window.open(url, '_blank', 'noopener,noreferrer');
    },
    [evMap],
  );
  const closePanel = useCallback(() => setPanel(null), []);

  const onEvidenceAdded = useCallback(
    (added: Evidence[]) => {
      if (added.length) ev.setData((cur) => [...(cur ?? []), ...added.filter((a) => !(cur ?? []).some((c) => c.id === a.id))]);
      void ev.reload(true);
    },
    [ev],
  );

  return (
    <div className={`reader ${panel ? 'reader-with-panel' : ''}`}>
      <div className="reader-main">
        <Card className="reader-card">
          <Tabs
            label="Investigation views"
            active={tab}
            onChange={setTab}
            tabs={[
              { id: 'report', label: 'Report' },
              { id: 'mindmap', label: 'Mind map' },
              { id: 'evidence', label: `Evidence${ev.data ? ` (${evidence.length})` : ''}` },
            ]}
          />
          <div role="tabpanel" aria-label="Report" hidden={tab !== 'report'}>
            {job.report_md ? (
              <ReportReader markdown={job.report_md} report={job.report} onCitation={(id, href) => openCitation(id, href)} />
            ) : (
              <p className="muted">The report is empty.</p>
            )}
          </div>
          {visited.has('mindmap') && (
            <div role="tabpanel" aria-label="Mind map" hidden={tab !== 'mindmap'}>
              <MindMapTab jobId={job.id} onOpenEvidence={(id) => openCitation(id)} />
            </div>
          )}
          <div role="tabpanel" aria-label="Evidence" hidden={tab !== 'evidence'}>
            <ErrorMessage error={ev.error} title="Could not load evidence" />
            {ev.loading && !ev.data ? <Spinner label="Loading evidence…" /> : <EvidencePanel evidence={evidence} onOpen={(id) => openCitation(id)} />}
          </div>
        </Card>
        {tab === 'report' && (
          <Card className="chat-card">
            <FollowUpChat jobId={job.id} evidence={evMap} onCitation={(id, href) => openCitation(id, href)} onEvidenceAdded={onEvidenceAdded} />
          </Card>
        )}
      </div>
      {panel && (
        <aside className="reader-panel" ref={panelRef} aria-label="Cited source">
          <CitationPanel target={panel} evidence={panel.evidenceId ? evMap.get(panel.evidenceId) : undefined} onClose={closePanel} />
        </aside>
      )}
    </div>
  );
}
