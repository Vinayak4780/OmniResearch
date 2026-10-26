import { useCallback, useEffect, useRef, useState } from 'react';
import { fmtDuration, useNow } from '../../hooks/useNow';
import { usePolling } from '../../hooks/usePolling';
import { api } from '../../services/api';
import type { ResearchEvent } from '../../types/api';
import type { NotebookReport } from '../../types/notebooks';
import { STAGES, buildSteps, progressInfo } from '../research/StageTimeline';
import { ProgressBar } from '../ui';

/** Report statuses that are still in progress. */
export const ACTIVE_REPORT = new Set(['queued', 'running']);

/** Compact live progress for a running report: stage k/14 and elapsed time. */
export function RunningProgress({ report }: { report: NotebookReport }) {
  const [events, setEvents] = useState<ResearchEvent[]>([]);
  const lastId = useRef(0);
  const now = useNow(1000, true);
  const fetchMore = useCallback(async () => {
    const more = await api.researchEvents(report.id, lastId.current);
    if (more.length) {
      lastId.current = more[more.length - 1].id;
      setEvents((e) => [...e, ...more]);
    }
  }, [report.id]);
  useEffect(() => {
    void fetchMore().catch(() => undefined); // show the current stage right away, then poll
  }, [fetchMore]);
  usePolling(fetchMore, 3000, true);
  const p = progressInfo(buildSteps(events, true), report.status);
  const started = Date.parse(report.created_at);
  return (
    <div className="nb-report-progress">
      <ProgressBar value={p.fraction} label={`Progress of “${report.question}”`} />
      <span className="small muted">
        Stage {Math.max(1, p.furthestIndex + 1)}/{STAGES.length}
        {p.round > 0 ? ` · round ${p.round}` : ''} · {Number.isFinite(started) ? fmtDuration(now - started) : ''}
      </span>
    </div>
  );
}
