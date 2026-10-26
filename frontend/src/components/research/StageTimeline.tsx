import { Fragment } from 'react';
import { fmtDuration, useNow } from '../../hooks/useNow';
import type { ResearchEvent, ResearchStatus } from '../../types/api';
import { fmtSeconds } from '../../utils/format';
import { ProgressBar } from '../ui';

export const STAGES = [
  'analyze_question',
  'plan_research',
  'decompose',
  'route_sources',
  'retrieve_local',
  'web_research',
  'rerank_evidence',
  'inspect_visuals',
  'assess_sufficiency',
  'follow_up',
  'compare_sources',
  'draft_report',
  'validate_citations',
  'finalize',
] as const;

export const STAGE_LABELS: Record<string, string> = {
  analyze_question: 'Analyze question',
  plan_research: 'Plan research',
  decompose: 'Decompose into sub-queries',
  route_sources: 'Route sources',
  retrieve_local: 'Retrieve from documents',
  web_research: 'Web research',
  rerank_evidence: 'Rerank evidence',
  inspect_visuals: 'Inspect figures',
  assess_sufficiency: 'Assess sufficiency',
  follow_up: 'Follow-up queries',
  compare_sources: 'Compare sources',
  draft_report: 'Draft report',
  validate_citations: 'Validate citations',
  finalize: 'Finalize report',
  system: 'System',
};

export interface Step {
  id: number;
  stage: string;
  status: 'running' | 'done' | 'system' | 'stopped';
  startedAt: string;
  endedAt?: string;
  message?: string;
  seconds?: number;
  data?: Record<string, unknown>;
  /** 0 = initial pass; N = N-th follow-up round (from "round N: …" messages of follow_up). */
  round: number;
}

const ROUND_RE = /^round\s+(\d+)\b/i;

/** Pair each stage's "started" event with its completion event (stages repeat in follow-up rounds). */
export function buildSteps(events: ResearchEvent[], jobActive: boolean): Step[] {
  const steps: Step[] = [];
  let round = 0;
  for (const ev of events) {
    if (ev.stage === 'system') {
      steps.push({ id: ev.id, stage: 'system', status: 'system', startedAt: ev.ts, endedAt: ev.ts, message: ev.message, data: ev.data, round });
      continue;
    }
    if (ev.message === 'started') {
      steps.push({ id: ev.id, stage: ev.stage, status: 'running', startedAt: ev.ts, round });
      continue;
    }
    let open: Step | undefined;
    for (let i = steps.length - 1; i >= 0; i--) {
      if (steps[i].stage === ev.stage && steps[i].status === 'running') {
        open = steps[i];
        break;
      }
    }
    const seconds = typeof ev.data?.seconds === 'number' ? ev.data.seconds : undefined;
    if (ev.stage === 'follow_up') {
      // data.round is authoritative (API round 3); older events only have it in the message text.
      const m = ROUND_RE.exec(ev.message);
      round = typeof ev.data?.round === 'number' ? ev.data.round : m ? Number(m[1]) : round + 1;
    }
    if (open) {
      Object.assign(open, { status: 'done', message: ev.message, seconds, data: ev.data, endedAt: ev.ts });
      if (ev.stage === 'follow_up') open.round = round;
    } else {
      steps.push({ id: ev.id, stage: ev.stage, status: 'done', startedAt: ev.ts, endedAt: ev.ts, message: ev.message, seconds, data: ev.data, round });
    }
  }
  if (!jobActive) for (const s of steps) if (s.status === 'running') s.status = 'stopped';
  return steps;
}

export interface ProgressInfo {
  /** 0..1, from the furthest pipeline stage reached (stage order, not time). */
  fraction: number;
  furthestIndex: number;
  current: Step | undefined;
  round: number;
  lastActivity: string | undefined;
}

export function progressInfo(steps: Step[], status: ResearchStatus): ProgressInfo {
  let furthest = -1;
  let furthestDone = true;
  for (const s of steps) {
    const i = STAGES.indexOf(s.stage as (typeof STAGES)[number]);
    if (i < 0) continue;
    if (i > furthest || (i === furthest && s.status === 'running')) {
      furthest = i;
      furthestDone = s.status === 'done';
    }
  }
  const running = steps.filter((s) => s.status === 'running');
  const last = steps[steps.length - 1];
  let fraction = furthest < 0 ? 0 : (furthest + (furthestDone ? 1 : 0.5)) / STAGES.length;
  if (status === 'completed') fraction = 1;
  return {
    fraction: Math.min(1, fraction),
    furthestIndex: furthest,
    current: running[running.length - 1],
    round: steps.reduce((m, s) => Math.max(m, s.round), 0),
    lastActivity: last ? (last.endedAt ?? last.startedAt) : undefined,
  };
}

function since(now: number, iso: string | undefined): number {
  const t = iso ? Date.parse(iso) : NaN;
  return Number.isFinite(t) ? Math.max(0, now - t) : 0;
}

function RunProgress({ steps, status, createdAt }: { steps: Step[]; status: ResearchStatus; createdAt: string }) {
  const now = useNow(1000, true);
  const p = progressInfo(steps, status);
  const running = steps.filter((s) => s.status === 'running');
  const idle = since(now, p.lastActivity);
  return (
    <div className="run-progress">
      <div className="run-progress-head">
        <div>
          <div className="run-current">
            {running.length ? (
              running.map((s) => (
                <span key={s.id} className="run-current-stage">
                  <span className="pulse-dot" aria-hidden="true" /> {STAGE_LABELS[s.stage] ?? s.stage}
                  <span className="muted small"> · {fmtDuration(since(now, s.startedAt))}</span>
                </span>
              ))
            ) : (
              <span className="muted">{status === 'queued' ? 'Queued — waiting to start' : 'Between stages…'}</span>
            )}
          </div>
          <div className="small muted">
            Stage {Math.max(0, p.furthestIndex + 1)} of {STAGES.length}
            {p.round > 0 && <strong className="round-badge"> · follow-up round {p.round}</strong>}
          </div>
        </div>
        <dl className="run-clock small">
          <div>
            <dt>Elapsed</dt>
            <dd className="num">{fmtDuration(since(now, createdAt))}</dd>
          </div>
          <div>
            <dt>Last activity</dt>
            <dd className={`num ${idle > 120000 ? 'text-warning' : ''}`}>{p.lastActivity ? `${fmtDuration(idle)} ago` : '—'}</dd>
          </div>
        </dl>
      </div>
      <ProgressBar value={p.fraction} label="Investigation progress by stage" />
      <p className="small muted">Real investigations usually take 5–10 minutes; model calls can run for a minute or more without new events.</p>
    </div>
  );
}

function StepDetails({ data }: { data?: Record<string, unknown> }) {
  if (!data) return null;
  const subs = Array.isArray(data.subqueries) ? (data.subqueries as { query?: string; source?: string; focus?: string }[]) : null;
  const gaps = Array.isArray(data.gaps) ? (data.gaps as string[]) : null;
  const plan = data.plan && typeof data.plan === 'object' ? (data.plan as { steps?: string[]; rationale?: string }) : null;
  const analysis = data.analysis && typeof data.analysis === 'object' ? (data.analysis as { key_concepts?: string[]; comparison_dimensions?: string[] }) : null;
  if (!subs?.length && !gaps?.length && !plan?.steps?.length && !analysis) return null;
  return (
    <details className="step-details small">
      <summary>Details</summary>
      {analysis && (
        <div>
          {analysis.key_concepts?.length ? <p>Key concepts: {analysis.key_concepts.join(', ')}</p> : null}
          {analysis.comparison_dimensions?.length ? <p>Compare by: {analysis.comparison_dimensions.join(', ')}</p> : null}
        </div>
      )}
      {plan?.steps?.length ? (
        <ol className="tight-list">
          {plan.steps.map((s, i) => (
            <li key={i}>{s}</li>
          ))}
        </ol>
      ) : null}
      {subs?.length ? (
        <ul className="tight-list">
          {subs.map((s, i) => (
            <li key={i}>
              {s.query} {s.source && <span className="muted">({s.source}{s.focus && s.focus !== 'text' ? `, ${s.focus}` : ''})</span>}
            </li>
          ))}
        </ul>
      ) : null}
      {gaps?.length ? (
        <ul className="tight-list">
          {gaps.map((g, i) => (
            <li key={i}>Gap: {g}</li>
          ))}
        </ul>
      ) : null}
    </details>
  );
}

export function StageTimeline({ events, status, createdAt }: { events: ResearchEvent[]; status: ResearchStatus; createdAt?: string }) {
  const active = status === 'running' || status === 'queued';
  const steps = buildSteps(events, active);
  const seen = new Set(steps.filter((s) => s.status === 'done').map((s) => s.stage));
  const running = new Set(steps.filter((s) => s.status === 'running').map((s) => s.stage));
  const stopped = new Set(steps.filter((s) => s.status === 'stopped').map((s) => s.stage));
  const rounds = steps.reduce((m, s) => Math.max(m, s.round), 0);
  const totalSeconds = steps.reduce((a, s) => a + (s.seconds ?? 0), 0);

  return (
    <div className="timeline-wrap">
      {active && createdAt && <RunProgress steps={steps} status={status} createdAt={createdAt} />}
      <ol className="pipeline" aria-label="Research pipeline">
        {STAGES.map((st) => {
          const state = running.has(st) ? 'active' : stopped.has(st) ? 'stopped' : seen.has(st) ? 'done' : 'pending';
          return (
            <li key={st} className={`pipeline-step pipeline-${state}`} title={`${STAGE_LABELS[st]}: ${state}`}>
              <span className="sr-only">{state}: </span>
              {STAGE_LABELS[st]}
              {st === 'follow_up' && rounds > 0 && <span className="pipeline-count"> ×{rounds}</span>}
            </li>
          );
        })}
      </ol>
      {steps.length === 0 ? (
        <p className="muted small">{active ? 'Waiting for the first stage to start…' : 'No events recorded.'}</p>
      ) : (
        <ol className="timeline" aria-live="polite">
          {steps.map((s, i) => (
            <Fragment key={s.id}>
              {s.round > 0 && (i === 0 || steps[i - 1].round !== s.round) && (
                <li className="timeline-round" aria-label={`Follow-up round ${s.round}`}>
                  <span>Follow-up round {s.round}</span>
                </li>
              )}
              <li className={`timeline-item timeline-${s.status}`}>
                <span className="timeline-marker" aria-hidden="true" />
                <div className="timeline-content">
                  <div className="timeline-head">
                    <span className="timeline-stage">{STAGE_LABELS[s.stage] ?? s.stage}</span>
                    <span className="timeline-time small muted">
                      {s.status === 'running' ? 'running…' : s.status === 'stopped' ? 'did not finish' : s.seconds !== undefined ? fmtSeconds(s.seconds) : ''}
                    </span>
                  </div>
                  {s.message && <div className="timeline-message">{s.message}</div>}
                  <StepDetails data={s.data} />
                </div>
              </li>
            </Fragment>
          ))}
        </ol>
      )}
      {!active && totalSeconds > 0 && <p className="small muted">Total stage time: {fmtSeconds(totalSeconds)} (parallel stages are counted separately)</p>}
    </div>
  );
}
