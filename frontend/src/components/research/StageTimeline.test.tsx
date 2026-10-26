import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { ResearchEvent } from '../../types/api';
import { StageTimeline, buildSteps, progressInfo } from './StageTimeline';

let n = 0;
const ev = (stage: string, message: string, data: Record<string, unknown> = {}): ResearchEvent => ({
  id: ++n,
  job_id: 'j',
  ts: new Date(Date.now() - 60000 + n * 1000).toISOString(),
  stage,
  message,
  data,
});

const events = [
  ev('analyze_question', 'started'),
  ev('analyze_question', 'intent', { seconds: 2 }),
  ev('retrieve_local', 'started'),
  ev('retrieve_local', '8 passages', { seconds: 1 }),
  ev('assess_sufficiency', 'started'),
  ev('assess_sufficiency', 'insufficient: 2 gap(s)', { seconds: 3 }),
  ev('follow_up', 'started'),
  ev('follow_up', 'round 1: 3 follow-up queries', { seconds: 1 }),
  ev('retrieve_local', 'started'),
  ev('retrieve_local', '4 passages', { seconds: 1 }),
  ev('assess_sufficiency', 'started'),
  ev('assess_sufficiency', 'insufficient: 1 gap(s)', { seconds: 2 }),
  ev('follow_up', 'started'),
  ev('follow_up', 'round 2: 1 follow-up queries', { seconds: 1 }),
  ev('retrieve_local', 'started'),
];

describe('live progress', () => {
  it('tags steps with their follow-up round', () => {
    const steps = buildSteps(events, true);
    expect(steps.map((s) => `${s.stage}:${s.round}`)).toEqual([
      'analyze_question:0',
      'retrieve_local:0',
      'assess_sufficiency:0',
      'follow_up:1',
      'retrieve_local:1',
      'assess_sufficiency:1',
      'follow_up:2',
      'retrieve_local:2',
    ]);
  });

  it('computes progress from the furthest stage reached and the current stage', () => {
    const p = progressInfo(buildSteps(events, true), 'running');
    expect(p.furthestIndex).toBe(9); // follow_up
    expect(p.fraction).toBeCloseTo(10 / 14);
    expect(p.current?.stage).toBe('retrieve_local');
    expect(p.round).toBe(2);
    expect(progressInfo([], 'queued').fraction).toBe(0);
    expect(progressInfo(buildSteps(events, false), 'completed').fraction).toBe(1);
  });

  it('renders round dividers, current stage, elapsed and last activity', () => {
    render(<StageTimeline events={events} status="running" createdAt={new Date(Date.now() - 125000).toISOString()} />);
    expect(screen.getByText('Follow-up round 1')).toBeInTheDocument();
    expect(screen.getByText('Follow-up round 2')).toBeInTheDocument();
    expect(screen.getByText(/follow-up round 2/)).toBeInTheDocument();
    expect(screen.getByText('Elapsed')).toBeInTheDocument();
    expect(screen.getByText(/^2m 0\ds$/)).toBeInTheDocument();
    expect(screen.getByText('Last activity')).toBeInTheDocument();
    expect(screen.getByRole('progressbar', { name: /Investigation progress/ })).toHaveAttribute('aria-valuenow', '71');
    expect(screen.getByText('×2')).toBeInTheDocument();
  });
});
