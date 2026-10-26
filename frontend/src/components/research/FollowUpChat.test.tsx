import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import { api } from '../../services/api';
import type { ChatAnswer, ChatMessage, Evidence } from '../../types/api';
import { FollowUpChat, checkWarnings } from './FollowUpChat';

const e3: Evidence = { key: 'k3', id: 'E3', source_type: 'local', text: 'INT4 weights need 7.4 GB.', document_id: 'doc1', page_no: 1 };
const evidence = new Map([['E3', e3]]);

const history: ChatMessage[] = [
  { id: 1, role: 'user', content: 'How much memory?', citations: [], links: {}, check: {}, created_at: '2026-10-24T10:00:00Z' },
  {
    id: 2,
    role: 'assistant',
    content: 'INT4 weights need 7.4 GB [E3] and reach 99.9% accuracy [E999].',
    citations: ['E3'],
    links: { E3: '/documents/doc1/view?page=1' },
    check: { answered_from_sources: true, unknown_citations: ['E999'], numbers_not_in_cited_evidence: ['99.9'], new_evidence: [] },
    created_at: '2026-10-24T10:00:05Z',
  },
];

function renderChat() {
  const onCitation = vi.fn();
  const onEvidenceAdded = vi.fn();
  render(
    <MemoryRouter>
      <FollowUpChat jobId="job1" evidence={evidence} onCitation={onCitation} onEvidenceAdded={onEvidenceAdded} />
    </MemoryRouter>,
  );
  return { onCitation, onEvidenceAdded };
}

describe('checkWarnings', () => {
  it('is empty for a clean answer', () => {
    expect(checkWarnings({ answered_from_sources: true, unknown_citations: [], numbers_not_in_cited_evidence: [], new_evidence: ['E12'] })).toEqual([]);
    expect(checkWarnings(undefined)).toEqual([]);
  });
  it('flags unsupported numbers, unknown citations and unanswered questions', () => {
    const w = checkWarnings({ answered_from_sources: false, unknown_citations: ['E999'], numbers_not_in_cited_evidence: ['99.9', '12'] });
    expect(w).toHaveLength(3);
    expect(w[0]).toMatch(/do not answer/);
    expect(w[1]).toMatch(/E999/);
    expect(w[2]).toMatch(/99\.9, 12/);
  });
});

describe('FollowUpChat', () => {
  it('loads history, renders citation chips and shows check warnings', async () => {
    const user = userEvent.setup();
    vi.spyOn(api, 'chatHistory').mockResolvedValue(history);
    const { onCitation } = renderChat();
    expect(await screen.findByText('How much memory?')).toBeInTheDocument();
    const chip = screen.getByRole('link', { name: '[E3]' });
    await user.click(chip);
    expect(onCitation).toHaveBeenCalledWith('E3', '/documents/doc1/view?page=1');
    // E999 is not a known evidence id: shown, but not clickable
    expect(screen.getByText('[E999]').closest('a')).toBeNull();
    const warn = screen.getByText('Check this answer').closest('.alert') as HTMLElement;
    expect(within(warn).getByText(/Numbers not found in the cited evidence: 99\.9/)).toBeInTheDocument();
    expect(within(warn).getByText(/does not exist: E999/)).toBeInTheDocument();
  });

  it('sends a question, shows a pending state, then the answer and refreshes evidence', async () => {
    const user = userEvent.setup();
    vi.spyOn(api, 'chatHistory').mockResolvedValue([]);
    let resolve!: (a: ChatAnswer) => void;
    const send = vi.spyOn(api, 'sendChat').mockImplementation(() => new Promise((r) => (resolve = r)));
    const { onEvidenceAdded } = renderChat();
    await waitFor(() => expect(api.chatHistory).toHaveBeenCalled());

    const box = screen.getByLabelText('Follow-up question');
    await user.type(box, 'Which GPU was used?{Enter}');
    expect(send).toHaveBeenCalledWith('job1', 'Which GPU was used?', expect.any(AbortSignal));
    expect(screen.getByText(/Searching the evidence and drafting an answer/)).toBeInTheDocument();
    expect(box).toBeDisabled();

    const newEv: Evidence = { key: 'k12', id: 'E12', source_type: 'local', text: 'RTX 4090', document_id: 'doc1', page_no: 2 };
    resolve({
      answer: 'An RTX 4090 [E12].',
      citations: ['E12'],
      links: { E12: '/documents/doc1/view?page=2' },
      check: { answered_from_sources: true, unknown_citations: [], numbers_not_in_cited_evidence: [], new_evidence: ['E12'] },
      evidence: [newEv],
    });
    expect(await screen.findByRole('link', { name: '[E12]' })).toHaveAttribute('href', '/documents/doc1/view?page=2');
    expect(screen.queryByText(/Searching the evidence/)).not.toBeInTheDocument();
    expect(screen.queryByText('Check this answer')).not.toBeInTheDocument();
    expect(screen.getByText(/New passages found for this question: E12/)).toBeInTheDocument();
    expect(onEvidenceAdded).toHaveBeenCalledWith([newEv]);
    expect(box).not.toBeDisabled();
  });

  it('warns when the answer is not from the sources and keeps the question on error', async () => {
    const user = userEvent.setup();
    vi.spyOn(api, 'chatHistory').mockResolvedValue([]);
    vi.spyOn(api, 'sendChat')
      .mockRejectedValueOnce(new Error('upstream timeout'))
      .mockResolvedValueOnce({ answer: 'The sources do not say.', citations: [], links: {}, check: { answered_from_sources: false, unknown_citations: [], numbers_not_in_cited_evidence: [] }, evidence: [] });
    renderChat();
    await waitFor(() => expect(api.chatHistory).toHaveBeenCalled());
    await user.type(screen.getByLabelText('Follow-up question'), 'What about TPUs?');
    await user.click(screen.getByRole('button', { name: 'Ask' }));
    expect(await screen.findByText('upstream timeout')).toBeInTheDocument();
    expect(screen.getByLabelText('Follow-up question')).toHaveValue('What about TPUs?');
    await user.click(screen.getByRole('button', { name: 'Retry' }));
    expect(await screen.findByText('The sources do not say.')).toBeInTheDocument();
    expect(screen.getByText(/sources do not answer this question/)).toBeInTheDocument();
  });
});
