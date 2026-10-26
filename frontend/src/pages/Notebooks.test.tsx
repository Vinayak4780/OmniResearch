import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { App } from '../App';
import { answerWarnings, friendlyProviderName } from '../components/notebook/ChatPanel';
import { answerToPlainText, numberCitations } from '../components/notebook/NotebookAnswer';
import { parseLinks } from '../components/notebook/AddSourcesDialog';
import { OUTPUT_POLL_MS } from '../components/notebook/StudioPanel';
import { trustParts } from '../components/notebook/StudioOutputs';
import { HealthProvider } from '../components/layout/HealthContext';
import { ApiError, api } from '../services/api';
import type { DocumentDetail, HealthResponse, PrivacyDestination, ResearchJob } from '../types/api';
import type { Notebook, NotebookCitation, NotebookMessage, NotebookOutput, NotebookSource, NotebookSummary } from '../types/notebooks';
import { NotebookPage } from './NotebookPage';
import { NotebooksPage } from './NotebooksPage';

// ------------------------------------------------------------------ fixtures

const HEALTH = {
  status: 'ok',
  database: true,
  vector_store: { ok: true, mode: 'embedded' },
  parser: { docling_installed: true, configured: 'auto' },
  paddleocr_installed: true,
  configured_tasks: { synthesis: true },
  admin_token_configured: false,
  max_upload_mb: 1,
} as unknown as HealthResponse;

const src = (id: string, extra: Partial<NotebookSource> = {}): NotebookSource => ({
  id,
  filename: `${id}.pdf`,
  title: `Title ${id}`,
  status: 'ready',
  error: null,
  page_count: 2,
  size_bytes: 1000,
  job: { status: 'succeeded', stage: 'done', progress: 1 },
  ...extra,
});

const SOURCES: NotebookSource[] = [
  src('d1', { title: 'Low-Bit Quantization' }),
  src('d2', { title: 'Paged KV Cache' }),
  src('d3', { title: null, filename: 'scan.pdf', status: 'processing', job: { status: 'running', stage: 'ocr', progress: 0.4 } }),
  src('d4', { title: 'Broken file', status: 'failed', error: 'Unreadable PDF' }),
];

const NB: Notebook = { id: 'nb-1', title: 'GPU serving', created_at: '2026-10-25T10:00:00Z', updated_at: '2026-10-25T10:00:00Z', sources: SOURCES };

const cite = (id: string, extra: Partial<NotebookCitation> = {}): NotebookCitation =>
  ({
    key: id,
    id,
    source_type: 'local',
    text: `Passage ${id}: INT4 group-wise quantization reduces weight memory from 26.0 GB to 7.4 GB.`,
    document_id: 'd1',
    filename: 'd1.pdf',
    doc_title: 'Low-Bit Quantization',
    page_no: 1,
    page_end: 1,
    bbox: [59.5, 84.2, 535.5, 168.4],
    link: '/documents/d1/view?page=1',
    ...extra,
  }) as NotebookCitation;

const ANSWER: NotebookMessage = {
  id: 2,
  role: 'assistant',
  content: 'INT4 weights need **7.4 GB** [E2], saving 18.6 GB [E1][E2]. Accuracy stays high [E9].',
  citations: { E1: cite('E1', { page_no: 2, page_end: 2 }), E2: cite('E2') },
  check: { answered_from_sources: true, unknown_citations: ['E9'], numbers_not_in_cited_evidence: ['18.6'], retrieval_warnings: [] },
  source_ids: ['d1', 'd2'],
  created_at: '2026-10-25T10:01:00Z',
};
const QUESTION: NotebookMessage = { id: 1, role: 'user', content: 'How much memory does INT4 save?', citations: {}, check: {}, source_ids: ['d1', 'd2'], created_at: '2026-10-25T10:00:30Z' };

const DOC: DocumentDetail = {
  id: 'd1',
  filename: 'd1.pdf',
  sha256: 'x',
  size_bytes: 1,
  collection: 'nb-1',
  page_count: 2,
  title: 'Low-Bit Quantization',
  status: 'ready',
  error: null,
  parser: 'docling',
  metadata: {},
  created_at: '',
  updated_at: '',
  pages: [{ page_no: 1, width: 595, height: 842, text_chars: 10, extraction_method: 'text_layer', ocr_status: null, ocr_error: null, ocr_provider: null, ocr_latency_ms: null }],
  jobs: [],
  element_types: [],
};

const EVENTS = [
  { id: 1, job_id: 'job-1', ts: new Date().toISOString(), stage: 'analyze_question', message: 'started', data: {} },
  { id: 2, job_id: 'job-1', ts: new Date().toISOString(), stage: 'analyze_question', message: 'ok', data: { seconds: 1 } },
  { id: 3, job_id: 'job-1', ts: new Date().toISOString(), stage: 'decompose', message: 'started', data: {} },
];

function mockNotebook(opts: { messages?: NotebookMessage[]; sources?: NotebookSource[]; outputs?: NotebookOutput[]; webAvailable?: boolean; destinations?: PrivacyDestination[] } = {}) {
  vi.spyOn(api, 'health').mockResolvedValue(HEALTH);
  vi.spyOn(api, 'notebook').mockResolvedValue({ ...NB, sources: opts.sources ?? SOURCES });
  const sources = vi.spyOn(api, 'notebookSources').mockResolvedValue(opts.sources ?? SOURCES);
  const messages = vi.spyOn(api, 'notebookMessages').mockResolvedValue(opts.messages ?? []);
  const notes = vi.spyOn(api, 'notebookNotes').mockResolvedValue([]);
  vi.spyOn(api, 'notebookReports').mockResolvedValue([]);
  const outputs = vi.spyOn(api, 'notebookOutputs').mockResolvedValue(opts.outputs ?? []);
  const preflight = vi.spyOn(api, 'preflight').mockResolvedValue({ problems: [], external_destinations: opts.destinations ?? [], web_available: opts.webAvailable ?? false, web_reason: opts.webAvailable ? '' : 'no web search provider configured' });
  const guide = vi.spyOn(api, 'notebookGuide').mockResolvedValue({
    summary: 'These sources compare quantization and KV-cache paging for serving LLMs on small GPUs.',
    questions: ['How much memory does INT4 save?', 'What does paging fix?', 'Which combination is best?'],
    sources_hash: 'h',
  });
  return { sources, messages, notes, guide, outputs, preflight };
}

function Where() {
  const l = useLocation();
  return <div data-testid="where">{l.pathname + l.search}</div>;
}

function renderNotebook(path = '/notebooks/nb-1', extra?: ReactNode) {
  return render(
    <HealthProvider>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/notebooks/:id" element={<NotebookPage />} />
          <Route path="/" element={<div>HOME</div>} />
        </Routes>
        <Where />
        {extra}
      </MemoryRouter>
    </HealthProvider>,
  );
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

// ------------------------------------------------------------------ pure helpers

describe('citation numbering', () => {
  it('numbers citations per message by first appearance and marks unknown ids', () => {
    const { markdown, order } = numberCitations(ANSWER.content, ANSWER.citations);
    expect(order).toEqual(['E2', 'E1']);
    expect(markdown).toBe('INT4 weights need **7.4 GB** [[1]](#nbcite-E2), saving 18.6 GB [[2]](#nbcite-E1)[[1]](#nbcite-E2). Accuracy stays high [[?]](#nbcite-E9).');
    expect(answerToPlainText(ANSWER.content, ANSWER.citations)).toBe(
      'INT4 weights need **7.4 GB** [1], saving 18.6 GB [2][1]. Accuracy stays high .\n\nSources:\n[1] Low-Bit Quantization, p. 1\n[2] Low-Bit Quantization, p. 2',
    );
  });

  it('words the automatic checks gently', () => {
    expect(answerWarnings(ANSWER.check)).toEqual([
      'Some numbers are calculated or not found in the cited passages: 18.6',
      'Some citation marks did not match a passage (shown as ?).',
    ]);
    expect(answerWarnings({ answered_from_sources: false, retrieval_warnings: ['dense retrieval skipped'] })).toEqual([
      "The selected sources don't fully answer this question.",
      'Search note: dense retrieval skipped',
    ]);
  });
});

// ------------------------------------------------------------------ home

describe('Notebooks home', () => {
  const LIST: NotebookSummary[] = [
    { id: 'demo', title: 'demo', created_at: '2026-10-20T00:00:00Z', updated_at: '2026-10-25T09:00:00Z', source_count: 3, report_count: 2 },
    { id: 'nb-2', title: 'Attention kernels', created_at: '2026-10-21T00:00:00Z', updated_at: '2026-10-24T09:00:00Z', source_count: 1, report_count: 0 },
  ];

  function renderHome() {
    return render(
      <MemoryRouter initialEntries={['/']}>
        <Routes>
          <Route path="/" element={<NotebooksPage />} />
          <Route path="/notebooks/:id" element={<div>NOTEBOOK</div>} />
        </Routes>
        <Where />
      </MemoryRouter>,
    );
  }

  it('lists notebooks as cards and creates a new one', async () => {
    vi.spyOn(api, 'notebooks').mockResolvedValue(LIST);
    const create = vi.spyOn(api, 'createNotebook').mockResolvedValue({ ...NB, id: 'nb-new', title: 'Untitled notebook', sources: [] });
    const user = userEvent.setup();
    renderHome();
    const card = (await screen.findByText('demo')).closest('li') as HTMLElement;
    expect(within(card).getByText(/3 sources · 2 reports/)).toBeInTheDocument();
    expect(within(card).getByRole('link', { name: 'demo' })).toHaveAttribute('href', '/notebooks/demo');
    expect(screen.getByText(/1 source$/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Create new notebook' }));
    expect(create).toHaveBeenCalledWith('Untitled notebook');
    expect(await screen.findByTestId('where')).toHaveTextContent('/notebooks/nb-new');
  });

  it('renames and deletes from the ⋯ menu (delete explains what is removed)', async () => {
    const list = vi.spyOn(api, 'notebooks').mockResolvedValue(LIST);
    const rename = vi.spyOn(api, 'renameNotebook').mockResolvedValue({ ...NB, id: 'demo', title: 'GPU memory' });
    const del = vi.spyOn(api, 'deleteNotebook').mockResolvedValue(undefined);
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);
    const user = userEvent.setup();
    renderHome();
    await user.click(await screen.findByRole('button', { name: 'More actions for demo' }));
    await user.click(screen.getByRole('menuitem', { name: 'Rename' }));
    const input = screen.getByLabelText('Notebook title');
    await user.clear(input);
    await user.type(input, 'GPU memory{Enter}');
    expect(rename).toHaveBeenCalledTimes(1);
    expect(rename).toHaveBeenCalledWith('demo', 'GPU memory');

    await user.click(screen.getByRole('button', { name: 'More actions for demo' }));
    await user.click(screen.getByRole('menuitem', { name: 'Delete' }));
    expect(confirm.mock.calls[0][0]).toMatch(/removes the notebook with its 3 sources, chat, notes and 2 reports/);
    expect(del).toHaveBeenCalledWith('demo');
    await waitFor(() => expect(list.mock.calls.length).toBeGreaterThanOrEqual(3));
  });

  it('shows a friendly empty state', async () => {
    vi.spyOn(api, 'notebooks').mockResolvedValue([]);
    renderHome();
    expect(await screen.findByText('Create your first notebook')).toBeInTheDocument();
  });
});

// ------------------------------------------------------------------ notebook

describe('Notebook: sources', () => {
  it('selects ready sources by default; selection drives source_ids; shows status pills and retry', async () => {
    mockNotebook();
    const ask = vi.spyOn(api, 'askNotebook').mockResolvedValue(ANSWER);
    const retry = vi.spyOn(api, 'retryDocument').mockResolvedValue({ document: {} as never, job_id: 'j' });
    const user = userEvent.setup();
    renderNotebook();
    const panel = await screen.findByRole('region', { name: 'Sources' });
    const selectAll = await within(panel).findByRole('checkbox', { name: 'Select all sources' });
    expect(selectAll).toBeChecked();
    expect(within(panel).getByRole('checkbox', { name: 'Use Low-Bit Quantization' })).toBeChecked();
    expect(within(panel).getByRole('checkbox', { name: 'Use scan.pdf' })).toBeDisabled();
    expect(within(panel).getByText(/Ocr 40%/)).toBeInTheDocument();
    expect(within(panel).getByText('Failed')).toBeInTheDocument();
    await user.click(within(panel).getByRole('button', { name: 'Retry Broken file' }));
    expect(retry).toHaveBeenCalledWith('d4');

    expect(screen.getByText('2 sources selected')).toBeInTheDocument();
    await user.click(within(panel).getByRole('checkbox', { name: 'Use Paged KV Cache' }));
    expect(screen.getByText('1 source selected')).toBeInTheDocument();
    expect((selectAll as HTMLInputElement).indeterminate).toBe(true);
    await user.type(screen.getByLabelText('Ask about your sources'), 'What about INT4?{Enter}');
    expect(ask).toHaveBeenCalledWith('nb-1', 'What about INT4?', ['d1'], expect.any(AbortSignal));

    // clicking the half-checked "select all" selects everything, clicking again clears
    await user.click(selectAll);
    expect(screen.getByText('2 sources selected')).toBeInTheDocument();
    await user.click(selectAll);
    expect(screen.getByText('0 sources selected')).toBeInTheDocument();
    expect(screen.getByLabelText('Ask about your sources')).toBeDisabled();
  });

  it('validates uploads (PDF only, size limit) and uploads into the notebook', async () => {
    mockNotebook();
    const upload = vi.spyOn(api, 'uploadDocument').mockResolvedValue({ document: {} as never, duplicate: false, job_id: 'j' });
    const user = userEvent.setup({ applyAccept: false });
    renderNotebook();
    await waitFor(() => expect(api.health).toHaveBeenCalled());
    await user.click(await screen.findByRole('button', { name: '+ Add sources' }));
    const dialog = screen.getByRole('dialog', { name: 'Add sources' });
    await waitFor(() => expect(within(dialog).getByText(/up to 1 MB each/)).toBeInTheDocument());
    const ok = new File(['%PDF-1.4'], 'paper.pdf', { type: 'application/pdf' });
    const big = new File([new Uint8Array(2 * 1024 * 1024)], 'huge.pdf', { type: 'application/pdf' });
    const txt = new File(['hi'], 'notes.txt', { type: 'text/plain' });
    await user.upload(within(dialog).getByLabelText('Choose PDF files'), [ok, big, txt]);
    await waitFor(() => expect(upload).toHaveBeenCalledTimes(1));
    expect(upload).toHaveBeenCalledWith(ok, 'nb-1');
    expect(within(dialog).getByText(/the limit is 1 MB/)).toBeInTheDocument();
    expect(within(dialog).getByText('Only PDF files are supported.')).toBeInTheDocument();
    expect(await within(dialog).findByText('Added')).toBeInTheDocument();
  });

  it('opens a source with its pages and passages', async () => {
    mockNotebook();
    vi.spyOn(api, 'chunks').mockResolvedValue([
      { id: 'c1', document_id: 'd1', chunk_index: 0, page_no: 1, page_end: 1, section_path: 'Abstract', element_type: 'text', extraction_method: 'text_layer', text: 'We study quantization.', bbox_json: null, source_element_id: null, chunk_version: 'v', content_hash: 'h' },
    ]);
    const user = userEvent.setup();
    renderNotebook();
    await user.click(await screen.findByRole('button', { name: 'Low-Bit Quantization' }));
    expect(await screen.findByText('We study quantization.')).toBeInTheDocument();
    expect(screen.getByAltText('Page 1 of Low-Bit Quantization')).toHaveAttribute('src', '/api/documents/d1/pages/1.png');
    await user.click(screen.getByRole('button', { name: 'Back to sources' }));
    expect(screen.getByRole('checkbox', { name: 'Select all sources' })).toBeInTheDocument();
  });
});

describe('Notebook: chat', () => {
  it('shows the guide; a suggested question is sent with the selected sources', async () => {
    const m = mockNotebook();
    let resolve!: (v: NotebookMessage) => void;
    const ask = vi.spyOn(api, 'askNotebook').mockImplementation(() => new Promise((r) => (resolve = r)));
    const user = userEvent.setup();
    renderNotebook();
    expect(await screen.findByText(/compare quantization and KV-cache paging/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'How much memory does INT4 save?' }));
    expect(ask).toHaveBeenCalledWith('nb-1', 'How much memory does INT4 save?', ['d1', 'd2'], expect.any(AbortSignal));
    expect(screen.getByText(/Reading your sources…/)).toBeInTheDocument();
    m.messages.mockResolvedValue([QUESTION, ANSWER]);
    await act(async () => resolve(ANSWER));
    expect((await screen.findAllByRole('button', { name: /Citation 1: Low-Bit Quantization · p\. 1/ })).length).toBe(2);
    expect(screen.queryByText(/Reading your sources…/)).not.toBeInTheDocument();
  });

  it('numbers citation chips per message, previews on hover and opens the cited page', async () => {
    mockNotebook({ messages: [QUESTION, ANSWER] });
    vi.spyOn(api, 'document').mockResolvedValue(DOC);
    const user = userEvent.setup();
    renderNotebook();
    const answer = (await screen.findByText('7.4 GB')).closest('li') as HTMLElement;
    const chips = within(answer).getAllByRole('button', { name: /^Citation/ });
    expect(chips.map((c) => c.textContent)).toEqual(['1', '2', '1']);
    expect(within(answer).getByTitle('E9 does not match any passage')).toHaveTextContent('?');
    fireEvent.mouseEnter(chips[1].parentElement!);
    expect(within(answer).getByRole('tooltip')).toHaveTextContent(/Low-Bit Quantization · p\. 2.*Passage E1/);
    await user.click(chips[0]);
    const drawer = await screen.findByRole('complementary', { name: 'Cited source' });
    expect(within(drawer).getByRole('heading', { name: 'Low-Bit Quantization' })).toBeInTheDocument();
    expect(within(drawer).getByText(/Passage E2: INT4/)).toBeInTheDocument();
    fireEvent.load(within(drawer).getByAltText(/Page 1 of/));
    expect(await within(drawer).findByTestId('bbox-highlight')).toBeInTheDocument();
    await user.click(within(drawer).getByRole('button', { name: 'Close cited source' }));
    expect(screen.queryByRole('complementary', { name: 'Cited source' })).not.toBeInTheDocument();
  });

  it('explains in plain words when no AI model is connected', async () => {
    const m = mockNotebook();
    m.guide.mockRejectedValue(new ApiError(503, "No provider configured for task 'chat'"));
    vi.spyOn(api, 'askNotebook').mockRejectedValue(new ApiError(503, "No provider configured for task 'chat'"));
    const user = userEvent.setup();
    renderNotebook();
    expect(await screen.findByText(/A summary of your sources needs an AI model/)).toBeInTheDocument();
    await user.type(screen.getByLabelText('Ask about your sources'), 'Anything?{Enter}');
    expect(await screen.findByText(/Chatting with your sources needs an AI model/)).toBeInTheDocument();
    expect(screen.getAllByRole('link', { name: 'Connect an AI service' })[0]).toHaveAttribute('href', '/settings/connect');
    expect(screen.queryByText(/No provider configured/)).not.toBeInTheDocument();
  });

  it('shows check warnings and saves an answer to notes', async () => {
    const m = mockNotebook({ messages: [QUESTION, ANSWER] });
    const add = vi.spyOn(api, 'addNotebookNote').mockResolvedValue({ id: 5, title: 'How much memory does INT4 save?', content: ANSWER.content, citations: ANSWER.citations, created_at: '' });
    const user = userEvent.setup();
    renderNotebook();
    expect(await screen.findByText('Some numbers are calculated or not found in the cited passages: 18.6')).toBeInTheDocument();
    m.notes.mockResolvedValue([{ id: 5, title: 'How much memory does INT4 save?', content: ANSWER.content, citations: ANSWER.citations, created_at: '2026-10-25T10:02:00Z' }]);
    await user.click(screen.getByRole('button', { name: 'Save to note' }));
    expect(add).toHaveBeenCalledWith('nb-1', { message_id: 2 });
    expect(await screen.findByRole('button', { name: 'Saved to notes ✓' })).toBeDisabled();
    const studio = screen.getByRole('region', { name: 'Studio' });
    const note = await within(studio).findByText('How much memory does INT4 save?');
    expect(within(studio).getByText(/2 citations/)).toBeInTheDocument();
    await user.click(note);
    expect(within(studio).getAllByRole('button', { name: /^Citation/ }).length).toBe(3); // working chips in the note
  });
});

describe('Notebook: studio', () => {
  it('has no deep research form any more; lists reports with progress and opens them', async () => {
    mockNotebook();
    const job = { id: 'job-1', question: 'Compare quantization and paging.', status: 'running', stage: 'decompose', error: null, created_at: new Date().toISOString(), updated_at: '', options: {}, report_md: null, report: null } as ResearchJob;
    vi.spyOn(api, 'notebookReports').mockResolvedValue([job]);
    vi.spyOn(api, 'researchEvents').mockResolvedValue(EVENTS);
    vi.spyOn(api, 'researchJob').mockResolvedValue(job);
    const user = userEvent.setup();
    renderNotebook();
    const studio = await screen.findByRole('region', { name: 'Studio' });
    expect(within(studio).queryByRole('heading', { name: 'Deep research' })).not.toBeInTheDocument();
    expect(within(studio).queryByLabelText('Research question')).not.toBeInTheDocument();
    const item = await within(studio).findByRole('button', { name: /Compare quantization and paging\./ });
    expect(await within(studio).findByText(/Stage 3\/14/)).toBeInTheDocument();
    await user.click(item);
    expect(screen.getByTestId('where')).toHaveTextContent('/notebooks/nb-1?report=job-1');
    expect(await screen.findByRole('heading', { name: 'Compare quantization and paging.' })).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: '← Back to notebook' }));
    expect(screen.getByTestId('where')).toHaveTextContent(/^\/notebooks\/nb-1$/);
    expect(screen.getByRole('region', { name: 'Chat' })).toBeInTheDocument();
  });
  it('adds a typed note', async () => {
    const m = mockNotebook();
    const add = vi.spyOn(api, 'addNotebookNote').mockResolvedValue({ id: 9, title: 'Idea', content: 'Try FP8', citations: {}, created_at: '' });
    const user = userEvent.setup();
    renderNotebook();
    const studio = await screen.findByRole('region', { name: 'Studio' });
    await user.click(within(studio).getByRole('button', { name: '+ Add note' }));
    await user.type(within(studio).getByLabelText('Note title'), 'Idea');
    await user.type(within(studio).getByLabelText('Note text'), 'Try FP8');
    m.notes.mockResolvedValue([{ id: 9, title: 'Idea', content: 'Try FP8', citations: {}, created_at: '2026-10-25T10:00:00Z' }]);
    await user.click(within(studio).getByRole('button', { name: 'Save note' }));
    expect(add).toHaveBeenCalledWith('nb-1', { content: 'Try FP8', title: 'Idea' });
    expect(await within(studio).findByText('Idea')).toBeInTheDocument();
  });
});

describe('Notebook: responsive layout', () => {
  it('switches to Sources | Chat | Studio tabs on narrow screens', async () => {
    vi.stubGlobal(
      'matchMedia',
      vi.fn((q: string) => ({ matches: q.includes('max-width: 999px'), media: q, addEventListener: () => {}, removeEventListener: () => {} })),
    );
    mockNotebook();
    const user = userEvent.setup();
    renderNotebook();
    const tabs = await screen.findByRole('tablist', { name: 'Notebook sections' });
    expect(within(tabs).getByRole('tab', { name: 'Chat' })).toHaveAttribute('aria-selected', 'true');
    expect(within(tabs).getByRole('tab', { name: 'Sources (4)' })).toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Studio' })).not.toBeInTheDocument();
    expect(screen.getByLabelText('Ask about your sources')).toBeInTheDocument();
    await user.click(within(tabs).getByRole('tab', { name: 'Studio' }));
    expect(screen.getByRole('heading', { name: 'Create from your sources' })).toBeInTheDocument();
    await user.click(within(tabs).getByRole('tab', { name: 'Sources (4)' }));
    expect(screen.getByRole('button', { name: '+ Add sources' })).toBeInTheDocument();
  });

  it('collapses side panels on wide screens', async () => {
    mockNotebook();
    const user = userEvent.setup();
    renderNotebook();
    await user.click(await screen.findByRole('button', { name: 'Collapse Sources' }));
    expect(screen.queryByRole('button', { name: '+ Add sources' })).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Expand Sources' }));
    expect(screen.getByRole('button', { name: '+ Add sources' })).toBeInTheDocument();
  });

  it('renames the notebook inline from the header', async () => {
    mockNotebook();
    const rename = vi.spyOn(api, 'renameNotebook').mockResolvedValue({ ...NB, title: 'GPU memory' });
    const user = userEvent.setup();
    renderNotebook();
    await user.click(await screen.findByRole('button', { name: 'GPU serving' }));
    const input = screen.getByLabelText('Notebook title');
    await user.clear(input);
    await user.type(input, 'GPU memory{Enter}');
    expect(rename).toHaveBeenCalledWith('nb-1', 'GPU memory');
    expect(await screen.findByRole('button', { name: 'GPU memory' })).toBeInTheDocument();
  });
});

describe('routes', () => {
  function renderApp(path: string) {
    vi.spyOn(api, 'health').mockResolvedValue(HEALTH);
    vi.spyOn(api, 'notebooks').mockResolvedValue([]);
    return render(
      <MemoryRouter initialEntries={[path]}>
        <App />
        <Where />
      </MemoryRouter>,
    );
  }

  it.each(['/research', '/documents', '/documents/d1'])('redirects %s to the notebooks home', async (path) => {
    renderApp(path);
    expect(await screen.findByRole('heading', { name: 'Notebooks' })).toBeInTheDocument();
    expect(screen.getByTestId('where')).toHaveTextContent(/^\/$/);
    const nav = screen.getByRole('navigation', { name: 'Main' });
    expect(within(nav).getAllByRole('link').map((a) => a.textContent)).toEqual(['Notebooks', 'Settings']);
  });

  it('redirects an old report link into its notebook (or home when unknown)', async () => {
    mockNotebook();
    const job = { id: 'job-7', question: 'Old question here', status: 'running', stage: 'plan', error: null, created_at: new Date().toISOString(), updated_at: '', options: { notebook_id: 'nb-1' }, report_md: null, report: null } as ResearchJob;
    vi.spyOn(api, 'researchJob').mockResolvedValue(job);
    vi.spyOn(api, 'researchEvents').mockResolvedValue([]);
    renderApp('/research/job-7');
    await waitFor(() => expect(screen.getByTestId('where')).toHaveTextContent('/notebooks/nb-1?report=job-7'));
    expect(await screen.findByRole('button', { name: '← Back to notebook' })).toBeInTheDocument();
  });

  it('sends unknown report links home', async () => {
    vi.spyOn(api, 'researchJob').mockRejectedValue(new Error('404'));
    renderApp('/research/nope');
    await waitFor(() => expect(screen.getByTestId('where')).toHaveTextContent(/^\/$/));
  });
});

// ------------------------------------------------------------------ add sources from links and the web

const NO_SEARCH = 'Web search needs a search service. Connect Tavily, Brave or Exa in Settings → Connect AI.';

async function openAddDialog(user: ReturnType<typeof userEvent.setup>, way?: 'Paste links' | 'Search the web') {
  await user.click(await screen.findByRole('button', { name: '+ Add sources' }));
  const dialog = screen.getByRole('dialog', { name: 'Add sources' });
  if (way) await user.click(within(dialog).getByRole('tab', { name: way }));
  return dialog;
}

describe('Notebook: add sources from links and the web', () => {
  it('parses pasted links (one per line, adds https://, drops blanks and repeats)', () => {
    expect(parseLinks(' https://a.com/x.pdf \n\nexample.org/page\r\nhttps://a.com/x.pdf\n')).toEqual(['https://a.com/x.pdf', 'https://example.org/page']);
  });

  it('pastes links and shows a result per link (added / already in notebook / error)', async () => {
    const m = mockNotebook();
    const add = vi.spyOn(api, 'addNotebookLinks').mockResolvedValue({
      results: [
        { url: 'https://a.com/x.pdf', ok: true, kind: 'pdf' },
        { url: 'https://example.org/page', ok: true, kind: 'web' },
        { url: 'https://dup.com/y.pdf', ok: true, kind: 'pdf', duplicate: true },
        { url: 'https://login.com/z', ok: false, error: 'This page needs a login, so it could not be read.' },
      ],
    });
    const user = userEvent.setup();
    renderNotebook();
    const dialog = await openAddDialog(user, 'Paste links');
    const box = within(dialog).getByLabelText('Links, one per line');
    await user.click(box);
    await user.paste('https://a.com/x.pdf\nexample.org/page\nhttps://dup.com/y.pdf\nhttps://login.com/z');
    const before = m.sources.mock.calls.length;
    await user.click(within(dialog).getByRole('button', { name: 'Add 4 links' }));
    expect(add).toHaveBeenCalledWith('nb-1', ['https://a.com/x.pdf', 'https://example.org/page', 'https://dup.com/y.pdf', 'https://login.com/z']);
    const results = await within(dialog).findByRole('list', { name: 'Link results' });
    const rows = within(results).getAllByRole('listitem');
    expect(rows[0]).toHaveTextContent('Added');
    expect(rows[1]).toHaveTextContent('Added (page saved as a PDF)');
    expect(rows[2]).toHaveTextContent('Already in this notebook');
    expect(rows[3]).toHaveTextContent('This page needs a login, so it could not be read.');
    await waitFor(() => expect(m.sources.mock.calls.length).toBeGreaterThan(before));
    // only the failed link is left in the box to fix and retry
    expect(box).toHaveValue('https://login.com/z');

    await user.clear(box);
    await user.paste(Array.from({ length: 11 }, (_, i) => `https://s${i}.com`).join('\n'));
    expect(within(dialog).getByRole('alert')).toHaveTextContent(/11 links — please add 10 or fewer/);
    expect(within(dialog).getByRole('button', { name: 'Add 11 links' })).toBeDisabled();
  });

  it('shows link sources with a globe, their domain and an "Open original" link', async () => {
    mockNotebook({ sources: [src('d9', { title: 'KV cache explained', source_url: 'https://www.example.com/blog/kv' })] });
    vi.spyOn(api, 'chunks').mockResolvedValue([]);
    const user = userEvent.setup();
    renderNotebook();
    const panel = await screen.findByRole('region', { name: 'Sources' });
    expect(await within(panel).findByText('example.com')).toBeInTheDocument();
    await user.click(within(panel).getByRole('button', { name: 'KV cache explained' }));
    const link = within(panel).getByRole('link', { name: 'Open original ↗' });
    expect(link).toHaveAttribute('href', 'https://www.example.com/blog/kv');
    expect(link).toHaveAttribute('target', '_blank');
    expect(link).toHaveAttribute('rel', 'noopener noreferrer');
  });

  const HITS = [
    { url: 'https://arxiv.org/pdf/1.pdf', title: 'Paged attention', snippet: 'We page the KV cache.', domain: 'arxiv.org', published_at: '2024-05-01', is_pdf: true, reason: null, already_added: false },
    { url: 'https://blog.dev/kv', title: 'KV cache 101', snippet: 'A gentle intro.', domain: 'blog.dev', published_at: null, is_pdf: false, reason: null, already_added: false },
    { url: 'https://old.org/a', title: 'Already here', snippet: 'Old.', domain: 'old.org', published_at: null, is_pdf: false, reason: null, already_added: true },
  ];

  it('fast web search: lists results, already-added rows are disabled, adds the selected links', async () => {
    mockNotebook();
    const discover = vi.spyOn(api, 'discoverSources').mockResolvedValue({ mode: 'fast', query: 'kv cache', queries: ['kv cache'], results: HITS, warnings: ['One search service was slow.'] });
    const add = vi.spyOn(api, 'addNotebookLinks').mockResolvedValue({ results: [{ url: 'https://arxiv.org/pdf/1.pdf', ok: true, kind: 'pdf' }, { url: 'https://blog.dev/kv', ok: true, kind: 'web' }] });
    const user = userEvent.setup();
    renderNotebook();
    const dialog = await openAddDialog(user, 'Search the web');
    expect(within(dialog).getByText('Fast: one search, a few seconds')).toBeInTheDocument();
    await user.type(within(dialog).getByLabelText('What are you looking for?'), 'kv cache');
    await user.click(within(dialog).getByRole('button', { name: 'Search' }));
    expect(discover).toHaveBeenCalledWith('nb-1', 'kv cache', 'fast', expect.any(AbortSignal));
    const list = await within(dialog).findByRole('list', { name: 'Search results' });
    expect(within(dialog).getByText('One search service was slow.')).toBeInTheDocument();
    expect(within(list).getByText('PDF')).toBeInTheDocument();
    expect(within(list).getByText('Already added')).toBeInTheDocument();
    const added = within(list).getByRole('checkbox', { name: 'Select Already here' });
    expect(added).toBeDisabled();
    expect(within(dialog).getByRole('button', { name: 'Add 0 selected' })).toBeDisabled();
    await user.click(within(list).getByRole('checkbox', { name: 'Select Paged attention' }));
    await user.click(within(list).getByRole('checkbox', { name: 'Select KV cache 101' }));
    await user.click(within(dialog).getByRole('button', { name: 'Add 2 selected' }));
    expect(add).toHaveBeenCalledWith('nb-1', ['https://arxiv.org/pdf/1.pdf', 'https://blog.dev/kv']);
    expect(await within(dialog).findByText('Added (page saved as a PDF)')).toBeInTheDocument();
    // newly added results now read "Already added"
    expect(within(list).getAllByText('Already added')).toHaveLength(3);
    expect(within(list).getByRole('checkbox', { name: 'Select Paged attention' })).toBeDisabled();
  });

  it('deep web search shows the searches and the AI reasons, preselects its picks, sends the chosen URLs', async () => {
    mockNotebook();
    const deepHits = HITS.map((h, i) => ({ ...h, reason: h.already_added ? null : `Reason ${i + 1}: directly answers the question.` }));
    const discover = vi.spyOn(api, 'discoverSources').mockResolvedValue({ mode: 'deep', query: 'kv cache', queries: ['kv cache', 'paged attention vLLM', 'kv cache quantization'], results: deepHits, warnings: [] });
    const add = vi.spyOn(api, 'addNotebookLinks').mockResolvedValue({ results: [{ url: 'https://blog.dev/kv', ok: true, kind: 'web' }] });
    const user = userEvent.setup();
    renderNotebook();
    const dialog = await openAddDialog(user, 'Search the web');
    await user.click(within(dialog).getByRole('radio', { name: 'Deep' }));
    expect(within(dialog).getByText('Deep: AI plans several searches and picks the best sources, up to a minute')).toBeInTheDocument();
    await user.type(within(dialog).getByLabelText('What are you looking for?'), 'kv cache{Enter}');
    expect(discover).toHaveBeenCalledWith('nb-1', 'kv cache', 'deep', expect.any(AbortSignal));
    expect(await within(dialog).findByText(/Searched for: kv cache · paged attention vLLM · kv cache quantization/)).toBeInTheDocument();
    expect(within(dialog).getByText('Reason 1: directly answers the question.')).toBeInTheDocument();
    expect(within(dialog).getByText('Reason 2: directly answers the question.')).toBeInTheDocument();
    expect(within(dialog).getByRole('checkbox', { name: 'Select Already here' })).toBeDisabled();
    await user.click(within(dialog).getByRole('checkbox', { name: 'Select Paged attention' })); // un-pick one of the AI's picks
    await user.click(within(dialog).getByRole('button', { name: 'Add 1 selected' }));
    expect(add).toHaveBeenCalledWith('nb-1', ['https://blog.dev/kv']);
  });

  it('explains how to connect a search service when none is connected', async () => {
    mockNotebook();
    vi.spyOn(api, 'discoverSources').mockRejectedValue(new ApiError(400, NO_SEARCH));
    const user = userEvent.setup();
    renderNotebook();
    const dialog = await openAddDialog(user, 'Search the web');
    await user.type(within(dialog).getByLabelText('What are you looking for?'), 'kv cache{Enter}');
    const link = await within(dialog).findByRole('link', { name: 'Connect Tavily, Brave or Exa' });
    expect(link).toHaveAttribute('href', '/settings/connect');
    expect(within(dialog).queryByText('Something went wrong')).not.toBeInTheDocument();
  });
});

// ------------------------------------------------------------------ chat with the web

describe('Notebook: chat composer (mode + web)', () => {
  it('Fast + Web sends web:true and allows asking with no sources selected', async () => {
    mockNotebook({ webAvailable: true });
    const ask = vi.spyOn(api, 'askNotebook').mockResolvedValue(ANSWER);
    const user = userEvent.setup();
    renderNotebook();
    const panel = await screen.findByRole('region', { name: 'Sources' });
    await user.click(await within(panel).findByRole('checkbox', { name: 'Select all sources' })); // clear the selection
    expect(screen.getByText('0 sources selected')).toBeInTheDocument();
    const input = screen.getByLabelText('Ask about your sources');
    expect(input).toBeDisabled();
    expect(screen.getByRole('radio', { name: /Fast/ })).toBeChecked();
    const web = screen.getByRole('button', { name: 'Web' });
    expect(web).toHaveAttribute('aria-pressed', 'false');
    await user.click(web);
    expect(web).toHaveAttribute('aria-pressed', 'true');
    expect(input).toBeEnabled();
    await user.type(input, 'What is new in vLLM?{Enter}');
    expect(ask).toHaveBeenCalledWith('nb-1', 'What is new in vLLM?', [], expect.any(AbortSignal), { web: true });
  });

  it('Deep research + Web starts a report with web_mode auto; without Web it uses never', async () => {
    mockNotebook({ webAvailable: true });
    const job = { id: 'job-1', question: 'Compare quantization and paging.', status: 'running', stage: 'decompose', error: null, created_at: new Date().toISOString(), updated_at: '', options: {}, report_md: null, report: null } as ResearchJob;
    const start = vi.spyOn(api, 'startNotebookReport').mockResolvedValue(job);
    const ask = vi.spyOn(api, 'askNotebook').mockResolvedValue(ANSWER);
    vi.spyOn(api, 'researchEvents').mockResolvedValue([]);
    const user = userEvent.setup();
    renderNotebook();
    const input = await screen.findByLabelText('Ask about your sources');
    await user.click(screen.getByRole('radio', { name: /Deep research/ }));
    expect(input).toHaveAttribute('placeholder', 'What should the report answer?');
    await user.type(input, 'Why');
    expect(screen.getByRole('button', { name: 'Start deep research' })).toBeDisabled(); // needs 5+ characters
    await user.type(input, ' INT4?{Enter}');
    expect(start).toHaveBeenLastCalledWith('nb-1', { question: 'Why INT4?', source_ids: ['d1', 'd2'], web_mode: 'never' });
    await user.click(screen.getByRole('button', { name: 'Web' }));
    await user.type(input, 'Compare quantization and paging.{Enter}');
    expect(start).toHaveBeenLastCalledWith('nb-1', { question: 'Compare quantization and paging.', source_ids: ['d1', 'd2'], web_mode: 'auto' });
    expect(ask).not.toHaveBeenCalled();
  });

  it('remembers the mode and web choice per notebook', async () => {
    mockNotebook({ webAvailable: true });
    const user = userEvent.setup();
    const { unmount } = renderNotebook();
    await user.click(await screen.findByRole('radio', { name: /Deep research/ }));
    await user.click(screen.getByRole('button', { name: 'Web' }));
    unmount();
    renderNotebook();
    expect(await screen.findByRole('radio', { name: /Deep research/ })).toBeChecked();
    await waitFor(() => expect(screen.getByRole('button', { name: 'Web' })).toHaveAttribute('aria-pressed', 'true'));
  });

  it('shows deep research in the chat with progress; Open report opens the reader', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const m = mockNotebook({ messages: [QUESTION, ANSWER] });
    const job = { id: 'job-1', question: 'Compare quantization and paging.', status: 'running', stage: 'decompose', error: null, created_at: '2026-10-25T10:05:00Z', updated_at: '', options: {}, report_md: null, report: null } as ResearchJob;
    const start = vi.spyOn(api, 'startNotebookReport').mockResolvedValue(job);
    const reports = vi.spyOn(api, 'notebookReports').mockResolvedValue([]);
    vi.spyOn(api, 'researchEvents').mockResolvedValue(EVENTS);
    vi.spyOn(api, 'researchJob').mockResolvedValue({ ...job, status: 'completed' });
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderNotebook();
    const chat = await screen.findByRole('region', { name: 'Chat' });
    await within(chat).findByText('7.4 GB');
    await user.click(within(chat).getByRole('radio', { name: /Deep research/ }));
    reports.mockResolvedValue([job]);
    await user.type(within(chat).getByLabelText('Ask about your sources'), 'Compare quantization and paging.{Enter}');
    expect(start).toHaveBeenCalled();
    const card = (await within(chat).findByText('Compare quantization and paging.')).closest('li') as HTMLElement;
    expect(within(card).getByText('Deep research')).toBeInTheDocument();
    expect(await within(card).findByText(/Stage 3\/14/)).toBeInTheDocument();
    // the card comes after the earlier answer in the timeline
    const items = within(within(chat).getByRole('list', { name: 'Chat' })).getAllByRole('listitem').filter((li) => li.parentElement?.getAttribute('aria-label') === 'Chat');
    expect(items[items.length - 1]).toBe(card);
    expect(within(card).queryByRole('button', { name: 'Open report' })).not.toBeInTheDocument();

    // finished → Open report
    m.messages.mockResolvedValue([QUESTION, ANSWER]);
    reports.mockResolvedValue([{ ...job, status: 'completed' }]);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5100);
    });
    await user.click(await within(chat).findByRole('button', { name: 'Open report' }));
    expect(screen.getByTestId('where')).toHaveTextContent('/notebooks/nb-1?report=job-1');
  });

  it('shows a failed deep research with its reason', async () => {
    mockNotebook();
    vi.spyOn(api, 'notebookReports').mockResolvedValue([
      { id: 'job-2', question: 'What changed in 2026?', status: 'failed', stage: 'plan', error: 'The AI model stopped responding.', created_at: '2026-10-25T10:05:00Z', updated_at: '' },
    ]);
    renderNotebook();
    const chat = await screen.findByRole('region', { name: 'Chat' });
    expect(await within(chat).findByText('The AI model stopped responding.')).toBeInTheDocument();
  });

  it('the Web chip explains how to connect a search service when none is connected', async () => {
    mockNotebook({ webAvailable: false });
    const user = userEvent.setup();
    renderNotebook();
    const web = await screen.findByRole('button', { name: 'Web' });
    await waitFor(() => expect(web).toHaveAttribute('aria-disabled', 'true'));
    await user.click(web);
    expect(web).toHaveAttribute('aria-pressed', 'false');
    expect(await screen.findByRole('link', { name: 'Connect Tavily, Brave or Exa' })).toHaveAttribute('href', '/settings/connect');
  });

  it('shows the no-search-service hint when the server says web search is not set up', async () => {
    mockNotebook({ webAvailable: true });
    vi.spyOn(api, 'askNotebook').mockRejectedValue(new ApiError(400, NO_SEARCH));
    const user = userEvent.setup();
    renderNotebook();
    await user.click(await screen.findByRole('button', { name: 'Web' }));
    await user.type(screen.getByLabelText('Ask about your sources'), 'Anything new?{Enter}');
    expect(await screen.findByRole('link', { name: 'Connect Tavily, Brave or Exa' })).toHaveAttribute('href', '/settings/connect');
  });

  it('shows one short privacy line with friendly provider names', async () => {
    expect(friendlyProviderName('nvidia-nim-chat')).toBe('NVIDIA');
    expect(friendlyProviderName('openai')).toBe('OpenAI');
    expect(friendlyProviderName('acme_cloud')).toBe('Acme');
    mockNotebook({
      destinations: [
        { task: 'chat', provider_id: 'nvidia-nim-chat', adapter: 'openai' },
        { task: 'embedding', provider_id: 'nvidia-embed', adapter: 'openai' },
        { task: 'rerank', provider_id: 'cohere-rerank', adapter: 'cohere' },
      ] as unknown as PrivacyDestination[],
    });
    renderNotebook();
    const note = await screen.findByTestId('privacy-note');
    expect(note).toHaveTextContent('Passages are sent to NVIDIA and Cohere');
    expect(note).toHaveAttribute('title', expect.stringContaining('nvidia-nim-chat, nvidia-embed, cohere-rerank'));
  });

  it('hides the privacy line when nothing leaves this computer', async () => {
    mockNotebook();
    renderNotebook();
    await screen.findByLabelText('Ask about your sources');
    await waitFor(() => expect(api.preflight).toHaveBeenCalled());
    expect(screen.queryByTestId('privacy-note')).not.toBeInTheDocument();
  });

  it('web citation chips preview the page and open the URL in a new tab, not the page drawer', async () => {
    const webAnswer: NotebookMessage = {
      ...ANSWER,
      content: 'vLLM added chunked prefill [E1] and INT4 helps [E2].',
      citations: {
        E1: cite('E1', {
          source_type: 'web',
          document_id: null,
          page_no: null,
          link: 'https://example.com/vllm-news',
          url: 'https://example.com/vllm-news',
          domain: 'example.com',
          web_title: 'What is new in vLLM',
          snippet_only: true,
          text: 'Chunked prefill is now on by default.',
        }),
        E2: cite('E2'),
      },
      check: { web: true },
    };
    mockNotebook({ messages: [QUESTION, webAnswer] });
    const user = userEvent.setup();
    renderNotebook();
    const chip = await screen.findByRole('link', { name: /Citation 1 \(web\): What is new in vLLM/ });
    expect(chip).toHaveAttribute('href', 'https://example.com/vllm-news');
    expect(chip).toHaveAttribute('target', '_blank');
    expect(chip).toHaveAttribute('rel', 'noopener noreferrer');
    expect(screen.getByText('Includes web results')).toBeInTheDocument();
    fireEvent.mouseEnter(chip.parentElement!);
    const tip = screen.getByRole('tooltip');
    expect(tip).toHaveTextContent('Web');
    expect(tip).toHaveTextContent('example.com');
    expect(tip).toHaveTextContent('What is new in vLLM');
    expect(tip).toHaveTextContent('Chunked prefill is now on by default.');
    expect(tip).toHaveTextContent(/search snippet only/i);
    const stop = (e: MouseEvent) => e.preventDefault(); // jsdom cannot navigate
    document.addEventListener('click', stop);
    await user.click(chip);
    document.removeEventListener('click', stop);
    expect(screen.queryByRole('complementary', { name: 'Cited source' })).not.toBeInTheDocument();
    // the local citation still has a regular chip button
    expect(screen.getByRole('button', { name: /Citation 2: Low-Bit Quantization/ })).toBeInTheDocument();
  });
});

// ------------------------------------------------------------------ studio outputs

const out = (id: string, extra: Partial<NotebookOutput> = {}): NotebookOutput => ({
  id,
  notebook_id: 'nb-1',
  kind: 'summary',
  title: 'Summary',
  status: 'completed',
  content: 'INT4 needs **7.4 GB** [E2] and saves 18.6 GB [E1]. Paging helps too [E99].',
  data: {},
  citations: { E1: cite('E1', { page_no: 2, page_end: 2 }), E2: cite('E2') },
  check: { unknown_citations: ['E99'], numbers_not_in_cited_evidence: ['18.6'], uncited_statements: 2, passages_used: 24 },
  source_ids: ['d1', 'd2'],
  error: null,
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  ...extra,
});

const MAP_OUT = out('o-map', {
  kind: 'mindmap',
  title: 'Mind map',
  content: 'Serving LLMs on small GPUs\n- Quantization\n- Paging',
  data: {
    central: 'Serving LLMs on small GPUs',
    branches: [
      { label: 'Quantization', citations: ['E2'], children: [{ label: 'INT4 weights', citations: ['E1'], children: [{ label: 'Group-wise scales', citations: [] }] }] },
      { label: 'Paging', citations: [], children: [{ label: 'Block tables', citations: [] }] },
    ],
  },
  check: { passages_used: 12 },
});

describe('Notebook: studio outputs', () => {
  it('words the trust bar gently and only shows non-zero parts', () => {
    expect(trustParts(out('x').check)).toEqual([
      'Checked: 24 passages',
      '2 statements without a citation',
      'numbers not found in cited passages: 18.6',
      'unknown citation: E99',
    ]);
    expect(trustParts({ passages_used: 1, uncited_statements: 0, numbers_not_in_cited_evidence: [], unknown_citations: [] })).toEqual(['Checked: 1 passage']);
  });

  it('creates an output from the checked sources, polls while it runs, then opens it in the reader', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const m = mockNotebook();
    const running = out('o1', { status: 'running', content: '', citations: {}, check: {} });
    const create = vi.spyOn(api, 'createNotebookOutput').mockResolvedValue(running);
    const addNote = vi.spyOn(api, 'addNotebookNote').mockResolvedValue({ id: 3, title: 'Summary', content: '', citations: {}, created_at: '' });
    vi.spyOn(api, 'document').mockResolvedValue(DOC);
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderNotebook();
    const studio = await screen.findByRole('region', { name: 'Studio' });
    for (const name of ['Summary', 'Briefing doc', 'Study guide', 'FAQ', 'Timeline', 'Mind map']) expect(within(studio).getByRole('button', { name: `Create ${name}` })).toBeEnabled();
    m.outputs.mockResolvedValue([running]);
    await user.click(within(studio).getByRole('button', { name: 'Create Summary' }));
    expect(create).toHaveBeenCalledWith('nb-1', 'summary', ['d1', 'd2']);
    const item = await within(studio).findByText(/Writing…/);
    expect(item.closest('li')).toHaveTextContent('Summary');

    m.outputs.mockResolvedValue([out('o1')]);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(OUTPUT_POLL_MS + 50);
    });
    await user.click(await within(studio).findByRole('button', { name: /^Summary/ }));
    expect(within(studio).getByRole('heading', { name: 'Summary' })).toBeInTheDocument();
    expect(within(studio).getByLabelText('What was checked')).toHaveTextContent(
      'Checked: 24 passages · 2 statements without a citation · numbers not found in cited passages: 18.6 · unknown citation: E99',
    );
    const chips = within(studio).getAllByRole('button', { name: /^Citation/ });
    expect(chips.map((c) => c.textContent)).toEqual(['1', '2']);
    expect(within(studio).getByTitle('E99 does not match any passage')).toBeInTheDocument();
    await user.click(chips[0]);
    expect(await screen.findByRole('complementary', { name: 'Cited source' })).toBeInTheDocument();

    await user.click(within(studio).getByRole('button', { name: 'Save to note' }));
    expect(addNote).toHaveBeenCalledWith('nb-1', { title: 'Summary', content: out('o1').content });
    expect(await within(studio).findByRole('button', { name: 'Saved to notes ✓' })).toBeDisabled();
    await user.click(within(studio).getByRole('button', { name: '← Back' }));
    expect(within(studio).getByRole('heading', { name: 'Outputs' })).toBeInTheDocument();
  });

  it('shows why an output failed and Retry creates a new one of the same kind', async () => {
    const failed = out('o2', { kind: 'timeline', title: 'Timeline', status: 'failed', error: 'The AI model took too long. Please try again.', source_ids: ['d1'] });
    mockNotebook({ outputs: [failed] });
    const create = vi.spyOn(api, 'createNotebookOutput').mockResolvedValue(out('o3', { kind: 'timeline', title: 'Timeline', status: 'running' }));
    const user = userEvent.setup();
    renderNotebook();
    const studio = await screen.findByRole('region', { name: 'Studio' });
    expect(await within(studio).findByText('The AI model took too long. Please try again.')).toBeInTheDocument();
    await user.click(within(studio).getByRole('button', { name: 'Retry Timeline' }));
    expect(create).toHaveBeenCalledWith('nb-1', 'timeline', ['d1']);
    expect(await within(studio).findByText(/Writing…/)).toBeInTheDocument();
  });

  it('renders a mind map with citation chips and collapsible branches', async () => {
    mockNotebook({ outputs: [MAP_OUT] });
    const user = userEvent.setup();
    renderNotebook();
    const studio = await screen.findByRole('region', { name: 'Studio' });
    await user.click(await within(studio).findByRole('button', { name: /^Mind map/ }));
    const tree = within(studio).getByRole('tree', { name: 'Serving LLMs on small GPUs' });
    expect(within(tree).getByText('Serving LLMs on small GPUs')).toBeInTheDocument();
    for (const label of ['Quantization', 'INT4 weights', 'Group-wise scales', 'Paging', 'Block tables']) expect(within(tree).getByRole('treeitem', { name: label })).toBeInTheDocument();
    expect(within(tree).getAllByRole('button', { name: /^Citation/ }).map((c) => c.textContent)).toEqual(['1', '2']);
    expect(within(studio).getByLabelText('What was checked')).toHaveTextContent('Checked: 12 passages');

    await user.click(within(tree).getByRole('button', { name: 'Collapse Quantization' }));
    expect(within(tree).getByRole('treeitem', { name: 'Quantization' })).toHaveAttribute('aria-expanded', 'false');
    expect(within(tree).queryByText('INT4 weights')).not.toBeInTheDocument();
    expect(within(tree).getByText('Block tables')).toBeInTheDocument();
    await user.click(within(tree).getByRole('button', { name: 'Expand Quantization' }));
    expect(within(tree).getByText('INT4 weights')).toBeInTheDocument();

    await user.click(within(studio).getByRole('button', { name: 'Collapse all' }));
    expect(within(tree).queryByText('INT4 weights')).not.toBeInTheDocument();
    expect(within(tree).queryByText('Block tables')).not.toBeInTheDocument();
    await user.click(within(studio).getByRole('button', { name: 'Expand all' }));
    expect(within(tree).getByText('Group-wise scales')).toBeInTheDocument();
  });

  it('deletes an output from the reader', async () => {
    mockNotebook({ outputs: [out('o1')] });
    const del = vi.spyOn(api, 'deleteNotebookOutput').mockResolvedValue(undefined);
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);
    const user = userEvent.setup();
    renderNotebook();
    const studio = await screen.findByRole('region', { name: 'Studio' });
    await user.click(await within(studio).findByRole('button', { name: /^Summary/ }));
    await user.click(within(studio).getByRole('button', { name: 'Delete' }));
    expect(confirm).toHaveBeenCalled();
    expect(del).toHaveBeenCalledWith('nb-1', 'o1');
    expect(await within(studio).findByText('Summaries, guides and mind maps you create appear here.')).toBeInTheDocument();
  });
});
