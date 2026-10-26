import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { App } from '../../App';
import { currentlyUsing } from '../../components/connect/CurrentlyUsing';
import { ROLES, prettyModel, roleOptions } from '../../components/connect/connectLogic';
import { EMPTY_DISCOVERY, applyEvent, mergeRecommendations, testingIds } from '../../components/connect/liveDiscovery';
import { ApiError, api } from '../../services/api';
import { createNdjsonParser, streamDiscover } from '../../services/discoverStream';
import type { ProviderDescriptor, ProviderRecord, RouteRow } from '../../types/api';
import type { DiscoverEvent, DiscoveredModel, LlmDiscovery, QuickPreset } from '../../types/quickconnect';
import { ConnectPage } from './ConnectPage';

// ------------------------------------------------------------------ fixtures

const preset = (p: Partial<QuickPreset> & Pick<QuickPreset, 'id' | 'name'>): QuickPreset => ({
  kind: 'llm',
  needs_key: true,
  key_url: `https://${p.id}.example/keys`,
  key_hint: '',
  base_url: `https://${p.id}.example/v1`,
  note: '',
  connected: false,
  ...p,
});

const PRESETS: QuickPreset[] = [
  preset({ id: 'nvidia', name: 'NVIDIA NIM', connected: true, key_hint: 'nvapi-…' }),
  preset({ id: 'openrouter', name: 'OpenRouter', key_hint: 'sk-or-…' }),
  preset({ id: 'ollama', name: 'Ollama (local)', needs_key: false, key_url: '', base_url: 'http://localhost:11434/v1' }),
  preset({ id: 'custom', name: 'Custom endpoint', needs_key: false, key_url: '', base_url: '' }),
  preset({ id: 'tavily', name: 'Tavily (web search)', kind: 'search', base_url: '' }),
];

const chat = (id: string, extra: Partial<DiscoveredModel> = {}): DiscoveredModel => ({ id, capabilities: ['chat'], available: null, json_ok: null, ...extra });

const MODELS: DiscoveredModel[] = [
  chat('nvidia/nemotron-3-ultra-550b'),
  chat('meta/llama-3.3-70b-instruct'),
  { id: 'qwen/qwen3-vl-235b', capabilities: ['chat', 'vision'], available: null, json_ok: null },
  chat('mistral/mistral-large'),
  chat('deepseek/deepseek-v4'),
  ...Array.from({ length: 8 }, (_, i) => chat(`misc/untested-${i + 1}`)),
  { id: 'nvidia/nv-embedqa-e5-v5', capabilities: ['embedding'], available: null, json_ok: null },
];
const TO_TEST = ['nvidia/nemotron-3-ultra-550b', 'meta/llama-3.3-70b-instruct', 'qwen/qwen3-vl-235b', 'mistral/mistral-large', 'deepseek/deepseek-v4', 'nvidia/nv-embedqa-e5-v5'];
const REC = { synthesis: 'nvidia/nemotron-3-ultra-550b', planning: 'meta/llama-3.3-70b-instruct', vision: 'qwen/qwen3-vl-235b', fallback: null, embedding: 'nvidia/nv-embedqa-e5-v5' };

const modelsEvent = (preset = 'openrouter'): DiscoverEvent => ({ type: 'models', preset, base_url: 'https://openrouter.example/v1', models: MODELS, to_test: TO_TEST });
let probeN = 0;
const probe = (id: string, result: Record<string, unknown>, recommended: Record<string, string | null> = {}): DiscoverEvent => ({
  type: 'probe',
  id,
  result: result as never,
  done: ++probeN,
  total: TO_TEST.length,
  recommended,
});

// ------------------------------------------------------------------ controllable NDJSON stream behind a fetch mock

interface Stream {
  url: string;
  body: Record<string, unknown>;
  signal: AbortSignal;
  push: (ev: DiscoverEvent | string) => Promise<void>;
  close: () => Promise<void>;
}

const enc = new TextEncoder();

function mockStreamFetch() {
  const streams: Stream[] = [];
  const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
    let ctrl!: ReadableStreamDefaultController<Uint8Array>;
    const rs = new ReadableStream<Uint8Array>({
      start(c) {
        ctrl = c;
      },
    });
    const s: Stream = {
      url,
      body: JSON.parse(String(init.body)),
      signal: init.signal as AbortSignal,
      push: async (ev) => {
        await act(async () => {
          try {
            ctrl.enqueue(enc.encode(typeof ev === 'string' ? ev : `${JSON.stringify(ev)}\n`));
          } catch {
            /* stream already cancelled */
          }
          await new Promise((r) => setTimeout(r, 0));
        });
      },
      close: async () => {
        await act(async () => {
          try {
            ctrl.close();
          } catch {
            /* already closed */
          }
          await new Promise((r) => setTimeout(r, 0));
        });
      },
    };
    streams.push(s);
    return new Response(rs, { status: 200, headers: { 'content-type': 'application/x-ndjson' } });
  });
  vi.stubGlobal('fetch', fetchMock);
  return { streams, fetchMock, last: () => streams[streams.length - 1] };
}

afterEach(() => {
  vi.unstubAllGlobals();
  probeN = 0;
});

function setup(presets = PRESETS) {
  vi.spyOn(api, 'quickPresets').mockResolvedValue(presets);
  vi.spyOn(api, 'routes').mockResolvedValue([]);
  vi.spyOn(api, 'providers').mockResolvedValue([]);
  vi.spyOn(api, 'catalog').mockResolvedValue([]);
  const apply = vi.spyOn(api, 'quickApply').mockResolvedValue({ routes: { synthesis: ['x'] }, notes: [] });
  const net = mockStreamFetch();
  const view = render(
    <MemoryRouter>
      <ConnectPage />
    </MemoryRouter>,
  );
  return { apply, user: userEvent.setup(), ...net, view };
}

const roleSelect = (label: RegExp) => screen.getByLabelText(label, { selector: 'select' }) as HTMLSelectElement;
const row = (id: string) => document.querySelector(`[data-model="${id}"]`) as HTMLElement;

// ------------------------------------------------------------------ pure pieces

describe('NDJSON parsing', () => {
  it('handles lines split across chunks, several lines per chunk, a final line without newline and junk', () => {
    const items: unknown[] = [];
    const bad: string[] = [];
    const p = createNdjsonParser((x) => items.push(x), (l) => bad.push(l));
    p.push('{"type":"mod');
    p.push('els","n":1}\n{"type":"probe","id":"a"}\n{"type"');
    expect(items).toEqual([{ type: 'models', n: 1 }, { type: 'probe', id: 'a' }]);
    p.push(':"probe","id":"b"}\n\nnot json\n');
    p.push('{"type":"done"}');
    expect(items).toHaveLength(3);
    p.flush();
    expect(items).toEqual([{ type: 'models', n: 1 }, { type: 'probe', id: 'a' }, { type: 'probe', id: 'b' }, { type: 'done' }]);
    expect(bad).toEqual(['not json']);
  });

  it('streamDiscover delivers events across chunk boundaries and error events; HTTP errors reject', async () => {
    const { last } = mockStreamFetch();
    const events: DiscoverEvent[] = [];
    const run = streamDiscover({ preset: 'nvidia' }, { onEvent: (e) => events.push(e) });
    await waitFor(() => expect(last()).toBeDefined());
    expect(last().url).toBe('/api/quick-connect/discover/stream');
    const line = JSON.stringify(modelsEvent()) + '\n';
    await last().push(line.slice(0, 25));
    expect(events).toHaveLength(0);
    await last().push(line.slice(25) + '{"type":"error","detail":"None of the tested models responded."}');
    await last().close();
    await run;
    expect(events.map((e) => e.type)).toEqual(['models', 'error']);
    expect(events[1]).toEqual({ type: 'error', detail: 'None of the tested models responded.' });

    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ detail: 'Admin token required' }), { status: 401 })));
    const err = await streamDiscover({ preset: 'nvidia' }, { onEvent: () => {} }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(401);
  });
});

describe('live discovery state', () => {
  it('applies events and merges recommendations without overriding user choices', () => {
    let s = applyEvent(EMPTY_DISCOVERY, modelsEvent());
    expect(testingIds(s).size).toBe(TO_TEST.length);
    s = applyEvent(s, probe('nvidia/nemotron-3-ultra-550b', { available: true, json_ok: true }, { synthesis: 'nvidia/nemotron-3-ultra-550b' }));
    expect(s.done).toBe(1);
    expect(testingIds(s).has('nvidia/nemotron-3-ultra-550b')).toBe(false);
    const a = mergeRecommendations({ synthesis: null, planning: 'misc/untested-1', vision: null, reranking: null, ocr: null, fallback: null, embedding: null }, new Set(['planning']), s.recommended, s.models);
    expect(a.synthesis).toBe('nvidia/nemotron-3-ultra-550b');
    expect(a.planning).toBe('misc/untested-1'); // user-picked, kept
    // a picked model that then fails its test is cleared
    s = applyEvent(s, probe('misc/untested-1', { available: false, detail: 'timeout' }, s.recommended));
    expect(mergeRecommendations(a, new Set(['planning']), s.recommended, s.models).planning).toBeNull();
    s = applyEvent(s, { type: 'error', detail: 'boom' });
    expect(s).toMatchObject({ finished: true, error: 'boom' });
    expect(testingIds(s).size).toBe(0);
  });

  it('orders options usable-first, marks testing ones and disables unavailable', () => {
    const models = MODELS.map((m) =>
      m.id === 'deepseek/deepseek-v4' ? { ...m, available: false, detail: 'auth: not enabled' } : m.id === 'meta/llama-3.3-70b-instruct' ? { ...m, available: true, json_ok: true } : m,
    );
    const opts = roleOptions(models, ROLES[0], null, '', null, new Set(['nvidia/nemotron-3-ultra-550b']));
    expect(opts[0]).toMatchObject({ id: 'meta/llama-3.3-70b-instruct', state: 'tested' });
    expect(opts.find((o) => o.id === 'nvidia/nemotron-3-ultra-550b')?.label).toMatch(/testing…/);
    expect(opts.at(-1)).toMatchObject({ id: 'deepseek/deepseek-v4', disabled: true });
    expect(prettyModel('nvidia/nemotron-3.5-super-vl-120b-a12b')).toBe('Nemotron 3.5 Super VL 120B A12B');
  });

  it('summarises roles and offers on/off only for local tools or switched-off ones', () => {
    const routes = [
      { task: 'synthesis', effective_provider_ids: ['nim-ultra'] },
      { task: 'text_embedding', effective_provider_ids: ['local-fastembed'] },
      { task: 'reranking', effective_provider_ids: ['local-cross'] },
      { task: 'web_search', effective_provider_ids: [] },
    ] as unknown as RouteRow[];
    const rec = (id: string, adapter: string, enabled: boolean, models: Record<string, string>, base_url?: string) =>
      ({ id, adapter, name: id, models, enabled, config: base_url ? { base_url } : {} }) as unknown as ProviderRecord;
    const providers = [
      rec('nim-ultra', 'openai_compatible', true, { chat: 'nvidia/nemotron-3-ultra-550b' }, 'https://nvidia.example/v1'),
      rec('local-fastembed', 'fastembed', false, { embedding: 'BAAI/bge-small-en-v1.5' }),
      rec('local-cross', 'fastembed_rerank', true, { reranking: 'x' }),
    ];
    const catalog = [{ adapter: 'fastembed', requires_network: false }, { adapter: 'fastembed_rerank', requires_network: false }, { adapter: 'openai_compatible', requires_network: true }] as ProviderDescriptor[];
    const by = Object.fromEntries(currentlyUsing(routes, providers, catalog, PRESETS).map((l) => [l.task, l]));
    expect(by.synthesis).toMatchObject({ value: 'Nemotron 3 Ultra 550B', where: 'NVIDIA NIM', toggle: undefined });
    expect(by.text_embedding).toMatchObject({ value: null, disabled: true, toggle: { ids: ['local-fastembed'], enable: true } });
    expect(by.reranking).toMatchObject({ where: 'on this computer', toggle: { ids: ['local-cross'], enable: false } });
    expect(by.web_search).toMatchObject({ missing: 'Not set up — connect Tavily, Brave or Exa', toggle: undefined });
  });
});

// ------------------------------------------------------------------ the flow

async function startOpenRouter(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole('button', { name: 'Connect OpenRouter' }));
  await user.type(screen.getByLabelText('OpenRouter API key'), 'sk-or-secret');
  await user.click(screen.getByRole('button', { name: 'Find models' }));
}

describe('Connect AI streaming flow', () => {
  it('shows models instantly, updates live, and allows connecting before testing is done', async () => {
    const { apply, user, last } = setup();
    await startOpenRouter(user);
    await waitFor(() => expect(last()).toBeDefined());
    expect(last().body).toEqual({ preset: 'openrouter', api_key: 'sk-or-secret' });
    expect(screen.getByText('Fetching the list of models…')).toBeInTheDocument();

    await last().push(modelsEvent());
    expect(screen.getByText('0 of 6 tested')).toBeInTheDocument();
    expect(within(row('nvidia/nemotron-3-ultra-550b')).getByRole('img', { name: 'testing' })).toBeInTheDocument();
    expect(screen.getByText(/8 more models listed by the service/)).toBeInTheDocument();
    expect(roleSelect(/Writes reports/)).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Connect' })).toBeDisabled();
    expect(screen.getByText(/Waiting for the first model that works/)).toBeInTheDocument();

    await last().push(probe('deepseek/deepseek-v4', { available: false, json_ok: null, detail: 'auth: model not enabled for this account' }, {}));
    expect(within(row('deepseek/deepseek-v4')).getByRole('img', { name: 'unavailable' })).toBeInTheDocument();
    expect(within(row('deepseek/deepseek-v4')).getByText(/model not enabled/)).toBeInTheDocument();

    await last().push(probe('nvidia/nemotron-3-ultra-550b', { available: true, json_ok: true }, { synthesis: 'nvidia/nemotron-3-ultra-550b', planning: 'nvidia/nemotron-3-ultra-550b' }));
    await last().push(probe('meta/llama-3.3-70b-instruct', { available: true, json_ok: true }, { ...REC, vision: null }));
    expect(screen.getByText('3 of 6 tested')).toBeInTheDocument();
    expect(screen.getByText(/Still checking 3 slower models — you don't need to wait/)).toBeInTheDocument();
    expect(within(row('meta/llama-3.3-70b-instruct')).getByRole('img', { name: 'works' })).toBeInTheDocument();
    expect(roleSelect(/Writes reports/)).not.toBeDisabled();
    expect(roleSelect(/Writes reports/).value).toBe('nvidia/nemotron-3-ultra-550b');
    expect(roleSelect(/Plans research/).value).toBe('meta/llama-3.3-70b-instruct'); // updated live
    expect((within(roleSelect(/Writes reports/)).getByRole('option', { name: /deepseek-v4/ }) as HTMLOptionElement).disabled).toBe(true);
    // embeddings stay local by default even though a cloud embedder is recommended
    expect(roleSelect(/Document search embeddings/).value).toBe('');

    // the user's choice survives later recommendation updates
    await user.selectOptions(roleSelect(/Backup model/), 'misc/untested-2');
    await last().push(probe('qwen/qwen3-vl-235b', { available: true, json_ok: true }, { ...REC, fallback: 'misc/untested-3' }));
    expect(roleSelect(/Reads charts/).value).toBe('qwen/qwen3-vl-235b');
    expect(roleSelect(/Backup model/).value).toBe('misc/untested-2');

    const stream = last();
    await user.click(screen.getByRole('button', { name: 'Connect' }));
    await waitFor(() => expect(apply).toHaveBeenCalledTimes(1));
    expect(stream.signal.aborted).toBe(true); // remaining tests cancelled
    expect(apply.mock.calls[0][0]).toEqual({
      preset: 'openrouter',
      api_key: 'sk-or-secret',
      assignments: {
        synthesis: 'nvidia/nemotron-3-ultra-550b',
        planning: 'meta/llama-3.3-70b-instruct',
        vision: 'qwen/qwen3-vl-235b',
        reranking: 'local',
        ocr: 'local',
        fallback: 'misc/untested-2',
        embedding: null,
      },
    });
    expect(await screen.findByText('All set')).toBeInTheDocument();
    expect(screen.getByText('Document search: stays on this computer')).toBeInTheDocument();
  });

  it('finishes with a summary, supports a cloud embedder, and Stop testing aborts', async () => {
    const { apply, user, last } = setup();
    apply.mockResolvedValue({ routes: {}, notes: ['Run Reindex to switch existing documents.'] });
    await startOpenRouter(user);
    await waitFor(() => expect(last()).toBeDefined());
    await last().push(modelsEvent());
    await last().push(probe('nvidia/nemotron-3-ultra-550b', { available: true, json_ok: true }, REC));
    await last().push(probe('nvidia/nv-embedqa-e5-v5', { available: true, dims: 1024 }, REC));
    const stream = last();
    await user.click(screen.getByRole('button', { name: 'Stop testing' }));
    expect(stream.signal.aborted).toBe(true);
    expect(await screen.findByText(/Stopped testing\. Tested 2 of 6 models: 2 work with your key/)).toBeInTheDocument();
    expect(within(row('meta/llama-3.3-70b-instruct')).getByRole('img', { name: 'not tested' })).toBeInTheDocument();

    await user.selectOptions(roleSelect(/Document search embeddings/), 'nvidia/nv-embedqa-e5-v5');
    expect(screen.getByText('Cloud embeddings')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Connect' }));
    await waitFor(() => expect(apply).toHaveBeenCalled());
    expect(apply.mock.calls[0][0].assignments.embedding).toBe('nvidia/nv-embedqa-e5-v5');
    expect(await screen.findByText('Run Reindex to switch existing documents.')).toBeInTheDocument();
  });

  it('aborts the stream when the user leaves the page', async () => {
    const { user, last, view } = setup();
    await startOpenRouter(user);
    await waitFor(() => expect(last()).toBeDefined());
    await last().push(modelsEvent());
    const stream = last();
    view.unmount();
    expect(stream.signal.aborted).toBe(true);
  });

  it('reuses the saved key for an already-connected service', async () => {
    const { apply, user, last } = setup();
    await user.click(await screen.findByRole('button', { name: 'Change models for NVIDIA NIM' }));
    expect(screen.queryByLabelText('NVIDIA NIM API key')).not.toBeInTheDocument();
    expect(screen.getByText('Your saved key will be used.')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Find models' }));
    await waitFor(() => expect(last()).toBeDefined());
    expect(last().body).toEqual({ preset: 'nvidia' });
    await last().push(modelsEvent('nvidia'));
    await last().push(probe('nvidia/nemotron-3-ultra-550b', { available: true, json_ok: true }, REC));
    await user.click(screen.getByRole('button', { name: 'Connect' }));
    await waitFor(() => expect(apply).toHaveBeenCalled());
    expect(apply.mock.calls[0][0]).not.toHaveProperty('api_key');
  });

  it('custom endpoint: address required, optional key, server errors shown', async () => {
    const { apply, user, last, streams } = setup();
    await user.click(await screen.findByRole('button', { name: 'Connect Custom endpoint' }));
    expect(screen.getByText(/OMNI_OUTBOUND_ALLOW_PRIVATE_HOSTS/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Find models' }));
    expect(screen.getByText(/Enter the address of your server/)).toBeInTheDocument();
    expect(streams).toHaveLength(0);

    await user.type(screen.getByLabelText('Address'), 'http://10.0.0.5:8000/v1');
    await user.click(screen.getByRole('button', { name: 'Find models' }));
    await waitFor(() => expect(last()).toBeDefined());
    expect(last().body).toEqual({ preset: 'custom', base_url: 'http://10.0.0.5:8000/v1' });
    await last().push({ type: 'error', detail: 'Could not list models from Custom endpoint: blocked: private address 10.0.0.5 is not allowed. Check the address.' });
    await last().close();
    expect(await screen.findByText(/private address 10\.0\.0\.5 is not allowed/)).toBeInTheDocument();

    // with an optional key, the same streaming flow follows
    await user.clear(screen.getByLabelText('Address'));
    await user.type(screen.getByLabelText('Address'), 'http://localhost:8000/v1');
    await user.type(screen.getByLabelText(/API key/), 'local-secret');
    await user.click(screen.getByRole('button', { name: 'Find models' }));
    await waitFor(() => expect(streams).toHaveLength(2));
    expect(last().body).toEqual({ preset: 'custom', api_key: 'local-secret', base_url: 'http://localhost:8000/v1' });
    await last().push({ ...modelsEvent('custom'), base_url: 'http://localhost:8000/v1' } as DiscoverEvent);
    await last().push(probe('nvidia/nemotron-3-ultra-550b', { available: true, json_ok: true }, REC));
    await user.click(screen.getByRole('button', { name: 'Connect' }));
    await waitFor(() => expect(apply).toHaveBeenCalled());
    expect(apply.mock.calls[0][0]).toMatchObject({ preset: 'custom', api_key: 'local-secret', base_url: 'http://localhost:8000/v1' });
  });

  it('search services: key → Test & connect → done (failed tests do not connect)', async () => {
    const { apply, user, last } = setup();
    await user.click(await screen.findByRole('button', { name: 'Connect Tavily (web search)' }));
    await user.type(screen.getByLabelText('Tavily (web search) API key'), 'tvly-bad');
    await user.click(screen.getByRole('button', { name: 'Test & connect' }));
    await waitFor(() => expect(last()).toBeDefined());
    await last().push({ type: 'done', result: { preset: 'tavily', kind: 'search', ok: false, detail: 'invalid API key', models: [], recommended: {} } });
    await last().close();
    expect(await screen.findByText(/The test search failed: invalid API key/)).toBeInTheDocument();
    expect(apply).not.toHaveBeenCalled();

    await user.clear(screen.getByLabelText('Tavily (web search) API key'));
    await user.type(screen.getByLabelText('Tavily (web search) API key'), 'tvly-good');
    await user.click(screen.getByRole('button', { name: 'Test & connect' }));
    await waitFor(() => expect(last().body).toEqual({ preset: 'tavily', api_key: 'tvly-good' }));
    await last().push({ type: 'done', result: { preset: 'tavily', kind: 'search', ok: true, detail: '2 test results', models: [], recommended: { web_search: 'tavily' } } });
    await last().close();
    await waitFor(() => expect(apply).toHaveBeenCalledWith({ preset: 'tavily', api_key: 'tvly-good', assignments: {} }));
    expect(await screen.findByText('Web search: Tavily (web search)')).toBeInTheDocument();
  });

  it('turns local tools on or off only when asked', async () => {
    vi.spyOn(api, 'quickPresets').mockResolvedValue(PRESETS);
    vi.spyOn(api, 'routes').mockResolvedValue([{ task: 'ocr', effective_provider_ids: ['local-paddleocr'] }] as unknown as RouteRow[]);
    vi.spyOn(api, 'providers').mockResolvedValue([
      { id: 'local-paddleocr', adapter: 'paddleocr_local', name: 'PaddleOCR', enabled: false, models: { ocr: 'PP-OCRv5' }, config: {} },
    ] as unknown as ProviderRecord[]);
    vi.spyOn(api, 'catalog').mockResolvedValue([{ adapter: 'paddleocr_local', requires_network: false }] as ProviderDescriptor[]);
    const update = vi.spyOn(api, 'updateProvider').mockResolvedValue({} as ProviderRecord);
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <ConnectPage />
      </MemoryRouter>,
    );
    expect(await screen.findByText('Turned off — scanned pages are skipped')).toBeInTheDocument();
    expect(screen.getByRole('switch', { name: /Reads scanned pages \(PaddleOCR\) \(off\)/ })).not.toBeChecked();
    expect(update).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: 'Turn on: Reads scanned pages' }));
    expect(update).toHaveBeenCalledWith('local-paddleocr', { enabled: true });
  });
});

describe('automatic ranking and scanned-page reading', () => {
  const REC2 = { ...REC, reranking: 'meta/llama-3.3-70b-instruct', ocr: 'qwen/qwen3-vl-235b' };

  async function ready(user: ReturnType<typeof userEvent.setup>, last: () => Stream, rec: Record<string, string | null>, extra: DiscoverEvent[] = []) {
    await startOpenRouter(user);
    await waitFor(() => expect(last()).toBeDefined());
    await last().push(modelsEvent());
    await last().push(probe('nvidia/nemotron-3-ultra-550b', { available: true, json_ok: true }, rec));
    for (const e of extra) await last().push(e);
  }

  it('pre-selects the recommended connected models and sends them', async () => {
    const { apply, user, last } = setup();
    await ready(user, last, REC2, [
      probe('meta/llama-3.3-70b-instruct', { available: true, json_ok: true }, REC2),
      probe('qwen/qwen3-vl-235b', { available: true, json_ok: true }, REC2),
    ]);
    const rank = roleSelect(/Ranks search results/);
    const ocr = roleSelect(/Reads scanned pages/);
    expect(rank.value).toBe('meta/llama-3.3-70b-instruct');
    expect(ocr.value).toBe('qwen/qwen3-vl-235b');
    expect(within(rank).getByRole('option', { name: 'Use tools on this computer' })).toBeInTheDocument();
    expect(within(rank).getByRole('option', { name: 'Off' })).toBeInTheDocument();
    expect(within(ocr).getByRole('option', { name: 'Use PaddleOCR on this computer' })).toBeInTheDocument();
    // only vision models can read scans
    expect(within(ocr).queryByRole('option', { name: /llama-3\.3/ })).not.toBeInTheDocument();
    expect(screen.getByText(/About 1–2 s per search/)).toBeInTheDocument();
    expect(screen.getByText(/stay available as backups/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Connect' }));
    await waitFor(() => expect(apply).toHaveBeenCalled());
    expect(apply.mock.calls[0][0].assignments).toMatchObject({ reranking: 'meta/llama-3.3-70b-instruct', ocr: 'qwen/qwen3-vl-235b' });
    expect(await screen.findByText('Result ranking: Llama 3.3 70B Instruct')).toBeInTheDocument();
    expect(screen.getByText('Scanned pages: Qwen3 VL 235B')).toBeInTheDocument();
  });

  it('sends "local" and "off" when chosen, and keeps them through later updates', async () => {
    const { apply, user, last } = setup();
    await ready(user, last, REC2);
    await user.selectOptions(roleSelect(/Ranks search results/), 'local');
    await user.selectOptions(roleSelect(/Reads scanned pages/), 'off');
    await last().push(probe('qwen/qwen3-vl-235b', { available: true, json_ok: true }, REC2));
    expect(roleSelect(/Ranks search results/).value).toBe('local');
    expect(roleSelect(/Reads scanned pages/).value).toBe('off');
    await user.click(screen.getByRole('button', { name: 'Connect' }));
    await waitFor(() => expect(apply).toHaveBeenCalled());
    expect(apply.mock.calls[0][0].assignments).toMatchObject({ reranking: 'local', ocr: 'off' });
    expect(await screen.findByText('Result ranking: tools on this computer')).toBeInTheDocument();
    expect(screen.getByText('Scanned pages: off')).toBeInTheDocument();
  });

  it('defaults scanned pages to PaddleOCR when the service has no vision model', async () => {
    const { apply, user, last } = setup();
    await ready(user, last, { ...REC, vision: null, reranking: 'nvidia/nemotron-3-ultra-550b', ocr: null });
    expect(roleSelect(/Reads scanned pages/).value).toBe('local');
    expect(roleSelect(/Ranks search results/).value).toBe('nvidia/nemotron-3-ultra-550b');
    await user.click(screen.getByRole('button', { name: 'Connect' }));
    await waitFor(() => expect(apply).toHaveBeenCalled());
    expect(apply.mock.calls[0][0].assignments).toMatchObject({ reranking: 'nvidia/nemotron-3-ultra-550b', ocr: 'local' });
  });

  it('merge logic: tool roles fall back to local, and only local for services that cannot do it', () => {
    let s = applyEvent(EMPTY_DISCOVERY, modelsEvent());
    s = applyEvent(s, probe('qwen/qwen3-vl-235b', { available: true, json_ok: true }, { ...REC, reranking: 'qwen/qwen3-vl-235b', ocr: 'qwen/qwen3-vl-235b' }));
    const none = { synthesis: null, planning: null, vision: null, reranking: null, ocr: null, fallback: null, embedding: null };
    expect(mergeRecommendations(none, new Set(), s.recommended, s.models)).toMatchObject({ reranking: 'qwen/qwen3-vl-235b', ocr: 'qwen/qwen3-vl-235b' });
    expect(mergeRecommendations(none, new Set(), s.recommended, s.models, false)).toMatchObject({ reranking: 'local', ocr: 'local' });
    // a picked model that fails its test falls back to the local tools
    s = applyEvent(s, probe('qwen/qwen3-vl-235b', { available: false, detail: 'timeout' }, { ...s.recommended, ocr: null, reranking: null }));
    expect(mergeRecommendations({ ...none, ocr: 'qwen/qwen3-vl-235b' }, new Set(['ocr']), s.recommended, s.models).ocr).toBe('local');
  });

  it('shows the cloud model in use and no "turned off" warning when only local fallbacks are off', () => {
    const routes = [
      { task: 'reranking', effective_provider_ids: ['nvidia-x-rerank', 'local-cross-encoder', 'local-lexical-rerank'] },
      { task: 'ocr', effective_provider_ids: ['nvidia-x-ocr', 'local-paddleocr'] },
    ] as unknown as RouteRow[];
    const rec = (id: string, adapter: string, enabled: boolean, models: Record<string, string>, base_url?: string) =>
      ({ id, adapter, name: id, models, enabled, config: base_url ? { base_url } : {} }) as unknown as ProviderRecord;
    const nim = 'https://nvidia.example/v1';
    const providers = [
      rec('nvidia-x-rerank', 'llm_rerank', true, { reranking: 'nvidia/nemotron-3.5-super-vl-120b-a12b' }, nim),
      rec('nvidia-x-ocr', 'vlm_ocr', true, { ocr: 'nvidia/nemotron-3.5-super-vl-120b-a12b' }, nim),
      rec('local-cross-encoder', 'fastembed_rerank', false, { reranking: 'x' }),
      rec('local-lexical-rerank', 'local_lexical_rerank', false, { reranking: 'y' }),
      rec('local-paddleocr', 'paddleocr_local', false, { ocr: 'PP-OCRv5' }),
    ];
    const catalog = ['fastembed_rerank', 'local_lexical_rerank', 'paddleocr_local'].map((a) => ({ adapter: a, requires_network: false })) as ProviderDescriptor[];
    const by = Object.fromEntries(currentlyUsing(routes, providers, catalog, PRESETS).map((l) => [l.task, l]));
    for (const t of ['reranking', 'ocr']) {
      expect(by[t]).toMatchObject({ value: 'Nemotron 3.5 Super VL 120B A12B', where: 'NVIDIA NIM', disabled: false, toggle: undefined, backups: 0 });
    }
  });
});

describe('settings routes', () => {
  function renderApp(path: string) {
    vi.spyOn(api, 'health').mockRejectedValue(new Error('offline'));
    vi.spyOn(api, 'quickPresets').mockResolvedValue(PRESETS);
    vi.spyOn(api, 'routes').mockResolvedValue([]);
    vi.spyOn(api, 'providers').mockResolvedValue([]);
    vi.spyOn(api, 'catalog').mockResolvedValue([]);
    render(
      <MemoryRouter initialEntries={[path]}>
        <App />
      </MemoryRouter>,
    );
  }

  it.each(['/settings/providers', '/settings/routing'])('redirects the removed %s page to Connect AI', async (path) => {
    renderApp(path);
    expect(await screen.findByRole('heading', { name: 'Connect AI' }, { timeout: 5000 })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Connect AI' })).toHaveAttribute('aria-current', 'page');
    expect(screen.queryByRole('link', { name: 'Custom providers' })).not.toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'Routing' })).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Provider health' })).toBeInTheDocument();
  });
});

// keep the LlmDiscovery type referenced for readers of this file
export type { LlmDiscovery };
