import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import { App } from '../../App';
import { ApiError, api } from '../../services/api';
import type { LocalCatalog, LocalJob, LocalModel } from '../../types/localmodels';
import { LocalModelsPage, fmtSize, jobProgressText } from './LocalModelsPage';

const model = (id: string, extra: Partial<LocalModel> = {}): LocalModel => ({
  id,
  description: 'Text embeddings, English',
  size_gb: 0.067,
  source: 'catalog',
  installed: false,
  in_use: false,
  recommended: false,
  job: null,
  ...extra,
});

function catalog(over: Partial<LocalCatalog> = {}): LocalCatalog {
  return {
    available: { ocr: true, embedding: true, reranking: true },
    in_use: { ocr: 'PP-OCRv5_mobile_det+en_PP-OCRv5_mobile_rec', embedding: null, reranking: null },
    jobs: [],
    kinds: {
      ocr: [
        model('PP-OCRv5_mobile_det+en_PP-OCRv5_mobile_rec', { description: 'English · Fast text detection', language: 'English', size_gb: 0.015, installed: true, in_use: true, recommended: true }),
        model('PP-OCRv5_mobile_det+latin_PP-OCRv5_mobile_rec', { description: 'Latin · Fast text detection', language: 'Latin', size_gb: 0.016 }),
      ],
      embedding: [
        model('BAAI/bge-small-en-v1.5', { recommended: true, dims: 384, license: 'mit' }),
        model('BAAI/bge-base-en-v1.5', { size_gb: 0.21, dims: 768, installed: true }),
        ...Array.from({ length: 9 }, (_, i) => model(`org/extra-${i + 1}`, { description: i === 8 ? 'Multilingual (~100 languages)' : 'Other' })),
      ],
      reranking: [model('Xenova/ms-marco-MiniLM-L-6-v2', { installed: true, in_use: false, description: 'Cross-encoder reranker' })],
    },
    ...over,
  };
}

function renderPage(path = '/settings/local-models') {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <LocalModelsPage />
    </MemoryRouter>,
  );
}

describe('helpers', () => {
  it('formats sizes and download progress', () => {
    expect(fmtSize(0.067)).toBe('67 MB');
    expect(fmtSize(2.24)).toBe('2.2 GB');
    expect(fmtSize(null)).toBe('size unknown');
    const job = { downloaded_mb: 23.4, expected_mb: 67, progress: 0.35 } as LocalJob;
    expect(jobProgressText(job)).toBe('23 / 67 MB');
    expect(jobProgressText({ ...job, expected_mb: null })).toBe('23 MB downloaded');
  });
});

describe('Models on this computer', () => {
  it('renders the catalog per job with status, sizes and actions', async () => {
    vi.spyOn(api, 'localModels').mockResolvedValue(catalog());
    const user = userEvent.setup();
    renderPage();
    expect(await screen.findByText('Models on this computer')).toBeInTheDocument();
    expect(screen.getByText(/Free, private, run on your CPU/)).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Reads scanned pages \(OCR\)/ })).toHaveAttribute('aria-selected', 'true');
    const inUse = document.querySelector('[data-model="PP-OCRv5_mobile_det+en_PP-OCRv5_mobile_rec"]') as HTMLElement;
    expect(within(inUse).getByText('In use')).toBeInTheDocument();
    expect(within(inUse).getByText('English.')).toBeInTheDocument();
    expect(within(inUse).getByText('15 MB')).toBeInTheDocument();
    expect(within(inUse).queryByRole('button', { name: /Remove/ })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Install PP-OCRv5_mobile_det+latin_PP-OCRv5_mobile_rec' })).toBeInTheDocument();

    await user.click(screen.getByRole('tab', { name: /Document search \(embeddings\)/ }));
    expect(screen.getByText(/rebuilds the search index automatically/)).toBeInTheDocument();
    expect(screen.getByText(/No model on this computer/)).toBeInTheDocument();
    const small = document.querySelector('[data-model="BAAI/bge-small-en-v1.5"]') as HTMLElement;
    expect(within(small).getByText('Recommended')).toBeInTheDocument();
    expect(within(small).getByText('67 MB')).toBeInTheDocument();
    expect(within(small).getByText('384 dimensions')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Use BAAI/bge-base-en-v1.5' })).toBeInTheDocument();
    // long lists are collapsed, with a filter
    expect(screen.getAllByRole('listitem').filter((li) => li.classList.contains('local-model'))).toHaveLength(8);
    await user.type(screen.getByRole('searchbox', { name: /Filter Document search/ }), 'multilingual');
    expect(screen.getByText('org/extra-9')).toBeInTheDocument();
    expect(screen.queryByText('org/extra-1')).not.toBeInTheDocument();
  });

  it('install → progress polling → done', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const job: LocalJob = { id: 'j1', kind: 'embedding', model: 'BAAI/bge-small-en-v1.5', status: 'downloading', error: null, downloaded_mb: 23, expected_mb: 67, progress: 0.34 };
      const withJob = (j: LocalJob | null, installed = false) => {
        const c = catalog();
        c.kinds.embedding[0] = { ...c.kinds.embedding[0], job: j, installed };
        c.jobs = j && j.status === 'downloading' ? [j] : [];
        return c;
      };
      const list = vi
        .spyOn(api, 'localModels')
        .mockResolvedValueOnce(catalog())
        .mockResolvedValueOnce(withJob(job))
        .mockResolvedValueOnce(withJob({ ...job, downloaded_mb: 50, progress: 0.75 }))
        .mockResolvedValue(withJob({ ...job, status: 'done', downloaded_mb: 67, progress: 1 }, true));
      const install = vi.spyOn(api, 'installLocalModel').mockResolvedValue(job);
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      renderPage('/settings/local-models?kind=embedding');
      await user.click(await screen.findByRole('button', { name: 'Install BAAI/bge-small-en-v1.5' }));
      expect(install).toHaveBeenCalledWith({ kind: 'embedding', model: 'BAAI/bge-small-en-v1.5' });
      expect(await screen.findByText('23 / 67 MB')).toBeInTheDocument();
      expect(screen.getByRole('progressbar', { name: /Downloading BAAI\/bge-small-en-v1.5/ })).toHaveAttribute('aria-valuenow', '34');
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1600);
      });
      expect(await screen.findByText('50 / 67 MB')).toBeInTheDocument();
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1600);
      });
      expect(await screen.findByRole('button', { name: 'Use BAAI/bge-small-en-v1.5' })).toBeInTheDocument();
      expect(screen.queryByText(/\/ 67 MB/)).not.toBeInTheDocument();
      const calls = list.mock.calls.length;
      await act(async () => {
        await vi.advanceTimersByTimeAsync(4000);
      });
      expect(list.mock.calls.length).toBe(calls); // polling stops once nothing is downloading
    } finally {
      vi.useRealTimers();
    }
  });

  it('searches Hugging Face and installs a result with its repo', async () => {
    vi.spyOn(api, 'localModels').mockResolvedValue(catalog());
    const search = vi.spyOn(api, 'searchLocalModels').mockResolvedValue([
      { id: 'jinaai/jina-embeddings-v2-small-en', repo: 'jinaai/jina-embeddings-v2-small-en', downloads: 123456, likes: 42, in_catalog: false, url: 'https://huggingface.co/jinaai/jina-embeddings-v2-small-en' },
    ]);
    const install = vi.spyOn(api, 'installLocalModel').mockRejectedValueOnce(new ApiError(400, 'jinaai/x has no ONNX weights'));
    install.mockResolvedValueOnce({ id: 'j', kind: 'embedding', model: 'jinaai/jina-embeddings-v2-small-en', status: 'downloading', error: null, downloaded_mb: 0, expected_mb: null, progress: null });
    const user = userEvent.setup();
    renderPage('/settings/local-models?kind=embedding');
    await user.type(await screen.findByPlaceholderText('Search Hugging Face…'), 'jina small');
    await user.click(screen.getByRole('button', { name: 'Search' }));
    expect(search).toHaveBeenCalledWith('embedding', 'jina small');
    const link = await screen.findByRole('link', { name: /jinaai\/jina-embeddings-v2-small-en/ });
    expect(link).toHaveAttribute('href', 'https://huggingface.co/jinaai/jina-embeddings-v2-small-en');
    expect(link).toHaveAttribute('target', '_blank');
    // grouping depends on the machine's locale (123,456 / 1,23,456 / 123.456)
    expect(screen.getByText(/\d[\d,.\s ]*456 downloads · 42 likes/)).toBeInTheDocument();
    const btn = screen.getByRole('button', { name: 'Install jinaai/jina-embeddings-v2-small-en from Hugging Face' });
    await user.click(btn);
    expect(await screen.findByText(/has no ONNX weights/)).toBeInTheDocument();
    await user.click(btn);
    expect(install).toHaveBeenLastCalledWith({ kind: 'embedding', model: 'jinaai/jina-embeddings-v2-small-en', repo: 'jinaai/jina-embeddings-v2-small-en' });
  });

  it('use on embedding confirms and shows the reindex note; remove-in-use errors are shown', async () => {
    vi.spyOn(api, 'localModels').mockResolvedValue(catalog());
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);
    const use = vi.spyOn(api, 'useLocalModel').mockResolvedValue({ provider_id: 'local-bge-base', task: 'text_embedding', notes: ['Rebuilding the search index with BAAI/bge-base-en-v1.5 in the background.'] });
    const remove = vi.spyOn(api, 'removeLocalModel').mockRejectedValue(new ApiError(400, 'This model is in use. Choose another model first.'));
    const user = userEvent.setup();
    renderPage('/settings/local-models?kind=embedding');
    await user.click(await screen.findByRole('button', { name: 'Use BAAI/bge-base-en-v1.5' }));
    expect(confirm).toHaveBeenCalledWith(expect.stringMatching(/search index is rebuilt/));
    expect(use).toHaveBeenCalledWith({ kind: 'embedding', model: 'BAAI/bge-base-en-v1.5' });
    expect(await screen.findByText('Rebuilding the search index with BAAI/bge-base-en-v1.5 in the background.')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Remove BAAI/bge-base-en-v1.5' }));
    expect(remove).toHaveBeenCalledWith({ kind: 'embedding', model: 'BAAI/bge-base-en-v1.5' });
    expect(await screen.findByText('This model is in use. Choose another model first.')).toBeInTheDocument();
  });

  it('explains when the server has no local runtime', async () => {
    vi.spyOn(api, 'localModels').mockResolvedValue(catalog({ available: { ocr: false, embedding: true, reranking: true } }));
    renderPage();
    expect(await screen.findByText(/Local models aren't available on this server/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^Install / })).not.toBeInTheDocument();
  });
});

describe('removed settings pages', () => {
  it.each(['/settings/web', '/settings/ocr'])('redirects %s to Connect AI and the nav lists Local models', async (path) => {
    vi.spyOn(api, 'health').mockRejectedValue(new Error('offline'));
    vi.spyOn(api, 'quickPresets').mockResolvedValue([]);
    vi.spyOn(api, 'routes').mockResolvedValue([]);
    vi.spyOn(api, 'providers').mockResolvedValue([]);
    vi.spyOn(api, 'catalog').mockResolvedValue([]);
    render(
      <MemoryRouter initialEntries={[path]}>
        <App />
      </MemoryRouter>,
    );
    expect(await screen.findByRole('heading', { name: 'Connect AI' }, { timeout: 5000 })).toBeInTheDocument();
    const nav = screen.getByRole('navigation', { name: 'Settings sections' });
    const links = within(nav).getAllByRole('link').map((a) => a.textContent);
    expect(links.slice(0, 2)).toEqual(['Connect AI', 'Local models']);
    expect(links).not.toContain('Web research');
    expect(links).not.toContain('Scanned pages (OCR)');
    await waitFor(() => expect(screen.getByText(/Web research turns on automatically once a search service is connected/)).toBeInTheDocument());
  });
});
