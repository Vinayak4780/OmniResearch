import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, api, buildUrl, formatDetail, request } from './api';
import { getAdminToken, setAdminToken } from './storage';

function jsonResponse(body: unknown, status = 200, headers: Record<string, string> = {}) {
  return new Response(body === undefined ? null : JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json', ...headers },
  });
}

describe('api client', () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const lastHeaders = () => (fetchMock.mock.calls.at(-1)![1] as RequestInit).headers as Record<string, string>;

  it('sends no Authorization header when no admin token is stored', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ status: 'ok' }));
    await api.health();
    expect(fetchMock).toHaveBeenCalledWith('/api/health', expect.objectContaining({ method: 'GET' }));
    expect(lastHeaders().Authorization).toBeUndefined();
  });

  it('attaches the stored admin token as a Bearer header on every request', async () => {
    setAdminToken('  s3cret-token ');
    expect(getAdminToken()).toBe('s3cret-token');
    fetchMock.mockImplementation(async () => jsonResponse([]));
    await api.providers();
    expect(lastHeaders().Authorization).toBe('Bearer s3cret-token');
    await api.documents({ q: 'kv cache', collection: 'papers' });
    expect(fetchMock.mock.calls.at(-1)![0]).toBe('/api/documents?q=kv+cache&collection=papers');
    expect(lastHeaders().Authorization).toBe('Bearer s3cret-token');
  });

  it('survives localStorage throwing', async () => {
    const spy = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('SecurityError');
    });
    fetchMock.mockResolvedValue(jsonResponse({ status: 'ok' }));
    await expect(api.health()).resolves.toEqual({ status: 'ok' });
    expect(lastHeaders().Authorization).toBeUndefined();
    spy.mockRestore();
  });

  it('serializes JSON bodies and passes FormData through untouched', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ id: 'job1' }, 201));
    await api.startResearch({ question: 'What is paged attention?', web_mode: 'never' });
    const init = fetchMock.mock.calls.at(-1)![1] as RequestInit;
    expect(init.method).toBe('POST');
    expect(init.body).toBe(JSON.stringify({ question: 'What is paged attention?', web_mode: 'never' }));
    expect((init.headers as Record<string, string>)['Content-Type']).toBe('application/json');

    fetchMock.mockResolvedValue(jsonResponse({ document: { id: 'd1' }, duplicate: false, job_id: 'j' }, 201));
    await api.uploadDocument(new File(['%PDF-1.4'], 'a.pdf', { type: 'application/pdf' }), 'papers');
    const up = fetchMock.mock.calls.at(-1)![1] as RequestInit;
    expect(up.body).toBeInstanceOf(FormData);
    expect((up.body as FormData).get('collection')).toBe('papers');
    expect((up.headers as Record<string, string>)['Content-Type']).toBeUndefined();
  });

  it('throws ApiError with status and string detail', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ detail: 'File exceeds 100 MB' }, 413));
    const err = await api.uploadDocument(new File(['x'], 'big.pdf'), 'default').catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(413);
    expect(err.detail).toBe('File exceeds 100 MB');
    expect(err.message).toBe('File exceeds 100 MB');
    expect(err.isAuthError).toBe(false);
  });

  it('flags 401/403 as auth errors', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ detail: 'Admin token required' }, 401));
    const err = (await api.providers().catch((e) => e)) as ApiError;
    expect(err.status).toBe(401);
    expect(err.isAuthError).toBe(true);
    fetchMock.mockResolvedValue(jsonResponse({ detail: 'Remote admin access requires OMNI_ADMIN_TOKEN to be configured' }, 403));
    expect(((await api.routes().catch((e) => e)) as ApiError).isAuthError).toBe(true);
  });

  it('keeps extra error-body fields and handles 204 responses', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ detail: 'Provider is still referenced', references: ['route:synthesis'] }, 409));
    const err = (await api.updateProvider('local-fastembed', { enabled: false }).catch((e) => e)) as ApiError;
    expect(err.status).toBe(409);
    expect(err.body?.references).toEqual(['route:synthesis']);
    const init = fetchMock.mock.calls.at(-1)![1] as RequestInit;
    expect(fetchMock.mock.calls.at(-1)![0]).toBe('/api/providers/local-fastembed');
    expect(init.method).toBe('PATCH');
    expect(init.body).toBe('{"enabled":false}');
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }));
    await expect(api.deleteResearch('job1')).resolves.toBeUndefined();
  });

  it('formats FastAPI validation error lists', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({ detail: [{ loc: ['body', 'question'], msg: 'String should have at least 5 characters', type: 'string_too_short' }] }, 422),
    );
    const err = (await api.startResearch({ question: 'x' }).catch((e) => e)) as ApiError;
    expect(err.status).toBe(422);
    expect(err.detail).toBe('question: String should have at least 5 characters');
  });

  it('handles non-JSON error bodies and network failures', async () => {
    fetchMock.mockResolvedValue(new Response('Bad Gateway from proxy', { status: 502, statusText: 'Bad Gateway' }));
    const err = (await request('/health').catch((e) => e)) as ApiError;
    expect(err.status).toBe(502);
    expect(err.detail).toBe('Bad Gateway from proxy');

    fetchMock.mockRejectedValue(new TypeError('Failed to fetch'));
    const net = (await request('/health').catch((e) => e)) as ApiError;
    expect(net).toBeInstanceOf(ApiError);
    expect(net.status).toBe(0);
    expect(net.detail).toMatch(/Network error/);
  });

  it('builds URLs and helpers', () => {
    expect(buildUrl('/research/abc/events', { after: 12 })).toBe('/api/research/abc/events?after=12');
    expect(buildUrl('/documents', { q: '', collection: undefined })).toBe('/api/documents');
    expect(formatDetail(null)).toBe('');
    expect(formatDetail({ a: 1 })).toBe('{"a":1}');
  });

  it('reads plain-text responses when asked', async () => {
    fetchMock.mockResolvedValue(new Response('# Report', { status: 200, headers: { 'content-type': 'text/markdown' } }));
    await expect(request<string>('/research/x/report.md', { text: true })).resolves.toBe('# Report');
  });
});
