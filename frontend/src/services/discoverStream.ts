/**
 * Streaming model discovery: POST /api/quick-connect/discover/stream returns NDJSON events
 * (models → probe… → done | error). Aborting the request cancels the remaining tests on the server.
 */
import type { DiscoverEvent, QuickDiscoverPayload } from '../types/quickconnect';
import { api, baseHeaders, buildUrl, errorFromResponse, networkError } from './api';

/** Incremental NDJSON parser: feed arbitrary text chunks, get one callback per complete JSON line. */
export function createNdjsonParser<T>(onItem: (item: T) => void, onBadLine?: (line: string) => void) {
  let buffer = '';
  const handle = (line: string) => {
    const s = line.trim();
    if (!s) return;
    try {
      onItem(JSON.parse(s) as T);
    } catch {
      onBadLine?.(s);
    }
  };
  return {
    push(chunk: string) {
      buffer += chunk;
      let nl = buffer.indexOf('\n');
      while (nl >= 0) {
        handle(buffer.slice(0, nl));
        buffer = buffer.slice(nl + 1);
        nl = buffer.indexOf('\n');
      }
    },
    /** Parse whatever is left after the stream ends (a final line without a trailing newline). */
    flush() {
      const rest = buffer;
      buffer = '';
      handle(rest);
    },
  };
}

export interface StreamOptions {
  signal?: AbortSignal;
  onEvent: (e: DiscoverEvent) => void;
}

/**
 * Run a streaming discovery, calling `onEvent` for each event. Resolves when the stream ends.
 * HTTP errors (401/403/422…) reject with ApiError; `{"type":"error"}` events are delivered to `onEvent`.
 * Falls back to the non-streaming endpoint when the browser cannot read response streams.
 */
export async function streamDiscover(payload: QuickDiscoverPayload, { signal, onEvent }: StreamOptions): Promise<void> {
  let res: Response;
  try {
    res = await fetch(buildUrl('/quick-connect/discover/stream'), {
      method: 'POST',
      headers: { ...baseHeaders('application/x-ndjson'), 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e;
    throw networkError(e);
  }
  if (!res.ok) throw await errorFromResponse(res);

  const reader = res.body?.getReader?.();
  if (!reader || typeof TextDecoder === 'undefined') {
    // Streaming not supported here: read the whole body if possible, else use the classic endpoint.
    const text = await res.text().catch(() => null);
    if (text !== null) {
      const p = createNdjsonParser<DiscoverEvent>(onEvent);
      p.push(text);
      p.flush();
      return;
    }
    const result = await api.quickDiscover(payload, signal);
    onEvent({ type: 'done', result });
    return;
  }

  const decoder = new TextDecoder();
  const parser = createNdjsonParser<DiscoverEvent>(onEvent);
  const onAbort = () => void reader.cancel().catch(() => undefined);
  signal?.addEventListener('abort', onAbort);
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      parser.push(decoder.decode(value, { stream: true }));
    }
    parser.push(decoder.decode());
    parser.flush();
  } catch (e) {
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    throw networkError(e);
  } finally {
    signal?.removeEventListener('abort', onAbort);
  }
}
