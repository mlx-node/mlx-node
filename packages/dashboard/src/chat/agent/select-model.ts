/**
 * The app's own calls to the local engine, over `node:http` to 127.0.0.1 only, with the token in the `x-api-key`
 * header and nowhere else:
 * - the model select call before a prompt goes to the agent: `POST /v1/messages/count_tokens` names the model, and the
 *   engine binds every later request (the agent's title helper included) to it. The engine sends the response headers
 *   only after the load, so the call has no header timeout of its own (`fetch` gives up after 300 s without headers);
 * - `POST /v1/models/unload`.
 */

import { request } from 'node:http';

/** A load longer than this fails the turn. */
export const MODEL_SELECT_MAX_MS = 1_800_000;
/** The unload call answers at once or with 409; this only bounds a hung engine. */
export const UNLOAD_MAX_MS = 60_000;
const UNLOAD_BODY_MAX = 64 * 1024;

/** What an engine call needs: the address, the token and the names the engine serves. */
export interface SelectTarget {
  url: string;
  token: string;
  models: ReadonlyArray<{ name: string }>;
}

/**
 * - `selected`: the model is in memory (200), loaded with no token counter (501), or swapped out by another
 *   conversation's select before the count ran (400): after this call loaded it, or when it was in memory and the count
 *   waited behind that select's load. After a 400 the agent's own request loads it. Calling again on a 400 loads it
 *   again and swaps the other conversation's model out under it, so two conversations on two models would fail each
 *   other.
 * - `unknown`: the engine does not serve the name (404), or the name is not in the endpoint file (no request).
 * - `auth`: the engine refused the token (401, 403).
 * - `failed`: the load failed (429, 500, any other status), or the ceiling was reached.
 * - `socket`: no answer (connection refused or dropped), or the address is not loopback (no request).
 * - `aborted`: the caller aborted; the socket is closed.
 */
export type SelectAnswer = 'selected' | 'unknown' | 'auth' | 'failed' | 'socket' | 'aborted';

export function answerOfStatus(status: number): SelectAnswer {
  if (status === 200 || status === 501 || status === 400) return 'selected';
  if (status === 404) return 'unknown';
  if (status === 401 || status === 403) return 'auth';
  return 'failed';
}

/** `path` on a loopback http engine URL, or null for any other URL. */
export function engineUrl(base: string, path: string): URL | null {
  let url: URL;
  try {
    url = new URL(base);
  } catch {
    return null;
  }
  if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1') return null;
  if (url.username !== '' || url.password !== '') return null;
  let prefix = url.pathname;
  while (prefix.endsWith('/')) prefix = prefix.slice(0, -1);
  url.pathname = `${prefix}${path}`;
  url.search = '';
  url.hash = '';
  return url;
}

export function selectUrl(base: string): URL | null {
  return engineUrl(base, '/v1/messages/count_tokens');
}

type PostResult = { status: number; body: string } | 'socket' | 'timeout' | 'aborted';

/** One POST. Resolves; never rejects. Reads at most `bodyMax` bytes of the answer. */
function post(
  url: URL,
  token: string,
  payload: string | null,
  options: { maxMs: number; signal?: AbortSignal; bodyMax: number },
): Promise<PostResult> {
  if (options.signal?.aborted === true) return Promise.resolve('aborted');
  return new Promise<PostResult>((resolve) => {
    let done = false;
    const headers: Record<string, string | number> = { 'x-api-key': token };
    if (payload !== null) {
      headers['content-type'] = 'application/json';
      headers['content-length'] = Buffer.byteLength(payload);
    }
    const req = request(url, { method: 'POST', agent: false, headers });
    const finish = (result: PostResult): void => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      options.signal?.removeEventListener('abort', onAbort);
      req.destroy();
      resolve(result);
    };
    const onAbort = (): void => finish('aborted');
    const timer = setTimeout(() => finish('timeout'), options.maxMs);
    options.signal?.addEventListener('abort', onAbort, { once: true });
    req.on('response', (res) => {
      const status = res.statusCode ?? 0;
      const chunks: Buffer[] = [];
      let size = 0;
      res.on('data', (chunk: Buffer) => {
        size += chunk.length;
        if (size <= options.bodyMax) chunks.push(chunk);
      });
      res.on('error', () => {});
      const end = (): void => finish({ status, body: Buffer.concat(chunks).toString('utf8') });
      res.on('end', end);
      res.on('close', end);
    });
    req.on('error', () => finish('socket'));
    req.end(payload ?? undefined);
  });
}

/** One select call. Resolves; never rejects. `maxMs` bounds the whole call; `signal` closes the socket at once. */
export async function selectModel(
  target: SelectTarget,
  modelId: string,
  options: { maxMs: number; signal?: AbortSignal },
): Promise<SelectAnswer> {
  if (!target.models.some((m) => m.name === modelId)) return 'unknown';
  const url = selectUrl(target.url);
  if (url === null) return 'socket';
  const body = JSON.stringify({ model: modelId, messages: [{ role: 'user', content: '.' }] });
  const result = await post(url, target.token, body, { ...options, bodyMax: 0 });
  if (result === 'timeout') return 'failed';
  if (result === 'socket' || result === 'aborted') return result;
  return answerOfStatus(result.status);
}

/** The unload call: the names the engine unloaded, or why it did not. */
export type UnloadAnswer = { unloaded: string[] } | 'busy' | 'auth' | 'failed' | 'socket';

export async function unloadModels(target: SelectTarget, options: { maxMs: number }): Promise<UnloadAnswer> {
  const url = engineUrl(target.url, '/v1/models/unload');
  if (url === null) return 'socket';
  const result = await post(url, target.token, null, { maxMs: options.maxMs, bodyMax: UNLOAD_BODY_MAX });
  if (result === 'socket' || result === 'aborted') return 'socket';
  if (result === 'timeout') return 'failed';
  if (result.status === 409) return 'busy';
  if (result.status === 401 || result.status === 403) return 'auth';
  if (result.status !== 200) return 'failed';
  let parsed: unknown;
  try {
    parsed = JSON.parse(result.body);
  } catch {
    return 'failed';
  }
  const names = typeof parsed === 'object' && parsed !== null ? (parsed as { unloaded?: unknown }).unloaded : null;
  if (!Array.isArray(names)) return 'failed';
  const known = new Set(target.models.map((m) => m.name));
  return { unloaded: names.filter((n): n is string => typeof n === 'string' && known.has(n)) };
}
