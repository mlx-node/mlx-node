import type { IncomingMessage } from 'node:http';
import { connect as netConnect, type AddressInfo } from 'node:net';

import { createServer, type ServerInstance } from '@mlx-node/server';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { extractPresentedToken, isAuthorized, tokensMatch } from '../../packages/server/src/auth.js';
import { resolveAuthToken } from '../../packages/server/src/server.js';

/**
 * Bearer-token auth + the `/health` disclosure split.
 *
 * Auth is enforced at the single choke point in `createHandler`'s returned
 * closure — after the CORS/OPTIONS early-return, before `routeRequest`. The
 * end-to-end tests drive a REAL `node:http` server over loopback rather than
 * a mocked `(req, res)` pair, because the properties under test are
 * header-level (`WWW-Authenticate`, CORS defaulting, how the parser folds
 * duplicate headers) and a mock would let a bug in any of that pass unnoticed.
 *
 * No model is ever registered: every request under test is rejected or
 * answered before the endpoint layer would touch the native addon.
 */

const TOKEN = 'sk-test-0123456789abcdef';

let servers: ServerInstance[] = [];

async function start(config: Parameters<typeof createServer>[0] = {}): Promise<string> {
  const instance = await createServer({ port: 0, host: '127.0.0.1', disableStore: true, ...config });
  servers.push(instance);
  const { port } = instance.server.address() as AddressInfo;
  return `http://127.0.0.1:${port}`;
}

afterEach(async () => {
  const pending = servers;
  servers = [];
  for (const instance of pending) {
    await instance.close({ timeoutMs: 250 }).catch(() => {});
  }
});

/**
 * Header-parsing unit tests.
 *
 * `node:http` joins repeated `x-api-key` / `authorization` values into a
 * single comma-separated STRING — verified empirically, it never hands back a
 * `string[]` for these two headers. The array branch is still reachable when
 * `createHandler` is mounted into a framework that normalises headers into
 * arrays, and TypeScript types the field as `string | string[] | undefined`
 * either way, so it is exercised directly here rather than through a socket
 * that cannot produce it.
 */
describe('extractPresentedToken', () => {
  function headers(h: Record<string, string | string[]>): IncomingMessage {
    return { headers: h } as unknown as IncomingMessage;
  }

  it('rejects a genuinely array-valued `x-api-key`', () => {
    expect(extractPresentedToken(headers({ 'x-api-key': [TOKEN, TOKEN] }))).toBeNull();
    expect(isAuthorized(headers({ 'x-api-key': [TOKEN] }), TOKEN)).toBe(false);
  });

  it('rejects a genuinely array-valued `authorization`', () => {
    expect(extractPresentedToken(headers({ authorization: [`Bearer ${TOKEN}`] }))).toBeNull();
  });

  it('does not fall back to `authorization` when `x-api-key` is present but unusable', () => {
    // Falling through would let a caller bypass a proxy that only sanitises
    // `x-api-key` by parking a valid token in `authorization`.
    expect(extractPresentedToken(headers({ 'x-api-key': [TOKEN], authorization: `Bearer ${TOKEN}` }))).toBeNull();
  });

  it('returns the raw credential without trimming', () => {
    expect(extractPresentedToken(headers({ authorization: `Bearer  ${TOKEN}` }))).toBe(` ${TOKEN}`);
    expect(extractPresentedToken(headers({ 'x-api-key': ` ${TOKEN} ` }))).toBe(` ${TOKEN} `);
  });

  it('rejects a bare scheme with no credential', () => {
    expect(extractPresentedToken(headers({ authorization: 'Bearer' }))).toBeNull();
    expect(extractPresentedToken(headers({ authorization: 'Bearer ' }))).toBeNull();
  });

  it('returns null when neither header is present', () => {
    expect(extractPresentedToken(headers({}))).toBeNull();
  });
});

describe('tokensMatch', () => {
  it('is exact — no prefix, suffix or substring match', () => {
    expect(tokensMatch(TOKEN, TOKEN)).toBe(true);
    expect(tokensMatch(TOKEN.slice(0, -1), TOKEN)).toBe(false);
    expect(tokensMatch(`${TOKEN}x`, TOKEN)).toBe(false);
    expect(tokensMatch(`x${TOKEN}`, TOKEN)).toBe(false);
    expect(tokensMatch(`${TOKEN}, ${TOKEN}`, TOKEN)).toBe(false);
    expect(tokensMatch('', TOKEN)).toBe(false);
  });

  it('compares bytes, so a same-glyph different-encoding token does not collide', () => {
    // U+00E9 is 2 UTF-8 bytes; "e" + U+0301 is 3. Same rendered glyph.
    // The length guard must reject before `timingSafeEqual`, which throws
    // outright on a size mismatch.
    expect(tokensMatch('\u00e9', 'e\u0301')).toBe(false);
    expect(tokensMatch('\u00e9', '\u00e9')).toBe(true);
  });

  it('never matches an empty expected token', () => {
    // `timingSafeEqual` on two zero-length buffers returns TRUE, so a server
    // misconfigured with an empty secret would have authenticated anyone who
    // sent an empty `x-api-key`. The empty string is not a credential in
    // either position.
    expect(tokensMatch('', '')).toBe(false);
    expect(tokensMatch(TOKEN, '')).toBe(false);
    expect(isAuthorized({ headers: { 'x-api-key': '' } } as unknown as IncomingMessage, '')).toBe(false);
  });
});

/**
 * `--auth-token "$VAR"` with `VAR` unset arrives here as `''`, and `''` used to
 * mean "auth is configured" to the bind guard and "everyone is authorised" to
 * the comparator: the two halves of a wildcard bind published under a
 * credential anyone can guess.
 */
describe('an empty explicit auth token is a configuration error', () => {
  it('is rejected by resolveAuthToken rather than resolving to a value', () => {
    expect(() => resolveAuthToken('')).toThrow(/empty string/);
    // Not merely "no auth": a token that silently vanishes would hand back the
    // unauthenticated, wildcard-CORS server the operator was trying not to
    // start. Both other inputs still resolve as before.
    expect(resolveAuthToken(TOKEN)).toBe(TOKEN);
    expect(resolveAuthToken(undefined)).toBeUndefined();
  });

  it('does not become "auth configured" for an env var either', () => {
    process.env.MLX_SERVER_AUTH_TOKEN = '';
    try {
      expect(resolveAuthToken(undefined)).toBeUndefined();
    } finally {
      delete process.env.MLX_SERVER_AUTH_TOKEN;
    }
  });

  it('fails createServer before it binds', async () => {
    await expect(createServer({ port: 0, host: '127.0.0.1', disableStore: true, authToken: '' })).rejects.toThrow(
      /empty string/,
    );
  });
});

describe('bearer auth', () => {
  it('accepts a matching `authorization: Bearer <token>`', async () => {
    const base = await start({ authToken: TOKEN });
    const res = await fetch(`${base}/v1/models`, { headers: { authorization: `Bearer ${TOKEN}` } });
    expect(res.status).toBe(200);
  });

  it('accepts the scheme case-insensitively', async () => {
    // RFC 7235 makes the auth-scheme token case-insensitive; Anthropic and
    // OpenAI SDKs both send `Bearer`, but proxies have been seen to lowercase.
    const base = await start({ authToken: TOKEN });
    for (const scheme of ['bearer', 'BEARER', 'BeArEr']) {
      const res = await fetch(`${base}/v1/models`, { headers: { authorization: `${scheme} ${TOKEN}` } });
      expect(res.status, scheme).toBe(200);
    }
  });

  it('accepts a matching `x-api-key`', async () => {
    const base = await start({ authToken: TOKEN });
    const res = await fetch(`${base}/v1/models`, { headers: { 'x-api-key': TOKEN } });
    expect(res.status).toBe(200);
  });

  it('prefers `x-api-key` over `authorization` when both are present', async () => {
    // Anthropic clients send `x-api-key`. Checking it first means a stale
    // `authorization` header injected by a proxy cannot shadow the real key.
    const base = await start({ authToken: TOKEN });
    const ok = await fetch(`${base}/v1/models`, {
      headers: { 'x-api-key': TOKEN, authorization: 'Bearer totally-wrong' },
    });
    expect(ok.status).toBe(200);

    const rejected = await fetch(`${base}/v1/models`, {
      headers: { 'x-api-key': 'totally-wrong', authorization: `Bearer ${TOKEN}` },
    });
    expect(rejected.status).toBe(401);
  });

  it('rejects a wrong token with 401 + WWW-Authenticate: Bearer', async () => {
    const base = await start({ authToken: TOKEN });
    const res = await fetch(`${base}/v1/models`, { headers: { authorization: 'Bearer nope' } });
    expect(res.status).toBe(401);
    expect(res.headers.get('www-authenticate')).toMatch(/^Bearer/);
  });

  it('rejects an absent credential with 401 + WWW-Authenticate: Bearer', async () => {
    const base = await start({ authToken: TOKEN });
    const res = await fetch(`${base}/v1/models`);
    expect(res.status).toBe(401);
    expect(res.headers.get('www-authenticate')).toMatch(/^Bearer/);
  });

  it('rejects a token that is a prefix of the configured one', async () => {
    // Guards against a `startsWith` / truncated-compare regression.
    const base = await start({ authToken: TOKEN });
    const res = await fetch(`${base}/v1/models`, { headers: { authorization: `Bearer ${TOKEN.slice(0, -1)}` } });
    expect(res.status).toBe(401);
  });

  it('rejects a token that merely contains the configured one', async () => {
    const base = await start({ authToken: TOKEN });
    const res = await fetch(`${base}/v1/models`, { headers: { authorization: `Bearer ${TOKEN}extra` } });
    expect(res.status).toBe(401);
  });

  it('rejects `authorization` with no scheme', async () => {
    const base = await start({ authToken: TOKEN });
    const res = await fetch(`${base}/v1/models`, { headers: { authorization: TOKEN } });
    expect(res.status).toBe(401);
  });

  it('rejects a non-Bearer scheme carrying the right secret', async () => {
    const base = await start({ authToken: TOKEN });
    const res = await fetch(`${base}/v1/models`, { headers: { authorization: `Basic ${TOKEN}` } });
    expect(res.status).toBe(401);
  });

  it('rejects duplicate credential headers', async () => {
    // Measured, not assumed: `node:http` joins repeated `x-api-key` values
    // with ", " into a single STRING (it does not produce a string[] here).
    // The joined value must fail the compare rather than being split or
    // prefix-matched — otherwise a caller could append a valid token to a
    // header a front proxy already inspected.
    const base = await start({ authToken: TOKEN });
    const res = await fetch(`${base}/v1/models`, {
      headers: [
        ['x-api-key', TOKEN],
        ['x-api-key', TOKEN],
      ],
    });
    expect(res.status).toBe(401);
  });

  it('gates POST endpoints too, without reading the body', async () => {
    const base = await start({ authToken: TOKEN });
    const res = await fetch(`${base}/v1/messages`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'nope', messages: [{ role: 'user', content: 'hi' }], max_tokens: 8 }),
    });
    expect(res.status).toBe(401);
  });

  it('gates unknown routes (no 404 oracle for unauthenticated callers)', async () => {
    const base = await start({ authToken: TOKEN });
    const res = await fetch(`${base}/v1/does-not-exist`);
    expect(res.status).toBe(401);
  });

  it('falls back to MLX_SERVER_AUTH_TOKEN when no config value is given', async () => {
    process.env.MLX_SERVER_AUTH_TOKEN = 'env-token-value';
    try {
      const base = await start({});
      expect((await fetch(`${base}/v1/models`)).status).toBe(401);
      expect((await fetch(`${base}/v1/models`, { headers: { 'x-api-key': 'env-token-value' } })).status).toBe(200);
    } finally {
      delete process.env.MLX_SERVER_AUTH_TOKEN;
    }
  });

  it('lets an explicit config token win over the env var', async () => {
    process.env.MLX_SERVER_AUTH_TOKEN = 'env-token-value';
    try {
      const base = await start({ authToken: TOKEN });
      expect((await fetch(`${base}/v1/models`, { headers: { 'x-api-key': 'env-token-value' } })).status).toBe(401);
      expect((await fetch(`${base}/v1/models`, { headers: { 'x-api-key': TOKEN } })).status).toBe(200);
    } finally {
      delete process.env.MLX_SERVER_AUTH_TOKEN;
    }
  });
});

describe('authToken: undefined preserves the pre-auth behaviour byte-for-byte', () => {
  // Hard requirement: adding the auth code path must be a no-op for every
  // existing deployment. Anything that changes status, headers or body when
  // no token is configured is a regression.

  it('serves every route with no credentials at all', async () => {
    const base = await start({ authToken: undefined });
    expect((await fetch(`${base}/v1/models`)).status).toBe(200);
    expect((await fetch(`${base}/health`)).status).toBe(200);
    expect((await fetch(`${base}/v1/health`)).status).toBe(200);
    expect((await fetch(`${base}/`, { method: 'HEAD' })).status).toBe(200);
    expect((await fetch(`${base}/v1/nope`)).status).toBe(404);
  });

  it('ignores credentials that would be wrong if auth were enabled', async () => {
    const base = await start({});
    const res = await fetch(`${base}/v1/models`, {
      headers: { authorization: 'Bearer garbage', 'x-api-key': 'also-garbage' },
    });
    expect(res.status).toBe(200);
  });

  it('never emits WWW-Authenticate', async () => {
    const base = await start({});
    const res = await fetch(`${base}/v1/models`);
    expect(res.headers.get('www-authenticate')).toBeNull();
  });

  it('keeps CORS on by default', async () => {
    const base = await start({});
    const res = await fetch(`${base}/v1/models`);
    expect(res.headers.get('access-control-allow-origin')).toBe('*');
  });

  it('keeps the OPTIONS preflight 204 short-circuit', async () => {
    const base = await start({});
    const res = await fetch(`${base}/v1/messages`, { method: 'OPTIONS' });
    expect(res.status).toBe(204);
  });

  it('serves the unchanged `/health` body shape', async () => {
    const base = await start({});
    const body = (await (await fetch(`${base}/health`)).json()) as Record<string, unknown>;
    expect(body.status).toBe('ok');
  });
});

describe('CORS defaulting under auth', () => {
  it('defaults CORS off once a token is configured', async () => {
    // Today every response carries `Access-Control-Allow-Origin: *`. On a
    // token-protected server that would invite any web origin to spend a
    // leaked token from the browser, so the default flips.
    const base = await start({ authToken: TOKEN });
    const res = await fetch(`${base}/v1/models`, { headers: { 'x-api-key': TOKEN } });
    expect(res.headers.get('access-control-allow-origin')).toBeNull();
  });

  it('drops the OPTIONS 204 short-circuit when CORS defaults off', async () => {
    const base = await start({ authToken: TOKEN });
    const res = await fetch(`${base}/v1/messages`, { method: 'OPTIONS' });
    expect(res.status).not.toBe(204);
  });

  it('honours an explicit `cors: true` alongside a token', async () => {
    const base = await start({ authToken: TOKEN, cors: true });
    const res = await fetch(`${base}/v1/models`, { headers: { 'x-api-key': TOKEN } });
    expect(res.headers.get('access-control-allow-origin')).toBe('*');
  });

  it('honours an explicit `cors: false` without a token', async () => {
    const base = await start({ cors: false });
    const res = await fetch(`${base}/v1/models`);
    expect(res.headers.get('access-control-allow-origin')).toBeNull();
  });
});

describe('GET /health disclosure split', () => {
  it('answers unauthenticated with only status/uptimeMs/pid', async () => {
    // The supervisor must be able to poll liveness before it holds a token,
    // but `models.resident` is user data and must stay behind auth.
    const base = await start({ authToken: TOKEN });
    const res = await fetch(`${base}/health`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as Record<string, unknown>;
    expect(Object.keys(body).sort()).toEqual(['pid', 'status', 'uptimeMs']);
    expect(body.status).toBe('ok');
    expect(typeof body.pid).toBe('number');
    expect(typeof body.uptimeMs).toBe('number');
  });

  it('answers unauthenticated on /v1/health too', async () => {
    const base = await start({ authToken: TOKEN });
    const body = (await (await fetch(`${base}/v1/health`)).json()) as Record<string, unknown>;
    expect(Object.keys(body).sort()).toEqual(['pid', 'status', 'uptimeMs']);
  });

  it('answers authenticated with the full ServerHealth body', async () => {
    const base = await start({ authToken: TOKEN });
    const res = await fetch(`${base}/health`, { headers: { 'x-api-key': TOKEN } });
    expect(res.status).toBe(200);
    const body = (await res.json()) as Record<string, unknown>;
    expect(body.status).toBe('ok');
    expect(body.models).toEqual({ resident: [], count: 0 });
    expect(body.work).toMatchObject({ inFlight: 0, writerActive: false, waitingWriters: 0 });
    expect(body.queue).toMatchObject({ saturated: false });
    expect(body.lastLoad).toBeNull();
  });

  it('does not leak resident model names to an unauthenticated poller', async () => {
    const base = await start({ authToken: TOKEN });
    const instance = servers[servers.length - 1]!;
    instance.registry.register('private-model-name', { chatStreamSessionStart(): void {} } as never);

    const anon = await (await fetch(`${base}/health`)).text();
    expect(anon).not.toContain('private-model-name');

    const authed = await (await fetch(`${base}/health`, { headers: { 'x-api-key': TOKEN } })).text();
    expect(authed).toContain('private-model-name');
  });

  it('rejects a WRONG token on /health rather than silently downgrading', async () => {
    // A supervisor that mistypes its token should learn that, not get a
    // 200 it will read as "everything is fine".
    const base = await start({ authToken: TOKEN });
    const res = await fetch(`${base}/health`, { headers: { 'x-api-key': 'wrong' } });
    expect(res.status).toBe(401);
    expect(res.headers.get('www-authenticate')).toMatch(/^Bearer/);
  });

  it('serves the full body on an unauthenticated server', async () => {
    const base = await start({});
    const body = (await (await fetch(`${base}/health`)).json()) as Record<string, unknown>;
    expect(body.status).toBe('ok');
    expect(body.models).toEqual({ resident: [], count: 0 });
  });
});

describe('`/` liveness probe is reachable without a token', () => {
  // `/` is a contentless stub — the router answers a constant
  // `{ service: 'mlx-node' }` and reads no state. Gating it would 401 every
  // client's startup liveness check for no security gain. Claude Code sends
  // `x-api-key` so it would pass either way; this is for everything else.

  it('answers GET / unauthenticated when a token is configured', async () => {
    const base = await start({ authToken: TOKEN });
    const res = await fetch(`${base}/`);
    expect(res.status).toBe(200);
    expect((await res.json()) as Record<string, unknown>).toEqual({ service: 'mlx-node' });
  });

  it('answers HEAD / unauthenticated — the probe Claude Code actually issues', async () => {
    const base = await start({ authToken: TOKEN });
    expect((await fetch(`${base}/`, { method: 'HEAD' })).status).toBe(200);
  });

  it('rejects a WRONG token on / rather than silently downgrading', async () => {
    // Same rule as /health: no credential degrades, a bad credential is a 401
    // so the caller learns about the typo.
    const base = await start({ authToken: TOKEN });
    const res = await fetch(`${base}/`, { headers: { 'x-api-key': 'wrong' } });
    expect(res.status).toBe(401);
    expect(res.headers.get('www-authenticate')).toMatch(/^Bearer/);
  });

  it('still gates every non-carved-out route', async () => {
    // Guards against the carve-out being widened by accident.
    const base = await start({ authToken: TOKEN });
    expect((await fetch(`${base}/v1/models`)).status).toBe(401);
    expect((await fetch(`${base}/v1/nope`)).status).toBe(401);
  });
});

/**
 * The auth gate has to classify the request path before it can apply the
 * `/health` carve-out, and it used to do that by building a `URL` whose base
 * came from the `Host` header. `Host` is attacker-controlled text on every
 * request: `Host: [` makes `new URL()` throw `ERR_INVALID_URL` out of an async
 * request listener, whose promise `http.createServer` discards — an unhandled
 * rejection, which under Node's default `--unhandled-rejections=throw` ends the
 * process. Any client that could open the socket could kill inference with one
 * request, and only on a server with auth ENABLED, because that is the only
 * branch that parsed the path.
 *
 * Driven over a raw socket: `fetch` (and every other HTTP client) refuses to
 * send a syntactically invalid `Host`, so the bug is unreachable from one.
 */
describe('a malformed Host header cannot take the server down', () => {
  /** One request, written verbatim, answered with the status line. */
  async function rawRequest(base: string, lines: string[]): Promise<string> {
    const { port } = new URL(base);
    return await new Promise<string>((resolve, reject) => {
      const socket = netConnect(Number(port), '127.0.0.1', () => {
        socket.write(`${lines.join('\r\n')}\r\n\r\n`);
      });
      let received = '';
      socket.on('data', (chunk: Buffer) => (received += chunk.toString('utf-8')));
      socket.on('close', () => resolve(received.split('\r\n')[0] ?? ''));
      socket.on('error', reject);
      // A hang IS the failure — pre-fix the listener rejected and nothing was
      // ever written back — so it must not wait for the suite timeout.
      setTimeout(() => {
        socket.destroy();
        resolve(received.split('\r\n')[0] ?? '<no response>');
      }, 4_000).unref();
    });
  }

  for (const host of ['[', '[::1', 'a b', '%%']) {
    it(`answers 401 instead of throwing on \`Host: ${host}\``, async () => {
      const base = await start({ authToken: TOKEN });
      expect(await rawRequest(base, ['GET /v1/models HTTP/1.1', `Host: ${host}`, 'Connection: close'])).toContain(
        '401',
      );
    });
  }

  it('still serves the next request afterwards', async () => {
    const base = await start({ authToken: TOKEN });
    await rawRequest(base, ['GET /v1/models HTTP/1.1', 'Host: [', 'Connection: close']);
    const res = await fetch(`${base}/v1/models`, { headers: { 'x-api-key': TOKEN } });
    expect(res.status).toBe(200);
  });

  it('keeps the /health carve-out working when Host is unparseable', async () => {
    // The carve-out is decided from the same pathname, so a fallback that
    // returned a constant would have quietly 401'd every liveness probe.
    const base = await start({ authToken: TOKEN });
    expect(await rawRequest(base, ['GET /health HTTP/1.1', 'Host: [', 'Connection: close'])).toContain('200');
  });

  it('routes normally on an unauthenticated server too', async () => {
    // `routeRequest` parsed the path the same way; there it was inside the
    // try/catch, so the symptom was a 500 rather than a crash. Same fix.
    const base = await start({});
    expect(await rawRequest(base, ['GET /v1/models HTTP/1.1', 'Host: [', 'Connection: close'])).toContain('200');
    expect(await rawRequest(base, ['GET /v1/nope HTTP/1.1', 'Host: [', 'Connection: close'])).toContain('404');
  });
});

describe('ServerInstance.health()', () => {
  it('returns the same shape as the authenticated endpoint', async () => {
    const base = await start({});
    const instance = servers[servers.length - 1]!;
    instance.registry.register('in-proc', { chatStreamSessionStart(): void {} } as never);

    const overHttp = (await (await fetch(`${base}/health`)).json()) as Record<string, unknown>;
    const inProc = instance.health();
    expect(Object.keys(inProc).sort()).toEqual(Object.keys(overHttp).sort());
    expect(inProc.models.resident).toEqual(['in-proc']);
  });
});
