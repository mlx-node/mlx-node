/**
 * Fake provider APIs on 127.0.0.1: each records every request (method, path, query, headers, body) and answers with
 * a scripted status, JSON or raw body, an optional delay, or a dropped socket. Unit tests and the dev harness point
 * the adapters and the key checks here; nothing reaches a real provider.
 */

import { createServer, type IncomingMessage, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';

export interface FakeRequest {
  method: string;
  path: string;
  query: Array<[string, string]>;
  headers: Record<string, string>;
  body: string;
}

export interface FakeAnswer {
  status?: number;
  json?: unknown;
  /** A raw body, sent instead of `json`. */
  body?: string | Buffer;
  delayMs?: number;
  headers?: Record<string, string>;
  /** Close the socket with no answer. */
  drop?: boolean;
}

export type FakeScript = (request: FakeRequest) => FakeAnswer;

export interface FakeProvider {
  baseUrl: string;
  requests: FakeRequest[];
  close(): Promise<void>;
}

function headersOf(message: IncomingMessage): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [name, value] of Object.entries(message.headers)) {
    if (typeof value === 'string') out[name] = value;
    else if (Array.isArray(value)) out[name] = value.join(', ');
  }
  return out;
}

export async function startFakeProvider(script: FakeScript | FakeAnswer): Promise<FakeProvider> {
  const requests: FakeRequest[] = [];
  const answerOf = typeof script === 'function' ? script : (): FakeAnswer => script;
  const server: Server = createServer((message, response) => {
    const chunks: Buffer[] = [];
    message.on('data', (chunk: Buffer) => chunks.push(chunk));
    message.on('end', () => {
      const url = new URL(message.url ?? '/', 'http://127.0.0.1');
      const request: FakeRequest = {
        method: message.method ?? 'GET',
        path: url.pathname,
        query: [...url.searchParams.entries()],
        headers: headersOf(message),
        body: Buffer.concat(chunks).toString('utf8'),
      };
      requests.push(request);
      const answer = answerOf(request);
      const send = (): void => {
        if (answer.drop === true) {
          message.socket.destroy();
          return;
        }
        const body = answer.body ?? JSON.stringify(answer.json ?? {});
        response.writeHead(answer.status ?? 200, { 'Content-Type': 'application/json', ...answer.headers });
        response.end(body);
      };
      if (answer.delayMs === undefined) send();
      else setTimeout(send, answer.delayMs);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address() as AddressInfo;
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    requests,
    close: () =>
      new Promise<void>((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}

/** The key a request carried, from whichever auth header its provider uses. */
export function keyOf(request: FakeRequest): string {
  const bearer = request.headers.authorization ?? '';
  if (bearer.startsWith('Bearer ')) return bearer.slice('Bearer '.length);
  return request.headers['x-api-key'] ?? request.headers['x-subscription-token'] ?? '';
}

/** A successful key check of each provider, as its API answers it. */
export const CHECK_OK: Readonly<Record<'exa' | 'deepseek' | 'brave' | 'firecrawl', unknown>> = {
  exa: { object: 'team', id: 'team-1', name: 'Team' },
  deepseek: { is_available: true, balance_infos: [{ currency: 'USD', total_balance: '12.50' }] },
  brave: { web: { results: [] } },
  firecrawl: {
    success: true,
    data: { remainingCredits: 742, planCredits: 1000, billingPeriodEnd: '2026-12-01T00:00:00Z' },
  },
};

export interface HarnessProviders {
  baseUrls: Record<'exa' | 'deepseek' | 'brave' | 'firecrawl', string>;
  /** A fake Firecrawl on this Mac, as `FIRECRAWL_CLOUD=off firecrawl-local serve` answers: type it into the This Mac field. */
  localUrl: string;
  close(): Promise<void>;
}

/**
 * The dev harness's providers: a key that starts with `bad` is rejected, one that starts with `nocredit` has no
 * credit, one that starts with `slow` answers after 12 s (the check gives up: "not checked"), any other is valid.
 * Beside them runs a fake Firecrawl on this Mac whose Cloud step answers `localCloud` (default off, which passes the
 * check); any other loopback port reads "not running".
 */
export async function startHarnessProviders(localCloud: FakeLocalCloud = 'off'): Promise<HarnessProviders> {
  const ids = ['exa', 'deepseek', 'brave', 'firecrawl'] as const;
  const servers = await Promise.all(
    ids.map((id) =>
      startFakeProvider((request) => {
        const key = keyOf(request);
        if (key.startsWith('bad')) return { status: 401, json: { error: 'unauthorized' } };
        if (key.startsWith('nocredit')) return { status: 402, json: { error: 'no credit' } };
        if (key.startsWith('slow')) return { delayMs: 12_000, json: CHECK_OK[id] };
        return { json: CHECK_OK[id] };
      }),
    ),
  );
  const baseUrls = {
    exa: servers[0]!.baseUrl,
    deepseek: servers[1]!.baseUrl,
    brave: servers[2]!.baseUrl,
    firecrawl: servers[3]!.baseUrl,
  };
  const local = await startFakeProvider(fakeLocalScript(fakeLocalAnswers(localCloud)));
  return {
    baseUrls,
    localUrl: local.baseUrl,
    close: async () => {
      await Promise.all([...servers, local].map((s) => s.close()));
    },
  };
}

/** What the fake Firecrawl on this Mac answers; a test may change a field between calls. */
export interface FakeLocalAnswers {
  health: FakeAnswer;
  /** The answer to a scrape of a loopback target (the check's probe). */
  probe: FakeAnswer;
  /** The answer to a scrape of any other target. */
  page: FakeAnswer;
  /** The answer to `GET /v2/team/credit-usage` (the check's Cloud read). */
  cloud: FakeAnswer;
}

/**
 * The Cloud step of a fake Firecrawl on this Mac. `off`: started with `FIRECRAWL_CLOUD=off` (or `search`), 501
 * `CLOUD_ONLY`. `fallback`: `FIRECRAWL_CLOUD` unset, the server's default, 403 `CROSS_SITE_REQUEST`. `unreadable`: an
 * answer the check does not know.
 */
export type FakeLocalCloud = 'off' | 'fallback' | 'unreadable';

export function fakeLocalCloud(mode: FakeLocalCloud): FakeAnswer {
  if (mode === 'off') return { status: 501, json: { success: false, code: 'CLOUD_ONLY', error: 'cloud is off' } };
  if (mode === 'fallback') {
    return { status: 403, json: { success: false, code: 'CROSS_SITE_REQUEST', error: 'from a web page' } };
  }
  return { status: 404, json: { success: false, code: 'NOT_FOUND' } };
}

/** `firecrawl-local serve` as its source answers: private targets refused, the Cloud step as `cloud` says. */
export function fakeLocalAnswers(cloud: FakeLocalCloud = 'off'): FakeLocalAnswers {
  return {
    health: { json: { status: 'ok', engine: 'firecrawl-local', version: '0.1.1' } },
    probe: { status: 403, json: { success: false, code: 'BLOCKED_TARGET', error: 'blocked' } },
    page: {
      json: {
        success: true,
        data: { markdown: '# Local page\nBody', metadata: { title: 'Local page', statusCode: 200 }, source: 'local' },
      },
    },
    cloud: fakeLocalCloud(cloud),
  };
}

const LOOPBACK_HOSTS: readonly string[] = ['127.0.0.1', 'localhost', '[::1]'];

/** Routes a request to the fake Firecrawl on this Mac to the matching field of `answers`, read at request time. */
export function fakeLocalScript(answers: FakeLocalAnswers): FakeScript {
  return (request) => {
    if (request.method === 'GET' && request.path === '/health') return answers.health;
    if (request.method === 'GET' && request.path === '/v2/team/credit-usage') return answers.cloud;
    if (request.method === 'POST' && request.path === '/v2/scrape') {
      let target = '';
      try {
        target = new URL((JSON.parse(request.body) as { url: string }).url).hostname;
      } catch {
        return { status: 400, json: { success: false, code: 'BAD_REQUEST' } };
      }
      return LOOPBACK_HOSTS.includes(target) ? answers.probe : answers.page;
    }
    return { status: 404, json: { success: false } };
  };
}
