/**
 * Firecrawl on this Mac: the loopback-only address rule and its saved form, and the check against a fake
 * `firecrawl-local serve` on 127.0.0.1 (health, the private-target probe, the Cloud setting). No request carries an
 * Authorization header, and every request goes to the saved address only.
 */

import { afterEach, describe, expect, it } from 'vite-plus/test';

import {
  checkLocal,
  LOCAL_DEFAULT_URL,
  type LocalCheckOptions,
  localOrigin,
  parseLocalAddress,
} from '../src/chat/web/local.js';
import {
  fakeLocalAnswers,
  fakeLocalCloud,
  fakeLocalScript,
  type FakeLocalAnswers,
  type FakeProvider,
  startFakeProvider,
  startHarnessProviders,
} from './helpers/fake-providers.js';

const servers: FakeProvider[] = [];

afterEach(async () => {
  await Promise.all(servers.splice(0).map((s) => s.close()));
});

async function fakeLocal(
  over: Partial<FakeLocalAnswers> = {},
): Promise<{ server: FakeProvider; answers: FakeLocalAnswers }> {
  const answers = { ...fakeLocalAnswers(), ...over };
  const server = await startFakeProvider(fakeLocalScript(answers));
  servers.push(server);
  return { server, answers };
}

/** A real fetch that records every URL it was asked for. */
function recordingFetch(): { fetch: typeof fetch; urls: string[] } {
  const urls: string[] = [];
  return {
    urls,
    fetch: (input, init) => {
      urls.push(new URL(input as string | URL).href);
      return fetch(input, init);
    },
  };
}

function opts(f: typeof fetch): LocalCheckOptions {
  return { fetch: f };
}

describe('parseLocalAddress', () => {
  it('accepts a loopback address with a port and gives its saved form', () => {
    expect(parseLocalAddress('http://127.0.0.1:3002')).toBe('http://127.0.0.1:3002');
    expect(parseLocalAddress('http://127.0.0.1:3002/')).toBe('http://127.0.0.1:3002');
    expect(parseLocalAddress(' http://localhost:3002 ')).toBe('http://127.0.0.1:3002');
    expect(parseLocalAddress('HTTP://LocalHost:3002')).toBe('http://127.0.0.1:3002');
    expect(parseLocalAddress('http://[::1]:4000')).toBe('http://[::1]:4000');
    expect(parseLocalAddress('http://127.0.0.1:1024')).toBe('http://127.0.0.1:1024');
    expect(parseLocalAddress('http://127.0.0.1:65535')).toBe('http://127.0.0.1:65535');
    expect(parseLocalAddress(LOCAL_DEFAULT_URL)).toBe(LOCAL_DEFAULT_URL);
  });

  it('refuses https, other hosts, a missing or low port, user info, a path, a query and a hash', () => {
    const refused = [
      'https://127.0.0.1:3002',
      'http://127.0.0.1',
      'http://127.0.0.1:',
      'http://127.0.0.1:80',
      'http://127.0.0.1:1023',
      'http://127.0.0.1:70000',
      'http://192.168.1.5:3002',
      'http://10.0.0.2:3002',
      'http://127.0.0.2:3002',
      'http://0.0.0.0:3002',
      'http://[::ffff:127.0.0.1]:3002',
      'http://example.com:3002',
      'http://localhost.evil.example:3002',
      'http://localhost.:3002',
      'http://localhost:3002@evil.example',
      'http://user@127.0.0.1:3002',
      'http://user:pw@127.0.0.1:3002',
      'http://127.0.0.1:3002/v2',
      'http://127.0.0.1:3002/.',
      'http://127.0.0.1:3002//',
      'http://127.0.0.1:3002?x=1',
      'http://127.0.0.1:3002/?',
      'http://127.0.0.1:3002#a',
      'http://127.0.0.1:30 02',
      'http://127.0.0.1:3002\\',
      'http:127.0.0.1:3002',
      'http:///127.0.0.1:3002',
      'localhost:3002',
      '127.0.0.1:3002',
      '',
      '   ',
    ];
    for (const text of refused) expect([text, parseLocalAddress(text)]).toEqual([text, null]);
  });
});

describe('localOrigin', () => {
  it('accepts only text that already is a saved form', () => {
    expect(localOrigin('http://127.0.0.1:3002')).toBe('http://127.0.0.1:3002');
    expect(localOrigin('http://[::1]:3002')).toBe('http://[::1]:3002');
    expect(localOrigin('http://localhost:3002')).toBeNull();
    expect(localOrigin('http://127.0.0.1:3002/')).toBeNull();
    expect(localOrigin(' http://127.0.0.1:3002')).toBeNull();
    expect(localOrigin('http://192.168.1.5:3002')).toBeNull();
    expect(localOrigin('')).toBeNull();
  });
});

describe('checkLocal', () => {
  it('Firecrawl Local that refuses private targets and has its Cloud off is ok, with its version', async () => {
    const { server } = await fakeLocal();
    const rec = recordingFetch();
    expect(await checkLocal(server.baseUrl, opts(rec.fetch))).toEqual({ check: 'ok', version: '0.1.1', cloud: 'off' });
    expect(server.requests.map((r) => [r.method, r.path])).toEqual([
      ['GET', '/health'],
      ['POST', '/v2/scrape'],
      ['GET', '/v2/team/credit-usage'],
    ]);
    expect(JSON.parse(server.requests[1]!.body)).toEqual({
      url: `${server.baseUrl}/health`,
      formats: ['markdown'],
      timeout: 3000,
    });
    expect(server.requests[2]!.headers['sec-fetch-site']).toBe('cross-site');
    expect(server.requests[2]!.body).toBe('');
    for (const request of server.requests) expect(request.headers.authorization).toBeUndefined();
    for (const url of rec.urls) expect(new URL(url).origin).toBe(server.baseUrl);
  });

  it('a Cloud that takes pages the server cannot load, or a Cloud setting it cannot read, is cloud-on', async () => {
    const fallback = await fakeLocal({ cloud: fakeLocalCloud('fallback') });
    expect(await checkLocal(fallback.server.baseUrl, opts(fetch))).toEqual({
      check: 'cloud-on',
      version: '0.1.1',
      cloud: 'fallback',
    });
    for (const cloud of [
      fakeLocalCloud('unreadable'),
      { status: 200, json: { success: true } },
      { status: 501, json: { code: 'OTHER' } },
      { status: 501, body: 'not json' },
      { ...fakeLocalCloud('off'), delayMs: 300 },
      { drop: true },
    ]) {
      const other = await fakeLocal({ cloud });
      expect([cloud, await checkLocal(other.server.baseUrl, { fetch, cloudMs: 50 })]).toEqual([
        cloud,
        { check: 'cloud-on', version: '0.1.1', cloud: null },
      ]);
    }
  });

  it('nothing listening is not-running', async () => {
    const { server } = await fakeLocal();
    const base = server.baseUrl;
    await servers.pop()!.close();
    expect(await checkLocal(base, opts(fetch))).toEqual({ check: 'not-running', version: null, cloud: null });
  });

  it('a health answer that is not the firecrawl-local engine is not-firecrawl, and the probe is never sent', async () => {
    const healths = [
      { json: { status: 'ok', engine: 'firecrawl', version: '2.11.0' } },
      { json: { status: 'ok' } },
      { body: '<html>ok</html>' },
      { status: 404, json: { engine: 'firecrawl-local' } },
      { status: 302, headers: { Location: 'http://127.0.0.1:1/health' }, body: '' },
    ];
    for (const health of healths) {
      const { server } = await fakeLocal({ health });
      expect([health, await checkLocal(server.baseUrl, opts(fetch))]).toEqual([
        health,
        { check: 'not-firecrawl', version: null, cloud: null },
      ]);
      expect(server.requests.map((r) => r.path)).toEqual(['/health']);
    }
  });

  it('a health 401 is token', async () => {
    const { server } = await fakeLocal({ health: { status: 401, json: { code: 'UNAUTHORIZED' } } });
    expect(await checkLocal(server.baseUrl, opts(fetch))).toEqual({ check: 'token', version: null, cloud: null });
  });

  it('a server that reads its own loopback address is open, and the Cloud setting is not read', async () => {
    const probes = [
      { json: { success: true, data: { markdown: 'ok', metadata: { statusCode: 200 } } } },
      { status: 502, json: { success: false, code: 'FETCH_FAILED' } },
      { status: 403, json: { success: false, code: 'SOMETHING_ELSE' } },
      { status: 403, json: { success: false, error: { code: 'BLOCKED_TARGET' } } },
      { status: 500, body: 'not json' },
    ];
    for (const probe of probes) {
      const { server } = await fakeLocal({ probe });
      expect([probe, await checkLocal(server.baseUrl, opts(fetch))]).toEqual([
        probe,
        { check: 'open', version: '0.1.1', cloud: null },
      ]);
      expect(server.requests.map((r) => r.path)).toEqual(['/health', '/v2/scrape']);
    }
  });

  it('a probe 401 is token; a probe past its time is not-running', async () => {
    const token = await fakeLocal({ probe: { status: 401, json: { code: 'UNAUTHORIZED' } } });
    expect(await checkLocal(token.server.baseUrl, opts(fetch))).toEqual({
      check: 'token',
      version: '0.1.1',
      cloud: null,
    });
    const slow = await fakeLocal({ probe: { ...fakeLocalAnswers().probe, delayMs: 300 } });
    expect(await checkLocal(slow.server.baseUrl, { fetch, probeMs: 50 })).toEqual({
      check: 'not-running',
      version: '0.1.1',
      cloud: null,
    });
  });

  it('keeps a version only when it is digits and dots, at most 16 characters', async () => {
    for (const [version, kept] of [
      ['0.1.1-beta', null],
      ['1.2.3.4.5.6.7.8.9', null],
      [12, null],
      ['10.20.30', '10.20.30'],
    ] as const) {
      const { server } = await fakeLocal({ health: { json: { status: 'ok', engine: 'firecrawl-local', version } } });
      expect((await checkLocal(server.baseUrl, opts(fetch))).version).toBe(kept);
    }
  });

  it('an address not in the saved form sends nothing', async () => {
    const { server } = await fakeLocal();
    const port = new URL(server.baseUrl).port;
    for (const text of [`http://localhost:${port}`, `${server.baseUrl}/`, 'http://192.168.1.5:3002']) {
      expect(await checkLocal(text, opts(fetch))).toEqual({ check: 'not-running', version: null, cloud: null });
    }
    expect(server.requests).toEqual([]);
  });
});

describe('the dev harness', () => {
  it('runs a fake Firecrawl on this Mac on a loopback port that passes the check, its Cloud off', async () => {
    const harness = await startHarnessProviders();
    try {
      expect(parseLocalAddress(harness.localUrl)).toBe(harness.localUrl);
      const result = await checkLocal(harness.localUrl, { fetch: (input, init) => fetch(input, init) });
      expect(result).toEqual({ check: 'ok', version: '0.1.1', cloud: 'off' });
    } finally {
      await harness.close();
    }
  });

  it('switches its Cloud step to fallback or to an answer the check cannot read', async () => {
    for (const [mode, cloud] of [
      ['fallback', 'fallback'],
      ['unreadable', null],
    ] as const) {
      const harness = await startHarnessProviders(mode);
      try {
        expect(await checkLocal(harness.localUrl, { fetch: (input, init) => fetch(input, init) })).toEqual({
          check: 'cloud-on',
          version: '0.1.1',
          cloud,
        });
      } finally {
        await harness.close();
      }
    }
  });
});
