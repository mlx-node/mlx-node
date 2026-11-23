/**
 * The `mlx_web` server as the agent runs it (a child process on stdio): the MCP methods it answers, the exact tool
 * schemas, `not-configured` with no provider, sloppy and wrong arguments, malformed and oversized frames, and a
 * stderr that only ever holds the fixed lines. Then the tool layer with a fake adapter, and the source rules: no
 * `console`, only `node:` and own imports.
 */

import { spawn } from 'node:child_process';
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { afterEach, describe, expect, it } from 'vite-plus/test';

import { STDERR_LINES } from '../src/chat/web/errors.js';
import { MAX_FRAME_BYTES } from '../src/chat/web/mcp.js';
import { formatErrorText, parseWebResultText } from '../src/chat/web/result-text.js';
import {
  MAX_CALLS_IN_FLIGHT,
  TOOL_DEFINITIONS,
  WebTools,
  readWebConfig,
  scrubEnv,
  searchArgs,
  type WebConfig,
  type WebDeps,
} from '../src/chat/web/tools.js';
import {
  fakeLocalAnswers,
  fakeLocalCloud,
  fakeLocalScript,
  type FakeLocalAnswers,
  type FakeProvider,
  startFakeProvider,
} from './helpers/fake-providers.js';

const WEB_DIR = fileURLToPath(new URL('../src/chat/web/', import.meta.url));
const BOOTSTRAP = join(WEB_DIR, 'server-ts-bootstrap.ts');
const KEY = 'sk-WEB-KEY-0123456789';

interface Run {
  replies: Array<Record<string, unknown>>;
  stderr: string;
  code: number | null;
}

/** Writes every chunk to a fresh server, closes stdin, and collects what came back. */
function run(chunks: string[], env: Record<string, string> = {}): Promise<Run> {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [BOOTSTRAP], {
      env: { PATH: process.env.PATH ?? '/usr/bin:/bin', NODE_NO_WARNINGS: '1', ...env },
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8').on('data', (d: string) => (stdout += d));
    child.stderr.setEncoding('utf8').on('data', (d: string) => (stderr += d));
    child.on('error', reject);
    child.on('close', (code) => {
      const replies = stdout
        .split('\n')
        .filter((l) => l !== '')
        .map((l) => JSON.parse(l) as Record<string, unknown>);
      resolve({ replies, stderr, code });
    });
    for (const chunk of chunks) child.stdin.write(chunk);
    child.stdin.end();
  });
}

const frame = (value: unknown): string => `${JSON.stringify(value)}\n`;
const callTool = (id: number, name: string, args: unknown): string =>
  frame({ jsonrpc: '2.0', id, method: 'tools/call', params: { name, arguments: args } });

function textOf(reply: Record<string, unknown> | undefined): string {
  const result = reply?.result as { content: Array<{ type: string; text: string }>; isError: boolean };
  expect(result.isError).toBe(false);
  expect(result.content).toHaveLength(1);
  return result.content[0]!.text;
}

describe('mlx_web over stdio', () => {
  it('answers the MCP methods and nothing else', async () => {
    const out = await run([
      frame({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18' } }),
      frame({ jsonrpc: '2.0', method: 'notifications/initialized' }),
      frame({ jsonrpc: '2.0', id: 2, method: 'tools/list' }),
      frame({ jsonrpc: '2.0', id: 3, method: 'ping' }),
      frame({ jsonrpc: '2.0', id: 4, method: 'resources/list' }),
    ]);
    expect(out.code).toBe(0);
    expect(out.stderr).toBe('');
    const byId = new Map(out.replies.map((r) => [r.id, r]));
    expect(out.replies).toHaveLength(4);
    expect(byId.get(1)!.result).toEqual({
      protocolVersion: '2025-06-18',
      capabilities: { tools: { listChanged: false } },
      serverInfo: { name: 'mlx_web', version: '1' },
    });
    expect(byId.get(2)!.result).toEqual({ tools: JSON.parse(JSON.stringify(TOOL_DEFINITIONS)) });
    expect(byId.get(3)!.result).toEqual({});
    expect(byId.get(4)!.error).toEqual({ code: -32601, message: 'Method not found' });
  });

  it('refuses a local address before any provider call, and never echoes a key', async () => {
    const out = await run(
      [
        callTool(1, 'web_fetch', { url: 'http://localhost:8080/admin' }),
        callTool(2, 'web_fetch', { url: 'https://10.0.0.1/' }),
        callTool(3, 'web_fetch', { url: 'https://nas.local./share' }),
      ],
      { MLX_WEB_FETCH_PROVIDER: 'firecrawl', MLX_WEB_FETCH_KEY: KEY },
    );
    expect(out.stderr).toBe('');
    expect(out.replies).toHaveLength(3);
    for (const reply of out.replies) {
      expect(parseWebResultText(textOf(reply))).toMatchObject({
        kind: 'error',
        code: 'blocked',
        provider: 'firecrawl',
      });
    }
    expect(JSON.stringify(out.replies)).not.toContain(KEY);
  });

  it('reads pages through Firecrawl on this Mac from its env, with no Authorization header', async () => {
    const local = await startFakeProvider(fakeLocalScript(fakeLocalAnswers()));
    try {
      const out = await run([callTool(1, 'web_fetch', { url: 'https://example.com/doc' })], {
        MLX_WEB_FETCH_PROVIDER: 'firecrawl',
        MLX_WEB_FETCH_KEY: KEY,
        MLX_WEB_FETCH_URL: local.baseUrl,
      });
      expect(out.stderr).toBe('');
      expect(parseWebResultText(textOf(out.replies[0]))).toMatchObject({
        kind: 'page',
        provider: 'firecrawl',
        title: 'Local page',
      });
      expect(local.requests.map((r) => r.path)).toEqual([
        '/health',
        '/v2/scrape',
        '/v2/team/credit-usage',
        '/v2/scrape',
      ]);
      for (const request of local.requests) expect(request.headers.authorization).toBeUndefined();
      expect(JSON.stringify(local.requests)).not.toContain(KEY);
    } finally {
      await local.close();
    }
  });

  it('says not-configured with no provider, and with a provider but no key', async () => {
    const envs: Array<Record<string, string>> = [
      {},
      { MLX_WEB_SEARCH_PROVIDER: 'exa', MLX_WEB_SEARCH_KEY: '', MLX_WEB_FETCH_PROVIDER: 'firecrawl' },
    ];
    for (const env of envs) {
      const out = await run(
        [callTool(1, 'web_search', { query: 'mlx news' }), callTool(2, 'web_fetch', { url: 'https://example.com' })],
        env,
      );
      expect(out.stderr).toBe('');
      for (const reply of out.replies) {
        expect(parseWebResultText(textOf(reply))).toMatchObject({
          kind: 'error',
          code: 'not-configured',
          provider: 'none',
        });
      }
      expect(JSON.stringify(out.replies)).not.toContain(KEY);
    }
  });

  it('answers wrong arguments and unknown tools with bad-request', async () => {
    const out = await run([
      callTool(1, 'web_search', { query: 5 }),
      callTool(2, 'web_search', { query: '   ' }),
      callTool(3, 'web_search', 'mlx'),
      callTool(4, 'web_fetch', { url: ['https://example.com'] }),
      callTool(5, 'web_fetch', { url: `https://example.com/${'a'.repeat(2048)}` }),
      callTool(6, 'bash', { command: 'id' }),
      frame({ jsonrpc: '2.0', id: 7, method: 'tools/call', params: { arguments: {} } }),
    ]);
    expect(out.stderr).toBe('');
    expect(out.replies).toHaveLength(7);
    for (const reply of out.replies) {
      expect(parseWebResultText(textOf(reply))).toMatchObject({ kind: 'error', code: 'bad-request' });
    }
  });

  it('drops malformed and oversized frames with one fixed stderr line each, and keeps serving', async () => {
    const big = `{"jsonrpc":"2.0","id":9,"method":"ping","pad":"${'x'.repeat(MAX_FRAME_BYTES)}"}\n`;
    const out = await run([
      '{not json\n',
      big.slice(0, 1000),
      big.slice(1000),
      '[1,2]\n',
      'null\n',
      frame({ jsonrpc: '2.0', id: { bad: true }, method: 'ping' }),
      frame({ jsonrpc: '2.0', id: 1, result: {} }),
      frame({ jsonrpc: '2.0', id: 2, method: 'ping' }),
    ]);
    expect(out.code).toBe(0);
    expect(out.stderr).toBe(STDERR_LINES.badFrame + STDERR_LINES.frameTooLarge);
    expect(out.replies.filter((r) => r.id === 2)).toEqual([{ jsonrpc: '2.0', id: 2, result: {} }]);
    expect(out.replies.filter((r) => r.id === 9)).toEqual([]);
    expect(out.replies.filter((r) => r.id === null).map((r) => (r.error as { code: number }).code)).toEqual([
      -32600, -32600, -32600,
    ]);
  });
});

const CONFIG: WebConfig = {
  searchProvider: 'exa',
  searchKey: KEY,
  fetchProvider: 'exa',
  fetchKey: KEY,
  fetchUrl: '',
};

/** A fetch that answers from `answer` and records each URL it was asked for. */
function fakeFetch(answer: (url: string) => Promise<Response>): { fetch: typeof fetch; urls: string[] } {
  const urls: string[] = [];
  return {
    urls,
    fetch: (input) => {
      const url = new URL(input as string | URL).href;
      urls.push(url);
      return answer(url);
    },
  };
}

function deps(fetchImpl: typeof fetch): WebDeps {
  const base = 'http://127.0.0.1:9';
  return { fetch: fetchImpl, baseUrls: { exa: base, deepseek: base, brave: base, firecrawl: base }, now: () => 0 };
}

describe('web tools', () => {
  it('turns a throwing fetch into a code with no provider text, and redacts an echoed key', async () => {
    const throwing = new WebTools(
      CONFIG,
      deps(() => Promise.reject(new Error(`401 for key ${KEY} at /Users/someone`))),
    );
    expect(await throwing.call('web_search', { query: 'q' })).toBe(formatErrorText('offline', 'exa'));
    const echo = new WebTools(
      CONFIG,
      deps(() =>
        Promise.resolve(
          Response.json({ results: [{ title: 't', url: 'https://example.com', text: `your key is ${KEY}` }] }),
        ),
      ),
    );
    const page = await echo.call('web_fetch', { url: 'https://example.com' });
    expect(page).not.toContain(KEY);
    expect(parseWebResultText(page)).toMatchObject({ kind: 'page', text: 'your key is [redacted]' });
  });

  it('refuses a fifth call while four are in flight', async () => {
    const release: Array<() => void> = [];
    const tools = new WebTools(
      CONFIG,
      deps(
        () =>
          new Promise((resolve) => {
            release.push(() => resolve(Response.json({ results: [] })));
          }),
      ),
    );
    const calls = Array.from({ length: MAX_CALLS_IN_FLIGHT }, () => tools.call('web_search', { query: 'q' }));
    expect(await tools.call('web_search', { query: 'q' })).toBe(formatErrorText('rate-limited', 'exa'));
    await new Promise((resolve) => setTimeout(resolve, 10));
    for (const r of release) r();
    expect(await Promise.all(calls)).toEqual(Array(MAX_CALLS_IN_FLIGHT).fill('WEB_SEARCH provider=exa results=0'));
  });

  it('forwards an address without its fragment, and sends nothing for a refused one', async () => {
    const fake = fakeFetch(() => Promise.resolve(Response.json({ results: [{ title: 'Docs', text: 'Body' }] })));
    const tools = new WebTools(CONFIG, deps(fake.fetch));
    const page = parseWebResultText(await tools.call('web_fetch', { url: ' https://example.com/docs#install ' }));
    expect(page).toMatchObject({ kind: 'page', url: 'https://example.com/docs' });
    expect(await tools.call('web_fetch', { url: 'http://127.1/' })).toBe(formatErrorText('blocked', 'exa'));
    expect(fake.urls).toEqual(['http://127.0.0.1:9/contents']);
  });

  it('reads sloppy search arguments', () => {
    expect(searchArgs({ query: '  mlx  ', count: '3', recency: 'week', extra: 1 })).toEqual({
      query: 'mlx',
      count: 3,
      recency: 'week',
    });
    expect(searchArgs({ query: 'q', count: 99 })).toMatchObject({ count: 8 });
    expect(searchArgs({ query: 'q', count: 0 })).toMatchObject({ count: 1 });
    expect(searchArgs({ query: 'q', count: 'many', recency: 'decade' })).toEqual({
      query: 'q',
      count: 5,
      recency: null,
    });
    expect(searchArgs({ query: 'é'.repeat(300) })!.query).toHaveLength(200);
  });

  it('scrubs the web keys and every token from the environment', () => {
    const env: NodeJS.ProcessEnv = {
      PATH: '/bin',
      MLX_WEB_SEARCH_KEY: KEY,
      MLX_WEB_SEARCH_PROVIDER: 'exa',
      XAI_API_KEY: 'x',
      OPENAI_API_KEY: 'y',
      HF_TOKEN: 'z',
      HOME: '/h',
    };
    scrubEnv(env);
    expect(env).toEqual({ PATH: '/bin', HOME: '/h' });
  });
});

describe('readWebConfig', () => {
  const BASE = 'http://127.0.0.1:3002';

  it('keeps MLX_WEB_FETCH_URL only for Firecrawl and only in the saved form, and then drops the fetch key', () => {
    expect(
      readWebConfig({ MLX_WEB_FETCH_PROVIDER: 'firecrawl', MLX_WEB_FETCH_KEY: KEY, MLX_WEB_FETCH_URL: BASE }),
    ).toEqual({ searchProvider: 'none', searchKey: '', fetchProvider: 'firecrawl', fetchKey: '', fetchUrl: BASE });
    for (const env of [
      { MLX_WEB_FETCH_PROVIDER: 'exa', MLX_WEB_FETCH_KEY: KEY, MLX_WEB_FETCH_URL: BASE },
      { MLX_WEB_FETCH_PROVIDER: 'firecrawl', MLX_WEB_FETCH_KEY: KEY, MLX_WEB_FETCH_URL: 'http://192.168.1.5:3002' },
      { MLX_WEB_FETCH_PROVIDER: 'firecrawl', MLX_WEB_FETCH_KEY: KEY, MLX_WEB_FETCH_URL: 'https://127.0.0.1:3002' },
      { MLX_WEB_FETCH_PROVIDER: 'firecrawl', MLX_WEB_FETCH_KEY: KEY, MLX_WEB_FETCH_URL: 'http://localhost:3002' },
    ]) {
      expect([env, readWebConfig(env)]).toMatchObject([env, { fetchKey: '', fetchUrl: '' }]);
    }
    expect(
      readWebConfig({ MLX_WEB_FETCH_PROVIDER: 'firecrawl', MLX_WEB_FETCH_KEY: KEY, MLX_WEB_FETCH_URL: '' }),
    ).toEqual({ searchProvider: 'none', searchKey: '', fetchProvider: 'firecrawl', fetchKey: KEY, fetchUrl: '' });
  });
});

describe('web tools with Firecrawl on this Mac', () => {
  const locals: FakeProvider[] = [];
  afterEach(async () => {
    await Promise.all(locals.splice(0).map((s) => s.close()));
  });

  async function local(): Promise<{ server: FakeProvider; answers: FakeLocalAnswers; tools: WebTools }> {
    const answers = fakeLocalAnswers();
    const server = await startFakeProvider(fakeLocalScript(answers));
    locals.push(server);
    const tools = new WebTools(
      { searchProvider: 'none', searchKey: '', fetchProvider: 'firecrawl', fetchKey: '', fetchUrl: server.baseUrl },
      deps((input, init) => fetch(input, init)),
    );
    return { server, answers, tools };
  }

  const paths = (server: FakeProvider): string[] => server.requests.map((r) => `${r.method} ${r.path}`);

  it('checks the server, then reads the page from it, with no key', async () => {
    const { server, tools } = await local();
    const page = parseWebResultText(await tools.call('web_fetch', { url: 'https://example.com/doc' }));
    expect(page).toMatchObject({ kind: 'page', provider: 'firecrawl', title: 'Local page', cloudRead: false });
    expect(paths(server)).toEqual(['GET /health', 'POST /v2/scrape', 'GET /v2/team/credit-usage', 'POST /v2/scrape']);
    expect(JSON.parse(server.requests[3]!.body)).toMatchObject({ url: 'https://example.com/doc' });
    for (const request of server.requests) expect(request.headers.authorization).toBeUndefined();
  });

  it('a server that reads private addresses gives local-down and gets no page address', async () => {
    const { server, answers, tools } = await local();
    answers.probe = { json: { success: true, data: { markdown: 'health', metadata: { statusCode: 200 } } } };
    expect(await tools.call('web_fetch', { url: 'https://example.com/doc' })).toBe(
      formatErrorText('local-down', 'firecrawl'),
    );
    expect(paths(server)).toEqual(['GET /health', 'POST /v2/scrape']);
    expect(JSON.stringify(server.requests)).not.toContain('example.com');
  });

  it('a server whose Cloud setting is not off, or cannot be read, gives local-cloud and gets no page address', async () => {
    for (const cloud of [
      fakeLocalCloud('fallback'),
      fakeLocalCloud('unreadable'),
      { status: 501, json: { success: false, code: 'OTHER' } },
      { drop: true },
    ]) {
      const { server, answers, tools } = await local();
      answers.cloud = cloud;
      expect([cloud, await tools.call('web_fetch', { url: 'https://example.com/doc' })]).toEqual([
        cloud,
        formatErrorText('local-cloud', 'firecrawl'),
      ]);
      expect(paths(server)).toEqual(['GET /health', 'POST /v2/scrape', 'GET /v2/team/credit-usage']);
      expect(JSON.stringify(server.requests)).not.toContain('example.com');
    }
  });

  it('checks again before every read', async () => {
    const { server, answers, tools } = await local();
    expect(parseWebResultText(await tools.call('web_fetch', { url: 'https://example.com/1' }))).toMatchObject({
      kind: 'page',
    });
    answers.probe = { json: { success: true, data: { markdown: 'health' } } };
    expect(await tools.call('web_fetch', { url: 'https://example.com/2' })).toBe(
      formatErrorText('local-down', 'firecrawl'),
    );
    answers.probe = fakeLocalAnswers().probe;
    expect(parseWebResultText(await tools.call('web_fetch', { url: 'https://example.com/3' }))).toMatchObject({
      kind: 'page',
    });
    const scraped = server.requests
      .filter((r) => r.path === '/v2/scrape')
      .map((r) => (JSON.parse(r.body) as { url: string }).url)
      .filter((url) => url.startsWith('https://example.com'));
    expect(scraped).toEqual(['https://example.com/1', 'https://example.com/3']);
    expect(server.requests.filter((r) => r.path === '/health')).toHaveLength(3);
  });

  it('a server that is not running gives local-down', async () => {
    const { server, tools } = await local();
    await locals.pop()!.close();
    expect(server.requests).toEqual([]);
    expect(await tools.call('web_fetch', { url: 'https://example.com/doc' })).toBe(
      formatErrorText('local-down', 'firecrawl'),
    );
  });

  it('a private or local page address is still blocked before any request', async () => {
    const { server, tools } = await local();
    for (const url of ['http://localhost:8080/admin', 'https://10.0.0.1/', 'http://127.1/', 'https://nas.local/x']) {
      expect(await tools.call('web_fetch', { url })).toBe(formatErrorText('blocked', 'firecrawl'));
    }
    expect(server.requests).toEqual([]);
  });

  it('search never goes to the local base', async () => {
    const answers = fakeLocalAnswers();
    const server = await startFakeProvider(fakeLocalScript(answers));
    locals.push(server);
    const fake = fakeFetch(() => Promise.resolve(Response.json({ results: [] })));
    const tools = new WebTools(
      { searchProvider: 'exa', searchKey: KEY, fetchProvider: 'firecrawl', fetchKey: '', fetchUrl: server.baseUrl },
      deps(fake.fetch),
    );
    expect(await tools.call('web_search', { query: 'q' })).toBe('WEB_SEARCH provider=exa results=0');
    expect(fake.urls).toEqual(['http://127.0.0.1:9/search']);
    expect(server.requests).toEqual([]);
  });

  it('cloud with no key, or a fetch URL that is not a saved loopback form, is not-configured and sends nothing', async () => {
    const fake = fakeFetch(() => Promise.resolve(Response.json({})));
    const configs: WebConfig[] = [
      { searchProvider: 'none', searchKey: '', fetchProvider: 'firecrawl', fetchKey: '', fetchUrl: '' },
      {
        searchProvider: 'none',
        searchKey: '',
        fetchProvider: 'firecrawl',
        fetchKey: KEY,
        fetchUrl: 'http://10.0.0.2:3002',
      },
      { searchProvider: 'none', searchKey: '', fetchProvider: 'exa', fetchKey: KEY, fetchUrl: 'http://127.0.0.1:3002' },
    ];
    for (const config of configs) {
      const tools = new WebTools(config, deps(fake.fetch));
      expect(await tools.call('web_fetch', { url: 'https://example.com' })).toBe(
        formatErrorText('not-configured', 'none'),
      );
    }
    expect(fake.urls).toEqual([]);
  });
});

describe('mlx_web source', () => {
  const files = readdirSync(WEB_DIR, { recursive: true })
    .map(String)
    .filter((f) => f.endsWith('.ts'));

  it('never names console', () => {
    for (const file of files)
      expect([file, readFileSync(join(WEB_DIR, file), 'utf8').includes('console')]).toEqual([file, false]);
  });

  it('imports only node: built-ins and its own folder', () => {
    for (const file of files) {
      const text = readFileSync(join(WEB_DIR, file), 'utf8');
      for (const line of text.split('\n')) {
        const at = line.indexOf(" from '");
        if (!line.startsWith('import ') && !(line.startsWith('} from ') || at !== -1)) continue;
        if (at === -1 && !line.startsWith('} from ')) continue;
        const start = line.indexOf("'") + 1;
        const spec = line.slice(start, line.indexOf("'", start));
        const inFolder = spec.startsWith('./') || (file.includes('/') && spec.startsWith('../'));
        const target = join(WEB_DIR, file, '..', spec);
        const own = inFolder && target.startsWith(WEB_DIR) && !target.slice(WEB_DIR.length).startsWith('..');
        expect([file, spec, spec.startsWith('node:') || own]).toEqual([file, spec, true]);
      }
    }
  });
});
