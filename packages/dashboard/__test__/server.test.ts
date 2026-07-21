import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { request } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { createDownloadSseSender } from '../src/api.js';
import type { DownloadEvent } from '../src/download.js';
import { startDashboardServer, type DashboardServer } from '../src/server.js';

const FIXTURE_SESSIONS = fileURLToPath(new URL('./fixtures/sessions', import.meta.url));
const FIXTURE_TRACES = fileURLToPath(new URL('./fixtures/traces', import.meta.url));

const MODEL_CONFIG = JSON.stringify({ model_type: 'qwen3', max_position_embeddings: 40960 });

/** 64-char lowercase-hex cold-cache block filename. */
function hexBlock(index: number): string {
  return `${index.toString(16).padStart(64, '0')}.safetensors`;
}

let base: string;
let sessionsRoot: string;
let tracesDir: string;
let modelsDir: string;
let cacheRoot: string;
let webRoot: string;
let server: DashboardServer;

/** Raw HTTP request so security tests can set forbidden `Host`/`Origin` headers. */
function rawRequest(
  port: number,
  method: string,
  path: string,
  headers: Record<string, string>,
): Promise<{ status: number; body: string }> {
  return new Promise((resolve, reject) => {
    const req = request({ host: '127.0.0.1', port, method, path, headers }, (res) => {
      const chunks: Buffer[] = [];
      res.on('data', (c: Buffer) => chunks.push(c));
      res.on('end', () => resolve({ status: res.statusCode ?? 0, body: Buffer.concat(chunks).toString('utf-8') }));
    });
    req.on('error', reject);
    req.end();
  });
}

/** Run a full incremental ingest and wait for it to land in the index. */
async function ingest(): Promise<void> {
  const res = await fetch(`${server.url}/api/ingest`, { method: 'POST' });
  expect(res.status).toBe(200);
}

beforeEach(async () => {
  base = mkdtempSync(join(tmpdir(), 'dash-server-'));
  sessionsRoot = join(base, 'sessions');
  cpSync(FIXTURE_SESSIONS, sessionsRoot, { recursive: true });
  tracesDir = join(base, 'traces');
  cpSync(FIXTURE_TRACES, tracesDir, { recursive: true });

  modelsDir = join(base, 'models');
  mkdirSync(join(modelsDir, 'model-a'), { recursive: true });
  writeFileSync(join(modelsDir, 'model-a', 'config.json'), MODEL_CONFIG);
  writeFileSync(join(modelsDir, 'model-a', 'model.safetensors'), Buffer.alloc(2048));

  cacheRoot = join(base, 'cache');
  mkdirSync(cacheRoot, { recursive: true });
  writeFileSync(join(cacheRoot, hexBlock(1)), Buffer.alloc(100));
  writeFileSync(join(cacheRoot, hexBlock(2)), Buffer.alloc(200));

  webRoot = join(base, 'web');
  mkdirSync(webRoot, { recursive: true });
  writeFileSync(join(webRoot, 'index.html'), '<!doctype html><title>mlx dashboard</title><div id="root"></div>');
  writeFileSync(join(webRoot, 'app.js'), 'console.log("app");');

  server = await startDashboardServer({
    port: 0,
    dbPath: ':memory:',
    sessionsRoot,
    tracesDir,
    modelsDir,
    cacheRoot,
    webRoot,
  });
});

afterEach(async () => {
  await server.close();
  rmSync(base, { recursive: true, force: true });
});

describe('dashboard server — models & catalog', () => {
  it('lists local models', async () => {
    const res = await fetch(`${server.url}/api/models`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as { models: Array<{ name: string; modelType: string }> };
    expect(body.models.map((m) => m.name)).toContain('model-a');
    expect(body.models.find((m) => m.name === 'model-a')?.modelType).toBe('qwen3');
  });

  it('serves the catalog with install state', async () => {
    const res = await fetch(`${server.url}/api/catalog`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as { items: Array<{ slug: string; installed: boolean; hfRepo: string }> };
    expect(body.items.length).toBeGreaterThan(0);
    for (const item of body.items) expect(typeof item.installed).toBe('boolean');
  });
});

describe('dashboard server — sessions', () => {
  it('lists ingested sessions', async () => {
    await ingest();
    const res = await fetch(`${server.url}/api/sessions`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as { sessions: Array<{ id: string; name: string | null; models: string[] }> };
    const ids = body.sessions.map((s) => s.id);
    expect(ids).toContain('fix-1');
    expect(ids).toContain('fix-2');
  });

  it('returns a session detail with transcript text', async () => {
    await ingest();
    const res = await fetch(`${server.url}/api/sessions/fix-1`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      session: { id: string };
      transcript: Array<{ role: string; text: string }>;
    };
    expect(body.session.id).toBe('fix-1');
    const texts = body.transcript.map((t) => t.text);
    expect(texts.some((t) => t.includes('Hello, world'))).toBe(true);
    expect(texts.some((t) => t.includes('Hi there'))).toBe(true);
  });

  it('404s an unknown session', async () => {
    await ingest();
    const res = await fetch(`${server.url}/api/sessions/does-not-exist`);
    expect(res.status).toBe(404);
  });

  it('renames a session (persists a session_info line and reflects the new name)', async () => {
    await ingest();
    const before = (await (await fetch(`${server.url}/api/sessions`)).json()) as {
      sessions: Array<{ id: string; path: string }>;
    };
    const filePath = before.sessions.find((s) => s.id === 'fix-1')!.path;

    const patch = await fetch(`${server.url}/api/sessions/fix-1`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'Renamed Session' }),
    });
    expect(patch.status).toBe(200);

    const detail = (await (await fetch(`${server.url}/api/sessions/fix-1`)).json()) as {
      session: { name: string | null };
    };
    expect(detail.session.name).toBe('Renamed Session');

    const fileText = readFileSync(filePath, 'utf-8');
    expect(fileText).toContain('"type":"session_info"');
    expect(fileText).toContain('Renamed Session');
  });

  it('deletes a session (file and rows removed)', async () => {
    await ingest();
    const before = (await (await fetch(`${server.url}/api/sessions`)).json()) as {
      sessions: Array<{ id: string; path: string }>;
    };
    const filePath = before.sessions.find((s) => s.id === 'fix-2')!.path;
    expect(existsSync(filePath)).toBe(true);

    const del = await fetch(`${server.url}/api/sessions/fix-2`, { method: 'DELETE' });
    expect(del.status).toBe(200);
    expect(existsSync(filePath)).toBe(false);

    const after = (await (await fetch(`${server.url}/api/sessions`)).json()) as { sessions: Array<{ id: string }> };
    expect(after.sessions.map((s) => s.id)).not.toContain('fix-2');
    expect((await fetch(`${server.url}/api/sessions/fix-2`)).status).toBe(404);
  });

  it('joins turns and traces for session metrics', async () => {
    await ingest();
    const res = await fetch(`${server.url}/api/sessions/fix-1/metrics`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      turns: Array<{ traceId: string | null; ttftMs: number | null }>;
      traces: unknown[];
    };
    const traced = body.turns.find((t) => t.traceId === 'trace-aaa');
    expect(traced).toBeDefined();
    expect(traced?.ttftMs).toBeCloseTo(120.5, 1);
  });
});

describe('dashboard server — metrics overview', () => {
  it('returns aggregate arrays', async () => {
    await ingest();
    const res = await fetch(`${server.url}/api/metrics/overview`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      tokensByDay: unknown[];
      throughputByModel: unknown[];
      modelShare: unknown[];
      totals: { turns: number; outputTokens: number };
    };
    expect(Array.isArray(body.tokensByDay)).toBe(true);
    expect(body.tokensByDay.length).toBeGreaterThan(0);
    expect(Array.isArray(body.throughputByModel)).toBe(true);
    expect(Array.isArray(body.modelShare)).toBe(true);
    expect(body.totals.turns).toBeGreaterThan(0);
    expect(body.totals.outputTokens).toBeGreaterThan(0);
  });
});

describe('dashboard server — cache', () => {
  it('scans and clears the cold cache', async () => {
    const res = await fetch(`${server.url}/api/cache`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as { disk: { entryCount: number; totalBytes: number } };
    expect(body.disk.entryCount).toBe(2);
    expect(body.disk.totalBytes).toBe(300);

    const del = await fetch(`${server.url}/api/cache`, { method: 'DELETE' });
    expect(del.status).toBe(200);
    const cleared = (await del.json()) as { removed: number; freedBytes: number };
    expect(cleared.removed).toBe(2);
    expect(cleared.freedBytes).toBe(300);

    const after = (await (await fetch(`${server.url}/api/cache`)).json()) as { disk: { entryCount: number } };
    expect(after.disk.entryCount).toBe(0);
  });
});

describe('dashboard server — downloads', () => {
  it('lists jobs and rejects a non-catalog repo', async () => {
    const list = (await (await fetch(`${server.url}/api/downloads`)).json()) as { jobs: unknown[] };
    expect(Array.isArray(list.jobs)).toBe(true);

    const bad = await fetch(`${server.url}/api/downloads`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo: 'someone/not-in-catalog' }),
    });
    expect(bad.status).toBe(400);
  });
});

describe('dashboard server — static SPA', () => {
  it('serves index.html at /', async () => {
    const res = await fetch(`${server.url}/`);
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toContain('text/html');
    expect(res.headers.get('cache-control')).toContain('no-cache');
    expect(await res.text()).toContain('mlx dashboard');
  });

  it('falls back to index.html for an unknown non-api path', async () => {
    const res = await fetch(`${server.url}/sessions/deep/link`);
    expect(res.status).toBe(200);
    expect(await res.text()).toContain('mlx dashboard');
  });

  it('serves a real asset with its content-type', async () => {
    const res = await fetch(`${server.url}/app.js`);
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toContain('javascript');
  });

  it('404s a JSON error for an unknown api path', async () => {
    const res = await fetch(`${server.url}/api/nope`);
    expect(res.status).toBe(404);
    expect(res.headers.get('content-type')).toContain('application/json');
  });

  it('reports health', async () => {
    const res = await fetch(`${server.url}/health`);
    expect(res.status).toBe(200);
    expect(((await res.json()) as { status: string }).status).toBe('ok');
  });
});

describe('dashboard server — local-origin guard', () => {
  it('rejects a mutating request with a cross-site Origin', async () => {
    const res = await rawRequest(server.port, 'POST', '/api/ingest', {
      Origin: 'https://evil.example',
    });
    expect(res.status).toBe(403);
  });

  it('allows a mutating request with no Origin (curl-style)', async () => {
    const res = await rawRequest(server.port, 'POST', '/api/ingest', {});
    expect(res.status).toBe(200);
  });

  it('rejects a mutating request with a non-local Host', async () => {
    const res = await rawRequest(server.port, 'POST', '/api/ingest', {
      Host: 'attacker.example:6590',
    });
    expect(res.status).toBe(403);
  });

  it('does not guard GET reads', async () => {
    const res = await rawRequest(server.port, 'GET', '/api/models', {
      Origin: 'https://evil.example',
    });
    expect(res.status).toBe(200);
  });
});

describe('dashboard server — loopback Host guard (all methods)', () => {
  it('rejects a GET read with a non-loopback Host (DNS-rebinding)', async () => {
    const res = await rawRequest(server.port, 'GET', '/api/models', { Host: 'evil.example' });
    expect(res.status).toBe(403);
  });

  it('allows a GET read with a loopback Host', async () => {
    const res = await rawRequest(server.port, 'GET', '/api/models', { Host: `127.0.0.1:${server.port}` });
    expect(res.status).toBe(200);
  });
});

describe('dashboard server — malformed Host header', () => {
  it('answers 400 for a malformed Host and stays alive', async () => {
    const bad = await rawRequest(server.port, 'GET', '/health', { Host: '[' });
    expect(bad.status).toBe(400);
    // The process must survive: a subsequent well-formed request still answers.
    const ok = await rawRequest(server.port, 'GET', '/health', { Host: `127.0.0.1:${server.port}` });
    expect(ok.status).toBe(200);
  });
});

describe('dashboard SSE progress — backpressure coalescing', () => {
  const progress = (received: number): DownloadEvent => ({
    type: 'progress',
    id: 'job',
    file: 'model.safetensors',
    receivedBytes: received,
    totalBytes: 1000,
    fileIndex: 0,
    fileCount: 1,
  });

  /** A writable whose backpressure is driven manually. */
  function fakeWritable(): {
    write(chunk: string): boolean;
    on(event: 'drain', listener: () => void): void;
    writes: string[];
    setCanWrite(v: boolean): void;
    drain(): void;
  } {
    const writes: string[] = [];
    let canWrite = true;
    let drainListener: (() => void) | null = null;
    return {
      writes,
      write(chunk: string): boolean {
        writes.push(chunk);
        return canWrite;
      },
      on(event: 'drain', listener: () => void): void {
        if (event === 'drain') drainListener = listener;
      },
      setCanWrite(v: boolean): void {
        canWrite = v;
      },
      drain(): void {
        drainListener?.();
      },
    };
  }

  it('coalesces a flood of progress frames while the socket is backpressured', () => {
    const res = fakeWritable();
    const send = createDownloadSseSender(res);

    send(progress(1));
    expect(res.writes.length).toBe(1); // drained synchronously

    res.setCanWrite(false);
    send(progress(2)); // written, returns false → now blocked
    expect(res.writes.length).toBe(2);

    for (let i = 3; i <= 1000; i++) send(progress(i)); // 998 more, all coalesced
    expect(res.writes.length).toBe(2); // bounded: nothing extra buffered to the socket

    res.setCanWrite(true);
    res.drain();
    expect(res.writes.length).toBe(3); // exactly one coalesced frame flushed
    expect(res.writes[2]).toContain('"receivedBytes":1000'); // carrying the latest bytes
  });

  it('keeps a terminal done frame after coalesced progress', () => {
    const res = fakeWritable();
    const send = createDownloadSseSender(res);

    res.setCanWrite(false);
    send(progress(1)); // written, returns false → blocked
    for (let i = 2; i <= 100; i++) send(progress(i)); // coalesced
    send({ type: 'done', id: 'job', outputDir: '/models/job' }); // queued, not dropped
    expect(res.writes.length).toBe(1);

    res.setCanWrite(true);
    res.drain();
    // Latest progress then the terminal frame, in order.
    expect(res.writes.length).toBe(3);
    expect(res.writes[1]).toContain('"receivedBytes":100');
    expect(res.writes[2]).toContain('event: done');
  });
});

describe('dashboard server — SSE downloads', () => {
  it('opens an event stream and cleans up on abort', async () => {
    const controller = new AbortController();
    const res = await fetch(`${server.url}/api/downloads/unknown-job/events`, { signal: controller.signal });
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toContain('text/event-stream');
    const reader = res.body!.getReader();
    const { value } = await reader.read();
    expect(new TextDecoder().decode(value)).toContain('connected');
    await reader.cancel();
    controller.abort();
  });
});
