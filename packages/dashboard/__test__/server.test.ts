import {
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  unlinkSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { request } from 'node:http';
import { networkInterfaces, tmpdir } from 'node:os';
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

  // Finding F: a rename must verify the indexed path still holds THIS session
  // before writing its name — a reused path must not stamp a foreign session.
  it('refuses to rename when the indexed path was reused by another session', async () => {
    await ingest();
    const before = (await (await fetch(`${server.url}/api/sessions`)).json()) as {
      sessions: Array<{ id: string; path: string }>;
    };
    const filePath = before.sessions.find((s) => s.id === 'fix-1')!.path;

    // Replace fix-1's file on disk with a different session (bypassing ingest):
    // the index still points fix-1 at a path that now holds session 'reused-B'.
    writeFileSync(
      filePath,
      `${JSON.stringify({ type: 'session', version: 3, id: 'reused-B', timestamp: '2026-07-09T10:00:00.000Z', cwd: '/w' })}\n${JSON.stringify({ type: 'message', id: 'b1', parentId: null, timestamp: '2026-07-09T10:00:01.000Z', message: { role: 'user', content: 'from B' } })}\n`,
    );

    const patch = await fetch(`${server.url}/api/sessions/fix-1`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'Hijack' }),
    });
    expect(patch.status).toBe(409);

    // Session B's file was never stamped with fix-1's requested name.
    const fileText = readFileSync(filePath, 'utf-8');
    expect(fileText).not.toContain('Hijack');
    expect(fileText).not.toContain('"type":"session_info"');
    expect(fileText).toContain('reused-B');
  });

  // Finding H: the detail transcript uses the same active-branch projection as
  // the index — a detached metadata leaf must not resurrect abandoned turns.
  it('detail transcript shows only the active branch under a detached metadata leaf', async () => {
    const forked = [
      { type: 'session', version: 3, id: 'detach-1', timestamp: '2026-07-08T10:00:00.000Z', cwd: '/w' },
      { type: 'message', id: 'u1', parentId: null, timestamp: '2026-07-08T10:00:01.000Z', message: { role: 'user', content: 'q' } },
      {
        type: 'message',
        id: 'a1',
        parentId: 'u1',
        timestamp: '2026-07-08T10:00:02.000Z',
        message: { role: 'assistant', content: [{ type: 'text', text: 'ABANDONED' }], model: 'gemma4', usage: { input: 999, output: 999 } },
      },
      {
        type: 'message',
        id: 'a2',
        parentId: 'u1',
        timestamp: '2026-07-08T10:00:03.000Z',
        message: { role: 'assistant', content: [{ type: 'text', text: 'ACTIVE' }], model: 'qwen3_5', usage: { input: 1, output: 2 } },
      },
      { type: 'session_info', id: 'si1', parentId: null, timestamp: '2026-07-08T10:00:04.000Z', name: 'Detached' },
    ];
    writeFileSync(
      join(sessionsRoot, '--w--', '2026-07-08T10-00-00_detach-1.jsonl'),
      `${forked.map((l) => JSON.stringify(l)).join('\n')}\n`,
    );
    await ingest();

    const body = (await (await fetch(`${server.url}/api/sessions/detach-1`)).json()) as {
      transcript: Array<{ text: string }>;
    };
    const texts = body.transcript.map((t) => t.text);
    expect(texts.some((t) => t.includes('ACTIVE'))).toBe(true);
    expect(texts.some((t) => t.includes('ABANDONED'))).toBe(false);
  });

  // Finding 5: a GET of the session detail must be READ-ONLY. A v1 session with a
  // malformed trailing line (the case where an open-for-write migrate would both
  // persist the v1→v3 rewrite and drop the malformed line) must be left byte-for-
  // byte unchanged on disk while still returning the valid transcript.
  it('detail GET does not rewrite a v1 session with a malformed trailing line', async () => {
    const dir = join(sessionsRoot, '--w--');
    mkdirSync(dir, { recursive: true });
    const file = join(dir, '2026-07-10T10-00-00_ro-1.jsonl');
    const header = JSON.stringify({ type: 'session', version: 1, id: 'ro-1', timestamp: '2026-07-10T10:00:00.000Z', cwd: '/w' });
    const user = JSON.stringify({ type: 'message', timestamp: '2026-07-10T10:00:01.000Z', message: { role: 'user', content: 'READ ONLY hi' } });
    const asst = JSON.stringify({
      type: 'message',
      timestamp: '2026-07-10T10:00:02.000Z',
      message: { role: 'assistant', content: [{ type: 'text', text: 'READ ONLY yo' }], model: 'qwen3_5', usage: { input: 5, output: 6 } },
    });
    const truncated = '{"type":"message","message":{"role":"asst';
    const original = `${header}\n${user}\n${asst}\n${truncated}`;
    writeFileSync(file, original);
    await ingest();
    const before = readFileSync(file);

    const res = await fetch(`${server.url}/api/sessions/ro-1`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as { transcript: Array<{ text: string }> };
    const texts = body.transcript.map((t) => t.text);
    expect(texts.some((t) => t.includes('READ ONLY hi'))).toBe(true);
    expect(texts.some((t) => t.includes('READ ONLY yo'))).toBe(true);

    // The GET never mutated the source of truth.
    const after = readFileSync(file);
    expect(after.equals(before)).toBe(true);
    expect(after.toString('utf-8')).toBe(original);
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

  // Finding 11b: a subagent (child) turn has no persisted session JSONL of its
  // own; its trace carries rootSessionId. The root session's metrics view must
  // surface it via root_session_id, not only its own session_id.
  it('includes subagent traces under the root session metrics', async () => {
    writeFileSync(
      join(tracesDir, '2026-07-02-child.jsonl'),
      `${JSON.stringify({
        v: 1,
        traceId: 'trace-child',
        ts: 1782036302000,
        sessionId: 'child-of-fix-1',
        rootSessionId: 'fix-1',
        model: 'qwen3_5',
        durationMs: 10,
        finishReason: 'stop',
        promptTokens: 1,
        cachedTokens: 0,
        outputTokens: 1,
        reasoningTokens: 0,
      })}\n`,
    );
    await ingest();
    const res = await fetch(`${server.url}/api/sessions/fix-1/metrics`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as { traces: Array<{ traceId: string }> };
    const traceIds = body.traces.map((t) => t.traceId);
    // Both the root's own trace (session_id match) and the child's (root match).
    expect(traceIds).toContain('trace-aaa');
    expect(traceIds).toContain('trace-child');
  });
});

// Finding 4: session-file symlink containment. Primary — a symlinked transcript
// is never indexed, so its external id is simply unknown (404). Defense-in-depth
// — a GET whose indexed path resolves outside the managed root (via a symlink
// swapped in after indexing) is refused (403) by the realpath containment guard.
describe('dashboard server — session symlink containment (Finding 4)', () => {
  it('never indexes a symlinked transcript, so its external id 404s on GET and PATCH', async () => {
    const externalFile = join(base, 'external.jsonl');
    writeFileSync(
      externalFile,
      `${[
        { type: 'session', version: 3, id: 'external-1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/secret' },
        {
          type: 'message',
          id: 'm1',
          parentId: null,
          timestamp: '2026-07-01T10:00:01.000Z',
          message: { role: 'user', content: 'secret transcript' },
        },
      ]
        .map((l) => JSON.stringify(l))
        .join('\n')}\n`,
    );
    symlinkSync(externalFile, join(sessionsRoot, '--w--', 'evil.jsonl'));
    await ingest();

    expect((await fetch(`${server.url}/api/sessions/external-1`)).status).toBe(404);
    const patch = await fetch(`${server.url}/api/sessions/external-1`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'x' }),
    });
    expect(patch.status).toBe(404);
  });

  it('refuses a GET whose indexed file was swapped to an external symlink', async () => {
    await ingest();
    const before = (await (await fetch(`${server.url}/api/sessions`)).json()) as {
      sessions: Array<{ id: string; path: string }>;
    };
    const filePath = before.sessions.find((s) => s.id === 'fix-1')!.path;

    // Swap the indexed real file for a symlink pointing outside the managed root.
    const externalFile = join(base, 'external-detail.jsonl');
    writeFileSync(externalFile, readFileSync(filePath));
    unlinkSync(filePath);
    symlinkSync(externalFile, filePath);

    // The row still points at filePath, but it now resolves outside the root.
    expect((await fetch(`${server.url}/api/sessions/fix-1`)).status).toBe(403);
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
  it('scans and clears the cold cache with an explicit {all:true}', async () => {
    const res = await fetch(`${server.url}/api/cache`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as { disk: { entryCount: number; totalBytes: number } };
    expect(body.disk.entryCount).toBe(2);
    expect(body.disk.totalBytes).toBe(300);

    const del = await fetch(`${server.url}/api/cache`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ all: true }),
    });
    expect(del.status).toBe(200);
    const cleared = (await del.json()) as { removed: number; freedBytes: number };
    expect(cleared.removed).toBe(2);
    expect(cleared.freedBytes).toBe(300);

    const after = (await (await fetch(`${server.url}/api/cache`)).json()) as { disk: { entryCount: number } };
    expect(after.disk.entryCount).toBe(0);
  });

  // Finding I: an ambiguous body must 400, never fall through to a whole wipe.
  it('rejects an ambiguous clear body instead of wiping the whole cache', async () => {
    const bodies: Array<string | undefined> = [
      undefined,
      '{}',
      JSON.stringify({ olderThanDays: '7' }),
      JSON.stringify({ olderThanDays: 0 }),
      JSON.stringify({ olderThanDays: -1 }),
    ];
    for (const b of bodies) {
      const del = await fetch(`${server.url}/api/cache`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        ...(b === undefined ? {} : { body: b }),
      });
      expect(del.status).toBe(400);
      // Nothing was cleared — both blocks survive.
      const after = (await (await fetch(`${server.url}/api/cache`)).json()) as { disk: { entryCount: number } };
      expect(after.disk.entryCount).toBe(2);
    }
  });

  it('evicts only blocks older than a positive olderThanDays', async () => {
    // Age one block past the 7-day cutoff; the other stays recent.
    const oldBlock = join(cacheRoot, hexBlock(1));
    const oldSec = (Date.now() - 10 * 86_400_000) / 1000;
    utimesSync(oldBlock, oldSec, oldSec);

    const del = await fetch(`${server.url}/api/cache`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ olderThanDays: 7 }),
    });
    expect(del.status).toBe(200);
    const evicted = (await del.json()) as { removed: number; freedBytes: number };
    expect(evicted.removed).toBe(1);
    expect(evicted.freedBytes).toBe(100);

    const after = (await (await fetch(`${server.url}/api/cache`)).json()) as { disk: { entryCount: number } };
    expect(after.disk.entryCount).toBe(1);
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

describe('dashboard server — wildcard bind Host allowlist', () => {
  function firstLocalIpv4(): string | undefined {
    for (const addrs of Object.values(networkInterfaces())) {
      for (const a of addrs ?? []) {
        if (a.family === 'IPv4' && !a.internal) return a.address;
      }
    }
    return undefined;
  }

  // Finding E: `--host 0.0.0.0` is a documented feature. A wildcard bind must
  // accept a LAN client whose Host carries a real local interface IP while still
  // rejecting a rebound attacker domain (no matching local IP). The loopback
  // default is covered by the existing rebinding tests and stays unchanged.
  it('allows a real local-interface Host and still rejects a rebound domain under a wildcard bind', async () => {
    let wild: DashboardServer;
    try {
      wild = await startDashboardServer({
        port: 0,
        host: '0.0.0.0',
        dbPath: ':memory:',
        sessionsRoot,
        tracesDir,
        modelsDir,
        cacheRoot,
        webRoot,
      });
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code;
      // Some sandboxes forbid a wildcard bind; an environment limit, not a defect.
      if (code === 'EPERM' || code === 'EACCES') return;
      throw err;
    }
    try {
      // A rebound attacker domain has no matching local IP → still 403.
      const evil = await rawRequest(wild.port, 'GET', '/api/models', { Host: `evil.example:${wild.port}` });
      expect(evil.status).toBe(403);

      // Loopback stays reachable under a wildcard bind.
      const loop = await rawRequest(wild.port, 'GET', '/api/models', { Host: `127.0.0.1:${wild.port}` });
      expect(loop.status).toBe(200);

      // A real local-interface IP (LAN reachability) is now allowed. If the host
      // has no external interface, skip that leg without weakening the assertions.
      const ip = firstLocalIpv4();
      if (ip !== undefined) {
        const lan = await rawRequest(wild.port, 'GET', '/api/models', { Host: `${ip}:${wild.port}` });
        expect(lan.status).toBe(200);
      }
    } finally {
      await wild.close();
    }
  });
});

describe('dashboard server — connectable display URL', () => {
  async function startWild(host: string): Promise<DashboardServer | undefined> {
    try {
      return await startDashboardServer({
        port: 0,
        host,
        dbPath: ':memory:',
        sessionsRoot,
        tracesDir,
        modelsDir,
        cacheRoot,
        webRoot,
      });
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code;
      // Some sandboxes forbid a wildcard/non-loopback bind; an environment limit.
      if (code === 'EPERM' || code === 'EACCES') return undefined;
      throw err;
    }
  }

  it('advertises the literal loopback host for a concrete bind (default 127.0.0.1)', () => {
    // beforeEach binds the default host; its URL must carry the literal host.
    expect(server.url).toBe(`http://127.0.0.1:${server.port}`);
  });

  it('advertises a bracketed [::1] for a concrete ::1 bind', async () => {
    const s = await startWild('::1');
    if (s === undefined) return; // IPv6 loopback bind blocked in this sandbox.
    try {
      expect(s.url).toBe(`http://[::1]:${s.port}`);
      const res = await fetch(`${s.url}/health`);
      expect(res.status).toBe(200);
    } finally {
      await s.close();
    }
  });

  // Finding 6: a wildcard bind has no connectable literal host — advertising the
  // raw wildcard yields a URL the server's own Host allowlist rejects
  // (`0.0.0.0` → 403 from classifyHost) or that `new URL` cannot even parse
  // (`::` → `:::`, '' → `:`). The returned URL must instead name a loopback the
  // allowlist accepts and a client can actually reach.
  it('advertises a connectable loopback URL for each wildcard bind', async () => {
    const cases: Array<{ host: string; authority: string }> = [
      { host: '0.0.0.0', authority: '127.0.0.1' },
      { host: '::', authority: '[::1]' },
      { host: '', authority: '127.0.0.1' },
    ];
    for (const { host, authority } of cases) {
      const s = await startWild(host);
      if (s === undefined) continue; // wildcard bind blocked in this sandbox.
      try {
        // Parses cleanly: the '' and '::' cases previously produced malformed URLs.
        const u = new URL(s.url);
        expect(u.protocol).toBe('http:');
        expect(u.hostname).toBe(authority); // URL keeps the [] for an IPv6 literal.
        expect(u.port).toBe(String(s.port));
        expect(s.url).toBe(`http://${authority}:${s.port}`);
        // Loopback is in the Host allowlist AND actually reachable → 200, unlike
        // the raw wildcard Host (0.0.0.0 → 403, ::/'' → malformed) it replaces.
        const res = await fetch(`${s.url}/health`);
        expect(res.status).toBe(200);
      } finally {
        await s.close();
      }
    }
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
