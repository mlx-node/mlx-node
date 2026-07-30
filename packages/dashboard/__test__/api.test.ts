/**
 * The behavioural gate for the whole dashboard API.
 *
 * Every assertion here drives `runtime.call` — the same entry point the Electron
 * MessagePort bridge feeds — so it tests API behaviour and not transport
 * plumbing. The one block that deliberately goes through the real RPC transport
 * is at the bottom: it holds the port bridge to the identical error-model
 * contract end to end, over a real structured clone.
 */

import {
  chmodSync,
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  realpathSync,
  rmSync,
  symlinkSync,
  unlinkSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MessageChannel } from 'node:worker_threads';

import { canonicalCacheRoot, coldTierRestoreFamilyList } from '@mlx-node/agent/catalog';
import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import type { ApiContext } from '../src/api/context.js';
import { dispatch, isApiPath } from '../src/api/dispatch.js';
import { openDashboardDb } from '../src/db/open.js';
import { DownloadsClosedError } from '../src/download.js';
import { TRACE_RETENTION_DAYS } from '../src/ingest/traces.js';
import { agentSessionsRoot } from '../src/paths.js';
import { createRpcClient } from '../src/rpc/client.js';
import { serveRuntimeOverPort } from '../src/rpc/host.js';
import { bindEventTargetPort } from '../src/rpc/port.js';
import { createDashboardRuntime, type DashboardRuntime } from '../src/runtime.js';
import { createTestClient, type TestClient } from './helpers/api-client.js';

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
let runtime: DashboardRuntime;
let api: TestClient;

/** Run a full incremental ingest and wait for it to land in the index. */
async function ingest(): Promise<void> {
  const res = await api.fetch('/api/ingest', { method: 'POST' });
  expect(res.status).toBe(200);
}

beforeEach(() => {
  base = mkdtempSync(join(tmpdir(), 'dash-api-'));
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

  runtime = createDashboardRuntime({
    dbPath: ':memory:',
    sessionsRoot,
    tracesDir,
    modelsDir,
    cacheRoot,
  });
  api = createTestClient(runtime);
});

afterEach(async () => {
  await runtime.close();
  rmSync(base, { recursive: true, force: true });
});

describe('dashboard api — models & catalog', () => {
  it('lists local models', async () => {
    const res = await api.fetch('/api/models');
    expect(res.status).toBe(200);
    const body = (await res.json()) as { models: Array<{ name: string; modelType: string }>; dir: string };
    expect(body.models.map((m) => m.name)).toContain('model-a');
    expect(body.models.find((m) => m.name === 'model-a')?.modelType).toBe('qwen3');
    // The directory is configurable, so the response names where it looked.
    expect(body.dir).toBe(modelsDir);
  });

  // The delete route distinguishes an unknown name from every other refusal ONLY
  // by sniffing `/not found/i` on the store's error message: an unknown checkpoint
  // is a 404, a refused-but-present one (reserved dir, symlink, not a checkpoint,
  // escaping path) is a 400. Nothing else in the store carries that distinction,
  // so the sniff is load-bearing.
  it('deletes a local model, 404s an unknown name and 400s a refused one', async () => {
    mkdirSync(join(modelsDir, 'not-a-checkpoint'), { recursive: true });

    const ok = await api.fetch('/api/models/model-a', { method: 'DELETE' });
    expect(ok.status).toBe(200);
    expect(await ok.json()).toEqual({ deleted: true, name: 'model-a' });
    expect(existsSync(join(modelsDir, 'model-a'))).toBe(false);

    const unknown = await api.fetch('/api/models/ghost-model', { method: 'DELETE' });
    expect(unknown.status).toBe(404);
    expect(((await unknown.json()) as { error: string }).error).toMatch(/not found/i);

    // Present on disk but not a checkpoint → a refusal, not a 404.
    const refused = await api.fetch('/api/models/not-a-checkpoint', { method: 'DELETE' });
    expect(refused.status).toBe(400);
    expect(((await refused.json()) as { error: string }).error).toMatch(/not a model checkpoint/i);
    expect(existsSync(join(modelsDir, 'not-a-checkpoint'))).toBe(true);
  });

  it('serves the catalog with install state', async () => {
    const res = await api.fetch('/api/catalog');
    expect(res.status).toBe(200);
    const body = (await res.json()) as { items: Array<{ slug: string; installed: boolean; hfRepo: string }> };
    expect(body.items.length).toBeGreaterThan(0);
    for (const item of body.items) expect(typeof item.installed).toBe('boolean');
  });
});

describe('dashboard api — sessions', () => {
  it('lists ingested sessions', async () => {
    await ingest();
    const res = await api.fetch('/api/sessions');
    expect(res.status).toBe(200);
    const body = (await res.json()) as { sessions: Array<{ id: string; name: string | null; models: string[] }> };
    const ids = body.sessions.map((s) => s.id);
    expect(ids).toContain('fix-1');
    expect(ids).toContain('fix-2');
  });

  it('reports the true total and honors limit/offset paging', async () => {
    await ingest();
    const all = (await (await api.fetch('/api/sessions')).json()) as {
      sessions: Array<{ id: string }>;
      total: number;
    };
    expect(all.total).toBe(2);
    expect(all.sessions).toHaveLength(2);

    // A capped page still reports the full match total (so the Overview tile is
    // accurate past the page size), and offset walks to the next row.
    const page = (await (await api.fetch('/api/sessions?limit=1')).json()) as {
      sessions: Array<{ id: string }>;
      total: number;
    };
    expect(page.total).toBe(2);
    expect(page.sessions).toHaveLength(1);

    const next = (await (await api.fetch('/api/sessions?limit=1&offset=1')).json()) as {
      sessions: Array<{ id: string }>;
      total: number;
    };
    expect(next.sessions).toHaveLength(1);
    expect(next.sessions[0].id).not.toBe(page.sessions[0].id);
  });

  it('serves a token total scoped to the sessions it counted, unlike the machine-wide metrics', async () => {
    // The Overview "Sessions" tile shows a count and a token total side by side.
    // The count can only ever describe THIS sessions root — ingest deletes rows
    // whose file left it — while `/api/metrics/overview` sums `traces`, a
    // machine-wide log under ~/.mlx-node/metrics/traces that no --session-dir
    // narrows. Serving the tokens beside the count is what makes the two halves
    // describable by one sentence.
    await ingest();
    const body = (await (await api.fetch('/api/sessions')).json()) as { total: number; tokens: number };
    expect(body.total).toBe(2);
    // fix-1: (100 + 50) + (180 + 60); fix-2: 40 + 20. The three fixture traces
    // are NOT delegated work — none carries a root_session_id pointing here — so
    // none of them belongs to these sessions.
    expect(body.tokens).toBe(450);

    const overview = (await (await api.fetch('/api/metrics/overview')).json()) as {
      totals: { inputTokens: number; outputTokens: number };
    };
    // The same window, machine-wide: the two orphan traces add 390 tokens that
    // belong to no session in this root. Pairing this number with `total` above
    // is the defect; the gap is exactly what the tile used to misreport.
    expect(overview.totals.inputTokens + overview.totals.outputTokens).toBe(840);
  });

  // The list token total and the per-session `/metrics` totals answer different
  // questions and MUST NOT be assumed equal: a fork copies its parent's turns
  // verbatim, so the list bills that shared history once while each session's own
  // page keeps its copy. Both numbers are pinned here so neither can drift into
  // the other's meaning unnoticed.
  it('bills inherited fork turns once in the list total, twice across the per-session ones', async () => {
    // fork-1 copies fix-1's m1..m4 VERBATIM (same entry ids, same mlxTraceId) and
    // adds one new inference of its own.
    const fork = [
      { type: 'session', version: 3, id: 'fork-1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' },
      {
        type: 'message',
        id: 'm1',
        parentId: null,
        timestamp: '2026-07-01T10:00:01.000Z',
        message: { role: 'user', content: 'Hello, world', timestamp: 1782036001000 },
      },
      {
        type: 'message',
        id: 'm2',
        parentId: 'm1',
        timestamp: '2026-07-01T10:00:02.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'Hi there' }],
          model: 'qwen3_5',
          usage: { input: 100, output: 50, cacheRead: 10, reasoning: 5 },
          timestamp: 1782036002000,
          mlxTraceId: 'trace-aaa',
        },
      },
      {
        type: 'message',
        id: 'm3',
        parentId: 'm2',
        timestamp: '2026-07-01T10:00:03.000Z',
        message: { role: 'user', content: 'Second question', timestamp: 1782036003000 },
      },
      {
        type: 'message',
        id: 'm4',
        parentId: 'm3',
        timestamp: '2026-07-01T10:00:04.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'An answer' }],
          model: 'qwen3_5',
          usage: { input: 180, output: 60, cacheRead: 100, reasoning: 12 },
          timestamp: 1782036004000,
        },
      },
      {
        type: 'message',
        id: 'm5',
        parentId: 'm4',
        timestamp: '2026-07-01T11:00:05.000Z',
        message: { role: 'user', content: 'Only on the fork', timestamp: 1782039605000 },
      },
      {
        type: 'message',
        id: 'm6',
        parentId: 'm5',
        timestamp: '2026-07-01T11:00:06.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'Fork-only answer' }],
          model: 'qwen3_5',
          usage: { input: 7, output: 3, cacheRead: 0, reasoning: 0 },
          timestamp: 1782039606000,
        },
      },
    ];
    writeFileSync(
      join(sessionsRoot, '--w--', '2026-07-01T11-00-00_fork-1.jsonl'),
      `${fork.map((l) => JSON.stringify(l)).join('\n')}\n`,
    );
    await ingest();

    const body = (await (await api.fetch('/api/sessions')).json()) as {
      total: number;
      tokens: number;
      sessions: Array<{ id: string; inputTokens: number; outputTokens: number }>;
    };
    expect(body.total).toBe(3);
    // Distinct inferences: trace-aaa (150) + entry m4 (240) — each held by BOTH
    // fix-1 and fork-1 but billed once — plus fix-2's n2 (60) and the fork's own
    // m6 (10).
    expect(body.tokens).toBe(460);

    // The per-row columns are raw per-session sums, so the fork's inherited
    // history shows up in both rows. Summing them is NOT the headline total.
    const byId = new Map(body.sessions.map((s) => [s.id, s.inputTokens + s.outputTokens]));
    expect(byId.get('fix-1')).toBe(390);
    expect(byId.get('fix-2')).toBe(60);
    expect(byId.get('fork-1')).toBe(400);
    const rowSum = [...byId.values()].reduce((a, b) => a + b, 0);
    expect(rowSum).toBe(850);

    // The per-session `/metrics` endpoint agrees with its own row, copy for copy.
    const perSession: number[] = [];
    for (const id of ['fix-1', 'fix-2', 'fork-1']) {
      const m = (await (await api.fetch(`/api/sessions/${id}/metrics`)).json()) as {
        turns: Array<{ inputTokens: number | null; outputTokens: number | null }>;
      };
      perSession.push(m.turns.reduce((sum, t) => sum + (t.inputTokens ?? 0) + (t.outputTokens ?? 0), 0));
    }
    expect(perSession).toEqual([390, 60, 400]);

    // The gap between the two readings is exactly the shared history (150 + 240),
    // counted once by the list and once more by the fork's own page.
    expect(rowSum - body.tokens).toBe(390);
  });

  it('reports no tokens for an empty sessions root while machine-wide metrics stay populated', async () => {
    // Same machine, same trace log, a --session-dir holding nothing. The session
    // list correctly says zero; the metrics overview still reports hundreds of
    // tokens. A tile that reads its headline from one and its subtitle from the
    // other renders "Sessions 0 · 530 tokens".
    const emptyRoot = join(base, 'empty-sessions');
    mkdirSync(emptyRoot, { recursive: true });
    const other = createDashboardRuntime({
      dbPath: ':memory:',
      sessionsRoot: emptyRoot,
      tracesDir,
      modelsDir,
      cacheRoot,
    });
    const otherApi = createTestClient(other);
    try {
      expect((await otherApi.fetch('/api/ingest', { method: 'POST' })).status).toBe(200);
      const body = (await (await otherApi.fetch('/api/sessions')).json()) as { total: number; tokens: number };
      expect(body.total).toBe(0);
      expect(body.tokens).toBe(0);

      const overview = (await (await otherApi.fetch('/api/metrics/overview')).json()) as {
        totals: { inputTokens: number; outputTokens: number };
      };
      expect(overview.totals.inputTokens + overview.totals.outputTokens).toBeGreaterThan(0);
    } finally {
      await other.close();
    }
  });

  it('returns a session detail with transcript text', async () => {
    await ingest();
    const res = await api.fetch('/api/sessions/fix-1');
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

  it('reports an invalid transcript rather than 500ing when a bare null precedes the header', async () => {
    // `null` is a syntactically valid JSONL record that `parseSessionEntries` keeps
    // verbatim, and `readSessionEntries` returns the array UNMIGRATED once any record
    // is a non-object. The detail handler then looked the header up with a raw
    // `.type` deref — before the topology gate that exists to reject exactly this —
    // so the one record whose deref throws produced a 500 carrying a raw TypeError
    // string. `isValidSessionTopology`'s own contract is to run "BEFORE any
    // `.type`/`.id` access"; this was the site that broke it.
    await ingest();
    const file = join(sessionsRoot, '--w--', '2026-07-01T10-00-00_fix-1.jsonl');
    writeFileSync(file, `null\n${readFileSync(file, 'utf8')}`);

    const res = await api.fetch('/api/sessions/fix-1');
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      session: { id: string };
      transcript: unknown[];
      transcriptError?: string;
    };
    // Not a 409: the header still names THIS row, so the file is this session,
    // merely malformed. The row's last-known-good metadata is still served — the
    // same answer every other non-object record already got.
    expect(body.session.id).toBe('fix-1');
    expect(body.transcript).toEqual([]);
    expect(body.transcriptError).toContain('Session tree is invalid');
  });

  it('answers a non-object record the same way whether or not its deref would throw', async () => {
    // The convention guard. `null` is the only first-record value whose `.type`
    // deref throws — `(123).type` is merely `undefined`, and an array IS an object
    // — so a fix that special-cases `null` into a 409 or a 404 would split one
    // malformed-file behaviour into two. Both must land on the same answer.
    await ingest();
    const file = join(sessionsRoot, '--w--', '2026-07-01T10-00-00_fix-1.jsonl');
    const original = readFileSync(file, 'utf8');
    for (const record of ['null', '123', '"str"', 'true', '[]']) {
      writeFileSync(file, `${record}\n${original}`);
      const res = await api.fetch('/api/sessions/fix-1');
      const body = (await res.json()) as { transcriptError?: string };
      expect([record, res.status]).toEqual([record, 200]);
      expect([record, body.transcriptError]).toEqual([
        record,
        'Session tree is invalid (cycle, duplicate id, or non-object message); transcript unavailable',
      ]);
    }
  });

  it('404s an unknown session', async () => {
    await ingest();
    const res = await api.fetch('/api/sessions/does-not-exist');
    expect(res.status).toBe(404);
  });

  it('renames a session (persists a session_info line and reflects the new name)', async () => {
    await ingest();
    const before = (await (await api.fetch('/api/sessions')).json()) as {
      sessions: Array<{ id: string; path: string }>;
    };
    const filePath = before.sessions.find((s) => s.id === 'fix-1')!.path;
    // The fixture file was just copied (recent mtime); age it past the liveness
    // window so this idle session renames instead of tripping the active-session
    // refusal (Finding 3).
    const old = Date.now() / 1000 - 600;
    utimesSync(filePath, old, old);

    const patch = await api.fetch('/api/sessions/fix-1', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'Renamed Session' }),
    });
    expect(patch.status).toBe(200);

    const detail = (await (await api.fetch('/api/sessions/fix-1')).json()) as {
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
    const before = (await (await api.fetch('/api/sessions')).json()) as {
      sessions: Array<{ id: string; path: string }>;
    };
    const filePath = before.sessions.find((s) => s.id === 'fix-1')!.path;

    // Replace fix-1's file on disk with a different session (bypassing ingest):
    // the index still points fix-1 at a path that now holds session 'reused-B'.
    writeFileSync(
      filePath,
      `${JSON.stringify({ type: 'session', version: 3, id: 'reused-B', timestamp: '2026-07-09T10:00:00.000Z', cwd: '/w' })}\n${JSON.stringify({ type: 'message', id: 'b1', parentId: null, timestamp: '2026-07-09T10:00:01.000Z', message: { role: 'user', content: 'from B' } })}\n`,
    );

    const patch = await api.fetch('/api/sessions/fix-1', {
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

  // Finding 3: renaming a session whose file has an incomplete trailing line must
  // be refused (409). SessionManager.open migrates + rewrites on construction,
  // persisting only the parsed entries, so opening would permanently truncate the
  // malformed trailing record. The file must be left byte-for-byte unchanged.
  it('refuses to rename a session with an incomplete trailing line and leaves the file byte-identical', async () => {
    const dir = join(sessionsRoot, '--w--');
    mkdirSync(dir, { recursive: true });
    const file = join(dir, '2026-07-12T10-00-00_rn-trunc.jsonl');
    const header = JSON.stringify({
      type: 'session',
      version: 3,
      id: 'rn-trunc',
      timestamp: '2026-07-12T10:00:00.000Z',
      cwd: '/w',
    });
    const user = JSON.stringify({
      type: 'message',
      id: 'm1',
      parentId: null,
      timestamp: '2026-07-12T10:00:01.000Z',
      message: { role: 'user', content: 'keep me' },
    });
    const asst = JSON.stringify({
      type: 'message',
      id: 'm2',
      parentId: 'm1',
      timestamp: '2026-07-12T10:00:02.000Z',
      message: {
        role: 'assistant',
        content: [{ type: 'text', text: 'and me' }],
        model: 'qwen3_5',
        usage: { input: 5, output: 6 },
      },
    });
    const truncated =
      '{"type":"message","id":"m3","parentId":"m2","timestamp":"2026-07-12T10:00:03.000Z","message":{"role":"asst';
    const original = `${header}\n${user}\n${asst}\n${truncated}`;
    writeFileSync(file, original);
    // Age it past the liveness window. Without this the file's fresh mtime trips
    // the ACTIVE-session refusal, which is also a 409 — so the assertion below
    // would pass even with the data-loss guard removed. Ageing it leaves the
    // data-loss guard as the only thing that can produce this 409.
    const old = Date.now() / 1000 - 600;
    utimesSync(file, old, old);
    await ingest();
    const before = readFileSync(file);

    const patch = await api.fetch('/api/sessions/rn-trunc', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'Should Not Apply' }),
    });
    expect(patch.status).toBe(409);

    // No write happened: byte-for-byte unchanged, no session_info stamped.
    const after = readFileSync(file);
    expect(after.equals(before)).toBe(true);
    expect(after.toString('utf-8')).toBe(original);
    expect(after.toString('utf-8')).not.toContain('"type":"session_info"');
    expect(after.toString('utf-8')).not.toContain('Should Not Apply');
  });

  // Finding 3 (liveness): renaming a session that appears actively written by a
  // live agent turn is refused (409). pi has no cross-process lock, so appending a
  // `session_info` to a file a concurrent turn is extending would race that turn.
  // An IDLE session (mtime well beyond the liveness window) renames normally.
  it('refuses to rename a session that looks active, but renames an idle one', async () => {
    const dir = join(sessionsRoot, '--w--');
    mkdirSync(dir, { recursive: true });
    const completeSession = (id: string, hour: string): object[] => [
      { type: 'session', version: 3, id, timestamp: `2026-07-13T${hour}:00:00.000Z`, cwd: '/w' },
      {
        type: 'message',
        id: 'u1',
        parentId: null,
        timestamp: `2026-07-13T${hour}:00:01.000Z`,
        message: { role: 'user', content: 'hi' },
      },
      {
        type: 'message',
        id: 'a1',
        parentId: 'u1',
        timestamp: `2026-07-13T${hour}:00:02.000Z`,
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'yo' }],
          model: 'qwen3_5',
          usage: { input: 5, output: 6 },
        },
      },
    ];

    // A complete, valid session with a very recent mtime → reaches (and trips) the
    // liveness pre-check rather than a records/identity 409.
    const liveFile = join(dir, '2026-07-13T10-00-00_rn-live.jsonl');
    writeFileSync(
      liveFile,
      `${completeSession('rn-live', '10')
        .map((l) => JSON.stringify(l))
        .join('\n')}\n`,
    );
    const now = Date.now() / 1000;
    utimesSync(liveFile, now, now);
    await ingest();
    const before = readFileSync(liveFile);

    const live = await api.fetch('/api/sessions/rn-live', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'Should Not Apply (active)' }),
    });
    expect(live.status).toBe(409);
    // The active session's file is byte-for-byte untouched — no session_info stamped.
    const after = readFileSync(liveFile);
    expect(after.equals(before)).toBe(true);
    expect(after.toString('utf-8')).not.toContain('"type":"session_info"');
    expect(after.toString('utf-8')).not.toContain('Should Not Apply');

    // An idle session (mtime well beyond the window) renames normally.
    const idleFile = join(dir, '2026-07-13T09-00-00_rn-idle.jsonl');
    writeFileSync(
      idleFile,
      `${completeSession('rn-idle', '09')
        .map((l) => JSON.stringify(l))
        .join('\n')}\n`,
    );
    const old = Date.now() / 1000 - 600; // 10 minutes ago
    utimesSync(idleFile, old, old);
    await ingest();

    const idle = await api.fetch('/api/sessions/rn-idle', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'Idle Renamed' }),
    });
    expect(idle.status).toBe(200);
    const detail = (await (await api.fetch('/api/sessions/rn-idle')).json()) as {
      session: { name: string | null };
    };
    expect(detail.session.name).toBe('Idle Renamed');
    expect(readFileSync(idleFile, 'utf-8')).toContain('Idle Renamed');
  });

  // Finding H: the detail transcript uses the same active-branch projection as
  // the index — a detached metadata leaf must not resurrect abandoned turns.
  it('detail transcript shows only the active branch under a detached metadata leaf', async () => {
    const forked = [
      { type: 'session', version: 3, id: 'detach-1', timestamp: '2026-07-08T10:00:00.000Z', cwd: '/w' },
      {
        type: 'message',
        id: 'u1',
        parentId: null,
        timestamp: '2026-07-08T10:00:01.000Z',
        message: { role: 'user', content: 'q' },
      },
      {
        type: 'message',
        id: 'a1',
        parentId: 'u1',
        timestamp: '2026-07-08T10:00:02.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'ABANDONED' }],
          model: 'gemma4',
          usage: { input: 999, output: 999 },
        },
      },
      {
        type: 'message',
        id: 'a2',
        parentId: 'u1',
        timestamp: '2026-07-08T10:00:03.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'ACTIVE' }],
          model: 'qwen3_5',
          usage: { input: 1, output: 2 },
        },
      },
      { type: 'session_info', id: 'si1', parentId: null, timestamp: '2026-07-08T10:00:04.000Z', name: 'Detached' },
    ];
    writeFileSync(
      join(sessionsRoot, '--w--', '2026-07-08T10-00-00_detach-1.jsonl'),
      `${forked.map((l) => JSON.stringify(l)).join('\n')}\n`,
    );
    await ingest();

    const body = (await (await api.fetch('/api/sessions/detach-1')).json()) as {
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
    const header = JSON.stringify({
      type: 'session',
      version: 1,
      id: 'ro-1',
      timestamp: '2026-07-10T10:00:00.000Z',
      cwd: '/w',
    });
    const user = JSON.stringify({
      type: 'message',
      timestamp: '2026-07-10T10:00:01.000Z',
      message: { role: 'user', content: 'READ ONLY hi' },
    });
    const asst = JSON.stringify({
      type: 'message',
      timestamp: '2026-07-10T10:00:02.000Z',
      message: {
        role: 'assistant',
        content: [{ type: 'text', text: 'READ ONLY yo' }],
        model: 'qwen3_5',
        usage: { input: 5, output: 6 },
      },
    });
    const truncated = '{"type":"message","message":{"role":"asst';
    const original = `${header}\n${user}\n${asst}\n${truncated}`;
    writeFileSync(file, original);
    await ingest();
    const before = readFileSync(file);

    const res = await api.fetch('/api/sessions/ro-1');
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
    const before = (await (await api.fetch('/api/sessions')).json()) as {
      sessions: Array<{ id: string; path: string }>;
    };
    const filePath = before.sessions.find((s) => s.id === 'fix-2')!.path;
    expect(existsSync(filePath)).toBe(true);

    const del = await api.fetch('/api/sessions/fix-2', { method: 'DELETE' });
    expect(del.status).toBe(200);
    expect(existsSync(filePath)).toBe(false);

    const after = (await (await api.fetch('/api/sessions')).json()) as { sessions: Array<{ id: string }> };
    expect(after.sessions.map((s) => s.id)).not.toContain('fix-2');
    expect((await api.fetch('/api/sessions/fix-2')).status).toBe(404);
  });

  // Delete carries NO liveness pre-check, unlike rename — see the note in
  // `handleSessionDelete`. A session written a moment ago (the just-finished or
  // just-crashed agent, the common cleanup case) must still delete outright rather
  // than 409 like the rename path does, so borrowing that window here is a
  // regression, not a hardening.
  it('deletes a session that was just written (no liveness refusal)', async () => {
    await ingest();
    const before = (await (await api.fetch('/api/sessions')).json()) as {
      sessions: Array<{ id: string; path: string }>;
    };
    const filePath = before.sessions.find((s) => s.id === 'fix-2')!.path;
    const now = Date.now() / 1000;
    utimesSync(filePath, now, now);

    const del = await api.fetch('/api/sessions/fix-2', { method: 'DELETE' });
    expect(del.status).toBe(200);
    expect(existsSync(filePath)).toBe(false);
    expect((await api.fetch('/api/sessions/fix-2')).status).toBe(404);
  });

  // A file that exists on disk but cannot be verified (unreadable / corrupt) must
  // NOT be half-deleted. The old code skipped the `rmSync` yet still dropped the
  // rows and replied `deleted`, so the transcript stayed on disk out of sync with
  // the DB: it could re-index on a later successful ingest (reappearing after we
  // said `deleted`) or linger as an orphan. Delete now refuses with 409, reconciles
  // the index, and leaves BOTH the row and the file intact. Reading is unrestricted
  // for root, so chmod 000 there is a no-op — skip.
  const canTestUnreadable = (process.getuid?.() ?? 0) !== 0;
  (canTestUnreadable ? it : it.skip)(
    'refuses to delete an unreadable session file, keeping the row and the file',
    async () => {
      await ingest();
      const before = (await (await api.fetch('/api/sessions')).json()) as {
        sessions: Array<{ id: string; path: string }>;
      };
      const filePath = before.sessions.find((s) => s.id === 'fix-2')!.path;

      // Unreadable in place: chmod touches ctime only, so mtime/size are unchanged
      // and the reconcile the handler runs cannot re-read it — its watermark keeps
      // the existing row rather than quarantining it.
      chmodSync(filePath, 0o000);
      try {
        const del = await api.fetch('/api/sessions/fix-2', { method: 'DELETE' });
        expect(del.status).toBe(409);
      } finally {
        chmodSync(filePath, 0o644); // restore so the assertions below (and cleanup) can read it
      }

      // Neither the file nor the row was removed.
      expect(existsSync(filePath)).toBe(true);
      expect((await api.fetch('/api/sessions/fix-2')).status).toBe(200);
    },
  );

  // The stale-row guard's legitimate case: a row whose PATH was reused by a NEWER
  // session must drop ONLY its own rows on delete — never `rmSync` the newer
  // session's file. This is `different`, not `unverifiable`, so it still succeeds.
  it('deletes only the stale rows when the path was reused by a newer session', async () => {
    await ingest();
    const before = (await (await api.fetch('/api/sessions')).json()) as {
      sessions: Array<{ id: string; path: string }>;
    };
    const filePath = before.sessions.find((s) => s.id === 'fix-2')!.path;

    // Overwrite fix-2's path with a brand-new VALID session (a different id), as if
    // pi reused the path — WITHOUT re-ingesting, so the stale fix-2 row still points
    // here and the delete must classify the file as `different`.
    const reused = [
      JSON.stringify({ type: 'session', version: 3, id: 'reused-B', timestamp: '2026-07-20T10:00:00.000Z', cwd: '/w' }),
      JSON.stringify({
        type: 'message',
        id: 'm1',
        parentId: null,
        timestamp: '2026-07-20T10:00:01.000Z',
        message: { role: 'user', content: 'I am the newer session' },
      }),
    ].join('\n');
    writeFileSync(filePath, `${reused}\n`);

    const del = await api.fetch('/api/sessions/fix-2', { method: 'DELETE' });
    expect(del.status).toBe(200);
    expect((await del.json()) as { deleted: boolean; id: string }).toMatchObject({ deleted: true, id: 'fix-2' });

    // The newer session's file was left untouched (content intact), and the stale
    // fix-2 row is gone; a fresh ingest then surfaces the newer session.
    expect(existsSync(filePath)).toBe(true);
    expect(readFileSync(filePath, 'utf-8')).toContain('reused-B');
    expect((await api.fetch('/api/sessions/fix-2')).status).toBe(404);
    await ingest();
    const after = (await (await api.fetch('/api/sessions')).json()) as { sessions: Array<{ id: string }> };
    expect(after.sessions.map((s) => s.id)).toContain('reused-B');
  });

  // A row whose file has already vanished must still be cleaned up — there is
  // nothing on disk left to orphan, so this is a plain success, not a conflict.
  it('deletes the rows when the session file is already gone', async () => {
    await ingest();
    const before = (await (await api.fetch('/api/sessions')).json()) as {
      sessions: Array<{ id: string; path: string }>;
    };
    const filePath = before.sessions.find((s) => s.id === 'fix-2')!.path;
    unlinkSync(filePath);

    const del = await api.fetch('/api/sessions/fix-2', { method: 'DELETE' });
    expect(del.status).toBe(200);
    expect((await del.json()) as { deleted: boolean; id: string }).toMatchObject({ deleted: true, id: 'fix-2' });
    expect((await api.fetch('/api/sessions/fix-2')).status).toBe(404);
  });

  it('renders image reads as thumbnails and raw-binary reads as chips (not garbage)', async () => {
    // A 1x1 PNG (base64) — the backend passes it through verbatim for the UI to inline.
    const png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';
    // Raw HEIC bytes a `read` returned verbatim in a text block: leading NUL control
    // bytes (→ classified binary) and the `ftypheic` magic (→ labelled HEIC).
    const nul = String.fromCharCode(0);
    const heicBytes = nul.repeat(3) + 'ftypheic' + nul.repeat(2) + 'mif1garbagebytes';
    const dir = join(sessionsRoot, '--imgproj--');
    mkdirSync(dir, { recursive: true });
    const lines = [
      { type: 'session', version: 3, id: 'img-1', timestamp: '2026-07-20T10:00:00.000Z', cwd: '/imgproj' },
      {
        type: 'message',
        id: 'u1',
        parentId: null,
        timestamp: '2026-07-20T10:00:01.000Z',
        message: { role: 'user', content: 'read the files' },
      },
      {
        type: 'message',
        id: 'r1',
        parentId: 'u1',
        timestamp: '2026-07-20T10:00:02.000Z',
        message: {
          role: 'toolResult',
          toolName: 'read',
          content: [
            { type: 'text', text: 'Read image file [image/png]' },
            {
              type: 'text',
              text: '[Image: original 6720x4480, displayed at 2000x1333. Multiply coordinates by 3.36 to map to original image.]',
            },
            { type: 'image', data: png, mimeType: 'image/png' },
          ],
        },
      },
      {
        type: 'message',
        id: 'r2',
        parentId: 'r1',
        timestamp: '2026-07-20T10:00:03.000Z',
        message: { role: 'toolResult', toolName: 'read', content: [{ type: 'text', text: heicBytes }] },
      },
    ];
    writeFileSync(join(dir, 'img.jsonl'), `${lines.map((l) => JSON.stringify(l)).join('\n')}\n`);
    await ingest();

    const detail = (await (await api.fetch('/api/sessions/img-1')).json()) as {
      transcript: Array<{
        role: string;
        text: string;
        images?: Array<{ mimeType: string; data: string }>;
        binaryNotes?: string[];
        imageNotes?: string[];
      }>;
    };

    const imageEntry = detail.transcript.find((e) => e.images !== undefined);
    expect(imageEntry?.images).toEqual([{ mimeType: 'image/png', data: png }]);
    // The coordinate-mapping note is split out of the rendered prose into imageNotes,
    // leaving only the plain action label — and never left inline in the text.
    expect(imageEntry?.text).toBe('Read image file [image/png]');
    expect(imageEntry?.imageNotes?.[0]).toMatch(/Multiply coordinates by 3\.36/);
    expect(imageEntry?.text ?? '').not.toContain('Multiply coordinates');

    const binaryEntry = detail.transcript.find((e) => e.binaryNotes !== undefined);
    expect(binaryEntry?.binaryNotes?.[0]).toMatch(/^HEIC · /);
    // The raw bytes are NOT dumped into rendered text.
    expect(binaryEntry?.text ?? '').toBe('');
    expect(JSON.stringify(detail.transcript)).not.toContain('garbagebytes');
  });

  it('summarizes tool-call args and joins them onto the result row title', async () => {
    const dir = join(sessionsRoot, '--callproj--');
    mkdirSync(dir, { recursive: true });
    const lines = [
      { type: 'session', version: 3, id: 'call-1', timestamp: '2026-07-20T10:00:00.000Z', cwd: '/callproj' },
      {
        type: 'message',
        id: 'u1',
        parentId: null,
        timestamp: '2026-07-20T10:00:01.000Z',
        message: { role: 'user', content: 'read it' },
      },
      {
        type: 'message',
        id: 'a1',
        parentId: 'u1',
        timestamp: '2026-07-20T10:00:02.000Z',
        message: {
          role: 'assistant',
          model: 'qwen3.6-27b-unsloth-mxfp4-mlx',
          content: [{ type: 'toolCall', id: 'call_abc', name: 'read', arguments: { path: '/callproj/src/lib.rs' } }],
        },
      },
      {
        type: 'message',
        id: 'r1',
        parentId: 'a1',
        timestamp: '2026-07-20T10:00:03.000Z',
        message: {
          role: 'toolResult',
          toolCallId: 'call_abc',
          toolName: 'read',
          content: [{ type: 'text', text: 'fn main() {}' }],
        },
      },
    ];
    writeFileSync(join(dir, 'call.jsonl'), `${lines.map((l) => JSON.stringify(l)).join('\n')}\n`);
    await ingest();

    const detail = (await (await api.fetch('/api/sessions/call-1')).json()) as {
      transcript: Array<{
        role: string;
        title?: string;
        toolName?: string;
        model?: string;
        toolCalls: Array<{ name: string; summary: string }>;
      }>;
    };

    // The assistant's tool call carries a one-line arg digest, and the entry
    // surfaces the model that produced it (for its per-model logo)…
    const callEntry = detail.transcript.find((e) => e.toolCalls.length > 0);
    expect(callEntry?.toolCalls[0]).toMatchObject({ name: 'read', summary: '/callproj/src/lib.rs' });
    expect(callEntry?.model).toBe('qwen3.6-27b-unsloth-mxfp4-mlx');
    // …and the result row recovers the same digest by joining on toolCallId.
    const resultEntry = detail.transcript.find((e) => e.role === 'toolResult' && e.toolName === 'read');
    expect(resultEntry?.title).toBe('/callproj/src/lib.rs');
  });

  it('renders the causal parent chain, not the wall clock', async () => {
    // A clock step backwards mid-session (an NTP correction) makes the tool result
    // and the closing reply older than the prompt that caused them. The branch walk
    // returns them in parent order; ordering by `ts` instead would render the result
    // card ABOVE the assistant bubble holding its call, and the reply above the
    // prompt — while the session list still shows the correct first message, because
    // the index derives `firstMessage` from the same chain WITHOUT sorting.
    //
    // The step-back is load-bearing: with a monotonic fixture this passes with or
    // without a sort, and would pin nothing.
    const dir = join(sessionsRoot, '--clockstep--');
    mkdirSync(dir, { recursive: true });
    const lines = [
      { type: 'session', version: 3, id: 'clock-1', timestamp: '2026-07-20T10:00:00.000Z', cwd: '/clockstep' },
      {
        type: 'message',
        id: 'u1',
        parentId: null,
        timestamp: '2026-07-20T10:00:10.000Z',
        message: { role: 'user', content: 'read it' },
      },
      {
        type: 'message',
        id: 'a1',
        parentId: 'u1',
        timestamp: '2026-07-20T10:00:11.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'toolCall', id: 'call_step', name: 'read', arguments: { path: '/clockstep/lib.rs' } }],
        },
      },
      {
        type: 'message',
        id: 'r1',
        parentId: 'a1',
        timestamp: '2026-07-20T10:00:05.000Z',
        message: {
          role: 'toolResult',
          toolCallId: 'call_step',
          toolName: 'read',
          content: [{ type: 'text', text: 'fn main() {}' }],
        },
      },
      {
        type: 'message',
        id: 'a2',
        parentId: 'r1',
        timestamp: '2026-07-20T10:00:06.000Z',
        message: { role: 'assistant', content: 'done' },
      },
    ];
    writeFileSync(join(dir, 'clock.jsonl'), `${lines.map((l) => JSON.stringify(l)).join('\n')}\n`);
    await ingest();

    const detail = (await (await api.fetch('/api/sessions/clock-1')).json()) as {
      session: { firstMessage: string | null };
      transcript: Array<{ role: string; text: string }>;
    };

    expect(detail.transcript.map((entry) => entry.role)).toEqual(['user', 'assistant', 'toolResult', 'assistant']);
    // The list row and the transcript must agree on where the conversation starts.
    expect(detail.session.firstMessage).toBe('read it');
    expect(detail.transcript[0]?.text).toBe('read it');
  });

  it('joins turns and traces for session metrics', async () => {
    await ingest();
    const res = await api.fetch('/api/sessions/fix-1/metrics');
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
    const res = await api.fetch('/api/sessions/fix-1/metrics');
    expect(res.status).toBe(200);
    const body = (await res.json()) as { traces: Array<{ traceId: string }> };
    const traceIds = body.traces.map((t) => t.traceId);
    // Both the root's own trace (session_id match) and the child's (root match).
    expect(traceIds).toContain('trace-aaa');
    expect(traceIds).toContain('trace-child');
  });

  // #7: a delegated subagent turn has a `traces` row (rootSessionId → root) but no
  // persisted `turns` row, so the counts/tokens/charts the SPA computes from the
  // `turns` array used to omit it — disagreeing with the child badge + throughput
  // it already shows (and with the global overview). The session metrics `turns`
  // set must UNION the trace-only child rows, net-of-cache and deduped (mirroring
  // the global fix in handleMetricsOverview).
  it('merges delegated subagent turns into the session metric turn set (net-of-cache, deduped)', async () => {
    writeFileSync(
      join(tracesDir, '2026-07-03-delegated.jsonl'),
      `${JSON.stringify({
        v: 1,
        traceId: 'trace-deleg',
        ts: 1782036402000,
        sessionId: 'child-session',
        rootSessionId: 'fix-1',
        model: 'child-model',
        durationMs: 20,
        finishReason: 'stop',
        promptTokens: 300,
        cachedTokens: 40,
        outputTokens: 90,
        reasoningTokens: 9,
        ttftMs: 55.5,
        decodeTps: 42,
      })}\n`,
    );
    await ingest();
    const res = await api.fetch('/api/sessions/fix-1/metrics');
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      turns: Array<{
        entryId: string | null;
        traceId: string | null;
        inputTokens: number | null;
        outputTokens: number | null;
        cachedTokens: number | null;
        reasoningTokens: number | null;
        model: string | null;
        ttftMs: number | null;
        decodeTps: number | null;
      }>;
    };

    const child = body.turns.filter((t) => t.traceId === 'trace-deleg');
    // Exactly one merged row for the delegated turn (deduped by trace_id).
    expect(child).toHaveLength(1);
    // Tokens are NET of cache (300 gross prompt − 40 cached = 260), never gross.
    expect(child[0].inputTokens).toBe(260);
    expect(child[0].outputTokens).toBe(90);
    expect(child[0].cachedTokens).toBe(40);
    expect(child[0].reasoningTokens).toBe(9);
    expect(child[0].model).toBe('child-model');
    // No persisted turn row → no entryId; its trace-derived fields are carried.
    expect(child[0].entryId).toBeNull();
    expect(child[0].ttftMs).toBeCloseTo(55.5, 1);
    expect(child[0].decodeTps).toBeCloseTo(42, 1);

    // The root's own persisted turn (trace-aaa) stays present exactly once — the
    // dedup excludes only trace-only rows, never a correlated turn.
    expect(body.turns.filter((t) => t.traceId === 'trace-aaa')).toHaveLength(1);
  });

  // #3 (R15): the trace-only UNION must admit ONLY genuine delegated children, never
  // resurrect an ABANDONED root turn. When a root session branches, ingestion drops
  // its abandoned assistant turn from `turns`, yet the trace lingers with
  // session_id == root_session_id == the root id. The old `session_id = ? OR
  // root_session_id = ?` predicate re-admitted it (it passes the trace-only dedup —
  // no longer in `turns`) and counted it as a fake delegated turn, inflating the
  // turn count and token totals. A genuine child runs on its own in-memory session,
  // so its trace has session_id = <child id> != root; the fixed predicate keys on
  // `root_session_id = ? AND session_id != ?`.
  it('excludes an abandoned root trace (session_id = root) but keeps genuine children', async () => {
    writeFileSync(
      join(tracesDir, '2026-07-04-branch.jsonl'),
      `${[
        // (b) ABANDONED root turn: no surviving `turns` row; trace lingers with
        // session_id == root_session_id == fix-1. Must NOT be counted.
        {
          v: 1,
          traceId: 'trace-abandoned-root',
          ts: 1782036502000,
          sessionId: 'fix-1',
          rootSessionId: 'fix-1',
          model: 'ghost-model',
          durationMs: 30,
          finishReason: 'stop',
          promptTokens: 9999,
          cachedTokens: 0,
          outputTokens: 8888,
          reasoningTokens: 0,
        },
        // (c) GENUINE delegated child: own in-memory session id != root.
        {
          v: 1,
          traceId: 'trace-real-child',
          ts: 1782036503000,
          sessionId: 'child-of-fix-1',
          rootSessionId: 'fix-1',
          model: 'child-model',
          durationMs: 20,
          finishReason: 'stop',
          promptTokens: 100,
          cachedTokens: 0,
          outputTokens: 40,
          reasoningTokens: 0,
        },
      ]
        .map((r) => JSON.stringify(r))
        .join('\n')}\n`,
    );
    await ingest();
    const res = await api.fetch('/api/sessions/fix-1/metrics');
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      turns: Array<{ traceId: string | null; inputTokens: number | null; outputTokens: number | null }>;
      traces: Array<{ traceId: string }>;
    };
    const ids = body.turns.map((t) => t.traceId);

    // The active root turn (trace-aaa) stays present, the genuine child is merged
    // exactly once.
    expect(body.turns.filter((t) => t.traceId === 'trace-aaa')).toHaveLength(1);
    expect(body.turns.filter((t) => t.traceId === 'trace-real-child')).toHaveLength(1);
    // The abandoned root trace is NEVER resurrected as a delegated turn.
    expect(ids).not.toContain('trace-abandoned-root');
    // Its inflated tokens never reach the session totals.
    const totalOut = body.turns.reduce((sum, t) => sum + (t.outputTokens ?? 0), 0);
    const totalIn = body.turns.reduce((sum, t) => sum + (t.inputTokens ?? 0), 0);
    expect(totalOut).toBeLessThan(8888);
    expect(totalIn).toBeLessThan(9999);

    // A12: the SAME abandoned trace must also be excluded from the `traces` array
    // the SPA derives model badges + avg TTFT/decode chips from — mirroring the
    // turns filter — so those chips agree with the transcript / turn charts. The
    // active root trace and the genuine delegated child remain.
    const traceIds = body.traces.map((t) => t.traceId);
    expect(traceIds).toContain('trace-aaa');
    expect(traceIds).toContain('trace-real-child');
    expect(traceIds).not.toContain('trace-abandoned-root');
  });
});

// The Sessions page's directory dropdown must offer every indexed directory, not
// only the ones that happen to appear on the page of rows it was served. The list
// is capped at 500 rows, so a directory whose sessions are all older than that cap
// would be missing from the dropdown — and the page's own footnote tells the user
// to reach older sessions with exactly that filter. The fixture is deliberately
// larger than the cap: below 501 rows nothing distinguishes the two.
describe('dashboard server — session directories past the page cap', () => {
  const OLD_DIR = '/tmp/only-old';
  const NEW_DIR = '/tmp/recent';

  /** Fake context over an in-memory index holding one row past the 500-row page cap. */
  function seededCtx(): { ctx: ApiContext; close: () => void } {
    const dash = openDashboardDb(':memory:');
    const insert = dash.sqlite.prepare(
      'INSERT INTO sessions (id, path, cwd, name, created, modified, message_count) VALUES (?, ?, ?, NULL, ?, ?, 1)',
    );
    // One session in OLD_DIR, older than every other row, then 500 newer ones in
    // NEW_DIR — exactly enough to push the old row off the served page.
    insert.run('old-1', `${OLD_DIR}/old-1.jsonl`, OLD_DIR, 1_000, 1_000);
    for (let i = 0; i < 500; i++) {
      insert.run(`new-${i}`, `${NEW_DIR}/new-${i}.jsonl`, NEW_DIR, 10_000 + i, 10_000 + i);
    }
    return { ctx: { dash } as unknown as ApiContext, close: dash.close };
  }

  it('serves every indexed directory even when its sessions fall outside the served page', async () => {
    const { ctx, close } = seededCtx();
    try {
      const res = await call(ctx, 'GET', '/api/sessions');
      expect(res.ok).toBe(true);
      const body = (res.ok ? res.body : null) as {
        sessions: Array<{ cwd: string }>;
        total: number;
        cwds: string[];
      };

      // The fixture really does exercise the cap: 501 matches, 500 rows served,
      // and the old directory appears on none of them.
      expect(body.total).toBe(501);
      expect(body.sessions).toHaveLength(500);
      expect(body.sessions.map((s) => s.cwd)).not.toContain(OLD_DIR);

      // …yet the directory list, which the dropdown is built from, still names it.
      expect(body.cwds).toContain(OLD_DIR);
      expect(body.cwds).toContain(NEW_DIR);
    } finally {
      close();
    }
  });

  it('keeps the directory list unfiltered so a chosen directory never hides the others', async () => {
    const { ctx, close } = seededCtx();
    try {
      const res = await dispatch(ctx, {
        method: 'GET',
        pathname: '/api/sessions',
        query: new URLSearchParams({ cwd: NEW_DIR }),
        body: null,
      });
      const body = (res.ok ? res.body : null) as { total: number; cwds: string[] };
      expect(body.total).toBe(500);
      // Narrowing to one directory must not strand the user there.
      expect(body.cwds).toEqual([NEW_DIR, OLD_DIR].sort((a, b) => (a < b ? -1 : 1)));
    } finally {
      close();
    }
  });
});

// Finding 4: session-file symlink containment. Primary — a symlinked transcript
// is never indexed, so its external id is simply unknown (404). Defense-in-depth
// — a GET whose indexed path resolves outside the managed root (via a symlink
// swapped in after indexing) is refused (403) by the realpath containment guard.
describe('dashboard api — session symlink containment (Finding 4)', () => {
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

    expect((await api.fetch('/api/sessions/external-1')).status).toBe(404);
    const patch = await api.fetch('/api/sessions/external-1', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'x' }),
    });
    expect(patch.status).toBe(404);
  });

  it('refuses a GET whose indexed file was swapped to an external symlink', async () => {
    await ingest();
    const before = (await (await api.fetch('/api/sessions')).json()) as {
      sessions: Array<{ id: string; path: string }>;
    };
    const filePath = before.sessions.find((s) => s.id === 'fix-1')!.path;

    // Swap the indexed real file for a symlink pointing outside the managed root.
    const externalFile = join(base, 'external-detail.jsonl');
    writeFileSync(externalFile, readFileSync(filePath));
    unlinkSync(filePath);
    symlinkSync(externalFile, filePath);

    // The row still points at filePath, but it now resolves outside the root.
    expect((await api.fetch('/api/sessions/fix-1')).status).toBe(403);
  });

  // Finding 5: an IN-ROOT swap passes realpath containment (the target is a real
  // session inside the root), so the detail handler must additionally require the
  // parsed header id to still be THIS row. A file (or in-root symlink) that now
  // holds a different session → 409, and reconciling the stale row → 404 next.
  it('409s a GET whose indexed file resolves in-root to a different session, then drops the row', async () => {
    await ingest();
    const before = (await (await api.fetch('/api/sessions')).json()) as {
      sessions: Array<{ id: string; path: string }>;
    };
    const aPath = before.sessions.find((s) => s.id === 'fix-1')!.path;

    // A genuine in-root session B (the symlink target — a regular file in the root).
    const bPath = join(sessionsRoot, '--w--', '2026-07-11T10-00-00_sess-B.jsonl');
    writeFileSync(
      bPath,
      `${[
        { type: 'session', version: 3, id: 'sess-B', timestamp: '2026-07-11T10:00:00.000Z', cwd: '/w' },
        {
          type: 'message',
          id: 'b1',
          parentId: null,
          timestamp: '2026-07-11T10:00:01.000Z',
          message: { role: 'user', content: 'B in root' },
        },
      ]
        .map((l) => JSON.stringify(l))
        .join('\n')}\n`,
    );

    // Swap A's real file for an in-root symlink to B: containment passes, header id won't.
    unlinkSync(aPath);
    symlinkSync(bPath, aPath);

    // fix-1's row resolves (in-root) to B, whose header id is sess-B, not fix-1 → 409.
    expect((await api.fetch('/api/sessions/fix-1')).status).toBe(409);
    // The 409 path reconciles: A's row (now a symlink, not a regular file) is dropped.
    expect((await api.fetch('/api/sessions/fix-1')).status).toBe(404);
    // B itself indexes and serves normally.
    expect((await api.fetch('/api/sessions/sess-B')).status).toBe(200);
  });
});

describe('dashboard api — ingest fault isolation', () => {
  // chmod 000 is a no-op for root, so the unreadable root would still list — skip.
  const canTestUnreadable = (process.getuid?.() ?? 0) !== 0;

  (canTestUnreadable ? it : it.skip)('still ingests traces when the session root is unreadable', async () => {
    // `doIngest` wraps BOTH halves in one try/catch, so a throw out of the session
    // scan means the trace ingest on the next line never runs: every periodic pass
    // fails identically, `/api/ingest` reports all-zero with a 200, and the metrics
    // pages go stale while the 30-day trace prune (which lives inside the trace
    // ingest) freezes. The session scan must degrade to a warning instead.
    await ingest();

    writeFileSync(
      join(tracesDir, 'late.jsonl'),
      `${JSON.stringify({
        v: 1,
        traceId: 'trace-after-chmod',
        ts: 1782039999000,
        model: 'qwen3_5',
        durationMs: 10,
        finishReason: 'stop',
        promptTokens: 1,
        cachedTokens: 0,
        outputTokens: 1,
        reasoningTokens: 0,
      })}\n`,
    );

    chmodSync(sessionsRoot, 0o000);
    try {
      const res = await api.fetch('/api/ingest', { method: 'POST' });
      expect(res.status).toBe(200);
      const body = (await res.json()) as {
        sessions: { removed: number; warnings: string[] };
        traces: { files: number; records: number };
      };
      // The trace half ran to completion and picked up the file written above.
      expect(body.traces.files).toBeGreaterThan(0);
      expect(body.traces.records).toBeGreaterThan(0);
      // The session half reported a skip, not a scan, and deleted nothing.
      expect(body.sessions.warnings.some((w) => w.includes('scan skipped'))).toBe(true);
      expect(body.sessions.removed).toBe(0);

      // An unreadable root is unknown, not empty: the indexed sessions survive it.
      const list = (await (await api.fetch('/api/sessions')).json()) as { total: number };
      expect(list.total).toBe(2);
    } finally {
      chmodSync(sessionsRoot, 0o755); // restore so afterEach cleanup can remove it
    }
  });
});

describe('dashboard api — metrics overview', () => {
  it('returns aggregate arrays', async () => {
    await ingest();
    const res = await api.fetch('/api/metrics/overview');
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

  // Finding 4: a forked session copies fix-1's turns verbatim (same entry ids and
  // mlxTraceId) into a new session file. The per-session views keep every copy,
  // but the GLOBAL overview must count each inference once.
  it('counts a forked inference once in the global overview, not per copy', async () => {
    const fork = [
      { type: 'session', version: 3, id: 'fork-1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' },
      {
        type: 'message',
        id: 'm1',
        parentId: null,
        timestamp: '2026-07-01T10:00:01.000Z',
        message: { role: 'user', content: 'Hello, world', timestamp: 1782036001000 },
      },
      {
        type: 'message',
        id: 'm2',
        parentId: 'm1',
        timestamp: '2026-07-01T10:00:02.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'Hi there' }],
          model: 'qwen3_5',
          usage: { input: 100, output: 50, cacheRead: 10, reasoning: 5 },
          timestamp: 1782036002000,
          mlxTraceId: 'trace-aaa',
        },
      },
      {
        type: 'message',
        id: 'm3',
        parentId: 'm2',
        timestamp: '2026-07-01T10:00:03.000Z',
        message: { role: 'user', content: 'Second question', timestamp: 1782036003000 },
      },
      {
        type: 'message',
        id: 'm4',
        parentId: 'm3',
        timestamp: '2026-07-01T10:00:04.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'An answer' }],
          model: 'qwen3_5',
          usage: { input: 180, output: 60, cacheRead: 100, reasoning: 12 },
          timestamp: 1782036004000,
        },
      },
    ];
    writeFileSync(
      join(sessionsRoot, '--w--', '2026-07-01T11-00-00_fork-1.jsonl'),
      `${fork.map((l) => JSON.stringify(l)).join('\n')}\n`,
    );
    await ingest();

    const body = (await (await api.fetch('/api/metrics/overview')).json()) as {
      modelShare: Array<{ model: string; turns: number; outputTokens: number }>;
      totals: {
        turns: number;
        inputTokens: number;
        outputTokens: number;
        cachedTokens: number;
        reasoningTokens: number;
      };
    };
    // fix-1 (2 turns) + fix-2 (1) + fork-1 (2 verbatim copies of fix-1) → 3 distinct
    // turns, PLUS the two trace-only fixture rows that carry no correlating `turns`
    // row (trace-bbb=gemma4, trace-ccc=lfm2), which the F3 union counts as delegated
    // subagent turns → 5 total.
    expect(body.totals.turns).toBe(5);
    expect(body.totals.inputTokens).toBe(570); // 100 + 180 + 40 (fork collapsed) + 200 + 50 (trace-only, net of cache)
    expect(body.totals.outputTokens).toBe(270); // 50 + 60 + 20 + 128 + 12
    expect(body.totals.cachedTokens).toBe(135); // 10 + 100 + 0 + 0 + 25
    expect(body.totals.reasoningTokens).toBe(17); // 5 + 12 + 0 + 0 + 0

    // Distinct real inferences per model stay separate: qwen3_5's forked copies
    // collapse to 2 (trace-aaa correlates, so it is not double-counted); gemma4 is
    // the n2 turn (20) plus the trace-only trace-bbb (128) → 2 turns / 148 output.
    const qwen = body.modelShare.find((m) => m.model === 'qwen3_5');
    const gemma = body.modelShare.find((m) => m.model === 'gemma4');
    expect(qwen?.turns).toBe(2);
    expect(qwen?.outputTokens).toBe(110);
    expect(gemma?.turns).toBe(2);
    expect(gemma?.outputTokens).toBe(148);
  });

  // F3: subagent turns run on an in-memory session manager → no session JSONL → no
  // `turns` row, but the shared provider still writes a `traces` row carrying token
  // columns. Those trace-only rows must be UNIONed into tokensByDay/modelShare/
  // turnTotals (which otherwise read `turns` only), each counted ONCE and never
  // double-counting a normal turn whose trace correlates 1:1 to a `turns` row.
  it('includes trace-only (subagent) turns in token aggregates without double-counting', async () => {
    const dayStart = Date.parse('2026-07-05T00:00:00.000Z');
    const dayEnd = Date.parse('2026-07-05T23:59:59.999Z');
    const midMs = Date.parse('2026-07-05T12:00:00.000Z');

    // A NORMAL turn: a persisted assistant turn whose trace correlates 1:1 (same
    // trace_id `corr-1`). Its authoritative tokens live in `turns`.
    const root = [
      { type: 'session', version: 3, id: 'sub-root', timestamp: '2026-07-05T12:00:00.000Z', cwd: '/w' },
      {
        type: 'message',
        id: 'sm1',
        parentId: null,
        timestamp: '2026-07-05T12:00:00.500Z',
        message: { role: 'user', content: 'root question', timestamp: midMs },
      },
      {
        type: 'message',
        id: 'sm2',
        parentId: 'sm1',
        timestamp: '2026-07-05T12:00:01.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'root answer' }],
          model: 'corr-model',
          usage: { input: 300, output: 90, cacheRead: 30, reasoning: 9 },
          timestamp: midMs,
          mlxTraceId: 'corr-1',
        },
      },
    ];
    writeFileSync(
      join(sessionsRoot, '--w--', '2026-07-05T12-00-00_sub-root.jsonl'),
      `${root.map((l) => JSON.stringify(l)).join('\n')}\n`,
    );

    // Two trace rows on the same day: `corr-1` correlates to the turn above (must NOT
    // double-count), `sub-1` is a subagent (trace-only: no `turns` row references it).
    writeFileSync(
      join(tracesDir, '2026-07-05-subagent.jsonl'),
      `${JSON.stringify({ v: 1, traceId: 'corr-1', ts: midMs, sessionId: 'sub-root', model: 'corr-model', promptTokens: 300, cachedTokens: 30, outputTokens: 90, reasoningTokens: 9 })}\n${JSON.stringify({ v: 1, traceId: 'sub-1', ts: midMs, rootSessionId: 'sub-root', model: 'sub-model', promptTokens: 500, cachedTokens: 40, outputTokens: 200, reasoningTokens: 15 })}\n`,
    );
    await ingest();

    // Scope to the seeded day so totals are deterministic regardless of other fixtures.
    const body = (await (await api.fetch(`/api/metrics/overview?from=${dayStart}&to=${dayEnd}`)).json()) as {
      tokensByDay: Array<{ day: string; input: number; output: number; cached: number; reasoning: number }>;
      modelShare: Array<{ model: string; turns: number; outputTokens: number }>;
      totals: {
        turns: number;
        inputTokens: number;
        outputTokens: number;
        cachedTokens: number;
        reasoningTokens: number;
      };
    };

    // turnTotals: normal turn (300 net/90/30/9) + trace-only sub-1 (500 gross −40
    // cache = 460 net / 200 / 40 / 15), each once. inputTokens is NET of cache on
    // both sides: turns store `usage.input` (already net), traces store gross
    // `promptTokens`, so the trace side clamps `MAX(prompt − cached, 0)`.
    expect(body.totals.turns).toBe(2);
    expect(body.totals.inputTokens).toBe(760); // 300 (net) + 460 (net of cache)
    expect(body.totals.outputTokens).toBe(290);
    expect(body.totals.cachedTokens).toBe(70);
    expect(body.totals.reasoningTokens).toBe(24);

    // tokensByDay: one 2026-07-05 bucket with the combined tokens (correlated once).
    const bucket = body.tokensByDay.find((d) => d.day === '2026-07-05');
    expect(bucket).toEqual({ day: '2026-07-05', input: 760, output: 290, cached: 70, reasoning: 24 });

    // modelShare: the correlated turn counted ONCE under corr-model (90, not 180),
    // and the trace-only subagent surfaced under sub-model.
    const corr = body.modelShare.find((m) => m.model === 'corr-model');
    const sub = body.modelShare.find((m) => m.model === 'sub-model');
    expect(corr).toEqual({ model: 'corr-model', turns: 1, outputTokens: 90 });
    expect(sub).toEqual({ model: 'sub-model', turns: 1, outputTokens: 200 });
  });

  // Finding 9-query: the overview must expose a day-bucketed throughput/TTFT trend
  // per model — one row per (model, day) with that day's averages — not just the
  // single range-wide average per model in `throughputByModel`.
  it('returns a per-model, per-day throughput trend, not one range-wide average', async () => {
    const day1 = Date.parse('2026-07-01T12:00:00.000Z');
    const day2 = Date.parse('2026-07-02T12:00:00.000Z');
    writeFileSync(
      join(tracesDir, '2026-07-02-trend.jsonl'),
      `${JSON.stringify({ v: 1, traceId: 'trend-d1', ts: day1, model: 'trend-model', decodeTps: 100, prefillTps: 500, ttftMs: 20 })}\n${JSON.stringify({ v: 1, traceId: 'trend-d2', ts: day2, model: 'trend-model', decodeTps: 200, prefillTps: 700, ttftMs: 40 })}\n`,
    );
    await ingest();

    const body = (await (await api.fetch('/api/metrics/overview')).json()) as {
      throughputTrend: Array<{
        model: string;
        day: string;
        decodeTps: number;
        prefillTps: number;
        ttftMs: number;
        samples: number;
      }>;
    };
    const rows = body.throughputTrend.filter((r) => r.model === 'trend-model');
    // One row per day (not a single averaged 150), ordered by day.
    expect(rows).toHaveLength(2);
    expect(rows[0].day).toBe('2026-07-01');
    expect(rows[1].day).toBe('2026-07-02');
    const d1 = rows.find((r) => r.day === '2026-07-01');
    const d2 = rows.find((r) => r.day === '2026-07-02');
    expect(d1?.decodeTps).toBe(100);
    expect(d1?.prefillTps).toBe(500);
    expect(d1?.ttftMs).toBe(20);
    expect(d1?.samples).toBe(1);
    expect(d2?.decodeTps).toBe(200);
    expect(d2?.samples).toBe(1);
  });

  // Finding F4: the trace perf columns (ttft_ms/prefill_tps/decode_tps) are
  // nullable and legitimately null for a valid trace recorded without timing.
  // A bucket whose traces carry no perf → AVG over an all-NULL group is NULL and
  // MUST surface as `null`, not a fabricated `0` (which would plot a fake
  // `0 tok/s` point and defeat the UI's `Number.isFinite` empty-state guard). A
  // sibling bucket that DOES carry perf must be unchanged (finite values).
  it('reports null (not 0) throughput for a bucket whose traces carry no perf', async () => {
    const day = Date.parse('2026-07-02T12:00:00.000Z');
    writeFileSync(
      join(tracesDir, '2026-07-02-nullperf.jsonl'),
      `${JSON.stringify({ v: 1, traceId: 'np-perf', ts: day, model: 'perf-model', decodeTps: 150, prefillTps: 600, ttftMs: 30 })}\n${JSON.stringify({ v: 1, traceId: 'np-null', ts: day, model: 'nullperf-model' })}\n`,
    );
    await ingest();

    const body = (await (await api.fetch('/api/metrics/overview')).json()) as {
      throughputTrend: Array<{
        model: string;
        day: string;
        decodeTps: number | null;
        prefillTps: number | null;
        ttftMs: number | null;
        samples: number;
      }>;
    };

    // The bucket WITH perf keeps its finite averages, unchanged.
    const perf = body.throughputTrend.find((r) => r.model === 'perf-model');
    expect(perf?.decodeTps).toBe(150);
    expect(perf?.prefillTps).toBe(600);
    expect(perf?.ttftMs).toBe(30);
    expect(perf?.samples).toBe(1);

    // The bucket with only null-perf traces: COUNT(*) still counts the row (the
    // bucket exists), but every AVG is NULL → surfaced as `null`, never `0`.
    const nullPerf = body.throughputTrend.find((r) => r.model === 'nullperf-model');
    expect(nullPerf).toBeDefined();
    expect(nullPerf?.samples).toBe(1);
    expect(nullPerf?.decodeTps).toBeNull();
    expect(nullPerf?.prefillTps).toBeNull();
    expect(nullPerf?.ttftMs).toBeNull();
    // Guard against the old `?? 0` regression re-appearing.
    expect(nullPerf?.decodeTps).not.toBe(0);
    expect(nullPerf?.prefillTps).not.toBe(0);
    expect(nullPerf?.ttftMs).not.toBe(0);
  });
});

describe('dashboard api — cache', () => {
  it('scans and clears the cold cache with an explicit {all:true}', async () => {
    const res = await api.fetch('/api/cache');
    expect(res.status).toBe(200);
    const body = (await res.json()) as { disk: { entryCount: number; totalBytes: number } };
    expect(body.disk.entryCount).toBe(2);
    expect(body.disk.totalBytes).toBe(300);

    const del = await api.fetch('/api/cache', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ all: true }),
    });
    expect(del.status).toBe(200);
    const cleared = (await del.json()) as { removed: number; freedBytes: number };
    expect(cleared.removed).toBe(2);
    expect(cleared.freedBytes).toBe(300);

    const after = (await (await api.fetch('/api/cache')).json()) as { disk: { entryCount: number } };
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
      const del = await api.fetch('/api/cache', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        ...(b === undefined ? {} : { body: b }),
      });
      expect(del.status).toBe(400);
      // Nothing was cleared — both blocks survive.
      const after = (await (await api.fetch('/api/cache')).json()) as { disk: { entryCount: number } };
      expect(after.disk.entryCount).toBe(2);
    }
  });

  it('evicts only blocks older than a positive olderThanDays', async () => {
    // Age one block past the 7-day cutoff; the other stays recent.
    const oldBlock = join(cacheRoot, hexBlock(1));
    const oldSec = (Date.now() - 10 * 86_400_000) / 1000;
    utimesSync(oldBlock, oldSec, oldSec);

    const del = await api.fetch('/api/cache', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ olderThanDays: 7 }),
    });
    expect(del.status).toBe(200);
    const evicted = (await del.json()) as { removed: number; freedBytes: number };
    expect(evicted.removed).toBe(1);
    expect(evicted.freedBytes).toBe(100);

    const after = (await (await api.fetch('/api/cache')).json()) as { disk: { entryCount: number } };
    expect(after.disk.entryCount).toBe(1);
  });
});

/**
 * F1 — the Cache page's hit rate and trend must describe THE CACHE IT IS
 * SHOWING.
 *
 * Before the root filter, `disk` was a filesystem scan of one root while
 * `trend` was an unfiltered `SUM()` over every trace row ever ingested: a
 * dashboard pointed at an empty cache dir reported `exists: false,
 * entryCount: 0` beside a hit rate summed across every cache directory the
 * machine had ever used. Nothing in the suite could have caught it — every
 * existing /api/cache assertion read `body.disk.*` and none had ever touched
 * `trend`.
 *
 * These fixtures deliberately make the WRONG answer much larger than the right
 * one, so a regression cannot hide inside a plausible-looking number.
 */
describe('dashboard api — cache trend scoping', () => {
  /** Canonical spelling of the fixture cache root, resolved independently of the
   *  implementation (`/var/folders/...` → `/private/var/folders/...` on macOS). */
  let canonicalCacheRootPath: string;

  /** Append one trace JSONL line shaped like `MetricsTrace.record()` output. */
  function writeTraceFile(name: string, records: Array<Record<string, unknown>>): void {
    const lines = records.map((r) => JSON.stringify({ v: 1, model: 'm', durationMs: 1, finishReason: 'stop', ...r }));
    writeFileSync(join(tracesDir, name), `${lines.join('\n')}\n`);
  }

  interface CacheBody {
    disk: { root: string; entryCount: number };
    trend: Array<{ day: string; hits: number; misses: number; bytesWritten: number; bytesRestored: number }>;
    scope: {
      root: string;
      trendWindowDays: number;
      legacy: { turns: number; hits: number; misses: number };
      otherRoots: { turns: number; hits: number; misses: number };
      unattributed: { turns: number; hits: number; misses: number };
      disabledTurns: number;
      unrootedSidecarCaptures: number;
    };
    health: {
      enqueued: number;
      queueDrops: number;
      evictions: number;
      corruptions: number;
      corruptionsTotal: number;
      queueDropsTotal: number;
      writeErrors: number;
      writeErrorsTotal: number;
      restoreDeclines: number;
      restoreSuppressed: number;
      sidecarCaptureReached: number;
      sidecarChainEmpty: number;
      sidecarBoundarySkips: number;
      sidecarAlreadyPersisted: number;
      sidecarEnqueued: number;
      sidecarQueueDrops: number;
      sidecarInstalled: number;
    };
    restoreFamilies: string[];
  }

  function totals(body: CacheBody): { hits: number; misses: number } {
    return body.trend.reduce((a, r) => ({ hits: a.hits + r.hits, misses: a.misses + r.misses }), {
      hits: 0,
      misses: 0,
    });
  }

  beforeEach(async () => {
    canonicalCacheRootPath = realpathSync(cacheRoot);
    // Start from an empty trace dir: the shared fixtures carry their own
    // (legitimately unattributed) cold rows, and this suite asserts exact
    // totals so a fixture edit can never quietly move them.
    for (const n of readdirSync(tracesDir)) if (n.endsWith('.jsonl')) unlinkSync(join(tracesDir, n));
    // 11 hits / 3 misses belong to THIS cache; 9999 hits belong to another cache
    // dir; 5000 hits predate attribution entirely. Anything unscoped reads 15010.
    writeTraceFile('2026-07-20-1.jsonl', [
      {
        traceId: 'mine-1',
        ts: Date.UTC(2026, 6, 20, 10),
        coldHits: 7,
        coldMisses: 2,
        coldBytesWritten: 100,
        coldBytesRestored: 700,
        coldRoot: canonicalCacheRootPath,
        coldEnabled: true,
        coldEnqueued: 5,
        coldQueueDrops: 3,
        coldEvictions: 2,
        coldCorruptions: 0,
        // DISTINCT non-zero cumulative totals across the two attributed rows so
        // MAX (5) and SUM (7) disagree. With 0 and 2 they were equal and the
        // reducer the acceptance bar depends on could be swapped for SUM with
        // the suite still green.
        coldCorruptionsTotal: 2,
        coldQueueDropsTotal: 4,
        // Write errors follow the corruption shape (SUM the deltas, MAX the
        // totals); declines follow neither — they are summed, because a
        // decline has a legitimate non-zero steady state.
        coldWriteErrors: 3,
        coldWriteErrorsTotal: 6,
        coldRestoreDeclines: 2,
        coldSidecarRestoreSuppressed: 1,
        // Every sidecar counter is a per-turn DELTA, so all eight SUM. The two
        // attributed rows carry different values for each so a reducer swapped
        // to MAX changes a number in the assertions below, and the eight sums
        // are pairwise distinct so a column aliased to a neighbour does too.
        // `coldSidecarEnqueued`/`coldSidecarQueueDrops` stay under the
        // object-scoped `coldEnqueued`/`coldQueueDrops` above, which they are a
        // subset of — the same admission bumps both.
        coldSidecarCaptureReached: 6,
        coldSidecarChainEmpty: 1,
        coldSidecarBoundarySkips: 2,
        coldSidecarAlreadyPersisted: 7,
        coldSidecarEnqueued: 2,
        coldSidecarQueueDrops: 2,
        coldSidecarInstalled: 4,
      },
      {
        traceId: 'mine-2',
        ts: Date.UTC(2026, 6, 21, 10),
        coldHits: 4,
        coldMisses: 1,
        coldBytesWritten: 50,
        coldBytesRestored: 400,
        coldRoot: canonicalCacheRootPath,
        coldEnabled: true,
        coldEnqueued: 3,
        coldQueueDrops: 5,
        coldEvictions: 1,
        coldCorruptions: 2,
        coldCorruptionsTotal: 5,
        coldQueueDropsTotal: 4,
        coldWriteErrors: 4,
        // Distinct from the other row's 6 so MAX (9) and SUM (15) disagree —
        // the same reason `coldCorruptionsTotal` carries 2 and 5.
        coldWriteErrorsTotal: 9,
        coldRestoreDeclines: 5,
        coldSidecarRestoreSuppressed: 2,
        coldSidecarCaptureReached: 4,
        coldSidecarChainEmpty: 3,
        coldSidecarBoundarySkips: 9,
        coldSidecarAlreadyPersisted: 9,
        coldSidecarEnqueued: 3,
        coldSidecarQueueDrops: 4,
        coldSidecarInstalled: 8,
      },
      {
        traceId: 'other-root-1',
        ts: Date.UTC(2026, 6, 20, 11),
        coldHits: 9999,
        coldMisses: 7,
        coldRoot: '/some/other/cache/mlx-paged-v1',
        coldEnabled: true,
        coldCorruptions: 55,
        coldCorruptionsTotal: 55,
        // Another root's faults must never leak into this cache's health.
        coldWriteErrors: 500,
        coldWriteErrorsTotal: 900,
        coldRestoreDeclines: 400,
        coldSidecarRestoreSuppressed: 300,
        // Another root's sidecar traffic, an order of magnitude larger, so a
        // projection that lost the `cold_root = ?` filter cannot look plausible.
        coldSidecarCaptureReached: 900,
        coldSidecarChainEmpty: 600,
        coldSidecarBoundarySkips: 500,
        coldSidecarAlreadyPersisted: 700,
        coldSidecarEnqueued: 400,
        coldSidecarQueueDrops: 350,
        coldSidecarInstalled: 800,
      },
      // No coldRoot AND no coldEnabled: written by a build that predates
      // attribution. Not "the tier was off" — genuinely unknown.
      { traceId: 'legacy-1', ts: Date.UTC(2026, 6, 19, 10), coldHits: 5000, coldMisses: 11 },
      // Tier explicitly off this turn: known, attributable, and zero — for
      // every BLOCK counter. The sidecar capture is the exception and the
      // reason this row carries one: `record_capture_reached()` is the first
      // statement of `capture_gemma4_sliding_cold_sidecar`, above its
      // `adapter.cold_tier()` guard, so a hybrid turn under
      // `--no-persist-cache` reaches the capture, counts it, and returns with
      // no tier to name a root from.
      {
        traceId: 'disabled-1',
        ts: Date.UTC(2026, 6, 20, 12),
        coldHits: 0,
        coldMisses: 0,
        coldEnabled: false,
        coldSidecarCaptureReached: 13,
      },
      // The partition hole: tier ON, root NULL. Matches none of `cold_root = ?`,
      // `cold_root IS NULL AND cold_enabled IS NULL`, `cold_root <> ?`, or
      // `cold_enabled = 0` — so before the fourth bucket these 777 hits were
      // reported by nothing at all.
      {
        traceId: 'orphan-1',
        ts: Date.UTC(2026, 6, 20, 13),
        coldHits: 777,
        coldMisses: 5,
        coldEnabled: true,
        coldSidecarCaptureReached: 21,
      },
    ]);
    await ingest();
  });

  it('scopes the hit/miss trend to the cache root being shown', async () => {
    const body = (await (await api.fetch('/api/cache')).json()) as CacheBody;
    // ONLY this root's rows. 15010 would mean the filter is gone.
    expect(totals(body)).toEqual({ hits: 11, misses: 3 });
    expect(body.scope.root).toBe(canonicalCacheRootPath);
    // Two days, two rows — neither the other root's day nor the legacy day.
    expect(body.trend.map((r) => r.day)).toEqual(['2026-07-20', '2026-07-21']);
    expect(body.trend.reduce((a, r) => a + r.bytesRestored, 0)).toBe(1100);
    expect(body.trend.reduce((a, r) => a + r.bytesWritten, 0)).toBe(150);
  });

  // The decision, pinned: a NULL-root row is EXCLUDED from the shown cache's
  // rate (counting it is the original defect) but REPORTED in its own bucket
  // (dropping it silently would make a real history vanish next to a root path
  // the user is staring at). It is bounded — trace JSONL retention prunes it —
  // so the bucket empties on its own.
  it('excludes legacy NULL-root rows from the rate but reports them as their own bucket', async () => {
    const body = (await (await api.fetch('/api/cache')).json()) as CacheBody;
    expect(totals(body).hits).toBe(11);
    expect(body.scope.legacy).toEqual({ turns: 1, hits: 5000, misses: 11 });
    expect(body.scope.trendWindowDays).toBe(TRACE_RETENTION_DAYS);
    // A turn that ran with the tier OFF is a different story from an
    // unattributable one, and must not be folded into `legacy`.
    expect(body.scope.disabledTurns).toBe(1);
  });

  /**
   * The buckets must PARTITION the table, not merely be disjoint. A row with
   * `cold_enabled = 1` and a NULL `cold_root` matched no arm, so its 777 hits
   * were reported by nothing: trend 0, legacy 0, otherRoots 0, disabledTurns 0.
   * That is the silent-zero the whole scope design exists to refuse, in the one
   * shape it could still take.
   *
   * Asserted as a CONSERVATION law against the table's own total rather than by
   * enumerating the arms, so any future fifth shape has to be accounted for too.
   */
  it('accounts for every trace row: the buckets partition the table, they do not just avoid overlap', async () => {
    const body = (await (await api.fetch('/api/cache')).json()) as CacheBody;
    expect(body.scope.unattributed).toEqual({ turns: 1, hits: 777, misses: 5 });
    const shown = totals(body);
    const accounted = {
      hits: shown.hits + body.scope.legacy.hits + body.scope.otherRoots.hits + body.scope.unattributed.hits,
      misses: shown.misses + body.scope.legacy.misses + body.scope.otherRoots.misses + body.scope.unattributed.misses,
    };
    // Every cold lookup the fixture wrote: 11 + 5000 + 9999 + 777 hits.
    expect(accounted).toEqual({ hits: 15_787, misses: 26 });
  });

  it('reports another cache directory as otherRoots rather than merging it in', async () => {
    const body = (await (await api.fetch('/api/cache')).json()) as CacheBody;
    expect(body.scope.otherRoots).toEqual({ turns: 1, hits: 9999, misses: 7 });
    expect(totals(body).hits).toBe(11);
  });

  // The writer and the reader are different PROCESSES resolving the root from
  // their own environments. A raw string compare is a silent-zero trap: on macOS
  // `/tmp` → `/private/tmp` alone makes two identical-looking spellings unequal.
  it('matches a symlinked cache root through canonicalization', async () => {
    const linkRoot = join(base, 'cache-link');
    symlinkSync(cacheRoot, linkRoot);
    const linked = createDashboardRuntime({
      dbPath: ':memory:',
      sessionsRoot,
      tracesDir,
      modelsDir,
      cacheRoot: linkRoot,
    });
    const linkedApi = createTestClient(linked);
    try {
      expect((await linkedApi.fetch('/api/ingest', { method: 'POST' })).status).toBe(200);
      const body = (await (await linkedApi.fetch('/api/cache')).json()) as CacheBody;
      // Rows were written under the REAL path; the dashboard was pointed at the
      // symlink. A lexical compare would report a flat zero here.
      expect(body.scope.root).toBe(canonicalCacheRootPath);
      expect(totals(body)).toEqual({ hits: 11, misses: 3 });
    } finally {
      await linked.close();
    }
  });

  // The four counters `coldCacheStats()` always exposed and the agent always
  // dropped. "corruptions must be 0" is the stated acceptance bar for admitting
  // a family to the restore allowlist and was previously uncheckable from
  // anything the user runs.
  it('surfaces the cold-tier counters the agent used to drop, scoped to this cache', async () => {
    const body = (await (await api.fetch('/api/cache')).json()) as CacheBody;
    expect(body.health.enqueued).toBe(8);
    // Object-scoped: 3 + 5 across the two attributed rows. Blocks and state
    // sidecars share this queue, so it is the SUPERSET of `sidecarQueueDrops`.
    expect(body.health.queueDrops).toBe(8);
    expect(body.health.evictions).toBe(3);
    expect(body.health.corruptions).toBe(2);
    // Cumulative totals reduce with MAX, not SUM (each process restarts at 0),
    // and are scoped — the other root's 55 corruptions must not leak in.
    //
    // The two attributed rows carry 2 and 5, so MAX is 5 and SUM would be 7:
    // asserting 5 is what makes the reducer, not just the scope, load-bearing.
    // Three processes that each saw 2 corruptions must report 2 under a label
    // that says "cumulative max", never 6.
    expect(body.health.corruptionsTotal).toBe(5);
    // Same shape for drops: both rows carry 4, so MAX is 4 and SUM would be 8.
    expect(body.health.queueDropsTotal).toBe(4);
  });

  /**
   * The two failure modes that had no number anywhere.
   *
   * A write that the queue accepted and the disk refused was reported by
   * nothing: the native writer is fail-open and swallows the error, so a root
   * that was read-only, full or unmounted produced `queueDrops 0`,
   * `corruptions 0` and a turn that exited 0 — a dashboard showing a perfectly
   * healthy cache holding nothing. A refused RESTORE was equally silent for
   * the opposite reason: it performs no lookup, so it moved neither hits nor
   * misses and rendered as "no lookups recorded", which reads as "nothing ran"
   * rather than "reuse was refused".
   *
   * The two reducers are asserted to DISAGREE with the alternative on purpose:
   * `writeErrorsTotal` is MAX over per-process totals (6 and 9 → 9, never 15),
   * while `restoreDeclines` is a SUM of deltas (2 + 5 → 7, never 5). Swapping
   * either reducer changes a number here.
   */
  it('surfaces write errors and restore declines, scoped and reduced correctly', async () => {
    const body = (await (await api.fetch('/api/cache')).json()) as CacheBody;
    // Deltas SUM across this cache's rows: 3 + 4. The other root's 500 must not
    // appear — a broken cache elsewhere is not this cache's alarm.
    expect(body.health.writeErrors).toBe(7);
    // Cumulative totals reduce with MAX: 6 and 9 → 9. SUM would say 15.
    expect(body.health.writeErrorsTotal).toBe(9);
    // Declines are SUMMED and have no MAX'd total, because unlike a corruption
    // a decline is normal on any prompt's first turn — a "did it ever happen"
    // latch would be pinned on from the first minute and mean nothing.
    expect(body.health.restoreDeclines).toBe(7);
    expect(body.health.restoreSuppressed).toBe(3);
  });

  /**
   * The other seven `coldSidecar*` counters, which the ingest mapping dropped
   * on the floor while only `restoreSuppressed` reached a column.
   *
   * `sidecarInstalled` is the reason this matters: every
   * `install_*_cold_sidecar` early-return falls through to a full O(prefix)
   * replay that produces CORRECT state, so a regression from "restored and
   * INSTALLED" to "restored and silently re-derived" leaves hits, cached
   * tokens, corruptions and the emitted text all unchanged. Nothing else in
   * the payload moves.
   *
   * All eight are per-turn DELTAS — none has a per-process cumulative twin —
   * so every one SUMS. The fixture's two attributed rows carry different
   * values for each counter, and the eight sums are pairwise distinct, so
   * swapping any reducer to MAX or aliasing any column to a neighbour changes
   * a number here. The other root's much larger traffic pins the scope.
   */
  it('projects every cold sidecar counter, summed per turn and scoped to this cache', async () => {
    const body = (await (await api.fetch('/api/cache')).json()) as CacheBody;
    // 6 + 4. MAX would say 6; the other root's 900 would say the scope is gone.
    expect(body.health.sidecarCaptureReached).toBe(10);
    // 1 + 3 (MAX 3), 2 + 9 (MAX 9), 7 + 9 (MAX 9).
    expect(body.health.sidecarChainEmpty).toBe(4);
    expect(body.health.sidecarBoundarySkips).toBe(11);
    expect(body.health.sidecarAlreadyPersisted).toBe(16);
    // Subsets of the object-scoped counters asserted above (8 enqueued, 8
    // dropped) — the same admission bumps both, so these are never added to them.
    expect(body.health.sidecarEnqueued).toBe(5);
    expect(body.health.sidecarQueueDrops).toBe(6);
    // 4 + 8. MAX would say 8, and the unscoped total would say 812.
    expect(body.health.sidecarInstalled).toBe(12);
  });

  /**
   * `captureReached` is the one sidecar counter that moves with NO tier in
   * hand. `record_capture_reached()` is the first statement of every family's
   * capture — `crates/mlx-core/src/models/gemma4/model.rs:2839` sits above the
   * `adapter.cold_tier()` guard at `:2843`, and qwen3_5 (`:2118` vs `:2122`)
   * and qwen3_5_moe (`:1497` vs `:1501`) are identical — so a hybrid turn run
   * under `--no-persist-cache`, or one whose tier failed to open, counts the
   * capture and then returns with no tier to name a root from. The writer only
   * emits `coldRoot` for a tier that was open
   * (`packages/agent/src/provider/index.ts:182`), so those turns land with a
   * NULL root and the root-scoped `health` query drops them.
   *
   * Reported as its own figure rather than folded into `health`: the scoping
   * exists so another cache's numbers never bleed into the shown one, and a
   * rootless row has no cache to bleed FROM. Dropping it silently is what makes
   * the page say "not reached — dense families" while the hybrid capture ran on
   * every turn and failed before acquiring a tier — the exact state these
   * counters were added to record.
   */
  it('reports sidecar captures that ran with no tier instead of dropping them', async () => {
    const body = (await (await api.fetch('/api/cache')).json()) as CacheBody;
    // 13 (tier off) + 21 (tier on, root unrecorded). The rooted rows' 10 must
    // NOT appear here, and the other root's 900 must not appear at all: a
    // rootless bucket that leaked either would read 23 / 34+10 / 934.
    expect(body.scope.unrootedSidecarCaptures).toBe(34);
    // The mirror assertion, and the one that pins the fix as ADDITIVE: the
    // shown cache's own figure stays exactly its two rooted rows.
    expect(body.health.sidecarCaptureReached).toBe(10);
  });

  // TRACE_RETENTION_DAYS is exported precisely so the window the UI prints and
  // the window the pruner enforces can never drift. Asserting the literal 30
  // passes whether `api.ts` imports the constant or hardcodes it, which defeats
  // the point — so pin the COUPLING: whatever the constant says, the payload
  // says.
  it('reports the trend window as the trace retention constant, not a copy of it', async () => {
    const body = (await (await api.fetch('/api/cache')).json()) as CacheBody;
    expect(body.scope.trendWindowDays).toBe(TRACE_RETENTION_DAYS);
    // Guard the guard: a constant that went 0/NaN would make the assert vacuous.
    expect(TRACE_RETENTION_DAYS).toBeGreaterThan(0);
  });

  // F5: the UI must not carry a fifth hardcoded copy of the allowlist, so the
  // server sends it. The browser bundle cannot import @mlx-node/agent (it
  // transitively loads the native addon), and a hardcoded array in cache.tsx
  // would sit outside the native drift guard.
  it('serves the cold-tier restore allowlist so the SPA never hardcodes it', async () => {
    const body = (await (await api.fetch('/api/cache')).json()) as CacheBody;
    expect(body.restoreFamilies).toEqual(coldTierRestoreFamilyList());
    expect(body.restoreFamilies.length).toBeGreaterThan(1);
  });
});

/**
 * The join key a real Control Panel window run actually uses.
 *
 * Every other cache test hands the runtime an explicit `cacheRoot`, but nothing
 * in production supplies one — the app's Control Panel entry constructs the runtime with
 * no cache root at all. So in production `ctx.cacheRoot` is `undefined` and the
 * scope root comes from
 * `coldCacheRoot(undefined)` → `defaultColdCacheDir()` → the `mlx-paged-v1`
 * child of `MLX_COLD_CACHE_DIR`, canonicalized. Whether THAT string equals what
 * the agent recorded was asserted nowhere — both sides were literals the tests
 * chose. A one-segment disagreement shows the Cache page a flat 0/0 trend beside
 * a populated disk scan: quieter than the 16189 the scoping fix was written to
 * kill, and just as wrong.
 *
 * `MLX_COLD_CACHE_DIR` is pointed at a SYMLINK here so the canonicalization step
 * is load-bearing on every platform, not only where `tmpdir()` happens to sit
 * under one.
 */
describe('dashboard api — production cache root (no explicit cacheRoot)', () => {
  interface DefaultCacheBody {
    disk: { root: string; exists: boolean; entryCount: number };
    trend: Array<{ hits: number; misses: number }>;
    scope: { root: string; otherRoots: { turns: number }; legacy: { turns: number } };
  }

  let envBase: string;
  let realColdDir: string;
  let linkedColdDir: string;
  /** Where the native tier actually opens: `<MLX_COLD_CACHE_DIR>/mlx-paged-v1`. */
  let managedRoot: string;
  let previousEnv: string | undefined;
  let defaulted: DashboardRuntime;
  let defaultedApi: TestClient;

  async function cacheBody(): Promise<DefaultCacheBody> {
    return (await (await defaultedApi.fetch('/api/cache')).json()) as DefaultCacheBody;
  }

  beforeEach(async () => {
    envBase = mkdtempSync(join(tmpdir(), 'dash-coldenv-'));
    realColdDir = join(envBase, 'cold');
    managedRoot = join(realColdDir, 'mlx-paged-v1');
    mkdirSync(managedRoot, { recursive: true });
    writeFileSync(join(managedRoot, hexBlock(9)), Buffer.alloc(512));
    linkedColdDir = join(envBase, 'cold-link');
    symlinkSync(realColdDir, linkedColdDir);

    previousEnv = process.env.MLX_COLD_CACHE_DIR;
    process.env.MLX_COLD_CACHE_DIR = linkedColdDir;

    // One trace row, written the way the agent writes it: the root goes through
    // `canonicalCacheRoot` in `provider/index.ts` before `record()` sees it, so
    // the WRITER side of the join is production code here, not a literal.
    for (const n of readdirSync(tracesDir)) if (n.endsWith('.jsonl')) unlinkSync(join(tracesDir, n));
    writeFileSync(
      join(tracesDir, '2026-07-22-default.jsonl'),
      `${JSON.stringify({
        v: 1,
        model: 'm',
        durationMs: 1,
        finishReason: 'stop',
        traceId: 'default-root-1',
        ts: Date.UTC(2026, 6, 22, 9),
        coldHits: 23,
        coldMisses: 5,
        coldEnabled: true,
        coldRoot: canonicalCacheRoot(managedRoot),
      })}\n`,
    );

    defaulted = createDashboardRuntime({
      dbPath: ':memory:',
      sessionsRoot,
      tracesDir,
      modelsDir,
      // Exactly what the app's Control Panel entry passes.
      cacheRoot: undefined,
    });
    defaultedApi = createTestClient(defaulted);
    expect((await defaultedApi.fetch('/api/ingest', { method: 'POST' })).status).toBe(200);
  });

  afterEach(async () => {
    await defaulted.close();
    if (previousEnv === undefined) delete process.env.MLX_COLD_CACHE_DIR;
    else process.env.MLX_COLD_CACHE_DIR = previousEnv;
    rmSync(envBase, { recursive: true, force: true });
  });

  it('scans the managed child of MLX_COLD_CACHE_DIR, never the directory itself', async () => {
    const body = await cacheBody();
    expect(body.disk.exists).toBe(true);
    expect(body.disk.entryCount).toBe(1);
    // Asserted structurally rather than against a repeated literal: the tier
    // always nests one managed level under the env var.
    expect(body.disk.root).not.toBe(linkedColdDir);
    expect(dirname(body.disk.root)).toBe(linkedColdDir);
  });

  it('joins the agent-written root when the CLI supplies no cacheRoot', async () => {
    const body = await cacheBody();
    expect(body.scope.root).toBe(canonicalCacheRoot(managedRoot));
    // JOINED, not merely "not crashed": the row's lookups land in the trend and
    // in neither exclusion bucket.
    expect(
      body.trend.reduce((a, r) => ({ hits: a.hits + r.hits, misses: a.misses + r.misses }), { hits: 0, misses: 0 }),
    ).toEqual({
      hits: 23,
      misses: 5,
    });
    expect(body.scope.otherRoots.turns).toBe(0);
    expect(body.scope.legacy.turns).toBe(0);
  });

  it('canonicalizes the derived root, so a symlinked MLX_COLD_CACHE_DIR still joins', async () => {
    const body = await cacheBody();
    // The reader was pointed at the symlink and the writer recorded the real
    // path. A lexical compare reports a flat zero here beside a disk scan that
    // just found a block.
    expect(body.disk.root.startsWith(linkedColdDir)).toBe(true);
    expect(body.scope.root.startsWith(realpathSync(realColdDir))).toBe(true);
    expect(body.scope.root).not.toBe(body.disk.root);
    expect(body.trend.length).toBe(1);
  });
});

describe('dashboard api — downloads', () => {
  it('lists jobs and rejects a non-catalog repo', async () => {
    const list = (await (await api.fetch('/api/downloads')).json()) as { jobs: unknown[] };
    expect(Array.isArray(list.jobs)).toBe(true);

    const bad = await api.fetch('/api/downloads', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo: 'someone/not-in-catalog' }),
    });
    expect(bad.status).toBe(400);
  });

  // Finding E: the cancel route is wired end-to-end through the real runtime and
  // its real DownloadManager; an unknown/terminal id is a 404 (nothing to
  // cancel). The known-id → 200 branch is exercised deterministically below
  // without a live download.
  it('404s DELETE /api/downloads/:id for an unknown job', async () => {
    const res = await api.fetch('/api/downloads/no-such-job', { method: 'DELETE' });
    expect(res.status).toBe(404);
  });

  // The progress stream is a transport capability, not a handler: a transport
  // that can stream intercepts this route before dispatch, and one that cannot
  // must reach the placeholder and be TOLD so. A plain `call` — which is exactly
  // what the port bridge issues, since it carries progress over `subscribe`
  // instead — must therefore see 503, never a misleading 404.
  it('answers the streaming route 503, not a misleading 404, for a caller that cannot stream', async () => {
    const res = await api.fetch('/api/downloads/some-job/events');
    expect(res.status).toBe(503);
    expect(((await res.json()) as { error: string }).error).toMatch(/not available over this transport/i);
  });
});

/** Dispatch one request against a stub context, with no runtime or socket. */
function call(ctx: ApiContext, method: string, pathname: string, body?: unknown): ReturnType<typeof dispatch> {
  return dispatch(ctx, { method, pathname, query: new URLSearchParams(), body });
}

// Finding E: the DELETE /api/downloads/:id handler invokes DownloadManager.cancel
// with the decoded id and maps its boolean to 200 (cancelled) / 404 (unknown or
// already-terminal) — verified at the route layer with a stub manager so no live
// download or network is needed.
describe('dashboard api — cancel download route', () => {
  function ctxWith(cancel: (id: string) => boolean): ApiContext {
    return { downloads: { cancel } } as unknown as ApiContext;
  }

  it('returns 200 and cancels a known job id', async () => {
    const seen: string[] = [];
    const ctx = ctxWith((id) => {
      seen.push(id);
      return id === 'known-job';
    });

    expect(isApiPath('/api/downloads/known-job')).toBe(true);
    const res = await call(ctx, 'DELETE', '/api/downloads/known-job');
    expect(seen).toEqual(['known-job']);
    expect(res.status).toBe(200);
    expect(res.ok).toBe(true);
    expect(res.ok ? res.body : null).toEqual({ cancelled: true, id: 'known-job' });
  });

  it('returns 404 for an unknown/terminal job id', async () => {
    const ctx = ctxWith(() => false);

    const res = await call(ctx, 'DELETE', '/api/downloads/ghost');
    expect(res.status).toBe(404);
    expect(res.ok).toBe(false);
    expect(res.ok ? '' : res.code).toBe('E_NOT_FOUND');
    expect(res.ok ? '' : res.message).toContain('ghost');
  });
});

// A route that only ACCEPTS work answers 202, not 200: starting a download queues
// a job that has not run yet. Asserted at the route layer with a stub manager so
// no multi-GB transfer is triggered — a live-repo POST would really download.
describe('dashboard api — download start is 202 Accepted', () => {
  it('answers 202 (not 200) for an accepted download start', async () => {
    const ctx = { downloads: { start: () => 'job-1' } } as unknown as ApiContext;
    const res = await call(ctx, 'POST', '/api/downloads', { repo: 'org/repo' });
    expect(res.status).toBe(202);
    expect(res.ok ? res.body : null).toEqual({ id: 'job-1', repo: 'org/repo' });
  });
});

// A KNOWN path reached with the wrong method must be 405, not 404: the dispatcher
// keeps matching after a method mismatch and remembers that the path shape (with
// its `:param` segments) existed. Losing that collapses every wrong-method call to
// a misleading "no such route".
describe('dashboard api — method-not-allowed vs not-found', () => {
  const ctx = {} as unknown as ApiContext;

  it('405s a known path reached with the wrong method', async () => {
    for (const [method, pathname] of [
      ['PUT', '/api/models'],
      ['POST', '/api/sessions/fix-1'],
      ['GET', '/api/ingest'],
      ['PATCH', '/api/downloads/some-id'],
    ] as const) {
      const res = await call(ctx, method, pathname);
      expect(res.status).toBe(405);
      expect(res.ok ? '' : res.code).toBe('E_METHOD_NOT_ALLOWED');
      expect(res.ok ? '' : res.message).toContain(method);
    }
  });

  it('404s a path no route has, whatever the method', async () => {
    for (const method of ['GET', 'POST', 'DELETE'] as const) {
      const res = await call(ctx, method, '/api/nope');
      expect(res.status).toBe(404);
      expect(res.ok ? '' : res.code).toBe('E_NOT_FOUND');
    }
  });

  it('does not treat a non-API path as a route', () => {
    expect(isApiPath('/')).toBe(false);
    expect(isApiPath('/sessions/deep/link')).toBe(false);
    expect(isApiPath('/health')).toBe(true);
    expect(isApiPath('/api/models')).toBe(true);
  });
});

// A POST that lands while `close()` is draining downloads is refused by the
// manager. That refusal is a server-lifecycle condition, not a malformed request,
// so it must not be reported as a 400 blaming the caller.
describe('dashboard api — start download route', () => {
  function postDownload(start: () => string): ReturnType<typeof dispatch> {
    const ctx = { downloads: { start } } as unknown as ApiContext;
    return call(ctx, 'POST', '/api/downloads', { repo: 'org/model' });
  }

  it('answers 503 once the download manager is shutting down', async () => {
    const res = await postDownload(() => {
      throw new DownloadsClosedError();
    });
    expect(res.status).toBe(503);
    expect(res.ok).toBe(false);
    expect(res.ok ? '' : res.code).toBe('E_UNAVAILABLE');
    expect(res.ok ? '' : res.message).toMatch(/shutting down/i);
  });

  it('still answers 400 for a repo the catalog does not carry', async () => {
    const res = await postDownload(() => {
      throw new Error('Repo "org/model" is not in the model catalog');
    });
    expect(res.status).toBe(400);
    expect(res.ok ? '' : res.code).toBe('E_BAD_REQUEST');
  });
});

describe('dashboard api — health and unknown paths', () => {
  it('404s an unknown api path', async () => {
    const res = await api.fetch('/api/nope');
    expect(res.status).toBe(404);
  });

  it('reports health', async () => {
    const res = await api.fetch('/health');
    expect(res.status).toBe(200);
    expect(((await res.json()) as { status: string }).status).toBe('ok');
  });

  it('also serves health under /api/health', async () => {
    const res = await api.fetch('/api/health');
    expect(res.status).toBe(200);
    expect(((await res.json()) as { status: string }).status).toBe('ok');
  });
});

// The suite above drives `runtime.call` in-process. This block runs the SAME
// error-model contract through the real RPC transport — two real ports, real
// structured clone — so an envelope the bridge mangles (a lost `code`, a
// re-derived status, a `bodyError` dropped on the floor) fails here rather than
// only in the renderer. Everything else stays in-process because it is API
// behaviour, not transport behaviour.
describe('dashboard api — the RPC transport carries the error model over a port', () => {
  let rpc: TestClient;
  let teardown: () => void;

  beforeEach(() => {
    const { port1, port2 } = new MessageChannel();
    port1.unref();
    port2.unref();
    const dispose = serveRuntimeOverPort(runtime, bindEventTargetPort(port2));
    const client = createRpcClient(bindEventTargetPort(port1), { onUnresponsive: () => port2.close() });
    rpc = createTestClient(client);
    teardown = () => {
      client.close();
      dispose();
    };
  });

  afterEach(() => {
    teardown();
  });

  it('answers a plain read over the port', async () => {
    const res = await rpc.fetch('/api/models');
    expect(res.status).toBe(200);
    expect((await res.json()) as { dir: string }).toMatchObject({ dir: modelsDir });
  });

  it('carries a handler 404 across with its message', async () => {
    const res = await rpc.fetch('/api/sessions/no-such-session');
    expect(res.status).toBe(404);
    expect((await res.json()) as { error: string }).toMatchObject({ error: expect.stringContaining('not found') });
  });

  it('carries a wrong method on a known path as 405, not 404', async () => {
    const res = await rpc.fetch('/api/models', { method: 'PUT' });
    expect(res.status).toBe(405);
    expect(((await res.json()) as { error: string }).error).toContain('PUT');
  });

  it('carries an unparseable request body as a 400 from the handler that needed it', async () => {
    const res = await rpc.fetch('/api/cache', { method: 'DELETE', rawBody: '{ this is not json' });
    expect(res.status).toBe(400);
    expect(((await res.json()) as { error: string }).error).toBe('Invalid JSON in request body');
    // Nothing was cleared by the rejected call.
    const after = (await (await api.fetch('/api/cache')).json()) as { disk: { entryCount: number } };
    expect(after.disk.entryCount).toBe(2);
  });

  it('checks the session id BEFORE the body, so an unknown id still 404s', async () => {
    // The rename handler looks the session up first; an unparseable body must not
    // pre-empt that 404 (the transport records the parse failure as data instead
    // of raising it at read time).
    const res = await rpc.fetch('/api/sessions/no-such-session', { method: 'PATCH', rawBody: '{ not json either' });
    expect(res.status).toBe(404);
  });

  it('reaches the database worker through the port, not just main-thread routes', async () => {
    // `/api/models` and `/api/sessions` are worker-owned: a reply proves the port
    // hop and the thread hop compose, which no in-process test can show.
    const res = await rpc.fetch('/api/sessions');
    expect(res.status).toBe(200);
    expect((await res.json()) as { total: number }).toMatchObject({ total: expect.any(Number) });
  });
});

// Finding 7: sessions written under a custom agent home (`PI_CODING_AGENT_DIR`)
// must be discoverable. `agentSessionsRoot()` resolves that env var the same way
// the agent/pi do, falling back to the default home.
describe('paths — agentSessionsRoot honors PI_CODING_AGENT_DIR', () => {
  const KEY = 'PI_CODING_AGENT_DIR';

  function withEnv(value: string | undefined, fn: () => void): void {
    const saved = process.env[KEY];
    if (value === undefined) delete process.env[KEY];
    else process.env[KEY] = value;
    try {
      fn();
    } finally {
      if (saved === undefined) delete process.env[KEY];
      else process.env[KEY] = saved;
    }
  }

  it('falls back to ~/.mlx-node/agent/sessions when unset', () => {
    withEnv(undefined, () => {
      expect(agentSessionsRoot()).toBe(join(homedir(), '.mlx-node', 'agent', 'sessions'));
    });
  });

  it('derives <PI_CODING_AGENT_DIR>/sessions for an absolute override', () => {
    withEnv('/tmp/custom-agent', () => {
      expect(agentSessionsRoot()).toBe(join('/tmp/custom-agent', 'sessions'));
    });
  });

  it('expands a ~/ override with the pi tilde rule', () => {
    withEnv('~/agent-home', () => {
      expect(agentSessionsRoot()).toBe(join(homedir(), 'agent-home', 'sessions'));
    });
  });
});
