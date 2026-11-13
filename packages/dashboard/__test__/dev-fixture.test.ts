/**
 * The dev harness fixture home goes through the real readers: trace and runtime ingest, model discovery, the cold
 * cache scan, the live-file parser and the agent sessions reader all find data in it, and nothing is pruned.
 */

import {
  existsSync,
  linkSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  assertFixtureHome,
  CATALOG_BLOCKED,
  CATALOG_INSTALL,
  fixtureArgs,
  fixturePaths,
  FIXTURE_MODELS,
  FIXTURE_MARKER,
  LIVE_SESSION_ID,
  LONG_SESSION_ID,
  refreshRuntime,
  resetHome,
  writeFixtureHome,
  writeLiveFile,
} from '../scripts/dev-fixture.js';
import { AgentSessionsReader } from '../src/agent-sessions/reader.js';
import { scanColdCache } from '../src/cache.js';
import { catalogWithState } from '../src/catalog.js';
import { MANAGED_BEGIN, MANAGED_END } from '../src/coding-agents.js';
import { openDashboardDb, type DashboardDb } from '../src/db/open.js';
import { ingestRuntime } from '../src/ingest/runtime.js';
import { ingestTraces } from '../src/ingest/traces.js';
import { classifyProcess, parseLiveFile, parseSupervisorFile } from '../src/live.js';
import { discoverLocalModels } from '../src/models.js';

let home: string;
let dash: DashboardDb | undefined;
const now = Date.now();

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), 'dash-dev-fixture-'));
});

afterEach(() => {
  dash?.sqlite.close();
  dash = undefined;
  rmSync(home, { recursive: true, force: true });
});

describe('dev fixture home', () => {
  it('ingests 30 days of v2 traces for ten models without pruning a file', async () => {
    const summary = writeFixtureHome(home, { now });
    const paths = fixturePaths(home);
    const files = readdirSync(paths.tracesDir);
    dash = openDashboardDb(':memory:');
    const result = await ingestTraces(dash, paths.tracesDir);
    expect(result.pruned).toBe(0);
    expect(result.warnings).toEqual([]);
    expect(result.files).toBe(files.length);
    expect(result.records).toBe(summary.traces);
    expect(result.records).toBeGreaterThan(30 * 90);
    const keys = dash.sqlite.prepare('SELECT DISTINCT model_key AS key FROM traces ORDER BY key').all() as Array<{
      key: string;
    }>;
    expect(keys.map((row) => row.key)).toEqual(FIXTURE_MODELS.map((model) => model.name).sort());
    const outcomes = dash.sqlite.prepare('SELECT DISTINCT outcome FROM traces ORDER BY outcome').all() as Array<{
      outcome: string;
    }>;
    expect(outcomes.map((row) => row.outcome)).toEqual(['cancelled', 'error', 'ok']);
    const first = JSON.parse(readFileSync(join(paths.tracesDir, files[0]!), 'utf8').split('\n')[0]!) as {
      v: number;
    };
    expect(first.v).toBe(2);
  });

  it('holds old, measured and unmeasured traces, and sessions with and without a measured prefill', async () => {
    writeFixtureHome(home, { now });
    const paths = fixturePaths(home);
    dash = openDashboardDb(':memory:');
    await ingestTraces(dash, paths.tracesDir);
    const count = (where: string): number =>
      (
        dash!.sqlite.prepare(`SELECT COUNT(*) AS n FROM traces WHERE outcome = 'ok' AND ${where}`).get() as {
          n: number;
        }
      ).n;
    const measured = 'prefill_ms > 0 AND prefill_tokens > 0';
    const week = String(now - 7 * 86_400_000);
    const day = String(now - 86_400_000);
    // Old records: the older process wrote no prefill window.
    expect(count(`pid = 41001`)).toBeGreaterThan(0);
    expect(count(`pid = 41001 AND ${measured}`)).toBe(0);
    // Measured: streamed turns of the newer process, the window inside the time to first token.
    expect(count(`pid = 41002 AND stream = 1 AND ${measured}`)).toBe(count('pid = 41002 AND stream = 1'));
    expect(count(`${measured} AND ttft_ms < prefill_ms`)).toBe(0);
    expect(count(`${measured} AND prefill_tokens >= 256`)).toBeGreaterThan(0);
    expect(count(`${measured} AND prefill_tokens < 256`)).toBeGreaterThan(0);
    // Not measured, new: non-streamed turns.
    expect(count(`pid = 41002 AND stream = 0`)).toBeGreaterThan(0);
    expect(count(`stream = 0 AND ${measured}`)).toBe(0);
    // A model without any measurement.
    expect(count(`model_key = 'Llama-3.2-3B-Instruct-4bit'`)).toBeGreaterThan(0);
    expect(count(`model_key = 'Llama-3.2-3B-Instruct-4bit' AND ${measured}`)).toBe(0);
    // 24h and 7d hold measured and non-streamed turns.
    for (const since of [day, week]) {
      expect(count(`ts >= ${since} AND ${measured}`)).toBeGreaterThan(0);
      expect(count(`ts >= ${since} AND stream = 0`)).toBeGreaterThan(0);
    }
    // The old lines still carry the TTFT-derived rate the server used to write.
    const oldFile = readdirSync(paths.tracesDir).find((name) => name.endsWith('-41001.jsonl'))!;
    const oldLine = JSON.parse(readFileSync(join(paths.tracesDir, oldFile), 'utf8').split('\n')[0]!) as Record<
      string,
      unknown
    >;
    expect('prefillMs' in oldLine).toBe(false);
    if (oldLine.outcome === 'ok') expect(typeof oldLine.prefillTps).toBe('number');

    const reader = new AgentSessionsReader({ agentHome: paths.agentHome, now: () => now });
    const long = reader.detail(LONG_SESSION_ID)!;
    expect(long.turns.every((t) => t.prefillMs !== null && t.prefillCalls === t.modelCalls)).toBe(true);
    const live = reader.detail(LIVE_SESSION_ID)!;
    expect(live.turns.every((t) => t.prefillMs === null && t.ttftMs !== null)).toBe(true);
    const children = long.session.subagents.flatMap((child) => reader.detail(child.id)?.turns ?? []);
    expect(
      children.some((t) => t.prefillCalls !== null && t.modelCalls !== null && t.prefillCalls < t.modelCalls),
    ).toBe(true);
    expect(children.some((t) => t.prefillMs === null)).toBe(true);
    // A turn's prefill windows fit inside the turn.
    const measuredTurns = [...long.turns, ...children].filter((t) => t.prefillMs !== null);
    expect(measuredTurns.length).toBeGreaterThan(0);
    expect(measuredTurns.filter((t) => t.elapsedMs === null || t.prefillMs! >= t.elapsedMs)).toEqual([]);
  });

  it('is deterministic for a seed and a clock', () => {
    const other = mkdtempSync(join(tmpdir(), 'dash-dev-fixture-b-'));
    try {
      writeFixtureHome(home, { now, seed: 7 });
      writeFixtureHome(other, { now, seed: 7 });
      const a = fixturePaths(home).tracesDir;
      const b = fixturePaths(other).tracesDir;
      const names = readdirSync(a).sort();
      expect(readdirSync(b).sort()).toEqual(names);
      const read = (dir: string, root: string, name: string): string =>
        readFileSync(join(dir, name), 'utf8').split(root).join('<home>');
      for (const name of names) expect(read(b, other, name) === read(a, home, name)).toBe(true);
    } finally {
      rmSync(other, { recursive: true, force: true });
    }
  });

  it('ingests the runtime series and serves a running live file with this process as its pid', async () => {
    writeFixtureHome(home, { now });
    const paths = fixturePaths(home);
    dash = openDashboardDb(':memory:');
    const runtime = await ingestRuntime(dash, paths.runtimeDir, { now });
    expect(runtime.pruned).toBe(0);
    expect(runtime.records).toBeGreaterThan(100);
    const live = parseLiveFile(JSON.parse(readFileSync(join(paths.runtimeDir, `live-${process.pid}.json`), 'utf8')));
    expect(live).not.toBeNull();
    expect(live!.models).toHaveLength(2);
    expect(live!.turns).toHaveLength(2);
    expect(classifyProcess(live!, now, (pid) => pid === process.pid)).toBe('running');
    const supervisor = parseSupervisorFile(
      JSON.parse(readFileSync(paths.supervisorPath, 'utf8')),
      (pid) => pid === process.pid,
    );
    expect(supervisor?.state).toBe('running');
  });

  it('rewrites the runtime series of a reused home for the new pid, so the live file and the series match', async () => {
    writeFixtureHome(home, { now: now - 3_600_000 });
    const paths = fixturePaths(home);
    const next = { pid: process.pid + 100_000, startedAt: now - 6 * 3_600_000 };
    refreshRuntime(home, next, now);
    const names = readdirSync(paths.runtimeDir);
    expect(names.filter((name) => name.startsWith('live-'))).toEqual([`live-${String(next.pid)}.json`]);
    const series = names.filter((name) => name.endsWith('.jsonl'));
    expect(series.some((name) => name.endsWith(`-${String(next.pid)}.jsonl`))).toBe(true);
    expect(series.some((name) => name.endsWith(`-${String(process.pid)}.jsonl`))).toBe(false);
    dash = openDashboardDb(':memory:');
    await ingestRuntime(dash, paths.runtimeDir, { now });
    const procs = dash.sqlite.prepare('SELECT pid, started_at AS startedAt FROM runtime_procs').all() as Array<{
      pid: number;
      startedAt: number;
    }>;
    expect(procs).toContainEqual({ pid: next.pid, startedAt: next.startedAt });
    const sched = dash.sqlite.prepare('SELECT COUNT(*) AS n FROM runtime_sched').get() as { n: number };
    expect(sched.n).toBeGreaterThan(0);
  });

  it('gives every Engine section data: rejections, a failed load, a turn in prefill and a moving decode rate', () => {
    const summary = writeFixtureHome(home, { now });
    const paths = fixturePaths(home);
    const records = readdirSync(paths.runtimeDir)
      .filter((name) => name.endsWith('.jsonl') && name.includes(String(summary.live.pid)))
      .flatMap((name) => readFileSync(join(paths.runtimeDir, name), 'utf8').split('\n'))
      .filter((line) => line.length > 0)
      .map((line) => JSON.parse(line) as { k: string; ts: number; ok?: boolean; dRejected?: number });
    const lastHour = records.filter((record) => record.ts >= now - 3_600_000);
    expect(lastHour.some((record) => record.k === 'sample' && (record.dRejected ?? 0) > 0)).toBe(true);
    expect(lastHour.some((record) => record.k === 'load' && record.ok === false)).toBe(true);
    const read = (at: number) => {
      writeLiveFile(home, summary.live, at);
      return parseLiveFile(JSON.parse(readFileSync(join(paths.runtimeDir, `live-${summary.live.pid}.json`), 'utf8')))!;
    };
    const live = read(now);
    expect(live.turns.map((turn) => turn.phase)).toEqual(['decode', 'prefill']);
    expect(live.counters.rejected).toBeGreaterThan(0);
    const prefill = live.turns[1]!;
    expect(prefill.prefillDone!).toBeLessThan(prefill.prefillTotal!);
    const rates = new Set([0, 2_000, 4_000, 6_000].map((offset) => read(now + offset).decodeTps));
    expect(rates.size).toBeGreaterThan(1);
    expect(read(now).decodeTps).toBe(live.decodeTps);
  });

  it('holds seven models, two companions, a skipped folder, a cold cache and 240 cloned sessions', () => {
    writeFixtureHome(home, { now });
    const paths = fixturePaths(home);
    const models = discoverLocalModels(paths.modelsDir);
    expect(models.models).toHaveLength(7);
    expect(models.companions).toHaveLength(2);
    expect([...models.warnings].sort()).toEqual([
      'partial-download: no readable config.json; skipped',
      `${CATALOG_BLOCKED.slug}: no readable config.json; skipped`,
    ]);
    const cache = scanColdCache(paths.cacheRoot);
    expect(cache.entryCount + cache.sidecarCount).toBe(140);
    expect(cache.sidecarCount).toBeGreaterThan(0);
    const sessions = new AgentSessionsReader({ agentHome: paths.agentHome, now: () => now }).list({ limit: 500 });
    expect(sessions.total).toBeGreaterThanOrEqual(240);
    expect(new Set(sessions.cwds).size).toBeGreaterThanOrEqual(5);
  });
});

describe('dev fixture content for the manage pages', () => {
  it('blocks one catalog slug with a one-file leftover that Reinstall can replace', () => {
    writeFixtureHome(home, { now });
    const entry = catalogWithState(fixturePaths(home).modelsDir).find((item) => item.slug === CATALOG_BLOCKED.slug);
    expect(entry?.blocker).toEqual({
      kind: 'directory',
      fileCount: 1,
      sizeBytes: CATALOG_BLOCKED.sizeBytes,
      truncated: false,
      names: [CATALOG_BLOCKED.file],
      listed: false,
      modelLike: false,
      holdsModel: false,
      linkTarget: null,
      dangling: false,
    });
  });

  it('installs one catalog entry that reads as installed with an update, with no network', () => {
    writeFixtureHome(home, { now });
    const entry = catalogWithState(fixturePaths(home).modelsDir).find((item) => item.slug === CATALOG_INSTALL.slug);
    expect(entry?.installed).toBe(true);
    expect(entry?.localRevision).not.toBeNull();
    // The marker predates the entry's tokenizer source, which the page reads as an update.
    expect(entry?.assetsRepo).toBeDefined();
    expect(entry?.localAssetsRepo).toBeNull();
  });

  it('writes a long session with subagents and a live session whose last turn is still running', () => {
    writeFixtureHome(home, { now });
    const reader = new AgentSessionsReader({ agentHome: fixturePaths(home).agentHome, now: () => now });
    const long = reader.detail(LONG_SESSION_ID)!;
    expect(long.turns.length).toBeGreaterThanOrEqual(40);
    const kinds = new Set(long.transcript.map((entry) => entry.kind));
    for (const kind of ['user', 'assistant', 'thinking', 'tool', 'plan', 'subagent', 'model', 'turn-end']) {
      expect(kinds.has(kind as (typeof long.transcript)[number]['kind'])).toBe(true);
    }
    expect(long.transcript.some((entry) => entry.kind === 'tool' && entry.status === 'failed')).toBe(true);
    expect(long.session.subagents.filter((child) => !child.exists)).toHaveLength(1);

    const live = reader.detail(LIVE_SESSION_ID)!;
    expect(live.session.live).toBe(true);
    const last = live.transcript.findLast((entry) => entry.kind === 'tool');
    expect(last?.kind === 'tool' && last.status).toBe('in_progress');
    expect(live.transcript.some((entry) => entry.kind === 'turn-end' && entry.ts > last!.ts)).toBe(false);
  });

  it('gives the agent timeline a turn for each case it draws', () => {
    writeFixtureHome(home, { now });
    const reader = new AgentSessionsReader({ agentHome: fixturePaths(home).agentHome, now: () => now });
    const timeline = (id: string, turn: number) => {
      const body = reader.timeline(id, turn);
      if (body === null || body === 'no-turn') throw new Error(`no timeline for turn ${String(turn)}`);
      return body;
    };
    const main = (body: ReturnType<typeof timeline>) => body.spans.filter((span) => span.lane === body.lanes[0]!.id);
    const handoffs = (body: ReturnType<typeof timeline>) =>
      body.handoffs.map((handoff) => `${handoff.kind}${handoff.failed ? '!' : ''}`).sort();

    const fanOut = timeline(LONG_SESSION_ID, 6);
    expect(fanOut.lanes).toHaveLength(5);
    expect(fanOut.lanes.filter((lane) => lane.status === 'failed')).toHaveLength(1);
    expect(fanOut.lanes.filter((lane) => lane.noSteps === 'missing')).toHaveLength(1);
    expect(new Set(fanOut.lanes.flatMap((lane) => (lane.model === null ? [] : [lane.model]))).size).toBeGreaterThan(1);
    expect(handoffs(fanOut)).toEqual(['return', 'return', 'return', 'return!', 'spawn', 'spawn', 'spawn', 'spawn']);

    expect(main(timeline(LONG_SESSION_ID, 7)).some((span) => span.kind === 'tool' && span.calls === 3)).toBe(true);

    const noTool = timeline(LONG_SESSION_ID, 8);
    expect(main(noTool).some((span) => span.kind === 'wait' && span.reason === 'subagent' && span.calls === 0)).toBe(
      true,
    );
    expect(handoffs(noTool)).toEqual(['return', 'spawn']);

    expect(main(timeline(LONG_SESSION_ID, 9)).some((span) => span.retry !== null)).toBe(true);
    expect(main(timeline(LONG_SESSION_ID, 12)).some((span) => span.status === 'unfinished' && span.inferredEnd)).toBe(
      true,
    );
    const outlives = timeline(LONG_SESSION_ID, 18);
    const out = outlives.lanes.find((lane) => lane.carriedOut)!;
    const outSpans = outlives.spans.filter((span) => span.lane === out.id);
    const length = outlives.range!.endMs! - outlives.range!.startMs;
    expect(outSpans.at(-1)).toMatchObject({ endMs: length, cutEnd: true, cutStart: false, inferredEnd: false });
    expect(outSpans.slice(0, -1).some((span) => span.cutEnd)).toBe(false);
    const carried = timeline(LONG_SESSION_ID, 19);
    const into = carried.lanes.find((lane) => lane.carriedIn)!;
    const inSpans = carried.spans.filter((span) => span.lane === into.id);
    expect(inSpans[0]).toMatchObject({ startMs: 0, cutStart: true, cutEnd: false, inferredStart: false });
    expect(inSpans.slice(1).some((span) => span.cutStart)).toBe(false);
    expect(handoffs(carried)).toEqual(['return']);
    expect(carried.spans.filter((span) => span.lane !== into.id).some((span) => span.cutStart || span.cutEnd)).toBe(
      false,
    );

    const untimed = timeline(LONG_SESSION_ID, 20);
    expect(untimed.range).toBeNull();
    expect(untimed.notes.untimedEvents).toBeGreaterThan(0);

    const many = timeline(LONG_SESSION_ID, 30);
    expect(many.lanes).toHaveLength(10);
    expect(many.lanes.at(-1)).toMatchObject({ noSteps: 'folded', subagents: 2 });
    expect(many.notes.foldedSubagents).toBe(2);

    // A child that ran in an earlier turn too: only the steps between this turn's spawn and return are drawn.
    const again = timeline(LONG_SESSION_ID, 42);
    expect(again.stopReason).toBe('cancelled');
    expect(again.lanes.some((lane) => lane.noSteps === 'empty')).toBe(true);
    const reused = again.lanes.find(
      (lane) => lane.role === 'subagent' && fanOut.lanes.some((first) => first.sessionId === lane.sessionId),
    )!;
    const spawned = again.handoffs.find((h) => h.lane === reused.id && h.kind === 'spawn')!.atMs;
    const returned = again.handoffs.find((h) => h.lane === reused.id && h.kind === 'return')!.atMs;
    const steps = again.spans.filter((span) => span.lane === reused.id);
    expect(steps.length).toBeGreaterThan(0);
    expect(steps.every((span) => span.startMs >= spawned && span.endMs !== null && span.endMs <= returned)).toBe(true);

    for (let turn = 1; turn <= reader.detail(LONG_SESSION_ID)!.turns.length; turn += 1) {
      const body = timeline(LONG_SESSION_ID, turn);
      expect(body.lanes.filter((lane) => lane.noSteps === 'out-of-range' || lane.clamped)).toEqual([]);
    }

    // Turn 26 starts a background workflow; its stop wakes the agent for turn 27, which nobody typed.
    const starts = timeline(LONG_SESSION_ID, 26);
    expect(starts.lanes.filter((lane) => lane.carriedOut)).toHaveLength(1);
    const detail = reader.detail(LONG_SESSION_ID)!;
    const wakes = detail.transcript.filter((entry) => entry.kind === 'wake');
    expect(wakes).toEqual([
      expect.objectContaining({ cause: 'workflow', workflow: { name: 'deep-research', status: 'complete' } }),
    ]);
    const woken = timeline(LONG_SESSION_ID, 27);
    const range = woken.range!;
    expect(range.turnStartMs).toBeGreaterThan(range.startMs);
    const turnStart = range.turnStartMs - range.startMs;
    const wokenMain = main(woken);
    expect(wokenMain[0]).toMatchObject({ kind: 'idle', startMs: 0, endMs: turnStart });
    expect(
      wokenMain.some((span) => span.kind === 'wait' && span.reason === 'subagent' && span.startMs < turnStart),
    ).toBe(false);
    expect(woken.lanes).toHaveLength(4);
    expect(woken.lanes.filter((lane) => lane.carriedIn)).toHaveLength(1);
    const carriedLane = woken.lanes.find((lane) => lane.carriedIn)!;
    expect(woken.spans.some((span) => span.lane === carriedLane.id)).toBe(true);

    const open = timeline(LIVE_SESSION_ID, reader.detail(LIVE_SESSION_ID)!.turns.length + 1);
    expect(open).toMatchObject({ open: true, live: true });
    expect(main(open).at(-1)).toMatchObject({ kind: 'wait', reason: 'subagent', endMs: null, status: 'running' });
    expect(open.lanes.slice(1).filter((lane) => lane.status === 'running')).toHaveLength(1);
  });

  it('keeps the live session registered to the pid that refreshes the runtime', () => {
    writeFixtureHome(home, { now });
    refreshRuntime(home, { pid: process.ppid, startedAt: now - 3_600_000 }, now);
    const registry = JSON.parse(
      readFileSync(join(fixturePaths(home).agentHome, 'active_sessions.json'), 'utf8'),
    ) as unknown;
    expect(registry).toEqual([{ session_id: LIVE_SESSION_ID, pid: process.ppid }]);
  });

  it('gives each coding agent its own state: current, old command, broken block', () => {
    writeFixtureHome(home, { now });
    const claude = readFileSync(join(home, '.claude', 'CLAUDE.md'), 'utf8');
    const codex = readFileSync(join(home, '.codex', 'AGENTS.md'), 'utf8');
    const grok = readFileSync(join(home, '.grok', 'AGENTS.md'), 'utf8');
    expect(claude.includes(MANAGED_BEGIN) && claude.includes(MANAGED_END)).toBe(true);
    expect(codex.includes(MANAGED_BEGIN) && codex.includes(MANAGED_END)).toBe(true);
    expect(codex).not.toEqual(claude.slice(claude.indexOf(MANAGED_BEGIN)));
    expect(grok.includes(MANAGED_BEGIN) && !grok.includes(MANAGED_END)).toBe(true);
  });
});

describe('dev fixture home guard', () => {
  it('refuses the user home, the real ~/.mlx-node, anything inside it and every ancestor', () => {
    const userHome = join(home, 'Users', 'dev');
    for (const target of [
      userHome,
      join(userHome, '.mlx-node'),
      join(userHome, '.mlx-node', 'models'),
      dirname(userHome),
      '/',
    ]) {
      expect(() => assertFixtureHome(target, userHome)).toThrow('refusing fixture home');
    }
    expect(() => assertFixtureHome(join(userHome, 'work', '.dev-home'), userHome)).not.toThrow();
    expect(() => assertFixtureHome(join(userHome, '.mlx-node-dev'), userHome)).not.toThrow();
  });

  it('refuses a home that reaches ~/.mlx-node through a symbolic link', () => {
    const userHome = join(home, 'user');
    mkdirSync(join(userHome, '.mlx-node'), { recursive: true });
    symlinkSync(join(userHome, '.mlx-node'), join(home, 'link'));
    expect(() => assertFixtureHome(join(home, 'link', 'sub'), userHome)).toThrow('refusing fixture home');
    expect(() => assertFixtureHome(join(home, 'link'), userHome)).toThrow('refusing fixture home');
  });

  it('writes nothing when a link in a marked fixture home leads outside it', () => {
    const userHome = join(home, 'user');
    const realModel = join(userHome, '.mlx-node', 'models', 'Qwen3-8B-4bit');
    const victim = join(realModel, 'model.safetensors');
    mkdirSync(realModel, { recursive: true });
    writeFileSync(victim, 'REAL');
    const notes = join(userHome, '.claude', 'CLAUDE.md');
    mkdirSync(dirname(notes), { recursive: true });
    writeFileSync(notes, 'my notes');

    const cases: Array<[string, string]> = [
      ['models', join(userHome, '.mlx-node', 'models')],
      [join('models', 'Qwen3-8B-4bit'), realModel],
      ['.claude', join(userHome, '.claude')],
      [join('cold', 'gone'), join(userHome, 'missing')],
    ];
    for (const [entry, target] of cases) {
      const fixture = mkdtempSync(join(home, 'fixture-'));
      writeFileSync(join(fixture, FIXTURE_MARKER), 'mlx-node dev fixture\n');
      mkdirSync(dirname(join(fixture, entry)), { recursive: true });
      symlinkSync(target, join(fixture, entry));
      expect(() => writeFixtureHome(fixture, { now })).toThrow('refusing fixture home');
      expect(readdirSync(fixture).sort()).toEqual([FIXTURE_MARKER, entry.split('/')[0]!].sort());
    }
    expect(readFileSync(victim, 'utf8')).toBe('REAL');
    expect(readdirSync(join(userHome, '.mlx-node', 'models'))).toEqual(['Qwen3-8B-4bit']);
    expect(readFileSync(notes, 'utf8')).toBe('my notes');
    expect(existsSync(join(userHome, 'missing'))).toBe(false);

    const hardLinked = mkdtempSync(join(home, 'fixture-'));
    writeFileSync(join(hardLinked, FIXTURE_MARKER), 'mlx-node dev fixture\n');
    mkdirSync(join(hardLinked, 'cold'));
    linkSync(victim, join(hardLinked, 'cold', 'entry.safetensors'));
    expect(() => writeFixtureHome(hardLinked, { now })).toThrow('another hard link');
    expect(readFileSync(victim, 'utf8')).toBe('REAL');
  });

  it('refuses the heartbeat write when the runtime folder links outside the home', () => {
    const outside = join(home, 'outside');
    mkdirSync(outside);
    const fixture = join(home, 'fixture');
    mkdirSync(join(fixture, 'metrics'), { recursive: true });
    symlinkSync(outside, join(fixture, 'metrics', 'runtime'));
    expect(() => writeLiveFile(fixture, { pid: process.pid, startedAt: now })).toThrow('outside');
    expect(readdirSync(outside)).toEqual([]);
  });

  it('checks --home while parsing the arguments', () => {
    expect(() => fixtureArgs(['--fresh', '--home', '/'], join(home, 'fixture'))).toThrow('refusing fixture home');
  });

  it('deletes only a fixture home, and never writes into a directory that holds other files', () => {
    const userHome = join(home, 'user');
    const real = join(home, 'real');
    mkdirSync(real);
    writeFileSync(join(real, 'keep.txt'), 'mine');
    expect(() => resetHome(real, userHome)).toThrow('not a dev fixture home');
    expect(() => writeFixtureHome(real, { now })).toThrow('not a dev fixture');
    expect(readdirSync(real)).toEqual(['keep.txt']);

    const fixture = join(home, 'fixture');
    writeFixtureHome(fixture, { now });
    resetHome(fixture, userHome);
    expect(existsSync(fixture)).toBe(false);
    resetHome(fixture, userHome);
  });
});
