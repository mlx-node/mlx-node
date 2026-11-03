import { execFile } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import { describe, expect, it } from 'vite-plus/test';

import {
  buildReport,
  legacyTurns,
  parseLog,
  quantiles,
  readFirstTokens,
  type FirstToken,
  type ProfileRow,
  type Report,
} from '../../scripts/dflash2-trace-summary.js';

const SCRIPT = fileURLToPath(new URL('../../scripts/dflash2-trace-summary.ts', import.meta.url));
const FIXTURE = fileURLToPath(new URL('../../scripts/fixtures/dflash2-trace-sample.log', import.meta.url));

function rawRecords(event: string): Record<string, number | string>[] {
  const out: Record<string, number | string>[] = [];
  for (const line of readFileSync(FIXTURE, 'utf8').split('\n')) {
    try {
      const record = JSON.parse(line) as Record<string, number | string>;
      if (record.event === event) out.push(record);
    } catch {
      // the fixture ends with a torn line
    }
  }
  return out;
}

function sum(records: Record<string, number | string>[], pick: (r: Record<string, number>) => number): number {
  return records.reduce((acc, record) => acc + pick(record as Record<string, number>), 0);
}

describe('dflash2-trace-summary on the fixture', () => {
  const report = buildReport(parseLog(readFileSync(FIXTURE, 'utf8')), undefined);
  const events = rawRecords('dflash2_prefix');
  const profiles = rawRecords('decode_profile_summary');

  it('totals equal the summed event fields', () => {
    expect(events).toHaveLength(9);
    const total = report.events!.total;
    expect(total.turns).toBe(events.length);
    expect(total.promptTokens).toBe(sum(events, (r) => r.prompt_tokens));
    expect(total.hitTokens).toBe(sum(events, (r) => r.hit_tokens));
    expect(total.prefilledTokens).toBe(sum(events, (r) => r.prefilled_tokens));
    expect(total.idealPrefill).toBe(sum(events, (r) => r.prompt_tokens - r.ideal_tokens));
    expect(total.waste).toBe(total.prefilledTokens - total.idealPrefill);
    expect(total.ttftMs.max).toBe(Math.max(...events.map((r) => r.ttft_ms as number)));
  });

  it('per-reason rows partition the totals', () => {
    const { reasons, total } = report.events!;
    expect(new Set(reasons.map((r) => r.reason))).toEqual(
      new Set([
        'cold_first',
        'hit:live',
        'cross_session',
        'rerender',
        'flat_dirty:paged_turn_stream_core',
        'invalidated:set_cache_owner_id',
      ]),
    );
    for (const key of ['turns', 'promptTokens', 'hitTokens', 'prefilledTokens', 'idealPrefill', 'waste'] as const) {
      expect(reasons.reduce((acc, r) => acc + r[key], 0)).toBe(total[key]);
    }
    const live = reasons.find((r) => r.reason === 'hit:live')!;
    expect(live).toMatchObject({ turns: 2, hitTokens: 4276 + 3810, prefilledTokens: 224 + 90, waste: 0 });
    const cross = reasons.find((r) => r.reason === 'cross_session')!;
    expect(cross).toMatchObject({ turns: 2, prefilledTokens: 4900 + 3420, idealPrefill: 4900 - 4656 + 3420 - 3088 });
  });

  it('reports the profile rows separately and skips the torn line', () => {
    expect(report.profile.rows).toBe(profiles.length);
    expect(report.profile.promptTokens).toBe(sum(profiles, (r) => r.prompt_tokens));
    expect(report.profile.legacy).toBeNull();
    expect(report.unparsedLines).toBe(1);
  });

  it('prints the same report from the CLI', async () => {
    const { stdout } = await promisify(execFile)('vp', ['exec', 'oxnode', SCRIPT, FIXTURE, '--json']);
    expect(JSON.parse(stdout) as Report).toEqual(report);
  });
});

describe('legacy classification of decode_profile_summary rows', () => {
  const start = Date.parse('2026-10-29T09:00:00.000Z');
  const row = (atSec: number, prompt: number, generated: number): ProfileRow => ({
    timestamp: new Date(start + atSec * 1000 + 2000).toISOString(),
    promptTokens: prompt,
    generatedTokens: generated,
    decodeMs: 2000,
    ttftMs: 1000,
  });
  const token = (session: string, atSec: number): FirstToken => ({ session, atMs: start + atSec * 1000 + 30 });

  it('classifies one flat state shared by two sessions and a side call', () => {
    const rows = [
      row(0, 1000, 50), // a cold
      row(10, 200, 40), // a live: cached 1050
      row(20, 300, 10), // side call
      row(30, 900, 20), // b cold
      row(40, 1500, 30), // a cross_session: previous context 1290
      row(50, 950, 10), // b cross_session: previous context 920
      row(60, 40, 10), // b live
    ];
    const tokens = [token('a', 0), token('a', 10), token('b', 30), token('a', 40), token('b', 50), token('b', 60)];
    const turns = legacyTurns(rows, tokens);
    expect(turns.map((t) => t.reason)).toEqual([
      'cold_first',
      'hit:live',
      'extra_request',
      'cold_first',
      'cross_session',
      'cross_session',
      'hit:live',
    ]);
    expect(turns[1]).toMatchObject({ hitTokens: 1050, promptTokens: 1250, prefilledTokens: 200, idealPrefill: 200 });
    expect(turns[4]).toMatchObject({ prefilledTokens: 1500, idealPrefill: 1500 - 1290 });
    expect(turns[5]).toMatchObject({ prefilledTokens: 950, idealPrefill: 950 - 920 });
    expect(turns[6]).toMatchObject({ hitTokens: 960, prefilledTokens: 40 });
  });

  it('marks a miss smaller than the session context unknown', () => {
    const rows = [row(0, 1000, 50), row(10, 500, 10), row(20, 400, 10)];
    const turns = legacyTurns(rows, [token('a', 0), token('b', 10), token('a', 20)]);
    expect(turns.map((t) => t.reason)).toEqual(['cold_first', 'cold_first', 'unknown']);
  });

  it('matches each first_token once and only within the tolerance', () => {
    const rows = [row(0, 100, 1), row(0.2, 100, 1), row(30, 100, 1)];
    const turns = legacyTurns(rows, [token('a', 0), { session: 'b', atMs: start + 30_000 + 900 }]);
    expect(turns.map((t) => t.reason)).toEqual(['cold_first', 'extra_request', 'extra_request']);
  });

  it('leaves rows unattributed without sessions', () => {
    expect(legacyTurns([row(0, 10, 1)], undefined)[0].reason).toBe('unattributed');
  });
});

it('takes nearest-rank quantiles', () => {
  expect(quantiles([5, 1, 4, 2, 3, 6, 7, 8, 9, 10])).toEqual({ p50: 5, p90: 9, p99: 10, max: 10 });
  expect(quantiles([])).toEqual({ p50: 0, p90: 0, p99: 0, max: 0 });
});

const baselineLog = process.env.MLX_DFLASH2_BASELINE_LOG;
const baselineSessions = process.env.MLX_DFLASH2_BASELINE_SESSIONS;

it.skipIf(baselineLog === undefined || baselineSessions === undefined)(
  'reproduces the recorded 4-session baseline',
  () => {
    const report = buildReport(parseLog(readFileSync(baselineLog!, 'utf8')), readFirstTokens(baselineSessions!));
    const legacy = report.profile.legacy!;
    const byReason = new Map(legacy.reasons.map((r) => [r.reason, r]));
    expect(byReason.get('cross_session')).toMatchObject({ turns: 41, prefilledTokens: 2_933_177 });
    expect(byReason.has('unknown')).toBe(false);
    expect(legacy.total.turns).toBe(55);
  },
);
