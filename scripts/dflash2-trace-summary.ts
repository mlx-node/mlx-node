#!/usr/bin/env oxnode
/**
 * Summarize DFlash2 prefix reuse from an `inference.log`.
 *
 *   oxnode scripts/dflash2-trace-summary.ts <inference.log> [--sessions <dir>] [--json]
 *
 * Two sources, reported separately:
 *
 * - `dflash2_prefix` events (one per DFlash2 turn): per miss reason (`hit:<kind>`
 *   when the turn resumed at its ideal), the turns, prompt / hit / prefilled
 *   tokens, the ideal prefill (`prompt_tokens - ideal_tokens`), the waste
 *   (prefilled - ideal) and TTFT quantiles.
 *
 * - `decode_profile_summary` rows (every family). For a DFlash2 row,
 *   `prompt_tokens` is the UNCACHED prefill length (`prefill.len()` in
 *   `dflash2_chat_turn`), not the prompt: it equals the full prompt only on a
 *   miss. `prefill_ms` also leaves out the last lazy prefill chunk, so TTFT is
 *   read from `ttft_ms`.
 *
 *   A log written before the `dflash2_prefix` event existed is classified from
 *   these rows alone, which needs `--sessions`: a directory of agent session
 *   folders whose `events.jsonl` holds `first_token` events. A row is matched
 *   to the session whose `first_token` lands at `timestamp - decode_ms`; an
 *   unmatched row is an `extra_request` (a side call such as a title request).
 *   Such a log came from one model-wide flat DFlash2 state, so a row whose
 *   previous row is the same session is a `hit:live` (cached = that session's
 *   previous prompt + generated tokens), a session's first row is `cold_first`,
 *   and any other row is a `cross_session` full miss whose ideal prefill is the
 *   tokens added since that session's previous turn. A miss smaller than the
 *   session's previous context contradicts this model and is `unknown`.
 */

/// <reference types="node" />

import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

export interface PrefixEvent {
  promptTokens: number;
  hitKind: string;
  hitTokens: number;
  prefilledTokens: number;
  idealTokens: number;
  missReason: string;
  ttftMs: number;
}

export interface ProfileRow {
  timestamp: string;
  promptTokens: number;
  generatedTokens: number;
  decodeMs: number;
  ttftMs: number;
}

export interface ParsedLog {
  events: PrefixEvent[];
  profiles: ProfileRow[];
  /** Lines that are not JSON (a torn last line of a live log). */
  unparsedLines: number;
}

export interface Turn {
  reason: string;
  promptTokens: number;
  hitTokens: number;
  prefilledTokens: number;
  idealPrefill: number;
  ttftMs: number;
}

export interface Quantiles {
  p50: number;
  p90: number;
  p99: number;
  max: number;
}

export interface ReasonSummary {
  reason: string;
  turns: number;
  promptTokens: number;
  hitTokens: number;
  prefilledTokens: number;
  idealPrefill: number;
  waste: number;
  ttftMs: Quantiles;
}

export interface Summary {
  reasons: ReasonSummary[];
  total: ReasonSummary;
}

export interface FirstToken {
  session: string;
  atMs: number;
}

export interface Report {
  events: Summary | null;
  profile: {
    rows: number;
    promptTokens: number;
    ttftMs: Quantiles;
    /** Present only for a log without `dflash2_prefix` events. */
    legacy: Summary | null;
    /** Whether `--sessions` attributed the legacy rows. */
    attributed: boolean;
  };
  unparsedLines: number;
}

/** A row's first token and the session's `first_token` event agree to within milliseconds. */
const ATTRIBUTION_TOLERANCE_MS = 500;

function num(record: Record<string, unknown>, key: string): number {
  const value = record[key];
  if (typeof value !== 'number') throw new Error(`field ${key} is ${JSON.stringify(value)}, expected a number`);
  return value;
}

function str(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  if (typeof value !== 'string') throw new Error(`field ${key} is ${JSON.stringify(value)}, expected a string`);
  return value;
}

export function parseLog(text: string): ParsedLog {
  const events: PrefixEvent[] = [];
  const profiles: ProfileRow[] = [];
  let unparsedLines = 0;
  for (const line of text.split('\n')) {
    if (line.trim() === '') continue;
    let record: Record<string, unknown>;
    try {
      record = JSON.parse(line) as Record<string, unknown>;
    } catch {
      unparsedLines += 1;
      continue;
    }
    if (record.event === 'dflash2_prefix') {
      events.push({
        promptTokens: num(record, 'prompt_tokens'),
        hitKind: str(record, 'hit_kind'),
        hitTokens: num(record, 'hit_tokens'),
        prefilledTokens: num(record, 'prefilled_tokens'),
        idealTokens: num(record, 'ideal_tokens'),
        missReason: str(record, 'miss_reason'),
        ttftMs: num(record, 'ttft_ms'),
      });
    } else if (record.event === 'decode_profile_summary') {
      profiles.push({
        timestamp: str(record, 'timestamp'),
        promptTokens: num(record, 'prompt_tokens'),
        generatedTokens: num(record, 'generated_tokens'),
        decodeMs: num(record, 'decode_ms'),
        ttftMs: num(record, 'ttft_ms'),
      });
    }
  }
  return { events, profiles, unparsedLines };
}

export function eventTurns(events: PrefixEvent[]): Turn[] {
  return events.map((event) => ({
    reason: event.missReason === 'none' ? `hit:${event.hitKind}` : event.missReason,
    promptTokens: event.promptTokens,
    hitTokens: event.hitTokens,
    prefilledTokens: event.prefilledTokens,
    idealPrefill: event.promptTokens - event.idealTokens,
    ttftMs: event.ttftMs,
  }));
}

export function readFirstTokens(sessionsDir: string): FirstToken[] {
  const out: FirstToken[] = [];
  for (const entry of readdirSync(sessionsDir, { withFileTypes: true })) {
    const path = join(sessionsDir, entry.name, 'events.jsonl');
    if (!entry.isDirectory() || !existsSync(path)) continue;
    for (const line of readFileSync(path, 'utf8').split('\n')) {
      if (!line.includes('"first_token"')) continue;
      const record = JSON.parse(line) as Record<string, unknown>;
      if (record.type === 'first_token') out.push({ session: entry.name, atMs: Date.parse(str(record, 'ts')) });
    }
  }
  return out;
}

/** The session each row belongs to, or `undefined` for a row no `first_token` matches. */
export function attributeRows(rows: ProfileRow[], firstTokens: FirstToken[]): (string | undefined)[] {
  const used = new Set<number>();
  return rows.map((row) => {
    const at = Date.parse(row.timestamp) - row.decodeMs;
    let best: number | undefined;
    for (const [index, token] of firstTokens.entries()) {
      const gap = Math.abs(token.atMs - at);
      if (used.has(index) || gap > ATTRIBUTION_TOLERANCE_MS) continue;
      if (best === undefined || gap < Math.abs(firstTokens[best].atMs - at)) best = index;
    }
    if (best === undefined) return undefined;
    used.add(best);
    return firstTokens[best].session;
  });
}

export function legacyTurns(rows: ProfileRow[], firstTokens: FirstToken[] | undefined): Turn[] {
  const plain = (row: ProfileRow, reason: string): Turn => ({
    reason,
    promptTokens: row.promptTokens,
    hitTokens: 0,
    prefilledTokens: row.promptTokens,
    idealPrefill: row.promptTokens,
    ttftMs: row.ttftMs,
  });
  if (firstTokens === undefined) return rows.map((row) => plain(row, 'unattributed'));

  const sessions = attributeRows(rows, firstTokens);
  const context = new Map<string, number>();
  return rows.map((row, index) => {
    const session = sessions[index];
    if (session === undefined) return plain(row, 'extra_request');
    const previous = context.get(session);
    let turn: Turn;
    if (previous === undefined) {
      turn = plain(row, 'cold_first');
    } else if (index > 0 && sessions[index - 1] === session) {
      turn = { ...plain(row, 'hit:live'), promptTokens: previous + row.promptTokens, hitTokens: previous };
    } else if (row.promptTokens < previous) {
      turn = plain(row, 'unknown');
    } else {
      turn = { ...plain(row, 'cross_session'), idealPrefill: row.promptTokens - previous };
    }
    context.set(session, turn.promptTokens + row.generatedTokens);
    return turn;
  });
}

export function quantiles(values: number[]): Quantiles {
  const sorted = [...values].sort((a, b) => a - b);
  const rank = (q: number): number =>
    sorted.length === 0 ? 0 : sorted[Math.min(sorted.length - 1, Math.ceil(q * sorted.length) - 1)];
  return { p50: rank(0.5), p90: rank(0.9), p99: rank(0.99), max: sorted.at(-1) ?? 0 };
}

function reduce(reason: string, turns: Turn[]): ReasonSummary {
  const sum = (pick: (turn: Turn) => number): number => turns.reduce((acc, turn) => acc + pick(turn), 0);
  const prefilledTokens = sum((turn) => turn.prefilledTokens);
  const idealPrefill = sum((turn) => turn.idealPrefill);
  return {
    reason,
    turns: turns.length,
    promptTokens: sum((turn) => turn.promptTokens),
    hitTokens: sum((turn) => turn.hitTokens),
    prefilledTokens,
    idealPrefill,
    waste: prefilledTokens - idealPrefill,
    ttftMs: quantiles(turns.map((turn) => turn.ttftMs)),
  };
}

export function summarize(turns: Turn[]): Summary {
  const groups = new Map<string, Turn[]>();
  for (const turn of turns) {
    const group = groups.get(turn.reason);
    if (group === undefined) groups.set(turn.reason, [turn]);
    else group.push(turn);
  }
  const reasons = [...groups].map(([reason, group]) => reduce(reason, group));
  reasons.sort((a, b) => b.prefilledTokens - a.prefilledTokens || a.reason.localeCompare(b.reason));
  return { reasons, total: reduce('total', turns) };
}

export function buildReport(parsed: ParsedLog, firstTokens: FirstToken[] | undefined): Report {
  return {
    events: parsed.events.length === 0 ? null : summarize(eventTurns(parsed.events)),
    profile: {
      rows: parsed.profiles.length,
      promptTokens: parsed.profiles.reduce((acc, row) => acc + row.promptTokens, 0),
      ttftMs: quantiles(parsed.profiles.map((row) => row.ttftMs)),
      legacy: parsed.events.length === 0 ? summarize(legacyTurns(parsed.profiles, firstTokens)) : null,
      attributed: firstTokens !== undefined,
    },
    unparsedLines: parsed.unparsedLines,
  };
}

const COLUMNS = ['reason', 'turns', 'prompt', 'hit', 'prefilled', 'ideal', 'waste', 'ttft p50', 'p90', 'p99', 'max'];

function table(summary: Summary): string {
  const count = (value: number): string => value.toLocaleString('en-US');
  const seconds = (ms: number): string => `${(ms / 1000).toFixed(1)}s`;
  const rows = [...summary.reasons, summary.total].map((r) => [
    r.reason,
    count(r.turns),
    count(r.promptTokens),
    count(r.hitTokens),
    count(r.prefilledTokens),
    count(r.idealPrefill),
    count(r.waste),
    seconds(r.ttftMs.p50),
    seconds(r.ttftMs.p90),
    seconds(r.ttftMs.p99),
    seconds(r.ttftMs.max),
  ]);
  const widths = COLUMNS.map((title, i) => Math.max(title.length, ...rows.map((row) => row[i].length)));
  const line = (cells: string[]): string =>
    cells.map((cell, i) => (i === 0 ? cell.padEnd(widths[i]) : cell.padStart(widths[i]))).join('  ');
  return [line(COLUMNS), ...rows.map(line)].join('\n');
}

export function renderReport(report: Report): string {
  const out: string[] = [];
  if (report.events !== null) {
    out.push('dflash2_prefix events', table(report.events), '');
  } else {
    out.push('dflash2_prefix events: none in this log', '');
  }
  const { profile } = report;
  out.push(
    `decode_profile_summary: ${profile.rows} rows, ${profile.promptTokens.toLocaleString('en-US')} uncached prompt tokens, ` +
      `ttft p50 ${(profile.ttftMs.p50 / 1000).toFixed(1)}s p90 ${(profile.ttftMs.p90 / 1000).toFixed(1)}s`,
  );
  if (profile.legacy !== null) {
    const heading = profile.attributed
      ? 'decode_profile_summary classified as one flat DFlash2 state'
      : 'decode_profile_summary (pass --sessions to classify)';
    out.push('', heading, table(profile.legacy));
  }
  if (report.unparsedLines > 0) out.push('', `skipped ${report.unparsedLines} non-JSON lines`);
  return out.join('\n');
}

function main(): void {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: { sessions: { type: 'string' }, json: { type: 'boolean', default: false } },
  });
  if (positionals.length !== 1) {
    throw new Error('usage: dflash2-trace-summary.ts <inference.log> [--sessions <dir>] [--json]');
  }
  const parsed = parseLog(readFileSync(positionals[0], 'utf8'));
  const firstTokens = values.sessions === undefined ? undefined : readFirstTokens(values.sessions);
  const report = buildReport(parsed, firstTokens);
  console.log(values.json ? JSON.stringify(report, null, 2) : renderReport(report));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
