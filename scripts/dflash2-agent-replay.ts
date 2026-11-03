#!/usr/bin/env oxnode
/**
 * Replay a recorded `mlx agent` workload (gate G2).
 *
 *   oxnode scripts/dflash2-agent-replay.ts --offline [--sessions <dir>] [--log <inference.log>] [--json]
 *   oxnode scripts/dflash2-agent-replay.ts [--sessions <dir>] [--log <inference.log>] [--limit <n>] [--json]
 *
 * Every session folder under `--sessions` (default: the 2026-10-29 splash run)
 * holds a `chat_history.jsonl`. The k-th model response of a session (its
 * `reasoning` rows plus the `assistant` row) answers one request: every row
 * before that response, rebuilt as an Anthropic `/v1/messages` body the way the
 * agent built it (`build_messages_request` in grok-build's
 * `xai-grok-sampling-types/src/conversation/messages.rs`):
 *
 * - `system` from the session's `system_prompt.txt` (every `system` row must equal it)
 * - each `user` row is its own user message
 * - a `reasoning` row is a `thinking` block (summary texts joined by `\n`) that
 *   opens the next assistant message; the `assistant` row adds its text and a
 *   `tool_use` block per tool call (arguments parsed as JSON)
 * - consecutive `tool_result` rows form one user message
 * - tools from the session's `tool_definitions.json`
 *
 * Request k maps to the k-th `first_token` event of the session's
 * `events.jsonl` (an agent loop that never streamed a token sent no answered
 * request). Each `decode_profile_summary` row of the recorded `inference.log`
 * is attributed to a session through those events (see
 * `dflash2-trace-summary.ts`); an unattributed row is an `extra_request` whose
 * content was not recorded.
 *
 * `--offline` loads no weights. It maps every body through the server's
 * `mapAnthropicRequest`, then `cargo test --test dflash2_replay_offline` renders
 * each request through the target tokenizer and classifies the log's row order
 * with the turn ledger under today's flat rule (one live state). It fails
 * unless:
 * - every session has exactly one request per `first_token` event and per
 *   attributed log row (the recorded run: 19 + 3 x 11);
 * - every request's replayed prefill equals the recorded uncached prompt
 *   (`decode_profile_summary.prompt_tokens`): a full-miss row checks the
 *   reconstructed prompt length, a hit row also the reused history;
 * - no request is classified `unknown` or `state_existed_unused`;
 * - every request that misses after its session's first is `cross_session`.
 *
 * Without `--offline` the script launches `mlx serve` on `--models-dir`
 * (default `/Volumes/P4510/.cache/models`), sends the requests one at a time in
 * `loop_started` order with `max_tokens: 64` and `temperature: 0`, then
 * summarizes the server's `inference.log` and runs the offline classifier on
 * the same request list. Unrecorded requests are not sent. The server writes
 * `inference.log` into `$MLX_DFLASH2_REPLAY_OUT`, which must be set; the
 * directory reaches the server through the environment only.
 *
 * `$MLX_DFLASH2_REPLAY_OUT` (default: a fresh temp directory) also receives
 * `requests.json`, the classifier's `offline.json` and `report.json`.
 * The target tokenizer is `--tokenizer`, else `tokenizer.json` next to
 * `$MLX_TEST_QWEN38_TARGET_PATH`, else the Qwen3.8-27B GGUF directory.
 */

/// <reference types="node" />

import { spawn, spawnSync } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

import type { ChatMessage, ToolDefinition } from '@mlx-node/core';

import { mapAnthropicRequest } from '../packages/server/src/mappers/anthropic-request.js';
import type { AnthropicContentBlock, AnthropicMessagesRequest } from '../packages/server/src/types-anthropic.js';
import {
  attributeRows,
  buildReport,
  legacyTurns,
  parseLog,
  renderReport,
  type FirstToken,
  type ProfileRow,
} from './dflash2-trace-summary.js';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const RECORDED_SESSIONS_DIR = join(homedir(), '.mlx-agent/sessions/%2FUsers%2Fbrooklyn%2Fworkspace%2Fgithub%2Fsplash');
const RECORDED_LOG = join(homedir(), '.mlx-node/logs/agent/2026-10-29T03-29-39-059Z-pid-63376/inference.log');
/** Requests per session in the recorded run: the main session and three subagents. */
const RECORDED_REQUESTS: Record<string, number> = {
  '01a185b5-5748-7721-a722-4cfc60df99a7': 19,
  '01a185c3-418e-7ea1-92d5-f35e9a11f1ef': 11,
  '01a185c3-418e-7ea1-92d5-f36748dac80a': 11,
  '01a185c3-418e-7ea1-92d5-f37b419c4bc9': 11,
};
const DEFAULT_TARGET_DIR = '/Volumes/P4510/.cache/models/qwen3.8-27b-gguf';
const DEFAULT_MODELS_DIR = '/Volumes/P4510/.cache/models';
const REPLAY_MAX_TOKENS = 64;
const SERVE_READY_TIMEOUT_MS = 120_000;
const INFERENCE_LOG_FILTER = 'mlx_core::inference=info,mlx_core::decode=info';

type TextPart = { type: 'text'; text: string };

export type HistoryRow =
  | { type: 'system'; content: string }
  | { type: 'user'; content: string | TextPart[]; synthetic_reason?: string }
  | {
      type: 'reasoning';
      summary?: { type: string; text: string }[];
      content?: { text: string }[] | null;
      encrypted_content?: string | null;
    }
  | {
      type: 'assistant';
      content?: string | null;
      tool_calls?: { id: string; name: string; arguments: string }[] | null;
      model_id?: string;
    }
  | { type: 'tool_result'; tool_call_id: string; content: string };

export interface ToolDefinitionRow {
  type?: string;
  function: { name: string; description?: string; parameters?: Record<string, unknown> };
}

export interface Session {
  id: string;
  system: string;
  tools: ToolDefinitionRow[];
  rows: HistoryRow[];
  loopStartedMs: number[];
  firstTokenMs: number[];
}

export interface AgentRequest {
  /** `<session id>#<k>`, k counted from 1. */
  label: string;
  session: string;
  index: number;
  body: AnthropicMessagesRequest;
  /** The agent loop that sent it: the last `loop_started` before its first token. */
  loopStartedMs: number;
  firstTokenMs: number;
}

/** The request shape `dflash2_replay_offline` reads (`RecordedRequest` in Rust). */
export interface OfflineRequest {
  label: string;
  session: string | null;
  messages?: object[];
  tools?: ToolDefinition[] | null;
  reasoning_effort?: string | null;
  prompt_tokens?: number;
}

/** One row of the classifier's result (`ReplayedTurn` in Rust). */
export interface OfflineTurn {
  label: string;
  session: string | null;
  prompt_tokens: number;
  generated_tokens: number | null;
  gen_suffix_len: number;
  replay_point: number | null;
  hit_kind: string;
  hit_tokens: number;
  prefilled_tokens: number;
  miss_reason: string;
  shadow_hit: number;
  ideal_tokens: number;
  ideal_paged_tokens: number;
  diverge_at: number;
}

export interface OfflineTotals {
  requests: number;
  promptTokens: number;
  prefilledToday: number;
  idealFlat: number;
  idealPaged: number;
  reasons: Record<string, number>;
}

function readJsonLines<T>(path: string): T[] {
  return readFileSync(path, 'utf8')
    .split('\n')
    .filter((line) => line.trim() !== '')
    .map((line, index) => {
      try {
        return JSON.parse(line) as T;
      } catch (error) {
        throw new Error(`${path}:${index + 1}: ${(error as Error).message}`, { cause: error });
      }
    });
}

export function readSession(dir: string, id: string): Session {
  const sessionDir = join(dir, id);
  const loopStartedMs: number[] = [];
  const firstTokenMs: number[] = [];
  for (const event of readJsonLines<{ ts: string; type: string }>(join(sessionDir, 'events.jsonl'))) {
    if (event.type === 'loop_started') loopStartedMs.push(Date.parse(event.ts));
    else if (event.type === 'first_token') firstTokenMs.push(Date.parse(event.ts));
  }
  return {
    id,
    system: readFileSync(join(sessionDir, 'system_prompt.txt'), 'utf8'),
    tools: JSON.parse(readFileSync(join(sessionDir, 'tool_definitions.json'), 'utf8')) as ToolDefinitionRow[],
    rows: readJsonLines<HistoryRow>(join(sessionDir, 'chat_history.jsonl')),
    loopStartedMs,
    firstTokenMs,
  };
}

export function readSessions(dir: string): Session[] {
  return readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && existsSync(join(dir, entry.name, 'chat_history.jsonl')))
    .map((entry) => readSession(dir, entry.name))
    .sort((a, b) => (a.firstTokenMs[0] ?? Infinity) - (b.firstTokenMs[0] ?? Infinity) || a.id.localeCompare(b.id));
}

/** `reasoning_item_text`: summary texts, then content texts, joined by a newline. */
export function reasoningText(row: Extract<HistoryRow, { type: 'reasoning' }>): string {
  return [...(row.summary ?? []).map((part) => part.text), ...(row.content ?? []).map((part) => part.text)].join('\n');
}

/** Tool-call ids keep only alphanumerics, `_` and `-`. */
function sanitizeToolCallId(id: string): string {
  return id.replace(/[^\p{L}\p{N}_-]/gu, '_');
}

function toolInput(args: string): Record<string, unknown> {
  try {
    const input = JSON.parse(args) as unknown;
    return input !== null && typeof input === 'object' ? (input as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

/** The Anthropic messages for `rows`, built the way the agent builds them. */
export function anthropicMessages(rows: HistoryRow[], system: string): AnthropicMessagesRequest['messages'] {
  type Message = AnthropicMessagesRequest['messages'][number];
  const messages: Message[] = [];
  let assistant: AnthropicContentBlock[] = [];
  let toolResults: AnthropicContentBlock[] = [];
  const flushAssistant = (): void => {
    if (assistant.length > 0) messages.push({ role: 'assistant', content: assistant });
    assistant = [];
  };
  const flushToolResults = (): void => {
    if (toolResults.length > 0) messages.push({ role: 'user', content: toolResults });
    toolResults = [];
  };
  for (const row of rows) {
    switch (row.type) {
      case 'system':
        if (row.content !== system) throw new Error('a system row differs from system_prompt.txt');
        flushAssistant();
        flushToolResults();
        break;
      case 'user': {
        if (row.synthetic_reason === 'agent_message') {
          throw new Error('an agent_message user row gets a label this replay does not reproduce');
        }
        flushAssistant();
        flushToolResults();
        const parts = typeof row.content === 'string' ? [{ type: 'text' as const, text: row.content }] : row.content;
        for (const part of parts) {
          if (part.type !== 'text') throw new Error(`unsupported user content part ${JSON.stringify(part.type)}`);
        }
        messages.push({ role: 'user', content: parts.map((part) => ({ type: 'text', text: part.text })) });
        break;
      }
      case 'reasoning': {
        flushToolResults();
        const thinking = reasoningText(row);
        const signature = row.encrypted_content ?? '';
        if (thinking !== '' || signature !== '') assistant.push({ type: 'thinking', thinking, signature });
        break;
      }
      case 'assistant':
        flushToolResults();
        if (row.content) assistant.push({ type: 'text', text: row.content });
        for (const call of row.tool_calls ?? []) {
          assistant.push({
            type: 'tool_use',
            id: sanitizeToolCallId(call.id),
            name: call.name,
            input: toolInput(call.arguments),
          });
        }
        break;
      case 'tool_result':
        flushAssistant();
        toolResults.push({
          type: 'tool_result',
          tool_use_id: sanitizeToolCallId(row.tool_call_id),
          content: row.content,
        });
        break;
      default:
        throw new Error(`unsupported chat_history row ${JSON.stringify((row as { type: unknown }).type)}`);
    }
  }
  flushAssistant();
  flushToolResults();
  return messages;
}

/** Where each model response starts: its first `reasoning` row, else its `assistant` row. */
export function responseStarts(rows: HistoryRow[]): number[] {
  const starts: number[] = [];
  let start: number | undefined;
  rows.forEach((row, index) => {
    if (row.type === 'reasoning') start ??= index;
    else if (row.type === 'assistant') {
      starts.push(start ?? index);
      start = undefined;
    } else start = undefined;
  });
  return starts;
}

export function sessionModel(session: Session): string {
  for (const row of session.rows) if (row.type === 'assistant' && row.model_id) return row.model_id;
  throw new Error(`${session.id}: no assistant row names its model`);
}

export function buildRequests(session: Session, maxTokens = REPLAY_MAX_TOKENS): AgentRequest[] {
  const starts = responseStarts(session.rows);
  if (starts.length !== session.firstTokenMs.length) {
    throw new Error(
      `${session.id}: ${starts.length} model responses but ${session.firstTokenMs.length} first_token events`,
    );
  }
  const model = sessionModel(session);
  const tools = session.tools.map((tool) => ({
    name: tool.function.name,
    ...(tool.function.description === undefined ? {} : { description: tool.function.description }),
    input_schema: tool.function.parameters ?? {},
  }));
  return starts.map((start, index) => {
    const firstTokenMs = session.firstTokenMs[index];
    const loopStartedMs = session.loopStartedMs.filter((at) => at <= firstTokenMs).at(-1);
    if (loopStartedMs === undefined)
      throw new Error(`${session.id}#${index + 1}: no loop_started before its first token`);
    return {
      label: `${session.id}#${index + 1}`,
      session: session.id,
      index,
      body: {
        model,
        max_tokens: maxTokens,
        temperature: 0,
        system: session.system,
        messages: anthropicMessages(session.rows.slice(0, start), session.system),
        tools,
      } as AnthropicMessagesRequest,
      loopStartedMs,
      firstTokenMs,
    };
  });
}

/** Rust `ChatMessage` field names. */
function nativeMessage(message: ChatMessage): object {
  return {
    role: message.role,
    content: message.content,
    ...(message.toolCalls === undefined ? {} : { tool_calls: message.toolCalls }),
    ...(message.toolCallId === undefined ? {} : { tool_call_id: message.toolCallId }),
    ...(message.isError === undefined ? {} : { is_error: message.isError }),
    ...(message.reasoningContent === undefined ? {} : { reasoning_content: message.reasoningContent }),
    ...(message.thinkingEnabled === undefined ? {} : { thinking_enabled: message.thinkingEnabled }),
  };
}

export function offlineRequest(request: AgentRequest): OfflineRequest {
  const { messages, config } = mapAnthropicRequest(request.body);
  return {
    label: request.label,
    session: request.session,
    messages: messages.map(nativeMessage),
    tools: config.tools ?? null,
    reasoning_effort: config.reasoningEffort ?? null,
  };
}

export type TimelineEntry = { request: AgentRequest; row: ProfileRow } | { extra: ProfileRow };

/**
 * The recorded run in log order. Each session's attributed rows map to its
 * requests in order; a session whose row count differs is an error.
 */
export function recordedTimeline(rows: ProfileRow[], sessions: Session[], requests: AgentRequest[]): TimelineEntry[] {
  const firstTokens: FirstToken[] = sessions.flatMap((session) =>
    session.firstTokenMs.map((atMs) => ({ session: session.id, atMs })),
  );
  const owners = attributeRows(rows, firstTokens);
  const next = new Map<string, number>();
  const timeline = rows.map((row, index): TimelineEntry => {
    const session = owners[index];
    if (session === undefined) return { extra: row };
    const k = next.get(session) ?? 0;
    next.set(session, k + 1);
    const request = requests.find((candidate) => candidate.session === session && candidate.index === k);
    if (request === undefined) throw new Error(`${session}: log row ${index} has no request ${k + 1}`);
    return { request, row };
  });
  for (const session of sessions) {
    const attributed = next.get(session.id) ?? 0;
    if (attributed !== session.firstTokenMs.length) {
      throw new Error(`${session.id}: ${attributed} attributed log rows for ${session.firstTokenMs.length} requests`);
    }
  }
  return timeline;
}

export function totals(turns: OfflineTurn[]): OfflineTotals {
  const reasons: Record<string, number> = {};
  let promptTokens = 0;
  let prefilledToday = 0;
  let idealFlat = 0;
  let idealPaged = 0;
  for (const turn of turns) {
    const reason = turn.miss_reason === 'none' ? `hit:${turn.hit_kind}` : turn.miss_reason;
    reasons[reason] = (reasons[reason] ?? 0) + 1;
    promptTokens += turn.prompt_tokens;
    prefilledToday += turn.prefilled_tokens;
    idealFlat += turn.prompt_tokens - turn.ideal_tokens;
    idealPaged += turn.prompt_tokens - turn.ideal_paged_tokens;
  }
  return { requests: turns.length, promptTokens, prefilledToday, idealFlat, idealPaged, reasons };
}

export interface OfflineCheck {
  failures: string[];
  crossSession: { misses: number; crossSession: number };
}

/** The `--offline` pass conditions over a replay of the recorded timeline. */
export function checkOffline(timeline: TimelineEntry[], turns: OfflineTurn[]): OfflineCheck {
  const failures: string[] = [];
  if (turns.length !== timeline.length) {
    failures.push(`classifier returned ${turns.length} turns for ${timeline.length} requests`);
    return { failures, crossSession: { misses: 0, crossSession: 0 } };
  }
  const seen = new Set<string>();
  let misses = 0;
  let crossSession = 0;
  timeline.forEach((entry, index) => {
    const turn = turns[index];
    if (!('request' in entry)) {
      if (turn.prompt_tokens !== entry.extra.promptTokens)
        failures.push(`log row ${index}: extra request length drifted`);
      return;
    }
    const { request, row } = entry;
    if (turn.label !== request.label) failures.push(`log row ${index}: ${turn.label} replayed for ${request.label}`);
    if (turn.prefilled_tokens !== row.promptTokens) {
      failures.push(
        turn.hit_tokens === 0
          ? `${request.label}: reconstructed prompt has ${turn.prompt_tokens} tokens and misses, the recording prefilled ${row.promptTokens}`
          : `${request.label}: replay prefills ${turn.prefilled_tokens} after a ${turn.hit_kind} hit of ${turn.hit_tokens}, the recording ${row.promptTokens}`,
      );
    }
    if (turn.miss_reason === 'unknown' || turn.miss_reason === 'state_existed_unused') {
      failures.push(`${request.label}: classified ${turn.miss_reason}`);
    }
    const first = !seen.has(request.session);
    seen.add(request.session);
    if (!first && turn.hit_tokens === 0) {
      misses += 1;
      if (turn.miss_reason === 'cross_session') crossSession += 1;
      else failures.push(`${request.label}: a later-turn miss classified ${turn.miss_reason}, not cross_session`);
    }
  });
  return { failures, crossSession: { misses, crossSession } };
}

function outDir(): string {
  const dir = process.env.MLX_DFLASH2_REPLAY_OUT;
  if (dir === undefined || dir === '') return mkdtempSync(join(tmpdir(), 'dflash2-agent-replay-'));
  mkdirSync(dir, { recursive: true });
  return dir;
}

function tokenizerPath(flag: string | undefined): string {
  if (flag !== undefined) return resolve(flag);
  const target = process.env.MLX_TEST_QWEN38_TARGET_PATH;
  return join(target ? dirname(target) : DEFAULT_TARGET_DIR, 'tokenizer.json');
}

/** Runs the Rust classifier over `requests` and returns its turns. Paths reach it through the environment. */
export function classifyOffline(requests: OfflineRequest[], dir: string, tokenizer: string): OfflineTurn[] {
  const requestsPath = join(dir, 'requests.json');
  const resultPath = join(dir, 'offline.json');
  writeFileSync(requestsPath, JSON.stringify(requests));
  const argv = ['test', '-p', 'mlx-core', '--test', 'dflash2_replay_offline', '--', '--ignored', '--nocapture'];
  console.error(`$ cargo ${argv.join(' ')}`);
  const run = spawnSync('cargo', argv, {
    cwd: ROOT,
    stdio: ['ignore', 'inherit', 'inherit'],
    env: {
      ...process.env,
      MLX_DFLASH2_REPLAY_REQUESTS: requestsPath,
      MLX_DFLASH2_REPLAY_TOKENIZER: tokenizer,
      MLX_DFLASH2_REPLAY_RESULT: resultPath,
    },
  });
  if (run.status !== 0) throw new Error(`dflash2_replay_offline failed (${run.status ?? run.signal})`);
  return JSON.parse(readFileSync(resultPath, 'utf8')) as OfflineTurn[];
}

function shortLabel(label: string): string {
  const [session, k] = label.split('#');
  return k === undefined ? label : `${session.slice(-4)}#${k}`;
}

function offlineTable(turns: OfflineTurn[]): string {
  const header = ['request', 'prompt', 'hit', 'prefilled', 'reason', 'shadow', 'ideal', 'ideal_paged'];
  const rows = turns.map((turn) => [
    shortLabel(turn.label),
    String(turn.prompt_tokens),
    turn.hit_tokens === 0 ? '-' : `${turn.hit_kind}:${turn.hit_tokens}`,
    String(turn.prefilled_tokens),
    turn.miss_reason,
    String(turn.shadow_hit),
    String(turn.ideal_tokens),
    String(turn.ideal_paged_tokens),
  ]);
  const widths = header.map((title, i) => Math.max(title.length, ...rows.map((row) => row[i].length)));
  return [header, ...rows].map((row) => row.map((cell, i) => cell.padEnd(widths[i])).join('  ')).join('\n');
}

function totalsLine(t: OfflineTotals): string {
  const count = (value: number): string => value.toLocaleString('en-US');
  const reasons = Object.entries(t.reasons)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([reason, n]) => `${reason}=${n}`)
    .join(' ');
  return (
    `${t.requests} requests, ${count(t.promptTokens)} prompt tokens: prefilled today ${count(t.prefilledToday)}, ` +
    `ideal flat ${count(t.idealFlat)}, ideal paged ${count(t.idealPaged)} | ${reasons}`
  );
}

function loadWorkload(sessionsDir: string): { sessions: Session[]; requests: AgentRequest[] } {
  const recorded = resolve(sessionsDir) === resolve(RECORDED_SESSIONS_DIR);
  // The default dir keeps growing with new agent runs; keep only the recorded ones.
  const sessions = readSessions(sessionsDir).filter((session) => !recorded || session.id in RECORDED_REQUESTS);
  if (sessions.length === 0) throw new Error(`${sessionsDir}: no session folders`);
  const requests = sessions.flatMap((session) => buildRequests(session));
  if (recorded) {
    for (const [id, expected] of Object.entries(RECORDED_REQUESTS)) {
      const got = requests.filter((request) => request.session === id).length;
      if (got !== expected) throw new Error(`${id}: ${got} requests, the recorded run made ${expected}`);
    }
  }
  return { sessions, requests };
}

function runOffline(sessionsDir: string, logPath: string, tokenizer: string, json: boolean): boolean {
  const { sessions, requests } = loadWorkload(sessionsDir);
  const rows = parseLog(readFileSync(logPath, 'utf8')).profiles;
  const timeline = recordedTimeline(rows, sessions, requests);
  const offline = timeline.map((entry, index) =>
    'request' in entry
      ? offlineRequest(entry.request)
      : { label: `extra@${index}`, session: null, prompt_tokens: entry.extra.promptTokens },
  );
  const dir = outDir();
  const turns = classifyOffline(offline, dir, tokenizer);
  const check = checkOffline(timeline, turns);
  const legacy = legacyTurns(
    rows,
    sessions.flatMap((s) => s.firstTokenMs.map((atMs) => ({ session: s.id, atMs }))),
  );
  const recordedCrossSession = legacy.filter((turn) => turn.reason === 'cross_session').length;
  if (check.crossSession.crossSession !== recordedCrossSession) {
    check.failures.push(
      `${check.crossSession.crossSession} cross_session misses replayed, the recorded log has ${recordedCrossSession}`,
    );
  }
  const report = {
    sessions: sessions.map((session) => ({ id: session.id, requests: session.firstTokenMs.length })),
    loopOrderMatchesLog: requestsInLoopOrder(requests).every(
      (request, i) => request === timeline.filter((entry) => 'request' in entry)[i].request,
    ),
    totals: totals(turns),
    crossSession: check.crossSession,
    failures: check.failures,
    turns,
  };
  writeFileSync(join(dir, 'report.json'), JSON.stringify(report, null, 2));
  if (json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log(offlineTable(turns));
    console.log('');
    console.log(`sessions: ${report.sessions.map((s) => `${s.id.slice(-4)}=${s.requests}`).join(' ')}`);
    console.log(totalsLine(report.totals));
    console.log(
      `later-turn misses: ${check.crossSession.crossSession}/${check.crossSession.misses} cross_session ` +
        `(recorded ${recordedCrossSession}); loop_started order matches the log: ${report.loopOrderMatchesLog}`,
    );
    console.log(`report: ${join(dir, 'report.json')}`);
    for (const failure of check.failures) console.log(`FAIL ${failure}`);
  }
  return check.failures.length === 0;
}

export function requestsInLoopOrder(requests: AgentRequest[]): AgentRequest[] {
  return [...requests].sort((a, b) => a.loopStartedMs - b.loopStartedMs || a.firstTokenMs - b.firstTokenMs);
}

/** `mlx serve` prints `… listening on http://HOST:PORT …`. */
function bannerBaseUrl(line: string): string | undefined {
  return /listening on (http:\/\/\S+)/.exec(line)?.[1];
}

async function startServer(
  modelsDir: string,
  logFile: string,
): Promise<{ url: string; token: string; stop: () => Promise<void> }> {
  const token = randomBytes(24).toString('base64url');
  const argv = ['packages/cli/src/cli.ts', 'serve', '--port', '0', '--auth-token', token, '--models-dir', modelsDir];
  console.error(
    `$ MLX_NODE_LOG=${INFERENCE_LOG_FILTER} MLX_NODE_LOG_FILE=$MLX_DFLASH2_REPLAY_OUT/inference.log oxnode ${argv.join(' ').replace(token, '<token>')}`,
  );
  // Its own process group: `oxnode` runs the CLI in a child process, and the
  // whole group has to go down with the replay.
  const child = spawn('oxnode', argv, {
    cwd: ROOT,
    detached: true,
    stdio: ['ignore', 'pipe', 'inherit'],
    env: { ...process.env, MLX_NODE_LOG: INFERENCE_LOG_FILTER, MLX_NODE_LOG_FILE: logFile },
  });
  const exited = new Promise<void>((resolveExit) => child.once('exit', () => resolveExit()));
  const stop = async (): Promise<void> => {
    if (child.exitCode === null && child.signalCode === null && child.pid !== undefined) {
      process.kill(-child.pid, 'SIGTERM');
    }
    await exited;
  };
  try {
    const url = await new Promise<string>((resolveUrl, reject) => {
      const timer = setTimeout(() => reject(new Error('mlx serve did not report its port')), SERVE_READY_TIMEOUT_MS);
      child.once('exit', (code) => reject(new Error(`mlx serve exited (${code}) before reporting its port`)));
      child.once('error', reject);
      const lines = createInterface({ input: child.stdout });
      lines.on('line', (line) => {
        process.stderr.write(`[serve] ${line}\n`);
        const base = bannerBaseUrl(line);
        if (base === undefined) return;
        clearTimeout(timer);
        resolveUrl(base);
      });
    });
    return { url, token, stop };
  } catch (error) {
    await stop();
    throw error;
  }
}

interface OnlineResult {
  label: string;
  status: number;
  wallMs: number;
  inputTokens?: number;
  cacheReadTokens?: number;
  error?: string;
}

async function send(url: string, token: string, request: AgentRequest): Promise<OnlineResult> {
  const started = performance.now();
  const response = await fetch(`${url}/v1/messages`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-api-key': token },
    body: JSON.stringify({ ...request.body, stream: false }),
  });
  const wallMs = performance.now() - started;
  const body = (await response.json()) as {
    usage?: { input_tokens?: number; cache_read_input_tokens?: number };
    error?: { message?: string };
  };
  return {
    label: request.label,
    status: response.status,
    wallMs,
    inputTokens: body.usage?.input_tokens,
    cacheReadTokens: body.usage?.cache_read_input_tokens,
    ...(response.ok ? {} : { error: body.error?.message ?? 'request failed' }),
  };
}

/** Fails before any request when the server does not list a model the workload names. */
async function requireModels(url: string, token: string, models: Set<string>): Promise<void> {
  const response = await fetch(`${url}/v1/models`, { headers: { 'x-api-key': token } });
  if (!response.ok) throw new Error(`GET /v1/models: HTTP ${response.status}`);
  const served = new Set(((await response.json()) as { data?: { id: string }[] }).data?.map((model) => model.id));
  for (const model of models) {
    if (!served.has(model)) throw new Error(`mlx serve does not list ${model}; it lists ${[...served].join(', ')}`);
  }
  console.error(`[serve] lists ${[...models].join(', ')}`);
}

async function runOnline(
  sessionsDir: string,
  modelsDir: string,
  tokenizer: string,
  limit: number | undefined,
  json: boolean,
): Promise<boolean> {
  const dir = process.env.MLX_DFLASH2_REPLAY_OUT;
  if (dir === undefined || dir === '') throw new Error('set MLX_DFLASH2_REPLAY_OUT to the directory for inference.log');
  mkdirSync(dir, { recursive: true });
  const { requests } = loadWorkload(sessionsDir);
  const ordered = requestsInLoopOrder(requests).slice(0, limit);
  const logFile = join(dir, 'inference.log');
  const server = await startServer(modelsDir, logFile);
  const results: OnlineResult[] = [];
  try {
    await requireModels(server.url, server.token, new Set(requests.map((request) => request.body.model)));
    for (const request of ordered) {
      const result = await send(server.url, server.token, request);
      results.push(result);
      console.error(
        `${shortLabel(request.label)} ${result.status} ${(result.wallMs / 1000).toFixed(1)}s input=${result.inputTokens} cached=${result.cacheReadTokens}${result.error ? ` ${result.error}` : ''}`,
      );
      if (result.error !== undefined) break;
    }
  } finally {
    await server.stop();
  }

  const offline = ordered.map(offlineRequest);
  const turns = classifyOffline(offline, dir, tokenizer);
  const parsed = parseLog(existsSync(logFile) ? readFileSync(logFile, 'utf8') : '');
  const failures = results.filter((result) => result.error !== undefined).map((r) => `${r.label}: ${r.error}`);
  if (parsed.events.length !== results.length) {
    failures.push(`${parsed.events.length} dflash2_prefix events for ${results.length} requests`);
  }
  parsed.events.forEach((event, i) => {
    const expected = turns[i]?.prompt_tokens;
    if (event.promptTokens !== expected) {
      failures.push(
        `${ordered[i].label}: the server rendered ${event.promptTokens} prompt tokens, the replay ${expected}`,
      );
    }
  });
  const report = {
    requests: results,
    server: buildReport(parsed, undefined),
    offline: totals(turns),
    failures,
  };
  writeFileSync(join(dir, 'report.json'), JSON.stringify(report, null, 2));
  if (json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log(renderReport(report.server));
    console.log('');
    console.log(`offline classifier, same request list: ${totalsLine(report.offline)}`);
    for (const failure of failures) console.log(`FAIL ${failure}`);
  }
  return failures.length === 0;
}

function requestLimit(value: string): number {
  const limit = Number(value);
  if (!Number.isSafeInteger(limit) || limit < 0) throw new Error(`--limit ${value}: expected a non-negative integer`);
  return limit;
}

async function main(): Promise<void> {
  const { values } = parseArgs({
    options: {
      offline: { type: 'boolean', default: false },
      sessions: { type: 'string' },
      log: { type: 'string' },
      tokenizer: { type: 'string' },
      'models-dir': { type: 'string' },
      limit: { type: 'string' },
      json: { type: 'boolean', default: false },
    },
  });
  const sessionsDir = resolve(values.sessions ?? process.env.MLX_DFLASH2_SESSIONS_DIR ?? RECORDED_SESSIONS_DIR);
  const tokenizer = tokenizerPath(values.tokenizer);
  const ok = values.offline
    ? runOffline(sessionsDir, resolve(values.log ?? RECORDED_LOG), tokenizer, values.json)
    : await runOnline(
        sessionsDir,
        resolve(values['models-dir'] ?? DEFAULT_MODELS_DIR),
        tokenizer,
        values.limit === undefined ? undefined : requestLimit(values.limit),
        values.json,
      );
  if (!ok) process.exitCode = 1;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
