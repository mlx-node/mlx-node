/**
 * Notices in the shared reducer: retry episodes from the agent's `retry_state` and `auto_recovery_*` one-shots (shapes
 * of mlx-agent v0.3.4 `extensions/notification.rs`), and the hold rules that keep a notice from starting a turn. The
 * same lines give the same entries live (ops applied as the page does) and from `updates.jsonl`.
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

import {
  HOOK_TEXT_CHARS,
  hookSource,
  NOTICE_TEXT_CHARS,
  parseUpdates,
  TranscriptReducer,
  type UpdateLine,
} from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry, TurnMetric } from '../src/agent-sessions/types.js';
import { liveFilter } from '../src/chat/agent/live-filter.js';
import { workGuardHookName, workGuardName } from '../src/chat/agent/profiles.js';
import { applyOp } from '../src/chat/conversation.js';
import { redactEntry } from '../src/chat/redact.js';
import { CHAT } from '../ui/src/lib/chat-strings.js';
import { hookName, hookRow } from '../ui/src/lib/chat-turns.js';

const RECORDED = fileURLToPath(new URL('./fixtures/chat-scenarios/recorded/', import.meta.url));

const SID = '01a20000-0000-7000-8000-0000000000f1';
const T0 = 1_791_643_279_000;
const STEP = 1_000;

type Update = Record<string, unknown>;
type Notice = Extract<TranscriptEntry, { kind: 'notice' }>;
type Retry = Extract<Notice, { code: 'retry' }>;

/**
 * One stored line. `prompt`: the typed prompt, which the host pushes live (the agent sends no live user chunk).
 * `diskOnly`: a line the agent never sends live (a further chunk of the stored prompt).
 */
interface Line {
  xai: boolean;
  update: Update;
  prompt?: string;
  diskOnly?: true;
  /** The line's own `agentTimestampMs` (a stored line); else one step after the line before. */
  ms?: number;
  /** More `_meta` of the line (a turn end's cancel category). */
  meta?: Update;
}

const prompt = (text: string, promptIndex = 0): Line => ({
  xai: false,
  update: { sessionUpdate: 'user_message_chunk', content: { type: 'text', text }, _meta: { promptIndex } },
  prompt: text,
});
const chunk = (text: string, promptIndex = 0): Line => ({
  xai: false,
  update: { sessionUpdate: 'user_message_chunk', content: { type: 'text', text }, _meta: { promptIndex } },
  diskOnly: true,
});
const text = (body: string): Line => ({
  xai: false,
  update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: body } },
});
const thought = (body: string): Line => ({
  xai: false,
  update: { sessionUpdate: 'agent_thought_chunk', content: { type: 'text', text: body } },
});
const toolCall = (id: string): Line => ({
  xai: false,
  update: { sessionUpdate: 'tool_call', toolCallId: id, title: 'Read README.md', status: 'pending' },
});
const retrying = (attempt: number, max = 3, errorType: string | null = 'idle_timeout', reason?: string): Line => ({
  xai: true,
  update: {
    sessionUpdate: 'retry_state',
    type: 'retrying',
    attempt,
    max_retries: max,
    reason: reason ?? 'Response stalled; retrying request',
    error_type: errorType,
  },
});
const failed = (message: string): Line => ({
  xai: true,
  update: { sessionUpdate: 'retry_state', type: 'failed', error_type: 'api', message },
});
const exhausted = (reason: string): Line => ({
  xai: true,
  update: { sessionUpdate: 'retry_state', type: 'exhausted', attempts: 3, reason, is_rate_limited: false },
});
const recovery = (attempt: number): Line => ({
  xai: true,
  update: { sessionUpdate: 'auto_recovery_started', attempt, max_retries: 2, error: 'task not done', delay_ms: 500 },
});
const recoveryOut = (error: string): Line => ({
  xai: true,
  update: { sessionUpdate: 'auto_recovery_exhausted', attempts: 2, error },
});
const done = (): Line => ({
  xai: true,
  update: { sessionUpdate: 'turn_completed', prompt_id: 'p-1', stop_reason: 'end_turn' },
});

const at = (n: number): number => T0 + n * STEP;

function diskText(lines: readonly Line[]): string {
  return lines
    .map((line, n) =>
      JSON.stringify({
        timestamp: Math.floor((line.ms ?? at(n)) / 1000),
        method: line.xai ? '_x.ai/session/update' : 'session/update',
        params: {
          sessionId: SID,
          update: line.update,
          _meta: { eventId: `${SID}-${n}`, agentTimestampMs: line.ms ?? at(n), ...line.meta },
        },
      }),
    )
    .map((l) => `${l}\n`)
    .join('');
}

/** The entries the page holds after the live ops, with a snapshot of each step's list. */
function live(lines: readonly Line[]): { entries: TranscriptEntry[]; steps: TranscriptEntry[][] } {
  const reducer = new TranscriptReducer();
  const entries: TranscriptEntry[] = [];
  const turns: TurnMetric[] = [];
  const steps: TranscriptEntry[][] = [];
  lines.forEach((line, n) => {
    if (line.diskOnly === true) return;
    const ops =
      line.prompt !== undefined
        ? reducer.userPrompt(line.prompt, [], line.ms ?? at(n))
        : reducer.apply({
            xai: line.xai,
            update: line.update,
            meta: { agentTimestampMs: line.ms ?? at(n), ...line.meta },
          });
    for (const op of ops) applyOp(entries, turns, op);
    steps.push(entries.map((e) => ({ ...e })));
  });
  return { entries, steps };
}

/** The reducer's live copy of the retry episode after each sent line (`null` while the turn has none). */
function liveRetries(lines: readonly Line[]): Array<Retry | null> {
  const reducer = new TranscriptReducer();
  const out: Array<Retry | null> = [];
  lines.forEach((line, n) => {
    if (line.diskOnly === true) return;
    if (line.prompt !== undefined) reducer.userPrompt(line.prompt, [], line.ms ?? at(n));
    else
      reducer.apply({ xai: line.xai, update: line.update, meta: { agentTimestampMs: line.ms ?? at(n), ...line.meta } });
    out.push(reducer.retryLive());
  });
  return out;
}

/** What the live and stored lists must share: a live prompt has no index and a live thought has its time. */
function comparable(entries: readonly TranscriptEntry[]): unknown[] {
  return entries.map((entry) => {
    if (entry.kind === 'user') return { ...entry, promptIndex: null };
    if (entry.kind === 'thinking') {
      const { timed: _timed, ...rest } = entry;
      return rest;
    }
    return entry;
  });
}

function both(lines: readonly Line[]): TranscriptEntry[] {
  const stored = parseUpdates(diskText(lines)).transcript;
  expect(comparable(live(lines).entries)).toEqual(comparable(stored));
  return stored;
}

/** The retry notices. */
const notices = (entries: readonly TranscriptEntry[]): Retry[] =>
  entries.filter((entry): entry is Retry => entry.kind === 'notice' && entry.code === 'retry');
/** Every notice, of any code. */
const allNotices = (entries: readonly TranscriptEntry[]): Notice[] =>
  entries.filter((entry): entry is Notice => entry.kind === 'notice');
const kinds = (entries: readonly TranscriptEntry[]): string[] => entries.map((entry) => entry.kind);

describe('retry episodes', () => {
  it('tries 1 and 2 are one entry with attempts 2; text after them closes it as recovered; the row goes at the end', () => {
    const entries = both([prompt('Fix it'), retrying(1), retrying(2), text('Done.'), done()]);
    expect(kinds(entries)).toEqual(['user', 'assistant', 'notice', 'turn-end']);
    expect(notices(entries)).toEqual([
      {
        kind: 'notice',
        ts: at(1),
        code: 'retry',
        cause: 'stalled',
        attempts: 2,
        max: 3,
        endTs: at(3),
        outcome: 'recovered',
      },
    ]);
  });

  it('a late text chunk between two tries closes the episode, and the next try reopens the same entry', () => {
    const lines = [prompt('Fix it'), retrying(1), text('Before the stall'), retrying(2), thought('Again'), done()];
    const entries = both(lines);
    expect(kinds(entries)).toEqual(['user', 'assistant', 'thinking', 'notice', 'turn-end']);
    expect(notices(entries)).toEqual([
      expect.objectContaining({ ts: at(1), attempts: 2, endTs: at(4), outcome: 'recovered' }),
    ]);
    // Live, the late chunk closed it for a moment, and the second try took it back: the stream's copy says so.
    const retries = liveRetries(lines);
    expect(retries[2]).toMatchObject({ attempts: 1, endTs: at(2), outcome: 'recovered' });
    expect(retries[3]).toEqual({
      kind: 'notice',
      ts: at(1),
      code: 'retry',
      cause: 'stalled',
      attempts: 2,
      max: 3,
    });
  });

  it('a tool call after a recovered episode ends it for good: the next try is a new episode', () => {
    const entries = both([
      prompt('Fix it'),
      retrying(1),
      text('Reading'),
      toolCall('c1'),
      retrying(2),
      text('ok'),
      done(),
    ]);
    expect(kinds(entries)).toEqual(['user', 'assistant', 'notice', 'tool', 'assistant', 'notice', 'turn-end']);
    expect(notices(entries).map((n) => [n.attempts, n.outcome])).toEqual([
      [1, 'recovered'],
      [2, 'recovered'],
    ]);
    // A try that does not follow the last one also starts a new episode.
    const fresh = both([prompt('Fix it'), retrying(1), text('back'), retrying(1), text('back again'), done()]);
    expect(notices(fresh)).toHaveLength(2);
  });

  it('a tool call closes an open episode as recovered', () => {
    const entries = both([prompt('Fix it'), retrying(1), toolCall('c1'), done()]);
    expect(notices(entries)).toEqual([expect.objectContaining({ endTs: at(2), outcome: 'recovered' })]);
  });

  it('failed after a try gives up with the agent words as detail; failed alone makes no entry', () => {
    const entries = both([
      prompt('Fix it'),
      retrying(1, 3, 'api', 'Server error; retrying request'),
      failed('API error (status 500)'),
      done(),
    ]);
    expect(notices(entries)).toEqual([
      {
        kind: 'notice',
        ts: at(1),
        code: 'retry',
        cause: 'server',
        attempts: 1,
        max: 3,
        endTs: at(2),
        outcome: 'gave-up',
        detail: 'API error (status 500)',
      },
    ]);
    expect(notices(both([prompt('Fix it'), failed('API error (status 403)'), done()]))).toEqual([]);
    expect(notices(both([prompt('Fix it'), exhausted('API error (status 429)'), done()]))).toEqual([]);
    const tired = both([
      prompt('Fix it'),
      retrying(3, 8, null, 'Too many requests in flight; waiting 9s before trying again'),
      exhausted('API error (status 429 Too Many Requests)'),
      done(),
    ]);
    expect(notices(tired)).toEqual([
      expect.objectContaining({
        cause: 'busy',
        attempts: 3,
        max: 8,
        outcome: 'gave-up',
        detail: 'API error (status 429 Too Many Requests)',
      }),
    ]);
  });

  it('the turn end closes an open episode with no outcome; a later turn starts clean', () => {
    const entries = both([prompt('Fix it'), retrying(1), done(), prompt('Again', 1), retrying(1), text('ok'), done()]);
    expect(kinds(entries)).toEqual(['user', 'notice', 'turn-end', 'user', 'assistant', 'notice', 'turn-end']);
    const [first, second] = notices(entries);
    expect(first).toEqual({
      kind: 'notice',
      ts: at(1),
      code: 'retry',
      cause: 'stalled',
      attempts: 1,
      max: 3,
      endTs: at(2),
    });
    expect(second).toMatchObject({ ts: at(4), attempts: 1, outcome: 'recovered' });
  });

  it('an episode the agent never ended stays open; the next prompt does not close it', () => {
    // The agent died mid-retry: no turn end, then a new prompt after its restart.
    const entries = both([prompt('Fix it'), retrying(1), prompt('Try once more', 1), text('Here it is.'), done()]);
    expect(kinds(entries)).toEqual(['user', 'notice', 'user', 'assistant', 'turn-end']);
    expect(notices(entries)[0]).not.toHaveProperty('endTs');
    expect(parseUpdates(diskText([prompt('Fix it'), retrying(1)])).openTurn).toBe(true);
  });

  it('auto-recovery uses the same episode rules, cause recovery, and its exhaustion gives up', () => {
    const entries = both([prompt('Fix it'), recovery(1), recovery(2), recoveryOut('task still not done'), done()]);
    expect(notices(entries)).toEqual([
      expect.objectContaining({
        cause: 'recovery',
        attempts: 2,
        max: 2,
        outcome: 'gave-up',
        detail: 'task still not done',
      }),
    ]);
  });

  it('the cause comes from error_type, else from the reason; the last try names it; other keeps the words', () => {
    const cause = (errorType: string | null, reason = 'x'): unknown =>
      notices(both([prompt('Fix it'), retrying(1, 3, errorType, reason), text('ok'), done()]))[0]?.cause;
    expect(cause('idle_timeout')).toBe('stalled');
    expect(cause('http')).toBe('connection');
    expect(cause('api')).toBe('server');
    expect(cause('serialization')).toBe('server');
    expect(cause('empty_response')).toBe('server');
    expect(cause('rate_limited')).toBe('busy');
    expect(cause(null, 'Too many requests in flight; waiting 2s before trying again')).toBe('busy');
    expect(cause('doom_loop_detected')).toBe('loop');
    expect(cause('auth')).toBe('other');
    const other = notices(
      both([
        prompt('Fix it'),
        retrying(1, 3, null, '  Re-authenticated after 401; retrying request '),
        text('ok'),
        done(),
      ]),
    );
    expect(other[0]).toMatchObject({ cause: 'other', detail: 'Re-authenticated after 401; retrying request' });
    // The sampler's try and the paced wait share one episode: the last line names cause, try and budget.
    const mixed = notices(
      both([
        prompt('Fix it'),
        retrying(1, 15, 'http', 'Connection problem; retrying request'),
        retrying(1, 8, null, 'Too many requests in flight; waiting 8s before trying again'),
        text('ok'),
        done(),
      ]),
    );
    expect(mixed).toEqual([expect.objectContaining({ cause: 'busy', attempts: 1, max: 8 })]);
    expect(mixed[0]).not.toHaveProperty('detail');
  });

  it("the agent's words are cut at 300 characters, and a cut end that could start a key is redacted", () => {
    const long = `${'x'.repeat(NOTICE_TEXT_CHARS - 4)} sk-abcdefgh`;
    const [entry] = notices(both([prompt('Fix it'), retrying(1), failed(long), done()]));
    expect(entry!.detail).toBe(`${long.slice(0, NOTICE_TEXT_CHARS)}…`);
    const shown = redactEntry(entry!, ['sk-abcdefghijklmnop']);
    expect(shown.kind === 'notice' && shown.code === 'retry' ? shown.detail : null).toBe(
      `${'x'.repeat(NOTICE_TEXT_CHARS - 4)} …`,
    );
  });

  it('a try with no count makes no entry; a missing budget is 0', () => {
    const odd: Line = {
      xai: true,
      update: { sessionUpdate: 'retry_state', type: 'retrying', attempt: 'x', reason: 'r' },
    };
    expect(notices(both([prompt('Fix it'), odd, done()]))).toEqual([]);
    const bare: Line = {
      xai: true,
      update: { sessionUpdate: 'retry_state', type: 'retrying', attempt: 2, reason: 'r' },
    };
    expect(notices(both([prompt('Fix it'), bare, text('ok'), done()]))[0]).toMatchObject({ attempts: 2, max: 0 });
  });

  it('the timeline list of tries is as before', () => {
    const parsed = parseUpdates(diskText([prompt('Fix it'), retrying(1), retrying(2, 15), failed('no'), done()]));
    expect(parsed.retries).toEqual([
      { ts: at(1), attempt: 1, max: 3 },
      { ts: at(2), attempt: 2, max: 15 },
    ]);
  });
});

describe('hold rules', () => {
  it('R1: a notice made while no turn is open goes in the next turn, at its end, live and on disk', () => {
    const lines = [prompt('one'), text('a'), done(), retrying(1), prompt('two', 1), text('b'), done()];
    const entries = both(lines);
    expect(kinds(entries)).toEqual(['user', 'assistant', 'turn-end', 'user', 'assistant', 'notice', 'turn-end']);
    expect(notices(entries)[0]).toMatchObject({ ts: at(3), endTs: at(5), outcome: 'recovered' });
    // Live, nothing shows before the prompt: the notice never makes a turn of its own.
    const { steps } = live(lines);
    expect(kinds(steps[3]!)).toEqual(['user', 'assistant', 'turn-end']);
    expect(kinds(steps[4]!)).toEqual(['user', 'assistant', 'turn-end', 'user']);
  });

  it('R1 live: a wake turn whose first line is a retry places it at the turn end; the live copy has it at once', () => {
    const reducer = new TranscriptReducer();
    const entries: TranscriptEntry[] = [];
    const turns: TurnMetric[] = [];
    const apply = (line: Line, n: number): void => {
      for (const op of reducer.apply({ xai: line.xai, update: line.update, meta: { agentTimestampMs: at(n) } }))
        applyOp(entries, turns, op);
    };
    apply(retrying(1), 0);
    expect(entries).toEqual([]);
    expect(reducer.retryLive()).toMatchObject({ attempts: 1 });
    // The host's own wake line (the agent stores a wake's prompt and sends none).
    apply(
      {
        xai: false,
        update: {
          sessionUpdate: 'user_message_chunk',
          content: { type: 'text', text: '' },
          _meta: { hideFromScrollback: true },
        },
      },
      1,
    );
    // The row waits for the turn's first call or its end; the stream's live copy says the try now.
    expect(kinds(entries)).toEqual(['wake']);
    expect(reducer.retryLive()).not.toHaveProperty('endTs');
    apply(text('woke up'), 2);
    expect(kinds(entries)).toEqual(['wake', 'assistant']);
    expect(reducer.retryLive()).toMatchObject({ endTs: at(2), outcome: 'recovered' });
    apply(done(), 3);
    expect(kinds(entries)).toEqual(['wake', 'assistant', 'notice', 'turn-end']);
    expect(notices(entries)[0]).toMatchObject({ endTs: at(2), outcome: 'recovered' });
    expect(reducer.retryLive()).toBeNull();
  });

  it('R2: on disk a notice between the chunks of one prompt waits, so the chunks still merge', () => {
    // Live the host pushes the whole typed text once.
    const first = { ...prompt('first part'), prompt: 'first part\nsecond part' };
    const entries = both([first, retrying(1), chunk('second part'), text('ok'), done()]);
    expect(kinds(entries)).toEqual(['user', 'assistant', 'notice', 'turn-end']);
    expect(entries[0]).toMatchObject({ kind: 'user', text: 'first part\nsecond part' });
  });

  it('R2: a notice held after the prompt is placed before the turn end, or at the end of an open file', () => {
    expect(kinds(both([prompt('Fix it'), retrying(1), done()]))).toEqual(['user', 'notice', 'turn-end']);
    const open = parseUpdates(diskText([prompt('Fix it'), retrying(1)]));
    expect(kinds(open.transcript)).toEqual(['user', 'notice']);
  });

  it('a notice after the last turn end with no prompt after it is never drawn', () => {
    expect(kinds(both([prompt('Fix it'), text('ok'), done(), retrying(1)]))).toEqual(['user', 'assistant', 'turn-end']);
  });

  it('live ops of a whole turn with retries equal its disk replay, and the seeded reducer goes on from it', () => {
    const lines = [
      prompt('Fix the build'),
      thought('Look first'),
      retrying(1),
      retrying(2),
      thought('Now'),
      toolCall('c1'),
      retrying(1, 15, 'http', 'Connection problem; retrying request'),
      text('Fixed.'),
      done(),
    ];
    const entries = both(lines);
    expect(notices(entries).map((n) => [n.cause, n.attempts, n.max, n.outcome])).toEqual([
      ['stalled', 2, 3, 'recovered'],
      ['connection', 1, 15, 'recovered'],
    ]);
    const seeded = new TranscriptReducer();
    seeded.seed(entries, []);
    seeded.userPrompt('Next', [], at(20));
    seeded.apply({ xai: true, update: retrying(1).update, meta: { agentTimestampMs: at(21) } });
    const after = seeded.snapshot().entries;
    expect(kinds(after.slice(entries.length))).toEqual(['user', 'notice']);
    expect(notices(after).slice(0, 2)).toEqual(notices(entries));
  });
});

describe("the real agent's own lines", () => {
  type Recorded = { dir: 'in' | 'out'; msg: Record<string, unknown> };

  it('recorded wire (f-u12): a `failed` with no try before it makes no notice; the turn end has the error', () => {
    const traffic = readFileSync(join(RECORDED, 'f-u12-new-and-load.jsonl'), 'utf8')
      .split('\n')
      .filter((l) => l !== '')
      .map((l) => JSON.parse(l) as Recorded);
    const reducer = new TranscriptReducer({ now: () => 1 });
    const entries: TranscriptEntry[] = [];
    const turns: TurnMetric[] = [];
    let failedLines = 0;
    let promptId: unknown = null;
    for (const { dir, msg } of traffic) {
      if (dir === 'out' && msg.method === 'session/prompt') {
        promptId = msg.id;
        const block = ((msg.params as Update).prompt as Update[])[0]!;
        for (const op of reducer.userPrompt(block.text as string, [], 1)) applyOp(entries, turns, op);
        continue;
      }
      if (dir === 'in' && msg.id === promptId && msg.method === undefined) promptId = null;
      if (dir !== 'in' || (msg.method !== 'session/update' && msg.method !== '_x.ai/session_notification')) continue;
      const params = msg.params as Update;
      const line: UpdateLine = {
        xai: msg.method !== 'session/update',
        update: params.update as Update,
        meta: params._meta as Update | undefined,
      };
      if (line.update.sessionUpdate === 'retry_state' && line.update.type === 'failed') failedLines += 1;
      if (liveFilter(line, { turnRunning: promptId !== null, modelId: 'qwen3.5-4b-nvfp4-mtp' }) === 'drop') continue;
      for (const op of reducer.apply(line)) applyOp(entries, turns, op);
    }
    expect(failedLines).toBe(1);
    expect(notices(entries)).toEqual([]);
    expect(entries.filter((e) => e.kind === 'turn-end')).toEqual([
      expect.objectContaining({ stopReason: 'error', error: 'response truncated by max_tokens' }),
    ]);
  });

  // Stored by mlx-agent v0.3.4: the retry and turn-end lines as written, with their own stamps; tool lines cut to
  // their ids. The agent sends a one-shot past tool lines made before it, so stamps run backwards at times.
  const stored = (xai: boolean, ms: number, update: Update): Line => ({ xai, ms, update });
  const tool = (ms: number, toolCallId: string, status?: string): Line =>
    stored(false, ms, {
      sessionUpdate: status === undefined ? 'tool_call' : 'tool_call_update',
      toolCallId,
      ...(status === undefined ? { title: 'Read' } : { status }),
    });
  const stall = (ms: number, attempt: number): Line =>
    stored(true, ms, {
      sessionUpdate: 'retry_state',
      type: 'retrying',
      attempt,
      max_retries: 3,
      reason: 'Response stalled; retrying request',
      error_type: 'idle_timeout',
    });

  it('a stall, then the agent gives up: the late tool line changes nothing, the turn end has the same words', () => {
    const entries = both([
      { ...prompt('Run the suite'), ms: 1_791_671_000_000 },
      tool(1_791_671_082_536, 'toolu_5f05'),
      tool(1_791_671_082_555, 'toolu_5f05', 'in_progress'),
      stall(1_791_672_602_164, 1),
      stored(true, 1_791_674_181_243, {
        sessionUpdate: 'retry_state',
        type: 'failed',
        error_type: 'idle_timeout',
        message: 'The model stopped responding after 600s.',
      }),
      tool(1_791_671_082_628, 'toolu_5f05', 'completed'),
      stored(true, 1_791_674_181_294, {
        sessionUpdate: 'turn_completed',
        prompt_id: '5e4f8f96-fc17-4bf8-ba28-105a52e184f2',
        stop_reason: 'error',
        agent_result: 'The model stopped responding after 600s.',
        error_kind: 'idle_timeout',
        elapsed_ms: 30_902_269,
      }),
    ]);
    expect(kinds(entries)).toEqual(['user', 'tool', 'notice', 'turn-end']);
    expect(notices(entries)).toEqual([
      {
        kind: 'notice',
        ts: 1_791_672_602_164,
        code: 'retry',
        cause: 'stalled',
        attempts: 1,
        max: 3,
        endTs: 1_791_674_181_243,
        outcome: 'gave-up',
        detail: 'The model stopped responding after 600s.',
      },
    ]);
  });

  it('two stalls, then the user stops the answer: one episode, ended with no outcome', () => {
    const entries = both([
      { ...prompt('Plan the change'), ms: 1_792_749_489_832 },
      stall(1_792_750_121_353, 1),
      stall(1_792_750_723_188, 2),
      stored(true, 1_792_750_848_621, {
        sessionUpdate: 'turn_completed',
        prompt_id: '3bf092dd-306b-4fd2-a840-dda74846c105',
        stop_reason: 'cancelled',
        elapsed_ms: 1_358_877,
      }),
    ]);
    expect(notices(entries)).toEqual([
      {
        kind: 'notice',
        ts: 1_792_750_121_353,
        code: 'retry',
        cause: 'stalled',
        attempts: 2,
        max: 3,
        endTs: 1_792_750_848_621,
      },
    ]);
  });

  it('a stall, a late tool line from before it, then a thought: recovered at the thought', () => {
    const entries = both([
      { ...prompt('Fix the test'), ms: 1_791_661_600_000 },
      tool(1_791_661_626_236, 'toolu_62cf'),
      tool(1_791_661_626_258, 'toolu_62cf', 'in_progress'),
      stall(1_791_663_059_892, 1),
      tool(1_791_661_626_326, 'toolu_62cf', 'completed'),
      stored(false, 1_791_664_510_571, {
        sessionUpdate: 'agent_thought_chunk',
        content: { type: 'text', text: 'The test passes now.' },
      }),
      { ...done(), ms: 1_791_664_600_000 },
    ]);
    expect(kinds(entries)).toEqual(['user', 'tool', 'thinking', 'notice', 'turn-end']);
    expect(notices(entries)[0]).toMatchObject({ endTs: 1_791_664_510_571, outcome: 'recovered' });
  });
});

describe('pictures left out or made smaller (#12)', () => {
  type Picture = Extract<Notice, { code: 'picture' }>;
  /** grok `SessionUpdate::ImageDropped { notes }` (`extensions/notification.rs`), sent and stored. */
  const dropped = (...notes: unknown[]): Line => ({ xai: true, update: { sessionUpdate: 'image_dropped', notes } });
  /** One `ImageCompressedEntry`. */
  const image = (index: number, over: Record<string, unknown> = {}): Record<string, unknown> => ({
    index,
    original_bytes: 2_202_010,
    compressed_bytes: 943_718,
    original_width: 4032,
    original_height: 3024,
    compressed_width: 2000,
    compressed_height: 1500,
    ...over,
  });
  const compressed = (images: unknown[], message: string): Line => ({
    xai: true,
    update: { sessionUpdate: 'image_compressed', images, message },
  });
  const pictures = (entries: readonly TranscriptEntry[]): Picture[] =>
    allNotices(entries).filter((entry): entry is Picture => entry.code === 'picture');
  const STRIP = 'This request failed over its images (or was too large); 1 image(s) were left out of the retry.';
  const SHRUNK = 'Downscaled Image 1: 2.1 MB (4032x3024) → 921.6 KB (2000x1500)';

  // Mutation: count = 1 → the two notes read as one picture.
  it('image_dropped: one notice, a count of its notes, the notes as the agent wrote them', () => {
    const one = 'Image 1 was dropped before send: unreadable Png header.';
    const two = 'Image 2 was dropped before send: below dimension floor (4x4).';
    const entries = both([prompt('What is in these?'), dropped(one, two), text('I see nothing.'), done()]);
    expect(kinds(entries)).toEqual(['user', 'assistant', 'notice', 'turn-end']);
    expect(pictures(entries)).toEqual([
      { kind: 'notice', ts: at(1), code: 'picture', what: 'dropped', count: 2, notes: [one, two] },
    ]);
  });

  it('blank and non-text notes are left out; none left makes no entry', () => {
    expect(pictures(both([prompt('Look'), dropped(' ', 7, STRIP), done()]))[0]).toMatchObject({
      count: 1,
      notes: [STRIP],
    });
    expect(allNotices(both([prompt('Look'), dropped(), dropped('', null), done()]))).toEqual([]);
    const bare: Line = { xai: true, update: { sessionUpdate: 'image_dropped' } };
    expect(allNotices(both([prompt('Look'), bare, done()]))).toEqual([]);
  });

  it('a note is cut at 300 characters; at most 6 notes are kept and the count stays whole', () => {
    const long = `Image 1 was dropped before send: ${'x'.repeat(NOTICE_TEXT_CHARS)}`;
    const many = Array.from({ length: 8 }, (_, n) => `Image ${n + 2} was dropped before send: unreadable Png header.`);
    const [entry] = pictures(both([prompt('Look'), dropped(long, ...many), done()]));
    expect(entry!.count).toBe(9);
    expect(entry!.notes).toHaveLength(6);
    expect(entry!.notes[0]).toBe(`${long.slice(0, NOTICE_TEXT_CHARS)}…`);
    expect(entry!.notes.slice(1)).toEqual(many.slice(0, 5));
  });

  // Mutation: drop the cut-note step in redactEntry → the end of a cut key stays.
  it('a cut note whose end could start a key is redacted', () => {
    const long = `${'x'.repeat(NOTICE_TEXT_CHARS - 4)} sk-abcdefgh`;
    const [entry] = pictures(both([prompt('Look'), dropped(long, 'sk-abcdefghijklmnop is not a picture'), done()]));
    const shown = redactEntry(entry!, ['sk-abcdefghijklmnop']);
    expect(shown.kind === 'notice' && shown.code === 'picture' ? shown.notes : null).toEqual([
      `${'x'.repeat(NOTICE_TEXT_CHARS - 4)} …`,
      '[redacted] is not a picture',
    ]);
  });

  // Mutation: read compressed_width as the original → the sizes are wrong.
  it('image_compressed with entries: shrunk, with each picture sizes and the message as its note', () => {
    const entries = both([prompt('Look'), compressed([image(1)], SHRUNK), text('A cat.'), done()]);
    expect(pictures(entries)).toEqual([
      {
        kind: 'notice',
        ts: at(1),
        code: 'picture',
        what: 'shrunk',
        count: 1,
        notes: [SHRUNK],
        sizes: [{ fromBytes: 2_202_010, toBytes: 943_718, from: [4032, 3024], to: [2000, 1500] }],
      },
    ]);
  });

  it('an entry with a count that is not a whole number gives no size, but still counts', () => {
    const odd = image(2, { compressed_width: -1 });
    const [entry] = pictures(both([prompt('Look'), compressed([image(1), odd, 'x'], 'Downscaled'), done()]));
    expect(entry).toMatchObject({ count: 3, notes: ['Downscaled'] });
    expect(entry!.sizes).toHaveLength(1);
    const [none] = pictures(both([prompt('Look'), compressed([odd], ''), done()]));
    expect(none).toEqual({ kind: 'notice', ts: at(1), code: 'picture', what: 'shrunk', count: 1, notes: [] });
  });

  // Mutation: drop the `images.length === 0` guard → a re-encode fallback makes a row.
  it('image_compressed with images: [] (a re-encode that failed, the original kept) makes no entry', () => {
    const fallback = 'Image 1 could not be re-encoded under the 1.5 MB limit; the original attachment was kept.';
    expect(allNotices(both([prompt('Look'), compressed([], fallback), text('ok'), done()]))).toEqual([]);
  });

  it('R1: picture lines made while no turn is open land in the next turn, at its end, live and on disk', () => {
    const lines = [
      prompt('one'),
      text('a'),
      done(),
      compressed([image(1)], SHRUNK),
      dropped(STRIP),
      prompt('two', 1),
      text('b'),
      done(),
    ];
    const entries = both(lines);
    expect(kinds(entries)).toEqual([
      'user',
      'assistant',
      'turn-end',
      'user',
      'assistant',
      'notice',
      'notice',
      'turn-end',
    ]);
    expect(pictures(entries).map((entry) => entry.what)).toEqual(['shrunk', 'dropped']);
    // Live, nothing shows before the prompt.
    expect(kinds(live(lines).steps[4]!)).toEqual(['user', 'assistant', 'turn-end']);
  });

  it('R2: on disk a picture line between the chunks of one prompt (as the real agent stores it) waits, so the chunks still merge', () => {
    const first = { ...prompt('first part'), prompt: 'first part\nsecond part' };
    const entries = both([first, dropped(STRIP), chunk('second part'), text('ok'), done()]);
    expect(kinds(entries)).toEqual(['user', 'assistant', 'notice', 'turn-end']);
    expect(entries[0]).toMatchObject({ kind: 'user', text: 'first part\nsecond part' });
  });

  it('a picture line closes no retry episode: only output does', () => {
    const entries = both([prompt('Look'), retrying(1), dropped(STRIP), text('ok'), done()]);
    expect(allNotices(entries).map((n) => n.code)).toEqual(['retry', 'picture']);
    expect(notices(entries)[0]).toMatchObject({ endTs: at(3), outcome: 'recovered' });
  });
});

type HookFailure = Extract<Notice, { code: 'hook' }>;

/** A `hook_execution` line (`hook_dispatch.rs` `send_hook_execution`); `runs` as the agent writes them. */
const hookRun = (event: string, runs: unknown[], extra: Update = {}): Line => ({
  xai: true,
  update: { sessionUpdate: 'hook_execution', event_name: event, ...extra, runs },
});
const failedRun = (name: string, error: string): Update => ({
  name,
  status: { status: 'failed', error, elapsed_ms: 1_000 },
});
const LINT = 'global/lint:stop[0].hooks[0]';
const NO_ESLINT = 'exit code 127: sh: eslint: command not found';
const hookNote = (message: string): Line => ({ xai: true, update: { sessionUpdate: 'hook_annotation', message } });
const hookFailures = (entries: readonly TranscriptEntry[]): HookFailure[] =>
  entries.filter((entry): entry is HookFailure => entry.kind === 'notice' && entry.code === 'hook');

describe('hook results (#11)', () => {
  it('a run that passed, was skipped or blocked makes no entry', () => {
    const entries = both([
      prompt('Fix it'),
      hookRun('user_prompt_submit', [
        { name: 'global/a:user_prompt_submit[0].hooks[0]', status: { status: 'success', elapsed_ms: 3 } },
      ]),
      hookRun('pre_tool_use', [{ name: 'global/b:pre_tool_use[0].hooks[0]', status: { status: 'skipped' } }], {
        tool_name: 'read_file',
      }),
      hookRun('stop', [
        {
          name: 'global/c:stop[0].hooks[0]',
          status: { status: 'failed', error: 'blocked: tests fail', elapsed_ms: 9, blocked: true },
        },
      ]),
      text('ok'),
      done(),
    ]);
    expect(kinds(entries)).toEqual(['user', 'assistant', 'turn-end']);
  });

  it('a failed run is one notice: the name without its spec path, the event, the tool, the error', () => {
    const entries = both([
      prompt('Fix it'),
      hookRun('pre_tool_use', [failedRun('global/guard:pre_tool_use[0].hooks[1]', 'timed out after 1000ms')], {
        tool_name: 'run_terminal_command',
      }),
      text('ok'),
      done(),
    ]);
    expect(hookFailures(entries)).toEqual([
      {
        kind: 'notice',
        ts: at(1),
        code: 'hook',
        event: 'pre_tool_use',
        tool: 'run_terminal_command',
        hook: 'global/guard',
        detail: 'timed out after 1000ms',
        times: 1,
        lastTs: at(1),
      },
    ]);
  });

  it('the same failure 3 times is one notice with times 3 and the newest time; live it is set in place', () => {
    const lines = [
      prompt('one'),
      text('a'),
      hookRun('stop', [failedRun(LINT, NO_ESLINT)]),
      done(),
      prompt('two', 1),
      text('b'),
      hookRun('stop', [failedRun(LINT, NO_ESLINT)]),
      done(),
      prompt('three', 2),
      text('c'),
      hookRun('stop', [failedRun(LINT, NO_ESLINT)]),
      done(),
    ];
    const entries = both(lines);
    expect(hookFailures(entries)).toEqual([
      expect.objectContaining({ ts: at(2), hook: 'global/lint', event: 'stop', times: 3, lastTs: at(10) }),
    ]);
    expect(entries.indexOf(hookFailures(entries)[0]!)).toBe(2);
    // Live, a repeat adds no row: the first turn's row changes.
    const { steps } = live(lines);
    expect(steps[9]!.length).toBe(steps[10]!.length);
    expect(steps[10]![2]).toMatchObject({ kind: 'notice', code: 'hook', times: 3 });
  });

  it('another error text, hook, event or tool is a second notice', () => {
    const entries = both([
      prompt('Fix it'),
      hookRun('stop', [failedRun(LINT, NO_ESLINT), failedRun(LINT, 'exit code 1: lint errors')]),
      hookRun('stop', [failedRun('global/other:stop[0].hooks[0]', NO_ESLINT)]),
      hookRun('session_end', [failedRun('global/lint:session_end[0].hooks[0]', NO_ESLINT)]),
      hookRun('post_tool_use', [failedRun('global/lint:post_tool_use[0].hooks[0]', NO_ESLINT)], { tool_name: 'write' }),
      hookRun('post_tool_use', [failedRun('global/lint:post_tool_use[0].hooks[0]', NO_ESLINT)], { tool_name: 'grep' }),
      done(),
    ]);
    expect(hookFailures(entries).map((n) => [n.hook, n.event, n.tool, n.detail, n.times])).toEqual([
      ['global/lint', 'stop', undefined, NO_ESLINT, 1],
      ['global/lint', 'stop', undefined, 'exit code 1: lint errors', 1],
      ['global/other', 'stop', undefined, NO_ESLINT, 1],
      ['global/lint', 'session_end', undefined, NO_ESLINT, 1],
      ['global/lint', 'post_tool_use', 'write', NO_ESLINT, 1],
      ['global/lint', 'post_tool_use', 'grep', NO_ESLINT, 1],
    ]);
  });

  it("the app's own guard reads as the app's command check", () => {
    const guard = workGuardHookName(workGuardName());
    const entries = both([
      prompt('ls'),
      hookRun('pre_tool_use', [failedRun(guard, 'timed out after 5000ms')], { tool_name: 'run_terminal_command' }),
      done(),
    ]);
    const row = hookFailures(entries)[0]!;
    expect(row.hook).toBe(guard.slice(0, guard.indexOf(':')));
    expect(hookName(row.hook)).toBe(CHAT.hook.ownGuard);
    expect(hookRow(row).text).toBe(`${CHAT.hook.ownGuard} · before the command`);
    // One character off and it is any other hook, shown as sent.
    expect(hookName(`${row.hook.slice(0, -8)}G/plugin`)).toBe(`${row.hook.slice(0, -8)}G/plugin`);
    expect(hookName('plugin/mlx-work-guard-0123/plugin')).toBe('plugin/mlx-work-guard-0123/plugin');
  });

  it('R1: a session_end failure after the last turn is held and shows in the next turn, at its end', () => {
    const lines = [
      prompt('one'),
      text('a'),
      done(),
      hookRun('session_end', [
        failedRun('global/settings:session_end[0].hooks[0]', 'session closed before the hook ran'),
      ]),
      prompt('two', 1),
      text('b'),
      done(),
    ];
    const entries = both(lines);
    expect(kinds(entries)).toEqual(['user', 'assistant', 'turn-end', 'user', 'assistant', 'notice', 'turn-end']);
    expect(entries[5]).toMatchObject({ code: 'hook', event: 'session_end', hook: 'global/settings' });
    const { steps } = live(lines);
    expect(kinds(steps[3]!)).toEqual(['user', 'assistant', 'turn-end']);
    // At the end of a file with no prompt after it, it is never drawn.
    expect(kinds(parseUpdates(diskText(lines.slice(0, 4))).transcript)).toEqual(['user', 'assistant', 'turn-end']);
  });

  // Mutation: place a hook row when it arrives (no `hookHeld`) → the stored list differs from the live one.
  it("a row goes before the first call made after it, else at the turn's end: the store's order gives the live list", () => {
    // Live, as the agent sends them: the answer, the stop batch, the end. The store still holds the answer's text when
    // the one-shot comes, so it writes the one-shot first (`persistence.rs` `maybe_merge_notification`).
    const stop = { ...hookRun('stop', [failedRun(LINT, NO_ESLINT)]), ms: at(3) };
    const answer = { ...text('Done.'), ms: at(2) };
    const end = { ...done(), ms: at(4) };
    const sent = [{ ...prompt('Fix it'), ms: at(0) }, answer, stop, end];
    const stored = [sent[0]!, stop, answer, end];
    const shown = live(sent).entries;
    expect(kinds(shown)).toEqual(['user', 'assistant', 'notice', 'turn-end']);
    expect(comparable(parseUpdates(diskText(stored)).transcript)).toEqual(comparable(shown));
    // A note that keeps the agent going: the next round's text still joins the answer, and the row goes before the
    // next call, which the store holds when the row comes.
    const note = { ...hookNote('↩ Stop blocked by hook `global/tests-green`, continuing: 2 tests fail'), ms: at(3) };
    const call = { ...toolCall('call-1'), ms: at(1) };
    const more = { ...text(' Running the tests.'), ms: at(4) };
    const next = { ...toolCall('call-2'), ms: at(5) };
    const line = [{ ...prompt('Fix it'), ms: at(0) }, call, { ...text('Done.'), ms: at(2) }, note, more, next];
    const sentNote = [...line, { ...done(), ms: at(6) }];
    const storedNote = [line[0]!, call, note, line[2]!, more, next, sentNote.at(-1)!];
    const shownNote = live(sentNote).entries;
    expect(kinds(shownNote)).toEqual(['user', 'tool', 'assistant', 'notice', 'tool', 'turn-end']);
    expect(shownNote[2]).toMatchObject({ text: 'Done. Running the tests.' });
    expect(comparable(parseUpdates(diskText(storedNote)).transcript)).toEqual(comparable(shownNote));
    // A pre_tool_use failure stored before the call it ran for: the call is older than the row, so it stays first.
    const pre = {
      ...hookRun('pre_tool_use', [failedRun('global/audit:pre_tool_use[0].hooks[0]', 'exit code 1')]),
      ms: at(2),
    };
    const preLines = [{ ...prompt('Fix it'), ms: at(0) }, call, pre, next, { ...done(), ms: at(6) }];
    const preStored = [preLines[0]!, pre, call, next, preLines[4]!];
    const shownPre = live(preLines).entries;
    expect(kinds(shownPre)).toEqual(['user', 'tool', 'notice', 'tool', 'turn-end']);
    expect(comparable(parseUpdates(diskText(preStored)).transcript)).toEqual(comparable(shownPre));
  });

  it("the user_prompt_submit line's prompt id still names a wake", () => {
    const wake = 'subagent-completed-sub-1';
    const lines: Line[] = [
      {
        xai: true,
        update: {
          sessionUpdate: 'subagent_spawned',
          child_session_id: 'child-1',
          subagent_id: 'sub-1',
          subagent_type: 'explore',
          description: 'Find the flaky test',
        },
      },
      hookRun('user_prompt_submit', [failedRun('global/a:user_prompt_submit[0].hooks[0]', 'exit code 1')], {
        prompt_id: wake,
      }),
      {
        xai: false,
        update: {
          sessionUpdate: 'user_message_chunk',
          content: { type: 'text', text: 'The helper ended.' },
          _meta: { hideFromScrollback: true },
        },
        diskOnly: true,
      },
    ];
    const stored = parseUpdates(diskText(lines)).transcript;
    expect(stored.find((e) => e.kind === 'wake')).toMatchObject({
      cause: 'subagent',
      subagent: { childId: 'child-1', description: 'Find the flaky test' },
    });
    expect(kinds(stored)).toEqual(['subagent', 'wake', 'notice']);
  });

  it('the error and a note are cut at 600 characters; a cut end that could start a key is redacted', () => {
    const long = `${'e'.repeat(HOOK_TEXT_CHARS - 4)}sk-live-123456`;
    const entries = both([
      prompt('Fix it'),
      hookRun('stop', [failedRun(LINT, long)]),
      hookNote(`\u26a0 ${long}`),
      done(),
    ]);
    const [failure] = hookFailures(entries);
    expect(failure!.detail).toBe(`${long.slice(0, HOOK_TEXT_CHARS)}…`);
    const note = allNotices(entries).find((n) => n.code === 'hook-note');
    expect(note).toMatchObject({ code: 'hook-note', text: `${long.slice(0, HOOK_TEXT_CHARS)}…` });
    const needles = ['sk-live-123456'];
    expect((redactEntry(failure!, needles) as HookFailure).detail).toBe(`${'e'.repeat(HOOK_TEXT_CHARS - 4)}…`);
    expect(redactEntry(note!, needles)).toMatchObject({ text: `${'e'.repeat(HOOK_TEXT_CHARS - 4)}…` });
  });

  it("a note drops the agent's leading ⚠ or ↩; a blank one makes none", () => {
    const entries = both([
      prompt('Fix it'),
      text('ok'),
      hookNote('\u21a9 Stop blocked by hook `global/tests-green`, continuing: 2 tests fail'),
      hookNote('\u26a0\ufe0f  Hook `global/budget` stopped the agent: token budget used'),
      hookNote('Remember to run the tests.'),
      hookNote('   '),
      done(),
    ]);
    expect(allNotices(entries).map((n) => (n.code === 'hook-note' ? n.text : n.code))).toEqual([
      'Stop blocked by hook `global/tests-green`, continuing: 2 tests fail',
      'Hook `global/budget` stopped the agent: token budget used',
      'Remember to run the tests.',
    ]);
    expect(kinds(entries)).toEqual(['user', 'assistant', 'notice', 'notice', 'notice', 'turn-end']);
  });

  it('a prompt a hook blocked: no chunk is stored, so its note and its end stay together; the end says a hook stopped it', () => {
    const blockedEnd: Line = {
      xai: true,
      update: { sessionUpdate: 'turn_completed', prompt_id: 'p-2', stop_reason: 'cancelled', elapsed_ms: 3 },
      meta: { cancellationCategory: 'HookDenied', cancellationContext: { hook_name: 'global/policy', reason: 'no' } },
      ms: at(5),
    };
    const blocked: Line[] = [
      hookRun(
        'user_prompt_submit',
        [
          {
            name: 'global/policy:user_prompt_submit[0].hooks[0]',
            status: { status: 'failed', error: 'blocked: no deploys on Friday', elapsed_ms: 30, blocked: true },
          },
        ],
        { prompt_id: 'p-2' },
      ),
      { ...hookNote('\u26a0 Prompt blocked by global/policy: no deploys on Friday'), ms: at(4) },
      blockedEnd,
    ];
    const lines = [prompt('one'), text('a'), done(), ...blocked, prompt('three', 1), text('c'), done()];
    const stored = parseUpdates(diskText(lines)).transcript;
    expect(kinds(stored)).toEqual([
      'user',
      'assistant',
      'turn-end',
      'notice',
      'turn-end',
      'user',
      'assistant',
      'turn-end',
    ]);
    expect(stored[3]).toMatchObject({
      code: 'hook-note',
      text: 'Prompt blocked by global/policy: no deploys on Friday',
    });
    expect(stored[4]).toEqual({
      kind: 'turn-end',
      ts: at(5),
      stopReason: 'cancelled',
      elapsedMs: 3,
      hookStopped: true,
    });
    // Live the host pushes the prompt itself: the same note right after it, then the same end.
    const shown = live([
      prompt('one'),
      text('a'),
      done(),
      { ...prompt('deploy to prod'), ms: at(3) },
      ...blocked,
    ]).entries;
    expect(kinds(shown)).toEqual(['user', 'assistant', 'turn-end', 'user', 'notice', 'turn-end']);
    expect(shown.slice(4)).toEqual(stored.slice(3, 5));
    // Any other cancel stays a plain cancel.
    const other = parseUpdates(
      diskText([prompt('one'), { ...blockedEnd, meta: { cancellationCategory: 'MidTurnAbort' } }]),
    );
    expect(other.transcript[1]).not.toHaveProperty('hookStopped');
  });

  it('a seeded conversation counts a live repeat on its stored row', () => {
    const lines = [prompt('one'), hookRun('stop', [failedRun(LINT, NO_ESLINT)]), done()];
    const disk = parseUpdates(diskText(lines));
    const reducer = new TranscriptReducer();
    reducer.seed(disk.transcript, disk.turns);
    const entries = disk.transcript.map((e) => ({ ...e }));
    const turns = disk.turns.map((t) => ({ ...t }));
    for (const op of reducer.userPrompt('two', [], at(10))) applyOp(entries, turns, op);
    const ops = reducer.apply({
      xai: true,
      update: hookRun('stop', [failedRun(LINT, NO_ESLINT)]).update,
      meta: { agentTimestampMs: at(11) },
    });
    expect(ops).toEqual([
      { op: 'set', index: 1, entry: expect.objectContaining({ code: 'hook', times: 2, lastTs: at(11) }) },
    ]);
    for (const op of ops) applyOp(entries, turns, op);
    expect(hookFailures(entries)).toHaveLength(1);
  });

  it('the name keeps everything but a spec path tail', () => {
    expect(hookSource('global/settings:session_start[0].hooks[0]')).toBe('global/settings');
    expect(hookSource('plugin/codex/hooks:stop[12].hooks[3]')).toBe('plugin/codex/hooks');
    expect(hookSource('a:b:pre_tool_use[0].hooks[0]')).toBe('a:b');
    for (const name of [
      'global/lint',
      ':stop[0].hooks[0]',
      'src:Stop[0].hooks[0]',
      'src:stop[0].hooks[]',
      'src:stop[a].hooks[0]',
      'src:stop[0].hooks[0]x',
      'src:stop0].hooks[0]',
      'src:[0].hooks[0]',
      'client:abc',
    ])
      expect(hookSource(name)).toBe(name);
  });
});

describe("one-shot rows in the agent store's order", () => {
  /**
   * The order the agent's store writes the sent lines in (`persistence.rs` `maybe_merge_notification`): it holds the
   * last ACP line until the next one comes and merges text of the same kind into it (the merged line keeps the last
   * stamp); an xAI line is written at once, so before the held line; the turn end drains the held line first.
   */
  function storeOrder(sent: readonly Line[]): Line[] {
    const stamped = sent.map((line, n) => ({ ...line, ms: line.ms ?? at(n) }));
    const out: Line[] = [];
    let held: Line | null = null;
    for (const line of stamped) {
      if (line.xai) {
        if (line.update.sessionUpdate === 'turn_completed' && held !== null) {
          out.push(held);
          held = null;
        }
        out.push(line);
        continue;
      }
      const kind = line.update.sessionUpdate;
      if (
        held !== null &&
        (kind === 'agent_message_chunk' || kind === 'agent_thought_chunk') &&
        held.update.sessionUpdate === kind
      ) {
        const body = (u: Update): string => String((u.content as { text: string }).text);
        held = {
          ...line,
          update: { ...line.update, content: { type: 'text', text: body(held.update) + body(line.update) } },
        };
        continue;
      }
      if (held !== null) out.push(held);
      held = line;
    }
    if (held !== null) out.push(held);
    return out;
  }

  const toolDone = (id: string): Line => ({
    xai: false,
    update: { sessionUpdate: 'tool_call_update', toolCallId: id, status: 'completed' },
  });
  const mode = (id: string): Line => ({
    xai: false,
    update: { sessionUpdate: 'current_mode_update', currentModeId: id },
  });
  const picture = (note: string): Line => ({ xai: true, update: { sessionUpdate: 'image_dropped', notes: [note] } });

  /** A merged text line keeps its last chunk's stamp, so a live text starts earlier than the stored one. */
  const order = (entries: readonly TranscriptEntry[]): unknown[] =>
    comparable(entries).map((entry) => {
      const e = entry as TranscriptEntry;
      if (e.kind !== 'assistant') return e;
      const { ts: _ts, ...rest } = e;
      return rest;
    });

  /** The live list of the sent lines equals the list read from what the store wrote of them. */
  function law(sent: readonly Line[]): TranscriptEntry[] {
    const shown = live(sent).entries;
    const disk = parseUpdates(diskText(storeOrder(sent))).transcript;
    expect(order(shown)).toEqual(order(disk));
    return shown;
  }

  // Mutation: place a retry episode when it comes (no `oneShots`) → the stored text is one line before the row, live
  // it is cut by it.
  it('a retry in the middle of the text: the row goes before the next call, live and on disk', () => {
    const shown = law([
      prompt('Fix it'),
      text('Before the stall'),
      retrying(1),
      text(' and after it.'),
      toolCall('call-1'),
      toolDone('call-1'),
      text('ok'),
      done(),
    ]);
    expect(kinds(shown)).toEqual(['user', 'assistant', 'notice', 'tool', 'assistant', 'turn-end']);
    expect(shown[1]).toMatchObject({ text: 'Before the stall and after it.' });
    expect(notices(shown)[0]).toMatchObject({ attempts: 1, outcome: 'recovered', endTs: at(3) });
  });

  it('a retry after a call and two tries: the row goes at the end, live and on disk', () => {
    const shown = law([
      prompt('Fix it'),
      toolCall('call-1'),
      toolDone('call-1'),
      retrying(1),
      retrying(2),
      text('ok'),
      done(),
    ]);
    expect(kinds(shown)).toEqual(['user', 'tool', 'assistant', 'notice', 'turn-end']);
    expect(notices(shown)[0]).toMatchObject({ attempts: 2, outcome: 'recovered' });
  });

  it('a picture line made after the prompt: the store writes it first; the row goes at the end both ways', () => {
    const sent = [prompt('Look'), picture('Image 1 was dropped before send: unreadable Png header.'), text('A cat.')];
    const shown = law([...sent, done()]);
    expect(kinds(shown)).toEqual(['user', 'assistant', 'notice', 'turn-end']);
    expect(storeOrder(sent).map((line) => line.update.sessionUpdate)).toEqual([
      'image_dropped',
      'user_message_chunk',
      'agent_message_chunk',
    ]);
  });

  it('a mode line keeps its place (an ACP line); a retry after it still goes at the end', () => {
    const shown = law([prompt('Plan it'), text('First.'), mode('plan'), retrying(1), text('Then.'), done()]);
    expect(kinds(shown)).toEqual(['user', 'assistant', 'plan-mode', 'assistant', 'notice', 'turn-end']);
  });

  it('a failed stop hook after the answer: the row goes at the end, live and on disk', () => {
    const shown = law([prompt('Fix it'), text('Done.'), hookRun('stop', [failedRun(LINT, NO_ESLINT)]), done()]);
    expect(kinds(shown)).toEqual(['user', 'assistant', 'notice', 'turn-end']);
  });
});
