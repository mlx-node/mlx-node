import { describe, expect, it } from 'vite-plus/test';

import {
  anthropicMessages,
  buildRequests,
  checkOffline,
  offlineRequest,
  recordedTimeline,
  responseStarts,
  type HistoryRow,
  type OfflineTurn,
  type Session,
} from '../../scripts/dflash2-agent-replay.js';
import type { ProfileRow } from '../../scripts/dflash2-trace-summary.js';

const SYSTEM = 'You are a subagent.';

const ROWS: HistoryRow[] = [
  { type: 'system', content: SYSTEM },
  { type: 'user', content: [{ type: 'text', text: 'context' }], synthetic_reason: 'project_instructions' },
  { type: 'user', content: [{ type: 'text', text: 'task' }] },
  { type: 'reasoning', summary: [{ type: 'summary_text', text: 'plan' }] },
  {
    type: 'assistant',
    content: '',
    tool_calls: [
      { id: 'toolu_a.1', name: 'read', arguments: '{"path":"b","line":2}' },
      { id: 'toolu_b', name: 'read', arguments: 'not json' },
    ],
    model_id: 'Qwen3.8-27B-UD-Q4_K_M',
  },
  { type: 'tool_result', tool_call_id: 'toolu_a.1', content: 'one' },
  { type: 'tool_result', tool_call_id: 'toolu_b', content: 'two' },
  {
    type: 'reasoning',
    summary: [
      { type: 'summary_text', text: 'first' },
      { type: 'summary_text', text: 'second' },
    ],
  },
  { type: 'assistant', content: 'done', tool_calls: [], model_id: 'Qwen3.8-27B-UD-Q4_K_M' },
];

function session(id: string, firstTokenMs: number[]): Session {
  return {
    id,
    system: SYSTEM,
    tools: [
      { type: 'function', function: { name: 'read', description: 'Read a file', parameters: { type: 'object' } } },
    ],
    rows: ROWS,
    loopStartedMs: firstTokenMs.flatMap((at) => [at - 50, at - 20]),
    firstTokenMs,
  };
}

function row(timestamp: number, promptTokens: number): ProfileRow {
  return { timestamp: new Date(timestamp).toISOString(), promptTokens, generatedTokens: 10, decodeMs: 0, ttftMs: 1 };
}

function turn(label: string, overrides: Partial<OfflineTurn>): OfflineTurn {
  return {
    label,
    session: null,
    prompt_tokens: 100,
    generated_tokens: null,
    gen_suffix_len: 3,
    replay_point: 96,
    hit_kind: 'none',
    hit_tokens: 0,
    prefilled_tokens: 100,
    miss_reason: 'cold_first',
    shadow_hit: 0,
    ideal_tokens: 0,
    ideal_paged_tokens: 0,
    diverge_at: -1,
    ...overrides,
  };
}

describe('dflash2-agent-replay request reconstruction', () => {
  it('starts each response at its reasoning rows', () => {
    expect(responseStarts(ROWS)).toEqual([3, 7]);
  });

  it('builds the messages the agent sends', () => {
    expect(anthropicMessages(ROWS.slice(0, 7), SYSTEM)).toEqual([
      { role: 'user', content: [{ type: 'text', text: 'context' }] },
      { role: 'user', content: [{ type: 'text', text: 'task' }] },
      {
        role: 'assistant',
        content: [
          { type: 'thinking', thinking: 'plan', signature: '' },
          { type: 'tool_use', id: 'toolu_a_1', name: 'read', input: { path: 'b', line: 2 } },
          { type: 'tool_use', id: 'toolu_b', name: 'read', input: {} },
        ],
      },
      {
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 'toolu_a_1', content: 'one' },
          { type: 'tool_result', tool_use_id: 'toolu_b', content: 'two' },
        ],
      },
    ]);
  });

  it('joins reasoning summary parts with a newline', () => {
    const messages = anthropicMessages([...ROWS, { type: 'user', content: 'next' }], SYSTEM);
    expect(messages[4]).toEqual({
      role: 'assistant',
      content: [
        { type: 'thinking', thinking: 'first\nsecond', signature: '' },
        { type: 'text', text: 'done' },
      ],
    });
  });

  it('rejects a system row that differs from system_prompt.txt', () => {
    expect(() => anthropicMessages([{ type: 'system', content: 'other' }], SYSTEM)).toThrow(/system_prompt/);
  });

  it('maps one request per first_token event and maps it through the server', () => {
    const requests = buildRequests(session('s', [1000, 2000]));
    expect(requests.map((r) => [r.label, r.loopStartedMs, r.body.messages.length])).toEqual([
      ['s#1', 980, 2],
      ['s#2', 1980, 4],
    ]);
    expect(requests[1].body.tools).toEqual([
      { name: 'read', description: 'Read a file', input_schema: { type: 'object' } },
    ]);
    const offline = offlineRequest(requests[1]);
    expect(offline.messages?.[3]).toEqual({
      role: 'assistant',
      content: '',
      reasoning_content: 'plan',
      tool_calls: [
        { id: 'call_a_1', name: 'read', arguments: '{"path":"b","line":2}' },
        { id: 'call_b', name: 'read', arguments: '{}' },
      ],
    });
    expect(offline.messages?.[4]).toEqual({ role: 'tool', content: 'one', tool_call_id: 'call_a_1' });
    expect(() => buildRequests(session('s', [1000, 2000, 3000]))).toThrow(/2 model responses but 3 first_token/);
  });
});

describe('dflash2-agent-replay offline checks', () => {
  const sessions = [session('a', [1000, 5000]), session('b', [3000, 7000])];
  const requests = sessions.flatMap((s) => buildRequests(s));
  const rows = [row(1000, 100), row(2000, 50), row(3000, 100), row(5000, 160), row(7000, 170)];

  it('orders the recorded run by log row and keeps unattributed rows as extras', () => {
    const timeline = recordedTimeline(rows, sessions, requests);
    expect(timeline.map((entry) => ('request' in entry ? entry.request.label : 'extra'))).toEqual([
      'a#1',
      'extra',
      'b#1',
      'a#2',
      'b#2',
    ]);
    expect(() => recordedTimeline(rows.slice(0, 4), sessions, requests)).toThrow(/b: 1 attributed log rows for 2/);
  });

  it('passes a replay that matches the recording', () => {
    const timeline = recordedTimeline(rows, sessions, requests);
    const turns = [
      turn('a#1', {}),
      turn('extra@1', { prompt_tokens: 50, prefilled_tokens: 50, miss_reason: 'extra_request' }),
      turn('b#1', {}),
      turn('a#2', { prompt_tokens: 160, prefilled_tokens: 160, miss_reason: 'cross_session' }),
      turn('b#2', { prompt_tokens: 170, prefilled_tokens: 170, miss_reason: 'cross_session' }),
    ];
    expect(checkOffline(timeline, turns)).toEqual({ failures: [], crossSession: { misses: 2, crossSession: 2 } });
  });

  it('names every request the replay gets wrong', () => {
    const timeline = recordedTimeline(rows, sessions, requests);
    const turns = [
      turn('a#1', { prompt_tokens: 101, prefilled_tokens: 101 }),
      turn('extra@1', { prompt_tokens: 50, prefilled_tokens: 50, miss_reason: 'extra_request' }),
      turn('b#1', { miss_reason: 'unknown' }),
      turn('a#2', { prompt_tokens: 160, prefilled_tokens: 160, miss_reason: 'flat_dirty' }),
      turn('b#2', { prompt_tokens: 270, hit_kind: 'live', hit_tokens: 90, prefilled_tokens: 180, miss_reason: 'none' }),
    ];
    expect(checkOffline(timeline, turns).failures).toEqual([
      'a#1: reconstructed prompt has 101 tokens and misses, the recording prefilled 100',
      'b#1: classified unknown',
      'a#2: a later-turn miss classified flat_dirty, not cross_session',
      'b#2: replay prefills 180 after a live hit of 90, the recording 170',
    ]);
  });
});
