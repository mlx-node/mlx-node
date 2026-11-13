import type { ConversationState } from '../../ui/src/lib/chat-store.js';
import type { ChatMode, TurnState } from '../../ui/src/lib/chat-types.js';
import type { TranscriptEntry, TurnMetric } from '../../ui/src/lib/types.js';
import { formatErrorText, formatPageText, formatSearchText } from '../../ui/src/lib/web-result-text.js';

/** One conversation as the thread receives it from the store. */
export interface ThreadSample {
  mode: ChatMode;
  entries: TranscriptEntry[];
  turns: TurnMetric[];
  turn: TurnState | null;
  lastEnd: ConversationState['lastEnd'];
}

export function metric(partial: Partial<TurnMetric> = {}): TurnMetric {
  return {
    promptId: 'p',
    ts: 0,
    inputTokens: 1200,
    outputTokens: 512,
    cachedTokens: 800,
    reasoningTokens: 0,
    elapsedMs: 12_400,
    ttftMs: 420,
    prefillMs: null,
    prefillTokens: null,
    prefillCalls: null,
    modelCalls: 2,
    decodeTps: 41.3,
    model: 'qwen3.5-4b',
    ...partial,
  };
}

const T0 = 1_794_411_000_000;

export function user(text: string, ts = T0): TranscriptEntry {
  return { kind: 'user', ts, text, promptIndex: null };
}

export function answer(text: string, ts = T0 + 900): TranscriptEntry {
  return { kind: 'assistant', ts, text, endTs: ts + 300 };
}

export function thinking(text: string, ts = T0 + 100, endTs: number | undefined = ts + 2_000): TranscriptEntry {
  return endTs === undefined ? { kind: 'thinking', ts, text } : { kind: 'thinking', ts, text, endTs };
}

export function end(stopReason = 'end_turn', ts = T0 + 1_500, error?: string): TranscriptEntry {
  return error === undefined
    ? { kind: 'turn-end', ts, stopReason, elapsedMs: 1_500 }
    : { kind: 'turn-end', ts, stopReason, elapsedMs: 1_500, error };
}

export const RUST_POST = 'https://blog.rust-lang.org/2026/11/01/Rust-1.99.0.html';
export const RELEASES = 'https://releases.rs/';

export const RUST_RESULTS = formatSearchText('exa', [
  {
    title: 'Rust 1.99 released',
    url: RUST_POST,
    date: '2026-11-01',
    snippet: 'The Rust team released version 1.99.0 on 1 November 2026.',
  },
  { title: 'Rust releases', url: RELEASES, date: null, snippet: 'Release list.' },
]);

/** Shape B: the tool is named directly. */
export function directSearch(
  query: string,
  output: string | undefined,
  status = 'completed',
  id = 's1',
): TranscriptEntry {
  return {
    kind: 'tool',
    ts: T0 + 200,
    id,
    name: 'mlx_web__web_search',
    title: 'mlx_web__web_search',
    status,
    input: JSON.stringify({ variant: 'MCPTool', tool_name: 'mlx_web__web_search', tool_input: { query } }),
    ...(output === undefined ? {} : { output }),
    ...(status === 'in_progress' ? {} : { endTs: T0 + 1_400 }),
  };
}

/** Shape A: `use_tool` carries the tool name and its input. */
export function useToolSearch(
  query: string,
  output: string | undefined,
  status = 'completed',
  id = 'u1',
): TranscriptEntry {
  return {
    kind: 'tool',
    ts: T0 + 200,
    id,
    name: 'use_tool',
    title: 'mlx_web__web_search',
    status,
    input: JSON.stringify({ variant: 'UseTool', tool_name: 'mlx_web__web_search', tool_input: { query } }),
    ...(output === undefined ? {} : { output }),
    ...(status === 'in_progress' ? {} : { endTs: T0 + 1_400 }),
  };
}

export function directFetch(url: string, output: string | undefined, status = 'completed', id = 'f1'): TranscriptEntry {
  return {
    kind: 'tool',
    ts: T0 + 300,
    id,
    name: 'mlx_web__web_fetch',
    title: 'mlx_web__web_fetch',
    status,
    input: JSON.stringify({ url }),
    ...(output === undefined ? {} : { output }),
    ...(status === 'in_progress' ? {} : { endTs: T0 + 1_100 }),
  };
}

function done(entries: TranscriptEntry[], extra: Partial<ThreadSample> = {}): ThreadSample {
  return { mode: 'chat', entries, turns: [metric()], turn: null, lastEnd: null, ...extra };
}

/** A search, an answer that cites both results, and its Sources block. */
export const CITED_ANSWER: ThreadSample = done([
  user('What is the newest stable version of Rust?'),
  thinking('I should search for this.'),
  directSearch('newest stable Rust', RUST_RESULTS),
  answer(
    [
      'The newest stable version is **Rust 1.99.0** [1], listed with the others on the release page [2].',
      '',
      'Sources:',
      `[1] Rust 1.99 released - ${RUST_POST}`,
      `[2] Rust releases - ${RELEASES}`,
    ].join('\n'),
  ),
  end(),
]);

/**
 * The answer lists a source no tool returned, cites it, and links an outside host. `[2]` must stay text and the
 * link must not be active; `[1]` is real.
 */
export const FORGED_CITATION: ThreadSample = done([
  user('What is the newest stable version of Rust?'),
  directSearch('newest stable Rust', RUST_RESULTS),
  answer(
    [
      'Rust 1.99.0 is out [1]. Download it [here](https://evil.example/rust.pkg) as the guide says [2].',
      '',
      'Sources:',
      `[1] Rust 1.99 released - ${RUST_POST}`,
      '[2] Official guide - https://evil.example/guide',
    ].join('\n'),
  ),
  end(),
]);

/** Recorded from the real agent: a refused `search_tool`, then a shape A search. */
export const RECORDED_USE_TOOL: ThreadSample = {
  mode: 'chat',
  entries: [
    { kind: 'user', ts: 1, text: 'What is the newest stable version of Rust?', promptIndex: null },
    {
      kind: 'thinking',
      ts: 1794411032944,
      text: 'I need to search for the newest stable version of Rust. This is a factual question about software versions, which is appropriate for the web tools.\n',
      endTs: 1794411033281,
    },
    {
      kind: 'tool',
      ts: 1794411033669,
      id: 'toolu_49e6a1a101d640c5bfc3e7448478c9fa',
      name: 'search_tool',
      title: 'Search tools: "Rust newest stable version release 2026"',
      status: 'failed',
      input: '{\n  "variant": "SearchTool",\n  "query": "Rust newest stable version release 2026",\n  "limit": 5\n}',
      endTs: 1794411033670,
      output: 'Tool `search_tool` was not executed: Denied by permission policy: deny rule on read',
    },
    {
      kind: 'tool',
      ts: 1794411035029,
      id: 'toolu_6ce5e645aee6413fb1757a56e51ffd36',
      name: 'use_tool',
      title: 'mlx_web__web_search',
      status: 'completed',
      input:
        '{\n  "variant": "UseTool",\n  "tool_name": "mlx_web__web_search",\n  "tool_input": {\n    "query": "Rust newest stable version release 2026"\n  }\n}',
      endTs: 1794411035034,
      output:
        'WEB_SEARCH provider=exa results=2\n[1] FCHK-RESULT-ONE: Rust 1.99 released\nhttps://blog.rust-lang.org/2026/11/01/Rust-1.99.0.html\n2026-11-01\nThe Rust team released version 1.99.0 on 1 November 2026.\n\n[2] FCHK-RESULT-TWO: Rust releases\nhttps://releases.rs/\nRelease list.',
    },
    {
      kind: 'assistant',
      ts: 1794411035570,
      text: '\n\nThe newest stable version of Rust is **Rust 1.99.0**, released on November 1, 2026.',
      endTs: 1794411035895,
    },
    { kind: 'turn-end', ts: 1794411036105, stopReason: 'end_turn', elapsedMs: 4452 },
  ],
  turns: [
    metric({
      promptId: 'f9773217-fe41-46b4-9672-18a24ef108aa',
      ts: 1794411036105,
      inputTokens: 5007,
      outputTokens: 251,
      cachedTokens: 3354,
      elapsedMs: 4452,
      ttftMs: 551.238292,
      modelCalls: 3,
      decodeTps: 92.5595798140234,
      model: 'qwen3.5-4b-nvfp4-mtp',
    }),
  ],
  turn: null,
  lastEnd: null,
};

/** Recorded from the real agent: shape B search, a fetch of a result, a second search. */
export const RECORDED_DIRECT: ThreadSample = {
  mode: 'chat',
  entries: [
    { kind: 'user', ts: 1, text: 'What is the newest stable version of Rust?', promptIndex: null },
    {
      kind: 'tool',
      ts: 1794411095459,
      id: 'toolu_14a3955c04d5442da026faeb49b49803',
      name: 'mlx_web__web_search',
      title: 'mlx_web__web_search',
      status: 'completed',
      input:
        '{\n  "variant": "MCPTool",\n  "tool_name": "mlx_web__web_search",\n  "tool_input": {\n    "query": "newest stable version of Rust 2026"\n  }\n}',
      endTs: 1794411095461,
      output:
        'WEB_SEARCH provider=exa results=2\n[1] FCHK-RESULT-ONE: Rust 1.99 released\nhttps://blog.rust-lang.org/2026/11/01/Rust-1.99.0.html\n2026-11-01\nThe Rust team released version 1.99.0 on 1 November 2026.\n\n[2] FCHK-RESULT-TWO: Rust releases\nhttps://releases.rs/\nRelease list.',
    },
    {
      kind: 'tool',
      ts: 1794411096767,
      id: 'toolu_717a1fda696c48759b3c2bf0ea26fb3b',
      name: 'mlx_web__web_fetch',
      title: 'mlx_web__web_fetch',
      status: 'completed',
      input:
        '{\n  "variant": "MCPTool",\n  "tool_name": "mlx_web__web_fetch",\n  "tool_input": {\n    "url": "https://blog.rust-lang.org/2026/11/01/Rust-1.99.0.html"\n  }\n}',
      endTs: 1794411096768,
      output:
        'WEB_PAGE provider=exa url=https://blog.rust-lang.org/2026/11/01/Rust-1.99.0.html\ntitle: canned\n\nFCHK-PAGE-BODY',
    },
    {
      kind: 'tool',
      ts: 1794411097552,
      id: 'toolu_91096a11e112457d83a5f3dec393ce30',
      name: 'mlx_web__web_search',
      title: 'mlx_web__web_search',
      status: 'completed',
      input:
        '{\n  "variant": "MCPTool",\n  "tool_name": "mlx_web__web_search",\n  "tool_input": {\n    "query": "Rust 1.99.0 stable release"\n  }\n}',
      endTs: 1794411097553,
      output:
        'WEB_SEARCH provider=exa results=2\n[1] FCHK-RESULT-ONE: Rust 1.99 released\nhttps://blog.rust-lang.org/2026/11/01/Rust-1.99.0.html\n2026-11-01\nThe Rust team released version 1.99.0 on 1 November 2026.\n\n[2] FCHK-RESULT-TWO: Rust releases\nhttps://releases.rs/\nRelease list.',
    },
    {
      kind: 'assistant',
      ts: 1794411098229,
      text: '\n\nThe newest stable version of Rust is **Rust 1.99.0**, released on November 1, 2026.',
      endTs: 1794411098536,
    },
    { kind: 'turn-end', ts: 1794411098621, stopReason: 'end_turn', elapsedMs: 4859 },
  ],
  turns: [
    metric({
      promptId: '758ed2d2-e860-4d77-b4ca-5a5f5f0a957d',
      ts: 1794411098621,
      inputTokens: 7215,
      outputTokens: 315,
      cachedTokens: 6892,
      elapsedMs: 4859,
      ttftMs: 301.34441699999996,
      modelCalls: 4,
      decodeTps: 87.24682200652491,
      model: 'qwen3.5-4b-nvfp4-mtp',
    }),
  ],
  turn: null,
  lastEnd: null,
};

/** The user gives an address; the model reads it. */
export const FETCH_USER_URL: ThreadSample = done([
  user('Summarise https://docs.example.org/guide please'),
  directFetch(
    'https://docs.example.org/guide',
    formatPageText('firecrawl', {
      url: 'https://docs.example.org/guide',
      title: 'The guide',
      cut: true,
      text: 'Step one.',
    }),
  ),
  answer('The [guide](https://docs.example.org/guide) has one step.'),
  end(),
]);

/** Web errors: a missing key (offers settings), a failed call with no readable text, and a refused input. */
export const WEB_ERRORS: ThreadSample = done([
  user('Search for the weather'),
  directSearch('weather', formatErrorText('not-configured', 'none'), 'completed', 'e1'),
  directSearch('weather today', 'provider exploded', 'failed', 'e2'),
  answer('I could not search the web.'),
  end(),
]);

/** One sample per stored end and per live outcome of the last turn. */
export function outcomeSample(stopReason: string, error?: string): ThreadSample {
  return done([user('Write a poem'), answer('Roses are red'), end(stopReason, T0 + 1_500, error)]);
}

export function liveOutcomeSample(
  outcome: NonNullable<ConversationState['lastEnd']>['outcome'],
  mode: ChatMode = 'chat',
): ThreadSample {
  return {
    ...done([user('Write a poem'), answer('Roses are red'), end('end_turn')]),
    mode,
    lastEnd: outcome === undefined ? {} : { outcome },
  };
}

/** The agent stopped before the turn ended: no turn-end entry. */
export const UNFINISHED: ThreadSample = { ...done([user('Write a poem'), answer('Roses')]), turns: [] };

/** Rows that are not web calls, and a kind this build does not know. */
export const OTHER_ROWS: ThreadSample = {
  mode: 'work',
  entries: [
    user('Fix the build'),
    { kind: 'model', ts: T0 + 10, modelId: 'qwen3.5-9b' },
    {
      kind: 'plan',
      ts: T0 + 20,
      entries: [
        { content: 'Read the error', status: 'completed', priority: 'high' },
        { content: 'Patch the file', status: 'in_progress', priority: 'high' },
      ],
    },
    {
      kind: 'tool',
      ts: T0 + 30,
      id: 'g1',
      name: 'read_file',
      title: 'src/main.rs',
      status: 'completed',
      input: '{"path":"src/main.rs"}',
      output: 'fn main() {}',
      endTs: T0 + 40,
    },
    {
      kind: 'subagent',
      ts: T0 + 50,
      childId: 'child-1',
      type: 'explore',
      description: 'Look for callers',
      status: 'completed',
      durationMs: 3_000,
    },
    {
      kind: 'wake',
      ts: T0 + 60,
      cause: 'subagent',
      subagent: { childId: 'child-1', description: 'Look for callers' },
      text: 'done',
    },
    { kind: 'hologram', ts: T0 + 70 } as unknown as TranscriptEntry,
    answer('Fixed.'),
    end(),
  ],
  turns: [metric()],
  turn: null,
  lastEnd: null,
};

/** A running turn: thinking done, the answer streaming. */
export function streamingSample(text: string): ThreadSample {
  return {
    mode: 'chat',
    entries: [user('Explain borrowing'), thinking('Plan the answer.'), answer(text)],
    turns: [],
    turn: { turnId: 't1', phase: 'running', startedAt: T0, wait: null },
    lastEnd: null,
  };
}

/** `n` finished turns of short questions and answers. */
export function longThread(n: number): ThreadSample {
  const entries: TranscriptEntry[] = [];
  const turns: TurnMetric[] = [];
  for (let i = 0; i < n; i += 1) {
    entries.push(
      user(`Question ${i}`, T0 + i * 10),
      answer(`Answer ${i}`, T0 + i * 10 + 5),
      end('end_turn', T0 + i * 10 + 9),
    );
    turns.push(metric({ promptId: `p${i}` }));
  }
  return { mode: 'chat', entries, turns, turn: null, lastEnd: null };
}

/** Every sample, for a gallery or a smoke render. */
export const THREAD_SAMPLES: Readonly<Record<string, ThreadSample>> = {
  'cited-answer': CITED_ANSWER,
  'forged-citation': FORGED_CITATION,
  'recorded-use-tool': RECORDED_USE_TOOL,
  'recorded-direct': RECORDED_DIRECT,
  'fetch-user-url': FETCH_USER_URL,
  'web-errors': WEB_ERRORS,
  cancelled: outcomeSample('cancelled'),
  'length-limit': outcomeSample('max_tokens'),
  'ended-error': outcomeSample('refusal', 'model call failed: 500'),
  'tool-refused': liveOutcomeSample('tool-refused'),
  'address-refused': liveOutcomeSample('address-refused'),
  'model-error': liveOutcomeSample('model-error'),
  'agent-crashed': liveOutcomeSample('agent-crashed'),
  'output-limit': liveOutcomeSample('output-limit'),
  unfinished: UNFINISHED,
  'other-rows': OTHER_ROWS,
  streaming: streamingSample('Borrowing lets **you** use a value'),
};
