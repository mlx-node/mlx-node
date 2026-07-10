/**
 * Spike B: warm cold-replay loop on a real local model.
 *
 * De-risks the mlx side of the `mlx agent` provider bridge: on EVERY pi LLM
 * call the production bridge will do
 *
 *   resetPreservingNativeCacheForWarmReuse(session)   // JS-state-only wipe
 *   session.primeHistory(fullHistory)                 // externally-built history
 *   session.startFromHistoryStream(config, signal)    // cold replay, warm KV
 *
 * and this spike proves (with real numbers) that KV-prefix reuse
 * (`cachedTokens` on the terminal stream chunk) survives those replays across
 * growing multi-turn histories, multi-tool fan-out replay, tool-call emission
 * + continuation, aborts, real pi tool schemas, and thinking turns.
 *
 * NOT production code. Run from the repo root:
 *
 *   oxnode packages/agent/spikes/spike-b-warm-replay.ts
 *
 * Model: smallest real local qwen3.5 instruct checkpoint (see MODEL_PATH;
 * override with SPIKE_MODEL=/path/to/model). All scenarios run strictly
 * sequentially in one process — GPU work is never concurrent.
 */
import { join } from 'node:path';

import { Qwen3Tokenizer } from '@mlx-node/core';
import {
  type ChatConfig,
  type ChatMessage,
  type ChatSession,
  type ChatStreamFinal,
  createToolDefinition,
  loadSession,
  type SessionCapableModel,
  type ToolCallResult,
  type ToolDefinition,
} from '@mlx-node/lm';
import { createBashToolDefinition, createEditToolDefinition } from '@earendil-works/pi-coding-agent';

const MODEL_PATH = process.env.SPIKE_MODEL ?? '/Users/brooklyn/.mlx-node/models/qwen3.5-0.8b-mlx-bf16';

// Production-mirroring constants: pi re-renders the same system prompt on
// every call, so all scenarios share one system message (which also lets us
// observe cross-scenario prefix reuse — production will hit the same case).
const SYSTEM = 'You are a concise assistant. Answer in at most two sentences.';

const BASE_CONFIG: ChatConfig = {
  maxNewTokens: 128,
  temperature: 0,
  reasoningEffort: 'none',
};

// ---------------------------------------------------------------------------
// Inline copy of the server-private warm-reuse helper.
//
// `resetPreservingNativeCacheForWarmReuse` is NOT exported from
// `@mlx-node/server` (module `packages/server/src/chat-session-warm-reuse.ts`
// is deliberately kept off the export map), so per the task brief this is a
// byte-faithful inline copy of its runtime behavior. Field names MUST match
// the private fields of `packages/lm/src/chat-session.ts`.
//
// Observation for the production bridge: the server helper does NOT wipe
// `lastAudioKey` (only `lastImagesKey`); irrelevant for a text-only agent but
// worth knowing if audio ever enters the bridge.
// ---------------------------------------------------------------------------
interface ChatSessionWarmReuseInternals {
  inFlight: boolean;
  history: unknown[];
  lastImagesKey: string | null;
  turnCount: number;
  unresolvedOkToolCallCount: number | null;
}

async function resetPreservingNativeCacheForWarmReuse(session: ChatSession<SessionCapableModel>): Promise<void> {
  const internals = session as unknown as ChatSessionWarmReuseInternals;
  if (internals.inFlight) {
    throw new Error(
      'ChatSession: cannot resetPreservingNativeCacheForWarmReuse() while a send() is in flight; await the previous call first',
    );
  }
  internals.history = [];
  internals.lastImagesKey = null;
  internals.turnCount = 0;
  internals.unresolvedOkToolCallCount = null;
}

// ---------------------------------------------------------------------------
// Verdict bookkeeping
// ---------------------------------------------------------------------------
type Verdict = 'PROVEN' | 'FAILED' | 'SURPRISE';

interface ItemResult {
  item: string;
  verdict: Verdict;
  evidence: string[];
}

const results: ItemResult[] = [];

function record(item: string, verdict: Verdict, evidence: string[]): void {
  results.push({ item, verdict, evidence });
  console.log(`\n=== [${verdict}] ${item} ===`);
  for (const line of evidence) console.log(`  - ${line}`);
}

function finalStats(final: ChatStreamFinal | null): string {
  if (!final) return 'final=<none>';
  return (
    `prompt=${final.promptTokens} cached=${final.cachedTokens ?? '<undefined>'} ` +
    `gen=${final.numTokens} reasoning=${final.reasoningTokens} finish=${final.finishReason}`
  );
}

// ---------------------------------------------------------------------------
// Warm replay driver: EXACTLY the production per-call pattern under test.
// ---------------------------------------------------------------------------
interface ReplayOutcome {
  final: ChatStreamFinal | null;
  deltaCount: number;
  visible: string;
}

async function warmReplay(
  session: ChatSession<SessionCapableModel>,
  history: ChatMessage[],
  config: ChatConfig,
  opts: { abortAfterDeltas?: number } = {},
): Promise<ReplayOutcome> {
  await resetPreservingNativeCacheForWarmReuse(session);
  session.primeHistory(history);
  const controller = new AbortController();
  let final: ChatStreamFinal | null = null;
  let deltaCount = 0;
  let visible = '';
  for await (const event of session.startFromHistoryStream(config, controller.signal)) {
    if (event.done) {
      final = event;
    } else {
      deltaCount++;
      if (event.isReasoning !== true) visible += event.text;
      if (opts.abortAfterDeltas !== undefined && deltaCount >= opts.abortAfterDeltas) {
        controller.abort();
      }
    }
  }
  return { final, deltaCount, visible };
}

/** Mirror of ChatSession's toAssistantToolCalls for building replay histories. */
function assistantMessageFromFinal(final: ChatStreamFinal): ChatMessage {
  const calls = final.toolCalls
    .filter((tc: ToolCallResult) => tc.status === 'ok')
    .map((tc: ToolCallResult) => ({
      id: tc.id,
      name: tc.name,
      arguments: typeof tc.arguments === 'string' ? tc.arguments : JSON.stringify(tc.arguments),
    }));
  const msg: ChatMessage = { role: 'assistant', content: final.text };
  if (calls.length > 0) msg.toolCalls = calls;
  return msg;
}

const weatherTool = createToolDefinition(
  'get_weather',
  'Get current weather for a city',
  {
    location: { type: 'string', description: 'City name' },
  },
  ['location'],
);

// ---------------------------------------------------------------------------
// Scenarios (strictly sequential)
// ---------------------------------------------------------------------------
async function main(): Promise<void> {
  console.log(`Spike B model: ${MODEL_PATH}`);
  const t0 = Date.now();
  const session = await loadSession(MODEL_PATH);
  console.log(`loadSession: ${Date.now() - t0}ms`);

  // ---- Items 1 + 2: warm replay x3 with growing history + output sanity ----
  {
    const failures: string[] = [];
    const evidence: string[] = [];
    const finals: (ChatStreamFinal | null)[] = [];
    const texts: string[] = [];

    const h1: ChatMessage[] = [
      { role: 'system', content: SYSTEM },
      { role: 'user', content: 'What is the capital of France? Answer in one short sentence.' },
    ];
    const r1 = await warmReplay(session, h1, BASE_CONFIG);
    finals.push(r1.final);
    texts.push(r1.final?.text ?? '');
    evidence.push(`call 1 (H1=${h1.length} msgs): ${finalStats(r1.final)}`);
    evidence.push(`call 1 text: ${JSON.stringify(r1.final?.text ?? '')}`);

    const h2: ChatMessage[] = [
      ...h1,
      { role: 'assistant', content: r1.final?.text ?? '' },
      { role: 'user', content: 'And what is the capital of Japan?' },
    ];
    const r2 = await warmReplay(session, h2, BASE_CONFIG);
    finals.push(r2.final);
    texts.push(r2.final?.text ?? '');
    evidence.push(`call 2 (H2=${h2.length} msgs): ${finalStats(r2.final)}`);
    evidence.push(`call 2 text: ${JSON.stringify(r2.final?.text ?? '')}`);

    const h3: ChatMessage[] = [
      ...h2,
      { role: 'assistant', content: r2.final?.text ?? '' },
      { role: 'user', content: 'Which of those two capital cities has the larger population?' },
    ];
    const r3 = await warmReplay(session, h3, BASE_CONFIG);
    finals.push(r3.final);
    texts.push(r3.final?.text ?? '');
    evidence.push(`call 3 (H3=${h3.length} msgs): ${finalStats(r3.final)}`);
    evidence.push(`call 3 text: ${JSON.stringify(r3.final?.text ?? '')}`);

    const cached = finals.map((f) => f?.cachedTokens);
    const prompts = finals.map((f) => f?.promptTokens ?? -1);
    if (cached[0] !== 0) {
      evidence.push(`SURPRISE: call 1 on a fresh model expected cachedTokens=0, got ${cached[0]}`);
    }
    if (!(typeof cached[1] === 'number' && cached[1] > 0)) {
      failures.push(`call 2 cachedTokens must be > 0, got ${cached[1]}`);
    }
    if (!(typeof cached[2] === 'number' && cached[2] > 0)) {
      failures.push(`call 3 cachedTokens must be > 0, got ${cached[2]}`);
    }
    if (typeof cached[1] === 'number' && typeof cached[2] === 'number' && !(cached[2] > cached[1])) {
      failures.push(`prefix reuse must grow with history: call3 cached=${cached[2]} <= call2 cached=${cached[1]}`);
    }
    // The production-critical property: re-prefill work must NOT grow with
    // history depth. If most of the prompt is re-prefilled every call, the
    // bridge design loses its warmth.
    const reprefill2 = prompts[1] - (cached[1] ?? 0);
    const reprefill3 = prompts[2] - (cached[2] ?? 0);
    evidence.push(`re-prefilled suffix: call2=${reprefill2} tokens, call3=${reprefill3} tokens`);
    if (!(reprefill3 < prompts[2] / 2)) {
      failures.push(`call 3 re-prefilled ${reprefill3} of ${prompts[2]} prompt tokens (>= half; reuse not effective)`);
    }
    record(
      '1. Warm replay x3: cachedTokens > 0 on calls 2 and 3',
      failures.length === 0 ? 'PROVEN' : 'FAILED',
      [...evidence, ...failures],
    );

    const sane = texts.every((t) => t.trim().length > 0);
    record(
      '2. Output sanity: non-empty, coherent finals on each turn',
      sane ? 'PROVEN' : 'FAILED',
      sane
        ? [`all 3 finals non-empty; eyeball texts above (Paris / Tokyo expected)`]
        : [`empty final text in turns: ${texts.map((t, i) => (t.trim() ? '' : i + 1)).filter(Boolean).join(',')}`],
    );
  }

  // ---- Item 3: multi-tool fan-out replay ----
  {
    const failures: string[] = [];
    const evidence: string[] = [];

    // (a) Evidence first: the live delta path REJECTS a fan-out — capture the
    // exact gate texts the production bridge routes around (no GPU work; the
    // gates throw before any native dispatch).
    await resetPreservingNativeCacheForWarmReuse(session);
    const fanoutAssistant: ChatMessage = {
      role: 'assistant',
      content: '',
      toolCalls: [
        { id: 'call_1', name: 'get_weather', arguments: '{"location":"Paris"}' },
        { id: 'call_2', name: 'get_weather', arguments: '{"location":"London"}' },
      ],
    };
    const unresolved: ChatMessage[] = [
      { role: 'system', content: SYSTEM },
      { role: 'user', content: 'What is the weather in Paris and in London? Use get_weather for each city.' },
      fanoutAssistant,
    ];
    session.primeHistory(unresolved);
    evidence.push(`pendingUnresolvedToolCallCount after fan-out prime: ${session.pendingUnresolvedToolCallCount}`);
    if (session.pendingUnresolvedToolCallCount !== 2) {
      failures.push(`expected pendingUnresolvedToolCallCount === 2, got ${session.pendingUnresolvedToolCallCount}`);
    }
    try {
      await session.sendStream('hello').next();
      failures.push('sendStream on an unresolved fan-out did NOT throw (gate missing?)');
    } catch (err) {
      evidence.push(`sendStream gate: ${JSON.stringify((err as Error).message)}`);
    }
    try {
      await session.sendToolResultStream('call_1', '{"ok":true}').next();
      failures.push('sendToolResultStream on a 2-call fan-out did NOT throw (gate missing?)');
    } catch (err) {
      evidence.push(`sendToolResultStream gate: ${JSON.stringify((err as Error).message)}`);
    }

    // (b) The production pattern: replay the RESOLVED fan-out through
    // warm-reset -> primeHistory -> startFromHistoryStream. Must not hit any
    // gate and must produce a sensible continuation.
    const resolved: ChatMessage[] = [
      ...unresolved,
      { role: 'tool', content: '{"location":"Paris","condition":"sunny","temp_c":22}', toolCallId: 'call_1' },
      { role: 'tool', content: '{"location":"London","condition":"rainy","temp_c":14}', toolCallId: 'call_2' },
    ];
    try {
      await resetPreservingNativeCacheForWarmReuse(session);
      session.primeHistory(resolved);
      evidence.push(
        `pendingUnresolvedToolCallCount after resolved prime: ${session.pendingUnresolvedToolCallCount} (null = gate disarmed)`,
      );
      if (session.pendingUnresolvedToolCallCount !== null) {
        failures.push(`resolved fan-out must derive null unresolved count, got ${session.pendingUnresolvedToolCallCount}`);
      }
      let final: ChatStreamFinal | null = null;
      for await (const event of session.startFromHistoryStream({ ...BASE_CONFIG, tools: [weatherTool] })) {
        if (event.done) final = event;
      }
      evidence.push(`fan-out replay: ${finalStats(final)}`);
      evidence.push(`fan-out continuation text: ${JSON.stringify(final?.text ?? '')}`);
      if (!final || final.text.trim().length === 0) {
        failures.push('fan-out replay produced no/empty final');
      } else if (!/paris|london|sunny|rain|22|14/i.test(final.text)) {
        failures.push('continuation does not reference the fabricated weather results');
      }
    } catch (err) {
      failures.push(`fan-out replay threw: ${(err as Error).message}`);
    }
    record('3. Multi-tool fan-out replay streams past the gates', failures.length === 0 ? 'PROVEN' : 'FAILED', [
      ...evidence,
      ...failures,
    ]);
  }

  // ---- Item 4: tool-call emission + replay-continue with the result ----
  {
    const failures: string[] = [];
    const evidence: string[] = [];
    const toolConfig: ChatConfig = { ...BASE_CONFIG, tools: [weatherTool] };
    const h: ChatMessage[] = [
      { role: 'system', content: SYSTEM },
      {
        role: 'user',
        content:
          'What is the current weather in Paris? You must call the get_weather tool — do not answer from memory.',
      },
    ];
    const r = await warmReplay(session, h, toolConfig);
    evidence.push(`emission call: ${finalStats(r.final)}`);
    evidence.push(`emission text: ${JSON.stringify(r.final?.text ?? '')}`);
    const calls = r.final?.toolCalls ?? [];
    evidence.push(
      `toolCalls: ${JSON.stringify(calls.map((c) => ({ id: c.id, name: c.name, status: c.status, arguments: c.arguments })))}`,
    );
    if (r.final?.finishReason !== 'tool_calls') {
      failures.push(`expected finishReason === 'tool_calls', got ${JSON.stringify(r.final?.finishReason)}`);
    }
    const ok = calls.find((c) => c.status === 'ok');
    if (!ok) {
      failures.push('no ok-status tool call parsed');
    } else {
      if (ok.name !== 'get_weather') failures.push(`tool name: expected get_weather, got ${ok.name}`);
      const args = typeof ok.arguments === 'string' ? JSON.parse(ok.arguments) : ok.arguments;
      if (!/paris/i.test(String((args as Record<string, unknown>).location ?? ''))) {
        failures.push(`arguments.location does not mention Paris: ${JSON.stringify(args)}`);
      }
    }

    if (ok && r.final) {
      const h2: ChatMessage[] = [
        ...h,
        assistantMessageFromFinal(r.final),
        { role: 'tool', content: '{"location":"Paris","condition":"sunny","temp_c":22}', toolCallId: ok.id },
      ];
      const r2 = await warmReplay(session, h2, toolConfig);
      evidence.push(`replay-continue with tool result: ${finalStats(r2.final)}`);
      evidence.push(`final answer: ${JSON.stringify(r2.final?.text ?? '')}`);
      if (!r2.final || r2.final.text.trim().length === 0) {
        failures.push('replay-continue produced no/empty final answer');
      } else if (!/sunny|22/i.test(r2.final.text)) {
        failures.push('final answer does not mention the fabricated result (sunny / 22)');
      }
      if (!(typeof r2.final?.cachedTokens === 'number' && r2.final.cachedTokens > 0)) {
        failures.push(`replay-continue cachedTokens must be > 0, got ${r2.final?.cachedTokens}`);
      }
    }
    record(
      "4. Tool-call emission: finishReason 'tool_calls' + JSON args + replay-continue",
      failures.length === 0 ? 'PROVEN' : 'FAILED',
      [...evidence, ...failures],
    );
  }

  // ---- Item 5: abort mid-stream, then session survives ----
  {
    const failures: string[] = [];
    const evidence: string[] = [];
    const h: ChatMessage[] = [
      { role: 'system', content: SYSTEM },
      { role: 'user', content: 'Count from 1 to 200 separated by commas, no other text.' },
    ];
    const aborted = await warmReplay(session, h, BASE_CONFIG, { abortAfterDeltas: 5 });
    evidence.push(
      `aborted stream: deltas=${aborted.deltaCount} final=${aborted.final ? finalStats(aborted.final) : '<none (expected)>'}`,
    );
    evidence.push(`partial text before abort: ${JSON.stringify(aborted.visible)}`);
    evidence.push(`session.turns after abort: ${session.turns} (0 = rollback held, primed state intact)`);
    if (aborted.final !== null) failures.push('generator yielded a final event after abort (expected none)');
    if (aborted.deltaCount < 5) failures.push(`expected >= 5 deltas before abort, got ${aborted.deltaCount}`);
    if (session.turns !== 0) failures.push(`expected turnCount rollback to 0 after abort, got ${session.turns}`);

    // Session must survive: the next warm replay runs to completion.
    const recovery = await warmReplay(
      session,
      [
        { role: 'system', content: SYSTEM },
        { role: 'user', content: 'Say the single word: recovered' },
      ],
      BASE_CONFIG,
    );
    evidence.push(`post-abort warm replay: ${finalStats(recovery.final)}`);
    evidence.push(`post-abort text: ${JSON.stringify(recovery.final?.text ?? '')}`);
    if (!recovery.final || recovery.final.text.trim().length === 0) {
      failures.push('post-abort warm replay failed to produce a final');
    }
    record(
      '5. Abort: no final event, session survives, next warm replay works',
      failures.length === 0 ? 'PROVEN' : 'FAILED',
      [...evidence, ...failures],
    );
  }

  // ---- Item 6: pi's REAL bash + edit schemas render through the jinja path ----
  {
    const failures: string[] = [];
    const evidence: string[] = [];
    try {
      // Real pi built-in tool definitions (TypeBox Type.Object compiles to a
      // plain JSON schema object at runtime: { type, properties, required }).
      const piDefs = [createBashToolDefinition(process.cwd()), createEditToolDefinition(process.cwd())];
      const mlxTools: ToolDefinition[] = piDefs.map((def) => {
        const params = def.parameters as unknown as {
          type: string;
          properties: Record<string, unknown>;
          required?: string[];
        };
        if (params.type !== 'object') throw new Error(`pi tool ${def.name}: schema root is ${params.type}`);
        return {
          type: 'function',
          function: {
            name: def.name,
            description: def.description,
            parameters: {
              type: 'object',
              // The NAPI quirk under test: properties MUST cross as a JSON string.
              properties: JSON.stringify(params.properties),
              required: params.required ?? [],
            },
          },
        };
      });
      evidence.push(`converted pi tools: ${mlxTools.map((t) => t.function.name).join(', ')}`);

      // Render through the same tokenizer+jinja path applyChatTemplateFromModelPath
      // uses (that helper is module-private in packages/lm/src/stream.ts; it
      // builds exactly this: Qwen3Tokenizer.fromPretrained(<model>/tokenizer.json)).
      const tokenizer = await Qwen3Tokenizer.fromPretrained(join(MODEL_PATH, 'tokenizer.json'));
      const tokens = await tokenizer.applyChatTemplate(
        [
          { role: 'system', content: SYSTEM },
          { role: 'user', content: 'List the files in the current directory.' },
        ],
        true,
        mlxTools,
        false,
      );
      const rendered = await tokenizer.decode(tokens, false);
      evidence.push(`rendered prompt: ${tokens.length} tokens`);
      for (const needle of ['"bash"', '"edit"', 'Bash command to execute', 'oldText']) {
        if (!rendered.includes(needle)) failures.push(`rendered prompt missing ${JSON.stringify(needle)}`);
      }
      if (failures.length === 0) {
        evidence.push('rendered prompt contains "bash", "edit", and both schemas’ property descriptions');
      }
      const toolsBlock = rendered.split('\n').filter((l) => l.includes('"function"'));
      evidence.push(`tool declaration lines in render: ${toolsBlock.length}`);

      // Secondary live evidence: the same schemas through ChatConfig.tools on a
      // real warm replay (observation only; a 0.8B model may or may not call).
      const live = await warmReplay(
        session,
        [
          { role: 'system', content: SYSTEM },
          { role: 'user', content: 'List the files in the current directory. Use the bash tool.' },
        ],
        { ...BASE_CONFIG, tools: mlxTools },
      );
      evidence.push(`live call with pi schemas: ${finalStats(live.final)}`);
      evidence.push(
        `live toolCalls: ${JSON.stringify((live.final?.toolCalls ?? []).map((c) => ({ name: c.name, status: c.status, arguments: c.arguments })))}`,
      );
    } catch (err) {
      failures.push(`render path threw: ${(err as Error).message}`);
    }
    record(
      '6. pi bash/edit TypeBox schemas render via TypeBox -> NAPI -> jinja',
      failures.length === 0 ? 'PROVEN' : 'FAILED',
      [...evidence, ...failures],
    );
  }

  // ---- Item 7: thinking replay (qwen3.5 is thinking-capable) ----
  {
    const failures: string[] = [];
    const evidence: string[] = [];
    // Default reasoning (no reasoningEffort:'none') so the turn REALLY thinks,
    // plus an explicit thinkingTokenBudget: at temp=0 the 0.8b never emits
    // </think> on its own (512/512 ramble observed, finish=length, no visible
    // text), so the engine's budget-forced </think> is what guarantees the turn
    // both thinks (reasoningTokens > 0) AND emits a visible answer. The prompt
    // stays BARE: "think step by step" nudges make the ramble worse and leak
    // pseudo-thinking into the visible text after the forced close.
    const thinkConfig: ChatConfig = { maxNewTokens: 256, temperature: 0, thinkingTokenBudget: 64 };
    const h1: ChatMessage[] = [
      { role: 'system', content: SYSTEM },
      { role: 'user', content: 'What is 17 + 25?' },
    ];
    const r1 = await warmReplay(session, h1, thinkConfig);
    evidence.push(`thinking turn 1: ${finalStats(r1.final)}`);
    evidence.push(
      `thinking present: ${r1.final?.thinking != null && r1.final.thinking.length > 0} ` +
        `(${r1.final?.reasoningTokens ?? 0} reasoning tokens)`,
    );
    evidence.push(`turn 1 visible text: ${JSON.stringify(r1.final?.text ?? '')}`);
    if (!r1.final) failures.push('thinking turn 1 produced no final');
    // Turn 1 MUST actually think — the item is meaningless without a real <think> block in the cache.
    if (!((r1.final?.reasoningTokens ?? 0) > 0)) {
      failures.push(`thinking turn 1 must produce reasoning tokens, got ${r1.final?.reasoningTokens}`);
    }
    if ((r1.final?.text ?? '').trim().length === 0) {
      failures.push(`thinking turn 1 produced no visible text (finish=${r1.final?.finishReason}; budget too small?)`);
    }

    const h2: ChatMessage[] = [
      ...h1,
      { role: 'assistant', content: r1.final?.text ?? '' }, // visible text only — thinking is stripped on replay
      { role: 'user', content: 'Now subtract 4 from that.' },
    ];
    const r2 = await warmReplay(session, h2, thinkConfig);
    evidence.push(`thinking replay turn 2: ${finalStats(r2.final)}`);
    evidence.push(`turn 2 visible text: ${JSON.stringify(r2.final?.text ?? '')}`);
    if (!r2.final) failures.push('thinking replay turn 2 produced no final');
    if ((r2.final?.text ?? '').trim().length === 0) failures.push('thinking replay turn 2 produced no visible text');
    const cached2 = r2.final?.cachedTokens;
    const prompt2 = r2.final?.promptTokens ?? 0;
    const prompt1 = r1.final?.promptTokens ?? 0;
    evidence.push(
      `prefix reuse across the thinking turn: cached=${cached2} of prompt=${prompt2} ` +
        `(re-prefilled ${prompt2 - (cached2 ?? 0)}; turn-1 prompt boundary=${prompt1})`,
    );
    // Invariant: the replay render strips <think>, so the KV cache diverges no later
    // than assistant-1's generated tokens — 0 < cached2 < prompt2 AND cached2 <= turn 1's promptTokens.
    if (!(typeof cached2 === 'number' && cached2 > 0)) {
      failures.push(`turn 2 cachedTokens must be > 0, got ${cached2}`);
    } else {
      if (!(cached2 < prompt2)) {
        failures.push(`turn 2 must re-prefill a suffix: cached=${cached2} !< prompt=${prompt2}`);
      }
      if (!(cached2 <= prompt1)) {
        failures.push(
          `cache must invalidate at the thinking assistant turn: cached=${cached2} > turn-1 prompt boundary=${prompt1}`,
        );
      }
    }
    record(
      '7. Thinking replay: turn 1 thinks; turn 2 reuse stops at the thinking-turn boundary',
      failures.length === 0 ? 'PROVEN' : 'FAILED',
      [...evidence, ...failures],
    );
  }

  // ---- Summary ----
  console.log('\n================ SPIKE B SUMMARY ================');
  let failed = 0;
  for (const r of results) {
    console.log(`[${r.verdict}] ${r.item}`);
    if (r.verdict === 'FAILED') failed++;
  }
  console.log(`Total: ${results.length} items, ${failed} FAILED, wall ${(Date.now() - t0) / 1000}s`);
  if (failed > 0) process.exitCode = 1;
}

main().catch((err) => {
  console.error('SPIKE_FATAL', err);
  process.exitCode = 2;
});
