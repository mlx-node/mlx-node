/**
 * Gated live smoke for the assembled provider seam: `makeMlxStreamSimple`
 * over a REAL `MlxModelHost` + real weights — one text turn plus one
 * tool round-trip (call → fabricated result → continuation).
 *
 * Availability convention (spike B): the smallest local qwen3.5
 * checkpoint, overridable via `MLX_AGENT_TEST_MODEL`. Skips cleanly when
 * no candidate exists. Turns run strictly sequentially on one shared
 * host — GPU work is never concurrent.
 */
import { existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { basename, join } from 'node:path';

import type {
  Api,
  AssistantMessage,
  AssistantMessageEvent,
  AssistantMessageEventStream,
  Context,
  Model,
  SimpleStreamOptions,
  Tool,
  ToolResultMessage,
  TSchema,
} from '@earendil-works/pi-ai';
import { detectModelType } from '@mlx-node/lm';
import { beforeAll, describe, expect, it } from 'vite-plus/test';

import { MlxModelHost } from '../src/provider/model-host.js';
import { makeMlxStreamSimple } from '../src/provider/stream-adapter.js';
import type { DiscoveredModelLike } from '../src/types.js';

const CANDIDATES = [
  process.env.MLX_AGENT_TEST_MODEL,
  join(homedir(), '.mlx-node', 'models', 'qwen3.5-0.8b-mlx-bf16'),
  join(homedir(), '.cache', 'models', 'qwen3.5-0.8b-mlx-bf16'),
].filter((p): p is string => typeof p === 'string' && p.length > 0);

const MODEL_PATH = CANDIDATES.find((p) => existsSync(join(p, 'config.json')));

const TURN_TIMEOUT = 240_000;
const OPTIONS: SimpleStreamOptions = { maxTokens: 128, temperature: 0 }; // no `reasoning` → reasoningEffort 'none'
const SYSTEM = 'You are a concise assistant. Answer in at most two sentences.';

async function collect(stream: AssistantMessageEventStream): Promise<AssistantMessageEvent[]> {
  const events: AssistantMessageEvent[] = [];
  for await (const event of stream) events.push(event);
  return events;
}

function finalMessage(events: AssistantMessageEvent[]): AssistantMessage {
  const last = events[events.length - 1]!;
  if (last.type === 'done') return last.message;
  if (last.type === 'error') return last.error;
  throw new Error(`stream did not terminate: last event ${last.type}`);
}

function visibleText(message: AssistantMessage): string {
  return message.content
    .filter((part): part is Extract<AssistantMessage['content'][number], { type: 'text' }> => part.type === 'text')
    .map((part) => part.text)
    .join('\n');
}

describe.skipIf(!MODEL_PATH)('mlx provider live smoke', () => {
  let streamSimple: ReturnType<typeof makeMlxStreamSimple>;
  let model: Model<Api>;

  beforeAll(async () => {
    const discovered: DiscoveredModelLike = {
      name: basename(MODEL_PATH!),
      path: MODEL_PATH!,
      modelType: await detectModelType(MODEL_PATH!),
    };
    const host = new MlxModelHost([discovered]);
    streamSimple = makeMlxStreamSimple(host);
    model = {
      id: discovered.name,
      name: discovered.name,
      api: 'mlx',
      provider: 'mlx',
      baseUrl: 'mlx://local',
      reasoning: true,
      input: ['text'],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 262144,
      maxTokens: 128,
    };
  });

  it(
    'streams a real text turn to a stop final',
    async () => {
      const context: Context = {
        systemPrompt: SYSTEM,
        messages: [
          {
            role: 'user',
            content: 'What is the capital of France? Answer in one short sentence.',
            timestamp: Date.now(),
          },
        ],
      };
      const events = await collect(streamSimple(model, context, OPTIONS));

      expect(events[0]!.type).toBe('start');
      expect(events[events.length - 1]!.type).toBe('done');
      expect(events.some((e) => e.type === 'text_delta')).toBe(true);

      const message = finalMessage(events);
      expect(message.stopReason).toBe('stop');
      expect(visibleText(message)).toMatch(/paris/i);
      expect(message.usage.output).toBeGreaterThan(0);
      expect(message.usage.totalTokens).toBeGreaterThan(0);
    },
    TURN_TIMEOUT,
  );

  it(
    'completes a tool round-trip: toolUse final, then a continuation quoting the fabricated result',
    async () => {
      const weatherTool: Tool = {
        name: 'get_weather',
        description: 'Get current weather for a city',
        parameters: {
          type: 'object',
          properties: { location: { type: 'string', description: 'City name' } },
          required: ['location'],
        } as unknown as TSchema,
      };
      const context: Context = {
        systemPrompt: SYSTEM,
        messages: [
          {
            role: 'user',
            content:
              'What is the current weather in Paris? You must call the get_weather tool — do not answer from memory.',
            timestamp: Date.now(),
          },
        ],
        tools: [weatherTool],
      };

      // Turn 1: the model must emit a tool call.
      const callEvents = await collect(streamSimple(model, context, OPTIONS));
      const callMessage = finalMessage(callEvents);
      expect(callMessage.stopReason).toBe('toolUse');
      expect(callEvents.some((e) => e.type === 'toolcall_end')).toBe(true);

      const toolCall = callMessage.content.find(
        (part): part is Extract<AssistantMessage['content'][number], { type: 'toolCall' }> => part.type === 'toolCall',
      );
      expect(toolCall).toBeDefined();
      expect(toolCall!.name).toBe('get_weather');
      expect(String(toolCall!.arguments.location ?? '')).toMatch(/paris/i);

      // Turn 2: replay with the fabricated tool result appended.
      const toolResult: ToolResultMessage = {
        role: 'toolResult',
        toolCallId: toolCall!.id,
        toolName: 'get_weather',
        content: [{ type: 'text', text: '{"location":"Paris","condition":"sunny","temp_c":22}' }],
        isError: false,
        timestamp: Date.now(),
      };
      const continueContext: Context = {
        ...context,
        messages: [...context.messages, callMessage, toolResult],
      };
      const continueEvents = await collect(streamSimple(model, continueContext, OPTIONS));
      const continueMessage = finalMessage(continueEvents);
      expect(continueMessage.stopReason).toBe('stop');
      expect(visibleText(continueMessage)).toMatch(/sunny|22/i);
      // Warm replay on the shared prefix: the second call must reuse KV.
      expect(continueMessage.usage.cacheRead).toBeGreaterThan(0);
    },
    TURN_TIMEOUT,
  );
});
