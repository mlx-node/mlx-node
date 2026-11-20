/**
 * An inference engine on 127.0.0.1 that answers the agent's model calls from a script, in the shapes the server
 * package writes: `GET /v1/models` (the host's `listModels` entry), the select and unload calls, and one model route
 * streamed as Anthropic SSE (`message_start`, progress `ping`s, content blocks, `message_delta`, `message_stop`).
 * The first model call names the route; a model call on any other path is answered 404 and kept in `strays`.
 * Nothing loads a model.
 */

import { randomUUID } from 'node:crypto';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';

import {
  DEFAULT_ENGINE_SCRIPT,
  SELECT_PATH,
  UNLOAD_PATH,
  type EngineRequest,
  type EngineScript,
  type FakeEngine,
} from './fake-engine.js';

type Json = Record<string, unknown>;

export interface ModelRequest {
  /** 0-based, in arrival order over every model call. */
  index: number;
  path: string;
  headers: Record<string, string>;
  body: Json;
  /** Every text of `system` joined by newlines. */
  system: string;
  /** The `name` of each offered tool. */
  tools: string[];
  messages: Json[];
  /** Milliseconds from the engine start to the request's end. */
  at: number;
  /** Milliseconds from the engine start to the end of the answer; null while it streams. */
  doneAt: number | null;
  /** The client closed the stream before the answer ended. */
  closedEarly: boolean;
}

export type ReplyBlock =
  | { type: 'text'; text: string }
  | { type: 'thinking'; thinking: string }
  | { type: 'tool_use'; name: string; input: Json };

export interface ModelReply {
  blocks: ReplyBlock[];
  /** Default `tool_use` when a block is a tool call, else `end_turn`. */
  stopReason?: string;
  /** Default {@link DEFAULT_INPUT_TOKENS}. */
  inputTokens?: number;
  /** Default {@link DEFAULT_OUTPUT_TOKENS}. */
  outputTokens?: number;
  /** After `message_start`, progress pings every 250 ms for this long, with `prefill_done` going up, then the blocks. */
  holdMs?: number;
}

export type ModelScript = (request: ModelRequest) => ModelReply;

export interface ScriptedModel {
  id: string;
  contextWindow: number;
}

export interface ScriptedEngine extends FakeEngine {
  models: ScriptedModel[];
  modelRequests: ModelRequest[];
  /** The path of the first model call; null before it. */
  modelPath(): string | null;
  /** Model calls on another path than the first one's. */
  strays: EngineRequest[];
}

export const PING_EVERY_MS = 250;
export const DEFAULT_INPUT_TOKENS = 100;
export const DEFAULT_OUTPUT_TOKENS = 10;

function headersOf(message: IncomingMessage): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [name, value] of Object.entries(message.headers)) {
    if (typeof value === 'string') out[name] = value;
    else if (Array.isArray(value)) out[name] = value.join(', ');
  }
  return out;
}

function isRecord(value: unknown): value is Json {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function systemText(body: Json): string {
  const system = body.system;
  if (typeof system === 'string') return system;
  if (!Array.isArray(system)) return '';
  return system
    .filter(isRecord)
    .map((b) => (typeof b.text === 'string' ? b.text : ''))
    .join('\n');
}

export function toolNames(body: Json): string[] {
  return Array.isArray(body.tools)
    ? body.tools.filter(isRecord).map((t) => (typeof t.name === 'string' ? t.name : ''))
    : [];
}

/** The blocks of a message's `content` (a string is one text block). */
export function blocksOf(message: Json | undefined): Json[] {
  if (message === undefined) return [];
  const content = message.content;
  if (typeof content === 'string') return [{ type: 'text', text: content }];
  return Array.isArray(content) ? content.filter(isRecord) : [];
}

/** All text of a message: its text blocks and the text inside its tool results. */
export function messageText(message: Json | undefined): string {
  const parts: string[] = [];
  for (const block of blocksOf(message)) {
    if (typeof block.text === 'string') parts.push(block.text);
    if (block.type === 'tool_result') {
      const inner = block.content;
      if (typeof inner === 'string') parts.push(inner);
      else if (Array.isArray(inner)) {
        for (const b of inner) if (isRecord(b) && typeof b.text === 'string') parts.push(b.text);
      }
    }
  }
  return parts.join('\n');
}

export function lastMessage(request: ModelRequest): Json | undefined {
  return request.messages.at(-1);
}

export function hasToolResult(message: Json | undefined): boolean {
  return blocksOf(message).some((b) => b.type === 'tool_result');
}

function sse(res: ServerResponse, event: string, data: Json): void {
  res.write(`event: ${event}\ndata: ${JSON.stringify({ type: event, ...data })}\n\n`);
}

async function stream(
  res: ServerResponse,
  request: ModelRequest,
  reply: ModelReply,
  live: () => boolean,
  onDone: () => void,
): Promise<void> {
  const model = typeof request.body.model === 'string' ? request.body.model : '';
  const inputTokens = reply.inputTokens ?? DEFAULT_INPUT_TOKENS;
  const outputTokens = reply.outputTokens ?? DEFAULT_OUTPUT_TOKENS;
  res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
  sse(res, 'message_start', {
    message: {
      id: `msg_${randomUUID()}`,
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: inputTokens, output_tokens: 0 },
    },
  });
  const holdMs = reply.holdMs ?? 0;
  const started = Date.now();
  let done = 0;
  while (Date.now() - started < holdMs) {
    await new Promise((resolve) => setTimeout(resolve, PING_EVERY_MS));
    if (!live()) return;
    done = Math.min(inputTokens, done + Math.max(1, Math.floor(inputTokens / 16)));
    sse(res, 'ping', {
      mlx_progress: {
        phase: 'prefill',
        elapsed_ms: Date.now() - started,
        prefill_total: inputTokens,
        prefill_done: done,
      },
    });
  }
  if (!live()) return;
  let index = 0;
  for (const block of reply.blocks) {
    if (block.type === 'text') {
      sse(res, 'content_block_start', { index, content_block: { type: 'text', text: '' } });
      sse(res, 'content_block_delta', { index, delta: { type: 'text_delta', text: block.text } });
    } else if (block.type === 'thinking') {
      sse(res, 'content_block_start', { index, content_block: { type: 'thinking', thinking: '', signature: '' } });
      sse(res, 'content_block_delta', { index, delta: { type: 'thinking_delta', thinking: block.thinking } });
    } else {
      sse(res, 'content_block_start', {
        index,
        content_block: { type: 'tool_use', id: `toolu_${randomUUID()}`, name: block.name, input: {} },
      });
      sse(res, 'content_block_delta', {
        index,
        delta: { type: 'input_json_delta', partial_json: JSON.stringify(block.input) },
      });
    }
    sse(res, 'content_block_stop', { index });
    index++;
  }
  const stopReason = reply.stopReason ?? (reply.blocks.some((b) => b.type === 'tool_use') ? 'tool_use' : 'end_turn');
  sse(res, 'message_delta', {
    delta: { stop_reason: stopReason, stop_sequence: null },
    usage: { output_tokens: outputTokens, input_tokens: inputTokens },
  });
  sse(res, 'message_stop', {});
  onDone();
  res.end();
}

export async function startScriptedEngine(models: ScriptedModel[], reply: ModelScript): Promise<ScriptedEngine> {
  const requests: EngineRequest[] = [];
  const modelRequests: ModelRequest[] = [];
  const strays: EngineRequest[] = [];
  let modelPath: string | null = null;
  let json: EngineScript = DEFAULT_ENGINE_SCRIPT;
  const t0 = Date.now();
  const server: Server = createServer((message, response) => {
    const chunks: Buffer[] = [];
    message.on('data', (chunk: Buffer) => chunks.push(chunk));
    message.on('end', () => {
      const url = new URL(message.url ?? '/', 'http://127.0.0.1');
      const request: EngineRequest = {
        method: message.method ?? 'GET',
        path: url.pathname,
        headers: headersOf(message),
        body: Buffer.concat(chunks).toString('utf8'),
        closedEarly: false,
        answered: false,
      };
      requests.push(request);
      response.on('close', () => {
        if (!request.answered) request.closedEarly = true;
      });
      const send = (status: number, body: unknown): void => {
        request.answered = true;
        response.writeHead(status, { 'Content-Type': 'application/json' });
        response.end(JSON.stringify(body));
      };
      if (request.method === 'GET' && request.path === '/v1/models') {
        const data = models.map((m) => ({
          id: m.id,
          object: 'model',
          created: Math.floor(t0 / 1000),
          owned_by: 'mlx-node',
          context_window: m.contextWindow,
          max_model_len: m.contextWindow,
          context_length: m.contextWindow,
          api_backend: 'messages',
          model_family: 'qwen3_5',
          supports_images: false,
        }));
        send(200, {
          object: 'list',
          data,
          models: data.map((m) => ({ name: m.id, description: 'Local MLX model', release_date: '' })),
        });
        return;
      }
      if (request.method === 'POST' && (request.path === SELECT_PATH || request.path === UNLOAD_PATH)) {
        const a = json(request);
        send(a.status ?? 200, a.json ?? {});
        return;
      }
      let body: unknown = null;
      try {
        body = JSON.parse(request.body);
      } catch {
        body = null;
      }
      if (request.method !== 'POST' || !isRecord(body) || !Array.isArray(body.messages)) {
        send(404, { error: 'not found' });
        return;
      }
      modelPath ??= request.path;
      if (request.path !== modelPath) {
        strays.push(request);
        send(404, { error: 'not found' });
        return;
      }
      const model: ModelRequest = {
        index: modelRequests.length,
        path: request.path,
        headers: request.headers,
        body,
        system: systemText(body),
        tools: toolNames(body),
        messages: body.messages.filter(isRecord),
        at: Date.now() - t0,
        doneAt: null,
        closedEarly: false,
      };
      modelRequests.push(model);
      response.on('close', () => {
        if (model.doneAt === null) model.closedEarly = true;
      });
      const answer = reply(model);
      request.answered = true;
      void stream(
        response,
        model,
        answer,
        () => !model.closedEarly,
        () => {
          model.doneAt = Date.now() - t0;
        },
      );
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}`,
    models,
    requests,
    modelRequests,
    strays,
    modelPath: () => modelPath,
    selects: () => requests.filter((r) => r.path === SELECT_PATH),
    script: (next) => {
      json = next;
    },
    close: () =>
      new Promise<void>((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}
