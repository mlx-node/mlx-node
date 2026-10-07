import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';

import type { Model } from '@earendil-works/pi-ai';
import { normalizeContext } from '@earendil-works/pi-ai';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { desktopStreamFactory } from '../src/provider/desktop-client.js';
import type { StreamSimpleHost } from '../src/provider/stream-adapter.js';

const servers: Server[] = [];
afterEach(async () => {
  await Promise.all(
    servers.splice(0).map(
      (s) =>
        new Promise<void>((resolve) => {
          s.closeAllConnections();
          s.close(() => resolve());
        }),
    ),
  );
});
const model: Model<'mlx'> = {
  id: 'local',
  name: 'local',
  provider: 'mlx',
  api: 'mlx',
  baseUrl: 'mlx://local',
  reasoning: true,
  input: ['text', 'image'],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 8192,
  maxTokens: 512,
};
function host() {
  return {
    modelInfo: () => ({ name: 'local', path: '/models/local', modelType: 'qwen3' }),
    runWithResident: vi.fn(() => {
      throw new Error('must not load weights');
    }),
    markResidentDirty: vi.fn(),
    consumeResidentDirty: vi.fn(),
    invalidateResident: vi.fn(),
  } satisfies StreamSimpleHost;
}
async function fixture(status = 200, ids = ['local']) {
  const requests: Record<string, any>[] = [];
  const server = createServer(async (req, res) => {
    if (req.headers.authorization !== 'Bearer secret' && req.headers['x-api-key'] !== 'secret') {
      res.writeHead(401).end();
      return;
    }
    if (req.url === '/health') {
      res.end(JSON.stringify({ pid: process.pid, models: { resident: ['local'] } }));
      return;
    }
    if (req.url === '/v1/models') {
      res.end(JSON.stringify({ data: ids.map((id) => ({ id })) }));
      return;
    }
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    requests.push(JSON.parse(Buffer.concat(chunks).toString()));
    if (status !== 200 && status !== 0) {
      res.writeHead(status).end(JSON.stringify({ error: { type: 'api_error', message: 'engine failed' } }));
      return;
    }
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    if (status === 0) {
      res.flushHeaders();
      return;
    }
    const events = [
      {
        type: 'message_start',
        message: {
          id: 'msg_1',
          type: 'message',
          role: 'assistant',
          model: 'local',
          content: [],
          usage: { input_tokens: 10, output_tokens: 0 },
        },
      },
      { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Hello' } },
      { type: 'content_block_stop', index: 0 },
      {
        type: 'content_block_start',
        index: 1,
        content_block: { type: 'tool_use', id: 'toolu_1', name: 'read', input: {} },
      },
      {
        type: 'content_block_delta',
        index: 1,
        delta: { type: 'input_json_delta', partial_json: '{"path":"README.md"}' },
      },
      { type: 'content_block_stop', index: 1 },
      { type: 'message_delta', delta: { stop_reason: 'tool_use' }, usage: { output_tokens: 7 } },
      { type: 'message_stop' },
    ];
    for (const event of events) res.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);
    res.end();
  });
  servers.push(server);
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
  const endpoint = {
    version: 1 as const,
    models: [{ name: 'local', path: '/models/local' }],
    pid: process.pid,
    url: `http://127.0.0.1:${(server.address() as AddressInfo).port}`,
    token: 'secret',
  };
  return { endpoint, requests };
}

describe('desktop inference transport', () => {
  it('streams text/tools/usage through the running engine without loading a local model', async () => {
    const { endpoint, requests } = await fixture();
    const local = host();
    const stream = desktopStreamFactory(async () => endpoint)(
      local,
      undefined,
      () => 'root',
      undefined,
      undefined,
      undefined,
      () => 0,
    );
    const result = await stream(
      model,
      normalizeContext({ messages: [{ role: 'user', content: 'hello', timestamp: 0 }] }),
      { reasoning: 'low', sessionId: 'child' },
    ).result();
    expect(result.api).toBe('mlx');
    expect(result.stopReason).toBe('toolUse');
    expect(result.content).toEqual([
      { type: 'text', text: 'Hello' },
      { type: 'toolCall', id: 'toolu_1', name: 'read', arguments: { path: 'README.md' } },
    ]);
    expect(result.usage.input).toBe(10);
    expect(result.usage.output).toBe(7);
    expect(requests[0].extra_body).toEqual({ reasoning_effort: 'low', thinking_budget: 0 });
    expect(requests[0].max_tokens).toBe(512);
    expect(requests[0].cache_salt).toContain(':child');
    expect(local.runWithResident).not.toHaveBeenCalled();
  });
  it('keeps tool-result and image follow-ups on the same engine', async () => {
    const { endpoint, requests } = await fixture();
    const stream = desktopStreamFactory(async () => endpoint)(host());
    const first = await stream(
      model,
      normalizeContext({ messages: [{ role: 'user', content: 'read', timestamp: 0 }] }),
    ).result();
    await stream(model, normalizeContext({
      messages: [
        { role: 'user', content: 'read', timestamp: 0 },
        first,
        {
          role: 'toolResult',
          toolCallId: 'toolu_1',
          toolName: 'read',
          content: [{ type: 'text', text: 'file contents' }],
          isError: false,
          timestamp: 1,
        },
        { role: 'user', content: [{ type: 'image', mimeType: 'image/png', data: 'aGVsbG8=' }], timestamp: 2 },
      ],
    })).result();
    expect(JSON.stringify(requests[1].messages)).toContain('tool_result');
    expect(JSON.stringify(requests[1].messages)).toContain('file contents');
    expect(JSON.stringify(requests[1].messages)).toContain('image/png');
    expect(requests[1].cache_salt).toBe(requests[0].cache_salt);
  });
  it('aborts an active HTTP stream without starting local inference', async () => {
    const { endpoint, requests } = await fixture(0);
    const local = host();
    const abort = new AbortController();
    const result = desktopStreamFactory(async () => endpoint)(local)(
      model,
      normalizeContext({ messages: [] }),
      { signal: abort.signal },
    ).result();
    await vi.waitFor(() => expect(requests).toHaveLength(1));
    abort.abort();
    expect((await result).stopReason).toBe('aborted');
    expect(local.runWithResident).not.toHaveBeenCalled();
  });
  it('rejects identically named checkpoints in a different models directory', async () => {
    const { endpoint, requests } = await fixture();
    const local = host();
    const result = await desktopStreamFactory(async () => ({
      ...endpoint,
      models: [{ name: 'local', path: '/other/local' }],
    }))(local)(model, normalizeContext({
      messages: [],
    })).result();
    expect(result.errorMessage).toContain('different model directory');
    expect(requests).toHaveLength(0);
    expect(local.runWithResident).not.toHaveBeenCalled();
  });
  it.each([429, 500])('never retries or loads locally after HTTP %s', async (status) => {
    const { endpoint, requests } = await fixture(status);
    const local = host();
    const result = await desktopStreamFactory(async () => endpoint)(local)(
      model,
      normalizeContext({ messages: [] }),
    ).result();
    expect(result.stopReason).toBe('error');
    expect(requests).toHaveLength(1);
    expect(local.runWithResident).not.toHaveBeenCalled();
  });
  it('rejects a model absent from the app instead of letting the server alias it', async () => {
    const { endpoint, requests } = await fixture(200, ['different']);
    const local = host();
    const result = await desktopStreamFactory(async () => endpoint)(local)(
      model,
      normalizeContext({ messages: [] }),
    ).result();
    expect(result.errorMessage).toContain('not available');
    expect(requests).toHaveLength(0);
    expect(local.runWithResident).not.toHaveBeenCalled();
  });
  it('handles app exit and pre-aborted requests without loading locally', async () => {
    const local = host();
    const read = vi.fn(async () => undefined);
    const stream = desktopStreamFactory(read)(local);
    expect((await stream(model, normalizeContext({ messages: [] })).result()).errorMessage).toContain('stopped');
    const abort = new AbortController();
    abort.abort();
    expect(
      (await stream(model, normalizeContext({ messages: [] }), { signal: abort.signal }).result()).stopReason,
    ).toBe('aborted');
    expect(read).toHaveBeenCalledOnce();
    expect(local.runWithResident).not.toHaveBeenCalled();
  });
});
