import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  AcpClient,
  AcpClosedError,
  AcpRpcError,
  AcpTimeoutError,
  MAX_GARBAGE_LINES,
  type AcpClientHandlers,
} from '../src/chat/acp/client.js';
import {
  readInitialize,
  readPermissionRequest,
  readSessionInfo,
  readSessionReply,
  readSessionUpdate,
} from '../src/chat/acp/messages.js';

function peer(handlers: AcpClientHandlers = {}): { client: AcpClient; sent: Array<Record<string, unknown>> } {
  const sent: Array<Record<string, unknown>> = [];
  const client = new AcpClient({ write: (frame) => sent.push(JSON.parse(frame) as Record<string, unknown>) }, handlers);
  return { client, sent };
}

afterEach(() => {
  vi.useRealTimers();
});

describe('AcpClient', () => {
  it('numbers its requests from 1 and resolves the matching reply', async () => {
    const { client, sent } = peer();
    const a = client.request('initialize', { x: 1 }, null);
    const b = client.request('session/new', {}, null);
    expect(sent.map((m) => m.id)).toEqual([1, 2]);
    client.receive(JSON.stringify({ jsonrpc: '2.0', id: 2, result: { sessionId: 's' } }));
    client.receive(JSON.stringify({ jsonrpc: '2.0', id: 1, result: { ok: true } }));
    await expect(a).resolves.toEqual({ ok: true });
    await expect(b).resolves.toEqual({ sessionId: 's' });
  });

  it('drops a reply with an unknown id', async () => {
    const { client } = peer();
    const p = client.request('initialize', {}, null);
    client.receive(JSON.stringify({ jsonrpc: '2.0', id: 99, result: {} }));
    client.receive(JSON.stringify({ jsonrpc: '2.0', id: 1, result: 'one' }));
    await expect(p).resolves.toBe('one');
  });

  it('keeps only the code and error_kind of an error reply, never its text', async () => {
    const { client } = peer();
    const p = client.request('session/prompt', {}, null);
    client.receive(
      JSON.stringify({
        jsonrpc: '2.0',
        id: 1,
        error: {
          code: -32603,
          message: 'secret marker',
          data: { message: 'secret marker', error_kind: 'max_tokens_truncation' },
        },
      }),
    );
    const error = await p.catch((e: unknown) => e);
    expect(error).toBeInstanceOf(AcpRpcError);
    expect(error).toMatchObject({ code: -32603, errorKind: 'max_tokens_truncation' });
    expect(String((error as Error).message)).not.toContain('secret');
    expect(JSON.stringify(error)).not.toContain('secret');
  });

  it('tells agent requests (method and id, own id space from 0) from notifications', async () => {
    const notes: string[] = [];
    const { client, sent } = peer({
      onNotification: (method) => notes.push(method),
      onRequest: async (method) =>
        method === 'session/request_permission' ? { result: { outcome: { outcome: 'cancelled' } } } : null,
    });
    client.receive(JSON.stringify({ jsonrpc: '2.0', method: 'session/update', params: {} }));
    client.receive(JSON.stringify({ jsonrpc: '2.0', id: 0, method: 'session/request_permission', params: {} }));
    client.receive(JSON.stringify({ jsonrpc: '2.0', id: 1, method: '_x.ai/exit_plan_mode', params: {} }));
    await new Promise((resolve) => setImmediate(resolve));
    expect(notes).toEqual(['session/update']);
    expect(sent).toEqual([
      { jsonrpc: '2.0', id: 0, result: { outcome: { outcome: 'cancelled' } } },
      { jsonrpc: '2.0', id: 1, error: { code: -32601, message: 'Method not found' } },
    ]);
  });

  it('answers every agent request -32601 when no handler is set', () => {
    const { client, sent } = peer();
    client.receive(JSON.stringify({ jsonrpc: '2.0', id: 5, method: 'fs/read_text_file', params: {} }));
    expect(sent).toEqual([{ jsonrpc: '2.0', id: 5, error: { code: -32601, message: 'Method not found' } }]);
  });

  it('skips blank lines and counts non-JSON lines; the 100th is a protocol error', () => {
    const reasons: string[] = [];
    const { client } = peer({ onProtocolError: (r) => reasons.push(r) });
    client.receive('');
    client.receive('   ');
    for (let i = 0; i < MAX_GARBAGE_LINES - 1; i++) client.receive('not json');
    client.receive('[1,2]');
    expect(client.garbageLines).toBe(MAX_GARBAGE_LINES);
    expect(reasons).toEqual(['garbage']);
  });

  it('99 non-JSON lines are not a protocol error', () => {
    const reasons: string[] = [];
    const { client } = peer({ onProtocolError: (r) => reasons.push(r) });
    for (let i = 0; i < MAX_GARBAGE_LINES - 1; i++) client.receive('nope');
    expect(reasons).toEqual([]);
  });

  it('a line whose handling throws is a protocol error, never a throw out of receive', () => {
    const errors: string[] = [];
    const client = new AcpClient(
      { write: () => {} },
      {
        onNotification: () => {
          throw new RangeError('Maximum call stack size exceeded');
        },
        onRequest: () => {
          throw new RangeError('Maximum call stack size exceeded');
        },
        onProtocolError: (reason) => errors.push(reason),
      },
    );
    expect(() => client.receive('{"jsonrpc":"2.0","method":"session/update","params":{}}')).not.toThrow();
    expect(() => client.receive('{"jsonrpc":"2.0","id":0,"method":"session/request_permission"}')).not.toThrow();
    expect(errors).toEqual(['unreadable']);
  });

  it('a request that times out rejects and reports a protocol error; the prompt has no deadline', async () => {
    vi.useFakeTimers();
    const reasons: string[] = [];
    const { client } = peer({ onProtocolError: (r) => reasons.push(r) });
    const info = client.request('_x.ai/session/info', {}, 5_000);
    const prompt = client.request('session/prompt', {}, null);
    const caught = info.catch((e: unknown) => e);
    vi.advanceTimersByTime(5_000);
    expect(await caught).toBeInstanceOf(AcpTimeoutError);
    expect(reasons).toEqual(['timeout']);
    vi.advanceTimersByTime(3_600_000);
    client.receive(JSON.stringify({ jsonrpc: '2.0', id: 2, result: { stopReason: 'end_turn' } }));
    await expect(prompt).resolves.toEqual({ stopReason: 'end_turn' });
  });

  it('close rejects every pending request and refuses new ones', async () => {
    const { client } = peer();
    const p = client.request('session/prompt', {}, null);
    client.close();
    await expect(p).rejects.toBeInstanceOf(AcpClosedError);
    await expect(client.request('initialize', {}, 10)).rejects.toBeInstanceOf(AcpClosedError);
  });
});

describe('message readers', () => {
  it('read the recorded initialize reply', () => {
    const facts = readInitialize({
      protocolVersion: 1,
      agentCapabilities: { loadSession: true, promptCapabilities: { image: false, embeddedContext: true } },
      _meta: {
        'x.ai/pluginDirs': true,
        agentVersion: '1.0.45',
        modelState: {
          currentModelId: 'q',
          availableModels: [
            { modelId: 'q', name: 'q', _meta: { totalContextTokens: 262144, agentType: 'grok-build-plan' } },
          ],
        },
      },
    });
    expect(facts).toEqual({
      protocolVersion: 1,
      loadSession: true,
      embeddedContext: true,
      pluginDirs: true,
      agentVersion: '1.0.45',
      models: [{ modelId: 'q', contextTokens: 262144 }],
    });
    expect(readInitialize({ _meta: { modelState: { availableModels: [{ id: 'x' }] } } }).models).toBeNull();
    expect(readInitialize(null).models).toBeNull();
  });

  it('read the cwd of a new reply and of a load reply (which has only the session detail)', () => {
    expect(
      readSessionReply({ sessionId: 's', models: { currentModelId: 'm' }, _meta: { currentWorkingDirectory: '/w' } }),
    ).toEqual({
      sessionId: 's',
      currentModelId: 'm',
      cwd: '/w',
    });
    expect(
      readSessionReply({
        models: { currentModelId: 'm' },
        _meta: { sessionId: 's', 'x.ai/sessionDetail': { cwd: '/w2' } },
      }),
    ).toEqual({ sessionId: 's', currentModelId: 'm', cwd: '/w2' });
  });

  it('accept session info only nested once more (reply.result.result)', () => {
    expect(readSessionInfo({ agentName: 'mlx-chat' })).toBeNull();
    expect(readSessionInfo({ result: { agentName: 'mlx-chat', context: { toolDefinitionsCount: 2 } } })).toMatchObject({
      agentName: 'mlx-chat',
      toolDefinitionsCount: 2,
    });
  });

  it('refuse a permission request with a malformed option list', () => {
    expect(
      readPermissionRequest({ sessionId: 's', toolCall: { toolCallId: 't' }, options: [{ name: 'x' }] }),
    ).toBeNull();
    expect(readPermissionRequest({ sessionId: 's', toolCall: { toolCallId: 't' }, options: [] })).toMatchObject({
      options: [],
    });
  });

  it('read updates from the three methods and nothing else', () => {
    const params = { sessionId: 's', update: { sessionUpdate: 'x' } };
    expect(readSessionUpdate('session/update', params)?.xai).toBe(false);
    expect(readSessionUpdate('_x.ai/session_notification', params)?.xai).toBe(true);
    expect(readSessionUpdate('_x.ai/session/update', params)?.xai).toBe(true);
    expect(readSessionUpdate('_x.ai/mcp/server_status', params)).toBeNull();
    expect(readSessionUpdate('session/update', { update: { sessionUpdate: 'x' } })).toBeNull();
  });
});
