import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  AcpClient,
  AcpClosedError,
  AcpRpcError,
  AcpSoftTimeoutError,
  AcpTimeoutError,
  MAX_GARBAGE_LINES,
  type AcpClientHandlers,
} from '../src/chat/acp/client.js';
import {
  readCompaction,
  readEffortConfig,
  readInitialize,
  readMonitorEvent,
  readPermissionRequest,
  readSessionInfo,
  readSessionReply,
  readSessionUpdate,
  readSpawnedChild,
  readSubagentCancel,
  readSubagentMessage,
  SUBAGENT_MESSAGE_KINDS,
  readTaskKill,
  readWorkflowStop,
  readWorkflowUpdate,
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

  it('keeps data.kind only for the listed compaction values, never the message', async () => {
    const { client } = peer();
    const kinds: unknown[] = ['compact_cancelled', 'compact_failed', 'compact_failed_secret', 'secret marker', 7, null];
    const replies = kinds.map(() => client.request('_x.ai/compact_conversation', { sessionId: 's' }, null));
    kinds.forEach((kind, n) => {
      client.receive(
        JSON.stringify({
          jsonrpc: '2.0',
          id: n + 1,
          error: { code: -32603, message: 'Internal error', data: { kind, message: 'compact failed: secret marker' } },
        }),
      );
    });
    const errors = await Promise.all(replies.map((p) => p.catch((e: unknown) => e)));
    expect(errors.map((e) => (e as AcpRpcError).kind)).toEqual([
      'compact_cancelled',
      'compact_failed',
      null,
      null,
      null,
      null,
    ]);
    for (const error of errors) {
      expect(error).toBeInstanceOf(AcpRpcError);
      expect(String((error as Error).message)).not.toContain('secret');
      expect(JSON.stringify(error)).not.toContain('secret');
    }
  });

  it('a soft deadline rejects without closing the client; the late reply still settles `late`', async () => {
    vi.useFakeTimers();
    const reasons: string[] = [];
    const { client, sent } = peer({ onProtocolError: (r) => reasons.push(r) });
    const reply = client.requestSoft('_x.ai/task/kill', { taskId: 't' }, 1_000);
    const caught = reply.catch((e: unknown) => e);
    vi.advanceTimersByTime(1_000);
    const error = await caught;
    expect(error).toBeInstanceOf(AcpSoftTimeoutError);
    expect(error).toMatchObject({ method: '_x.ai/task/kill' });
    expect(reasons).toEqual([]);
    expect(client.isClosed).toBe(false);
    expect(client.isPending(1)).toBe(true);
    client.receive(JSON.stringify({ jsonrpc: '2.0', id: 1, result: { status: 'killed' } }));
    await expect((error as AcpSoftTimeoutError).late).resolves.toEqual({ status: 'killed' });
    expect(sent).toEqual([{ jsonrpc: '2.0', id: 1, method: '_x.ai/task/kill', params: { taskId: 't' } }]);

    const inTime = client.requestSoft('_x.ai/recap', {}, 1_000);
    client.receive(JSON.stringify({ jsonrpc: '2.0', id: 2, result: { ok: true } }));
    await expect(inTime).resolves.toEqual({ ok: true });
    const refused = client.requestSoft('_x.ai/recap', {}, 1_000).catch((e: unknown) => e);
    client.receive(JSON.stringify({ jsonrpc: '2.0', id: 3, error: { code: -32601, message: 'Method not found' } }));
    expect(await refused).toBeInstanceOf(AcpRpcError);
    vi.advanceTimersByTime(5_000);
    expect(reasons).toEqual([]);
  });

  // Mutation: make `harden` arm no timer → the request waits on and no protocol error comes.
  it('harden gives a request sent with no deadline the hard one from now; nothing for one with a reply or a deadline', async () => {
    vi.useFakeTimers();
    const reasons: string[] = [];
    const { client } = peer({ onProtocolError: (r) => reasons.push(r) });
    const open = client.start('session/load', {}, null);
    const caught = open.reply.catch((e: unknown) => e);
    vi.advanceTimersByTime(60_000);
    expect(client.isPending(open.id)).toBe(true);
    client.harden(open.id, 1_000);
    client.harden(open.id, 50_000);
    vi.advanceTimersByTime(999);
    expect(reasons).toEqual([]);
    vi.advanceTimersByTime(1);
    expect(await caught).toBeInstanceOf(AcpTimeoutError);
    expect(reasons).toEqual(['timeout']);
    expect(client.isPending(open.id)).toBe(false);

    const answered = client.start('session/load', {}, null);
    client.receive(JSON.stringify({ jsonrpc: '2.0', id: answered.id, result: { ok: true } }));
    client.harden(answered.id, 10);
    vi.advanceTimersByTime(100);
    await expect(answered.reply).resolves.toEqual({ ok: true });
    expect(reasons).toEqual(['timeout']);
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
        'x.ai/workflowControl': true,
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
      workflowControl: true,
      agentVersion: '1.0.45',
      models: [{ modelId: 'q', contextTokens: 262144, efforts: null }],
    });
    expect(readInitialize({ _meta: { 'x.ai/workflowControl': 'true' } }).workflowControl).toBe(false);
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
      efforts: null,
      effort: null,
    });
    expect(
      readSessionReply({
        models: { currentModelId: 'm' },
        _meta: { sessionId: 's', 'x.ai/sessionDetail': { cwd: '/w2' } },
      }),
    ).toEqual({ sessionId: 's', currentModelId: 'm', cwd: '/w2', efforts: null, effort: null });
  });

  it('read the effort option of a session reply, and each model’s listed efforts at initialize', () => {
    // As agent 0.3.4 replies for a model whose `/v1/models` row lists levels (chat-thinking-levels-real-agent).
    const effort = {
      id: 'reasoning_effort',
      name: 'Reasoning Effort',
      category: 'thought_level',
      type: 'select',
      currentValue: 'high',
      options: [
        { value: 'none', name: 'Off' },
        { value: 'low', name: 'Low' },
        { value: 'high', name: 'High', description: 'slow' },
      ],
    };
    const model = { id: 'model', name: 'Model', category: 'model', type: 'select', currentValue: 'm', options: [] };
    expect(readSessionReply({ sessionId: 's', configOptions: [model, effort] })).toMatchObject({
      efforts: ['none', 'low', 'high'],
      effort: 'high',
    });
    expect(readSessionReply({ sessionId: 's', configOptions: [model] })).toMatchObject({ efforts: null, effort: null });
    expect(readEffortConfig([{ ...effort, type: 'boolean' }])).toBeNull();
    expect(readEffortConfig([{ ...effort, options: [{ value: 3 }, { value: 'low', name: 'Low' }] }])).toEqual({
      ids: ['low'],
      current: 'high',
    });
    expect(readEffortConfig('x')).toBeNull();

    const facts = readInitialize({
      _meta: {
        modelState: {
          availableModels: [
            {
              modelId: 'q',
              _meta: {
                totalContextTokens: 8,
                supportsReasoningEffort: true,
                reasoningEfforts: [
                  { id: 'none', value: 'none', label: 'Off', description: null, default: false },
                  { id: 'high', value: 'high', label: 'High', description: null, default: true },
                  { value: 'low' },
                ],
              },
            },
            { modelId: 'p', _meta: { reasoningEfforts: 'high' } },
          ],
        },
      },
    });
    expect(facts.models).toEqual([
      {
        modelId: 'q',
        contextTokens: 8,
        efforts: [
          { id: 'none', value: 'none', default: false },
          { id: 'high', value: 'high', default: true },
        ],
      },
      { modelId: 'p', contextTokens: null, efforts: null },
    ]);
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

  it('read the requester of a helper ask as the fork sends it; a present malformed one makes the request unreadable', () => {
    const base = {
      sessionId: 'parent',
      toolCall: { toolCallId: 't', kind: 'execute', rawInput: { command: 'ls' } },
      options: [{ optionId: 'allow-once', name: 'Allow', kind: 'allow_once' }],
    };
    const requester = { sessionId: 'child', label: 'researcher-0', runId: 'wf_1', cwd: '/work/repo' };
    expect(readPermissionRequest(base)?.requester).toBeNull();
    expect(
      readPermissionRequest({ ...base, _meta: { hookAsk: { hookName: 'guard' }, 'x.ai/requester': requester } }),
    ).toMatchObject({ sessionId: 'parent', hookAsk: { hookName: 'guard', reason: null }, requester });
    expect(
      readPermissionRequest({ ...base, _meta: { 'x.ai/requester': { ...requester, runId: null } } })?.requester,
    ).toEqual({ ...requester, runId: null });
    const junk: unknown[] = [
      null,
      'child',
      [],
      { ...requester, sessionId: '' },
      { ...requester, sessionId: 7 },
      { ...requester, label: undefined },
      { ...requester, cwd: '' },
      { ...requester, cwd: null },
      { ...requester, runId: '' },
      { ...requester, runId: 3 },
      { sessionId: 'child', label: 'x', cwd: '/w' },
    ];
    for (const value of junk) {
      expect(readPermissionRequest({ ...base, _meta: { 'x.ai/requester': value } })).toBeNull();
    }
  });

  it('read a workflow update, a spawned child and the stop answer; other shapes are null', () => {
    const update = {
      sessionUpdate: 'workflow_updated',
      run_id: 'wf_1',
      revision: 3,
      name: 'deep-research',
      status: 'active',
      phases: [],
    };
    expect(readWorkflowUpdate(update)).toEqual({
      runId: 'wf_1',
      revision: 3,
      name: 'deep-research',
      status: 'active',
      lastEvent: null,
    });
    expect(readWorkflowUpdate({ ...update, name: undefined })).toMatchObject({ name: '' });
    expect(readWorkflowUpdate({ ...update, last_event: 'workflow_outcome_ignored' })).toMatchObject({
      lastEvent: 'workflow_outcome_ignored',
    });
    for (const bad of [
      { ...update, sessionUpdate: 'workflow_started' },
      { ...update, run_id: '' },
      { ...update, run_id: 1 },
      { ...update, revision: -1 },
      { ...update, revision: 1.5 },
      { ...update, revision: '3' },
      { ...update, status: '' },
      { ...update, status: null },
    ]) {
      expect(readWorkflowUpdate(bad)).toBeNull();
    }
    expect(readSpawnedChild({ sessionUpdate: 'subagent_spawned', child_session_id: 'c1' })).toBe('c1');
    expect(readSpawnedChild({ sessionUpdate: 'subagent_spawned', child_session_id: '' })).toBeNull();
    expect(readSpawnedChild({ sessionUpdate: 'subagent_finished', child_session_id: 'c1' })).toBeNull();
    expect(readWorkflowStop({ outcome: 'stopped', runId: 'wf_1', name: 'deep-research' })).toBe('stopped');
    expect(readWorkflowStop({ outcome: 'rejected', code: 'workflow_control_unknown_run', detail: 'x' })).toBe(
      'rejected',
    );
    expect(readWorkflowStop({ outcome: 'Stopped' })).toBeNull();
    expect(readWorkflowStop(null)).toBeNull();
  });

  // The `_x.ai/task/kill` reply as v0.3.4 sends it (`extensions/task.rs` KillTaskResponse in `ExtMethodResult`).
  it('read the task kill outcome; the error envelope and any other shape are null', () => {
    for (const outcome of ['killed', 'already_exited', 'not_found'] as const) {
      expect(readTaskKill({ result: { taskId: 't1', outcome } })).toBe(outcome);
    }
    for (const bad of [
      { result: null, error: 'session not found' },
      { result: { taskId: 't1', outcome: 'Killed' } },
      { result: { taskId: 't1', outcome: 'alreadyExited' } },
      { result: { taskId: 't1' } },
      { result: 'killed' },
      { outcome: 'killed' },
      null,
      'killed',
    ]) {
      expect(readTaskKill(bad)).toBeNull();
    }
  });

  // The `_x.ai/subagent/cancel` reply as v0.3.5 sends it (`extensions/task.rs` CancelSubagentResponse in
  // `ExtMethodResult`). Mutation: read the legacy `cancelled` bool → `already_finished` reads as a failed stop (503).
  // Mutation: take any kind → the agent's `unknown` reads as stopped and the row waits for an end that never comes.
  it('read the subagent cancel outcome by its kind; the error envelope, the bool alone and any other shape are null', () => {
    const id = '01a1264a-4c3b-7cd3-833e-26c6f665d1db';
    expect(readSubagentCancel({ result: { subagentId: id, cancelled: true, outcome: { kind: 'cancelled' } } })).toBe(
      'cancelled',
    );
    expect(
      readSubagentCancel({
        result: { subagentId: id, cancelled: false, outcome: { kind: 'already_finished', status: 'completed' } },
      }),
    ).toBe('already_finished');
    expect(readSubagentCancel({ result: { subagentId: id, cancelled: false, outcome: { kind: 'not_found' } } })).toBe(
      'not_found',
    );
    for (const bad of [
      { result: null, error: 'session not found' },
      { result: { subagentId: id, cancelled: true } },
      { result: { subagentId: id, cancelled: false, outcome: { kind: 'unknown' } } },
      { result: { subagentId: id, outcome: 'cancelled' } },
      { subagentId: id, cancelled: true, outcome: { kind: 'cancelled' } },
      null,
    ]) {
      expect(readSubagentCancel(bad)).toBeNull();
    }
  });

  // The `_x.ai/subagent/message` reply as v0.3.5 sends it (`extensions/subagent_message.rs` SendSubagentMessageOutcome,
  // `kind`-tagged, in `ExtMethodResult`). Mutation: read `result.outcome` → every message reads as an error.
  it('read each subagent message kind; the error envelope and any other shape are null', () => {
    for (const kind of SUBAGENT_MESSAGE_KINDS) expect(readSubagentMessage({ result: { kind } })).toBe(kind);
    expect(readSubagentMessage({ result: { kind: 'accepted', messageId: 'm1' } })).toBe('accepted');
    expect(readSubagentMessage({ result: { kind: 'limit', maxBytes: 32_768, observedBytes: 40_000 } })).toBe('limit');
    for (const bad of [
      { result: null, error: 'session not found' },
      { result: { kind: 'Accepted' } },
      { result: { kind: 'notActive' } },
      { result: { outcome: { kind: 'accepted' } } },
      { kind: 'accepted' },
      null,
    ]) {
      expect(readSubagentMessage(bad)).toBeNull();
    }
  });

  it('read a monitor_event as the agent sends it (task id, description, raw text); any other shape is null', () => {
    const MON = '01a20df5-8562-75b2-aaf5-d3c1e2ec9a01';
    const line = {
      sessionUpdate: 'monitor_event',
      task_id: MON,
      description: 'Watch the build log',
      event_text: 'error TS2345',
    };
    expect(readMonitorEvent(line)).toEqual({ taskId: MON, description: 'Watch the build log', text: 'error TS2345' });
    for (const bad of [
      { ...line, sessionUpdate: 'task_completed' },
      { ...line, task_id: '' },
      { ...line, task_id: 7 },
      { ...line, event_text: null },
      { sessionUpdate: 'monitor_event', task_id: MON, event_text: 'x' },
    ]) {
      expect(readMonitorEvent(bad)).toBeNull();
    }
  });

  it('read updates from the update methods and nothing else', () => {
    const params = { sessionId: 's', update: { sessionUpdate: 'x' } };
    expect(readSessionUpdate('session/update', params)?.xai).toBe(false);
    expect(readSessionUpdate('_x.ai/session_notification', params)?.xai).toBe(true);
    expect(readSessionUpdate('_x.ai/session/update', params)?.xai).toBe(true);
    for (const method of [
      '_x.ai/task_backgrounded',
      '_x.ai/task_completed',
      '_x.ai/scheduled_task_created',
      '_x.ai/scheduled_task_fired',
      '_x.ai/scheduled_task_deleted',
      '_x.ai/monitor_event',
    ]) {
      expect(readSessionUpdate(method, params)?.xai).toBe(true);
    }
    expect(readSessionUpdate('_x.ai/mcp/server_status', params)).toBeNull();
    expect(readSessionUpdate('session/update', { update: { sessionUpdate: 'x' } })).toBeNull();
  });

  it("read the agent's four compaction updates as the fork sends them, and junk counts as null", () => {
    expect(
      readCompaction({
        sessionUpdate: 'auto_compact_started',
        tokens_used: 222_822,
        context_window: 262_144,
        percentage: 85,
        reason: 'Context window 85% full',
      }),
    ).toEqual({ kind: 'started', tokensUsed: 222_822, contextWindow: 262_144 });
    expect(
      readCompaction({
        sessionUpdate: 'auto_compact_completed',
        tokens_before: 222_822,
        tokens_after: 31_204,
        elapsed_ms: 1_221_000,
        summary_preview: null,
      }),
    ).toEqual({ kind: 'completed', tokensBefore: 222_822, tokensAfter: 31_204, elapsedMs: 1_221_000 });
    // Manual `/compact`: tokens_before and elapsed_ms may be missing.
    expect(readCompaction({ sessionUpdate: 'auto_compact_completed', tokens_after: 9 })).toEqual({
      kind: 'completed',
      tokensBefore: null,
      tokensAfter: 9,
      elapsedMs: null,
    });
    expect(
      readCompaction({
        sessionUpdate: 'auto_compact_failed',
        error: "it'll retry on the next turn, or start a new session using /new.\nstream idle timeout after 600s",
      }),
    ).toEqual({
      kind: 'failed',
      error: "it'll retry on the next turn, or start a new session using /new.\nstream idle timeout after 600s",
      retry: true,
    });
    expect(
      readCompaction({ sessionUpdate: 'auto_compact_failed', error: 'this conversation is too large to compact.' }),
    ).toEqual({ kind: 'failed', error: 'this conversation is too large to compact.', retry: false });
    expect(readCompaction({ sessionUpdate: 'auto_compact_cancelled', reason: 'user_cancelled' })).toEqual({
      kind: 'cancelled',
    });
    expect(
      readCompaction({ sessionUpdate: 'auto_compact_started', tokens_used: -1, context_window: '262144' }),
    ).toEqual({ kind: 'started', tokensUsed: null, contextWindow: null });
    expect(
      readCompaction({
        sessionUpdate: 'auto_compact_completed',
        tokens_before: 1.5,
        tokens_after: null,
        elapsed_ms: {},
      }),
    ).toEqual({ kind: 'completed', tokensBefore: null, tokensAfter: null, elapsedMs: null });
    expect(readCompaction({ sessionUpdate: 'auto_compact_failed', error: 7 })).toEqual({
      kind: 'failed',
      error: '',
      retry: false,
    });
    expect(readCompaction({ sessionUpdate: 'memory_flush_started' })).toBeNull();
    expect(readCompaction({ sessionUpdate: 'auto_continue_completed', total_tokens: 1 })).toBeNull();
  });
});
