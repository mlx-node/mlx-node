import type {
  ExtensionAPI,
  ExtensionContext,
  BeforeAgentStartEvent,
  ToolCallEvent,
  ToolResultEvent,
} from '@earendil-works/pi-coding-agent';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  createDelegationExtension,
  delegateCallerPermissions,
  delegationBlocker,
} from '../src/extensions/delegation.js';

const previousExitCode = process.exitCode;
const writeStderr = vi.fn(() => true);

beforeEach(() => {
  for (const key of [
    'CODEX_THREAD_ID',
    'CODEX_PERMISSION_PROFILE',
    'CODEX_SANDBOX',
    'CODEX_SANDBOX_NETWORK_DISABLED',
    'MLX_AGENT_AUTO_APPROVE',
  ])
    vi.stubEnv(key, undefined);
  writeStderr.mockClear();
  vi.spyOn(process.stderr, 'write').mockImplementation(writeStderr);
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
  process.exitCode = previousExitCode;
});

function fixture(callerApproved = false) {
  const handlers = new Map<string, (event: never, ctx: ExtensionContext) => unknown>();
  const appendEntry = vi.fn();
  const extension = createDelegationExtension({ callerApproved });
  if (typeof extension === 'function') throw new Error('Expected named extension');
  void extension.factory({
    on: (name: string, handler: (event: never, ctx: ExtensionContext) => unknown) => handlers.set(name, handler),
    appendEntry,
  } as unknown as ExtensionAPI);
  const abort = vi.fn();
  const ctx = {
    abort,
    sessionManager: { getSessionFile: () => '/sessions/delegated-task.jsonl', getSessionId: () => 'delegated-task' },
  } as unknown as ExtensionContext;
  return {
    abort,
    appendEntry,
    toolCall: (toolName: string) =>
      handlers.get('tool_call')!(
        { type: 'tool_call', toolName, toolCallId: 'call-1', input: {} } as ToolCallEvent as never,
        ctx,
      ),
    toolResult: (text: string, isError: boolean) =>
      handlers.get('tool_result')!(
        {
          type: 'tool_result',
          toolName: 'bash',
          toolCallId: 'call-1',
          input: {},
          content: [{ type: 'text', text }],
          isError,
        } as ToolResultEvent as never,
        ctx,
      ),
    start: () =>
      handlers.get('before_agent_start')!({ systemPrompt: 'Worker prompt' } as BeforeAgentStartEvent as never, ctx),
  };
}

describe('delegate caller permissions', () => {
  it('persists a delegate boundary for each invocation without adding it to the model prompt', () => {
    const gate = fixture();
    const result = gate.start();
    gate.start();
    expect(gate.appendEntry.mock.calls).toEqual([
      ['mlx-delegate-session', { version: 1, sessionId: 'delegated-task' }],
      ['mlx-delegate-session', { version: 1, sessionId: 'delegated-task' }],
    ]);
    expect(JSON.stringify(result)).not.toContain('mlx-delegate-session');
  });
  it.each([':danger-full-access', ':workspace-write', ':read-only', 'custom-network-policy'])(
    'preserves the opaque active Codex profile %s without elevating it',
    (profile) => {
      expect(
        delegateCallerPermissions({
          CODEX_THREAD_ID: 'thread',
          CODEX_PERMISSION_PROFILE: profile,
          CODEX_SANDBOX_NETWORK_DISABLED: '1',
        }),
      ).toEqual({ source: 'codex', profile, networkDisabled: true });
    },
  );

  it.each([undefined, 'unknown'])('recognizes a Codex thread without a known profile or sandbox (%s)', (sandbox) => {
    expect(delegateCallerPermissions({ CODEX_THREAD_ID: 'thread', CODEX_SANDBOX: sandbox })).toEqual({
      source: 'codex',
      profile: 'inherited',
      networkDisabled: false,
    });
  });

  it('requires a nonempty Codex thread id even when permission metadata is present', () => {
    expect(delegateCallerPermissions({})).toBeUndefined();
    expect(delegateCallerPermissions({ CODEX_THREAD_ID: ' ' })).toBeUndefined();
    expect(delegateCallerPermissions({ CODEX_PERMISSION_PROFILE: ':danger-full-access' })).toBeUndefined();
    expect(delegateCallerPermissions({ CODEX_SANDBOX: 'seatbelt' })).toBeUndefined();
  });

  it.each(['seatbelt', 'landlock'])('recognizes the older %s child sandbox marker', (sandbox) => {
    expect(delegateCallerPermissions({ CODEX_THREAD_ID: 'thread', CODEX_SANDBOX: sandbox })?.profile).toBe(
      `sandbox:${sandbox}`,
    );
  });

  it('uses caller execution without setting auto-approval or rewriting sandbox variables', () => {
    vi.stubEnv('CODEX_THREAD_ID', 'thread');
    vi.stubEnv('CODEX_PERMISSION_PROFILE', ':read-only');
    vi.stubEnv('CODEX_SANDBOX_NETWORK_DISABLED', '1');
    const gate = fixture();
    expect(gate.toolCall('bash')).toBeUndefined();
    expect(process.env.MLX_AGENT_AUTO_APPROVE).toBeUndefined();
    expect(process.env.CODEX_PERMISSION_PROFILE).toBe(':read-only');
    expect(process.env.CODEX_SANDBOX_NETWORK_DISABLED).toBe('1');
    expect(gate.start()).toMatchObject({ systemPrompt: expect.stringContaining('caller disables network access') });
  });

  it('allows thread-id-only callers without inventing a profile or clearing the network restriction', () => {
    vi.stubEnv('CODEX_THREAD_ID', 'thread');
    vi.stubEnv('CODEX_SANDBOX_NETWORK_DISABLED', '1');
    const gate = fixture();
    expect(gate.toolCall('bash')).toBeUndefined();
    expect(process.env.MLX_AGENT_AUTO_APPROVE).toBeUndefined();
    expect(process.env.CODEX_PERMISSION_PROFILE).toBeUndefined();
    expect(process.env.CODEX_SANDBOX).toBeUndefined();
    expect(process.env.CODEX_SANDBOX_NETWORK_DISABLED).toBe('1');
    expect(gate.start()).toMatchObject({ systemPrompt: expect.stringContaining('caller disables network access') });
  });

  it('stops at the first unapproved tool and rejects later calls without more inference or repeated diagnostics', () => {
    const gate = fixture();
    expect(gate.toolCall('read')).toBeUndefined();
    expect(gate.toolCall('bash')).toMatchObject({
      block: true,
      terminate: true,
      reason: expect.stringContaining('No caller permission context'),
    });
    expect(gate.abort).toHaveBeenCalledOnce();
    expect(gate.toolCall('subagent')).toMatchObject({ block: true, terminate: true });
    expect(gate.toolCall('read')).toMatchObject({ block: true, terminate: true });
    expect(gate.appendEntry).toHaveBeenCalledOnce();
    expect(writeStderr).toHaveBeenCalledOnce();
    expect(writeStderr).toHaveBeenCalledWith(expect.stringContaining('/sessions/delegated-task.jsonl'));
    expect(process.exitCode).toBe(1);
  });

  it('does not let a model-written environment change grant permission during a run', () => {
    const gate = fixture();
    vi.stubEnv('CODEX_THREAD_ID', 'thread');
    vi.stubEnv('CODEX_PERMISSION_PROFILE', ':danger-full-access');
    vi.stubEnv('MLX_AGENT_AUTO_APPROVE', '1');
    expect(gate.toolCall('bash')).toMatchObject({ block: true });
  });

  it('accepts explicit caller approval without Codex metadata or changing the environment', () => {
    const gate = fixture(true);
    expect(gate.toolCall('bash')).toBeUndefined();
    expect(process.env.CODEX_THREAD_ID).toBeUndefined();
    expect(process.env.MLX_AGENT_AUTO_APPROVE).toBeUndefined();
    expect(gate.start()).toMatchObject({ systemPrompt: expect.stringContaining('does not copy') });
    expect(gate.toolCall('subagent')).toMatchObject({ block: true, terminate: true });
  });

  it('retains explicit caller approval but never enables recursive subagents', () => {
    vi.stubEnv('MLX_AGENT_AUTO_APPROVE', '1');
    const gate = fixture();
    expect(gate.toolCall('bash')).toBeUndefined();
    expect(gate.toolCall('subagent')).toMatchObject({
      block: true,
      reason: expect.stringContaining('cannot create subagents'),
    });
  });

  it('ends a blocked sandbox operation with a persisted handoff and nonzero exit status', () => {
    const gate = fixture();
    expect(gate.toolResult('Network access was denied by the Codex sandbox network proxy.', true)).toMatchObject({
      isError: true,
      content: [{ type: 'text', text: expect.stringContaining('Continue this task in the calling agent') }],
    });
    expect(gate.abort).toHaveBeenCalledOnce();
    expect(gate.appendEntry).toHaveBeenCalledWith(
      'mlx-delegate-handoff',
      expect.objectContaining({ reason: expect.stringContaining('Network access was denied') }),
    );
    expect(process.exitCode).toBe(1);
  });

  it('does not interpret successful CI log content or ordinary command mistakes as permission blockers', () => {
    const gate = fixture();
    expect(gate.toolResult('CI log: permission denied', false)).toBeUndefined();
    expect(gate.toolResult('Unknown JSON field: headCommit', true)).toBeUndefined();
    expect(gate.toolResult('CI checks failed; command exited with code 1', true)).toBeUndefined();
    expect(gate.abort).not.toHaveBeenCalled();
  });

  it.each([
    'Permission denied',
    'Operation not permitted',
    'HTTP 403: Resource not accessible by integration',
    'To get started run gh auth login',
    'error connecting to api.github.com',
  ])('recognizes the actionable blocker %s', (text) => expect(delegationBlocker(text)).toBe(true));
});
