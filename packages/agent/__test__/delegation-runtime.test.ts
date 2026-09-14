import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { createAssistantMessageEventStream, type AssistantMessage } from '@earendil-works/pi-ai';
import {
  createAgentSession,
  createBashTool,
  DefaultResourceLoader,
  ModelRuntime,
  SessionManager,
  SettingsManager,
} from '@earendil-works/pi-coding-agent';
import { Type } from 'typebox';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { createDelegationExtension } from '../src/extensions/delegation.js';

const savedExitCode = process.exitCode;
afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
  process.exitCode = savedExitCode;
});

describe('delegate runtime permission handoff', () => {
  it.each([
    'missing-caller',
    'caller-approved',
    'caller-denied',
    'sandbox-denied',
    'pipeline-denied',
    'authorized',
    'thread-id-only',
    'thread-id-denied',
  ] as const)('handles %s in the real agent loop without permission retry turns', async (scenario) => {
    for (const name of ['CODEX_THREAD_ID', 'CODEX_PERMISSION_PROFILE', 'CODEX_SANDBOX', 'MLX_AGENT_AUTO_APPROVE'])
      vi.stubEnv(name, undefined);
    const callerApproved = scenario.startsWith('caller-');
    if (scenario !== 'missing-caller' && !callerApproved) vi.stubEnv('CODEX_THREAD_ID', 'test-thread');
    if (scenario !== 'missing-caller' && !callerApproved && !scenario.startsWith('thread-id-')) {
      vi.stubEnv('CODEX_PERMISSION_PROFILE', ':workspace-write');
    }
    const authorized = scenario === 'authorized' || scenario === 'thread-id-only' || scenario === 'caller-approved';
    vi.spyOn(process.stderr, 'write').mockImplementation(() => true);
    const root = await mkdtemp(join(tmpdir(), 'mlx-delegate-runtime-'));
    const command =
      scenario === 'pipeline-denied' ? "printf 'Permission denied\\n' >&2; false | cat" : 'gh pr view 147';
    const settingsManager = SettingsManager.inMemory();
    const resourceLoader = new DefaultResourceLoader({
      cwd: root,
      agentDir: root,
      settingsManager,
      noExtensions: true,
      noSkills: true,
      noContextFiles: true,
      noThemes: true,
      noPromptTemplates: true,
      extensionFactories: [createDelegationExtension({ callerApproved })],
    });
    let session: Awaited<ReturnType<typeof createAgentSession>>['session'] | undefined;
    try {
      await resourceLoader.reload();
      const runtime = await ModelRuntime.create({
        authPath: join(root, 'auth.json'),
        modelsPath: join(root, 'models.json'),
        allowModelNetwork: false,
      });
      const model = {
        id: 'scripted-delegate',
        name: 'scripted-delegate',
        api: 'mlx',
        provider: 'mlx',
        baseUrl: 'mlx://local',
        reasoning: false,
        input: ['text'] as ['text'],
        contextWindow: 65536,
        maxTokens: 1024,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      };
      let inferences = 0;
      runtime.registerProvider('mlx', {
        api: 'mlx',
        baseUrl: 'mlx://local',
        apiKey: 'local-test',
        models: [model],
        streamSimple: () => {
          inferences++;
          if (inferences > 2) throw new Error('Unexpected retry inference');
          const message: AssistantMessage = {
            role: 'assistant',
            api: 'mlx',
            provider: 'mlx',
            model: model.id,
            content:
              inferences === 1
                ? [{ type: 'toolCall', id: 'call-bash', name: 'bash', arguments: { command } }]
                : [{ type: 'text', text: 'CI passed' }],
            stopReason: inferences === 1 ? 'toolUse' : 'stop',
            timestamp: Date.now(),
            usage: {
              input: 1,
              output: 1,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 2,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
            },
          };
          const stream = createAssistantMessageEventStream();
          stream.push({ type: 'start', partial: message });
          stream.push({ type: 'done', reason: message.stopReason === 'toolUse' ? 'toolUse' : 'stop', message });
          stream.end();
          return stream;
        },
      });
      const execute = vi.fn(async (id: string, args: { command: string }) => {
        if (scenario === 'pipeline-denied') return createBashTool(root).execute(id, args);
        if (scenario === 'sandbox-denied' || scenario === 'thread-id-denied' || scenario === 'caller-denied')
          throw new Error('Network access was denied by the Codex sandbox network proxy.');
        return { content: [{ type: 'text' as const, text: 'All checks passed' }], details: {} };
      });
      ({ session } = await createAgentSession({
        cwd: root,
        agentDir: root,
        model,
        modelRuntime: runtime,
        resourceLoader,
        settingsManager,
        sessionManager: SessionManager.inMemory(root),
        tools: ['bash'],
        customTools: [
          {
            name: 'bash',
            label: 'Scripted bash',
            description: 'Controlled tool fixture; only the harmless pipeline case executes a shell.',
            parameters: Type.Object({ command: Type.String() }),
            execute,
          },
        ],
      }));
      await session.bindExtensions({ mode: 'print' });
      await session.prompt('Check PR #147');
      expect(inferences).toBe(authorized ? 2 : 1);
      expect(execute).toHaveBeenCalledTimes(scenario === 'missing-caller' ? 0 : 1);
      if (scenario !== 'missing-caller')
        expect(execute.mock.calls[0]![1].command).toBe(`set -e -o pipefail\n${command}`);
      const handoffs = session.sessionManager
        .getEntries()
        .filter((entry) => entry.type === 'custom' && entry.customType === 'mlx-delegate-handoff');
      expect(handoffs).toHaveLength(authorized ? 0 : 1);
      if (!authorized) expect(process.exitCode).toBe(1);
    } finally {
      session?.dispose();
      await rm(root, { recursive: true, force: true });
    }
  });
});
