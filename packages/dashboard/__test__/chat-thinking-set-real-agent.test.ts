/**
 * The host's thinking-level set call on the real agent binary (`MLX_AGENT_E2E_BIN`) with a scripted engine and no
 * model. The launch is marked mlx-agent-v0.3.5 by hand: 0.3.4 takes `session/set_config_option` and pushes
 * `config_option_update` the same way and differs only in the field it sends the level in (`output_config.effort`
 * instead of `extra_body.reasoning_effort`), so its request shows the level the host set. Skipped without the variable.
 */

import { mkdirSync, mkdtempSync, realpathSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { spawnAgent, type SpawnAgent } from '../src/chat/acp/process.js';
import { AGENT_ARGS } from '../src/chat/agent/profiles.js';
import { endpointModel, fakeHost, noProviders, type Json } from './helpers/chat-fake.js';
import { startScriptedEngine, type ModelReply, type ScriptedEngine } from './helpers/scripted-engine.js';

const BIN = process.env.MLX_AGENT_E2E_BIN;
const MODEL = 'qwen3.5-4b';
/** A qwen3_5 row's `reasoningEfforts` as the engine writes them (`reasoningEffortsFor`, pinned in thinking-levels.test.ts). */
const QWEN_EFFORTS = [
  { value: 'none', label: 'Off' },
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High', default: true },
];

function stdoutLines(into: Json[]): SpawnAgent {
  return (command) => {
    const child = spawnAgent(command);
    let partial = '';
    child.stdout?.on('data', (chunk: Buffer) => {
      partial += chunk.toString();
      let nl = partial.indexOf('\n');
      while (nl !== -1) {
        const line = partial.slice(0, nl);
        partial = partial.slice(nl + 1);
        if (line.trim() !== '') {
          try {
            into.push(JSON.parse(line) as Json);
          } catch {
            into.push({ unparsed: line });
          }
        }
        nl = partial.indexOf('\n');
      }
    });
    return child;
  };
}

describe.skipIf(BIN === undefined)('Thinking level set on the real agent', () => {
  it('the agent lists the levels, takes the set before the prompt, pushes the option, and runs at that level', async () => {
    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-level-set-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    const out: Json[] = [];
    let engine: ScriptedEngine | null = null;
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: 'mlx-agent-v0.3.5', versionVerified: true },
      startEngine: async () => {
        engine = await startScriptedEngine(
          [{ id: MODEL, contextWindow: 262_144, reasoningEfforts: QWEN_EFFORTS }],
          (q): ModelReply =>
            q.tools.length === 1 && q.tools[0] === 'session_title'
              ? { blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Level set' } }] }
              : { blocks: [{ type: 'text', text: 'Hello.' }] },
        );
        return engine;
      },
      env: { TMPDIR: `${agentTmp}/` },
      endpointModels: [endpointModel(MODEL, `/models/${MODEL}`)],
      spawn: stdoutLines(out),
    });
    try {
      const cwd = real.project;
      await real.host.start();
      expect(real.host.models()).toEqual([
        {
          modelId: MODEL,
          contextTokens: 262_144,
          thinking: { levels: ['none', 'low', 'medium', 'high'], default: 'high' },
        },
      ]);
      const { sessionId } = await real.host.createSession({ modelId: MODEL, cwd });
      const efforts = (): string[] =>
        engine!.modelRequests
          .filter((q) => q.body.model === MODEL && !(q.tools.length === 1 && q.tools[0] === 'session_title'))
          .map((q) => {
            const effort = (q.body.output_config as Json | undefined)?.effort;
            return typeof effort === 'string' ? effort : 'unset';
          });

      const low = real.host.startTurn(sessionId, 'Level probe: say hello.', { modelId: MODEL, cwd, thinking: 'low' });
      expect((await low.done).outcome).toBe('completed');
      expect(real.events.notices).toEqual([]);
      expect(real.events.cleared).toEqual([{ sessionId, code: 'thinking-not-set' }]);
      expect(new Set(efforts())).toEqual(new Set(['low']));
      const pushed = out.filter(
        (m) =>
          m.method === 'session/update' && ((m.params as Json).update as Json).sessionUpdate === 'config_option_update',
      );
      expect(pushed.length).toBeGreaterThan(0);
      const option = (((pushed.at(-1)!.params as Json).update as Json).configOptions as Json[]).find(
        (o) => o.id === 'reasoning_effort',
      );
      expect(option).toMatchObject({ currentValue: 'low' });
      // Taken by the host: never a transcript line.
      expect(real.events.updates.map((u) => u.line.update.sessionUpdate)).not.toContain('config_option_update');

      // No level picked: the default goes back before the next prompt.
      const before = efforts().length;
      const back = real.host.startTurn(sessionId, 'Level probe again.', { modelId: MODEL, cwd });
      expect((await back.done).outcome).toBe('completed');
      expect(new Set(efforts().slice(before))).toEqual(new Set(['high']));

      // A level the model does not list steps down as the page shows it: Max is High on this row.
      const off = real.host.startTurn(sessionId, 'Level probe off.', { modelId: MODEL, cwd, thinking: 'none' });
      expect((await off.done).outcome).toBe('completed');
      const atMax = efforts().length;
      const max = real.host.startTurn(sessionId, 'Level probe max.', { modelId: MODEL, cwd, thinking: 'max' });
      expect((await max.done).outcome).toBe('completed');
      expect(new Set(efforts().slice(atMax))).toEqual(new Set(['high']));
      expect(real.events.notices).toEqual([]);
    } finally {
      await real.close();
      rmSync(outer, { recursive: true, force: true });
    }
  }, 120_000);
});
