/**
 * The engine's thinking levels on the real agent binary (`MLX_AGENT_E2E_BIN`) with a scripted engine and no model:
 * the `/v1/models` row carries `reasoningEfforts` as an engine with `thinkingLevels` writes them; the agent reads them
 * into a `reasoning_effort` session option at the default level, and sends that level as `extra_body.reasoning_effort`
 * (the row says `owned_by: "mlx-node"`) beside `output_config.effort` and an adaptive thought. Its request differs from
 * the one for a row with no levels in those three fields alone; the plain row's request has none of them. Skipped
 * without the variable.
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
/** A row with no levels: what a by-hand `mlx serve` lists. */
const PLAIN = 'qwen3.5-9b';
const PROMPT = 'Thinking levels probe: say hello.';
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

describe.skipIf(BIN === undefined)('Thinking levels on the real agent', () => {
  it('the agent offers the listed levels at the default one and sends it as extra_body.reasoning_effort', async () => {
    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-levels-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    const out: Json[] = [];
    let engine: ScriptedEngine | null = null;
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
      startEngine: async () => {
        engine = await startScriptedEngine(
          [
            { id: MODEL, contextWindow: 262_144, reasoningEfforts: QWEN_EFFORTS },
            { id: PLAIN, contextWindow: 262_144 },
          ],
          (q): ModelReply =>
            q.tools.length === 1 && q.tools[0] === 'session_title'
              ? { blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Levels probe' } }] }
              : { blocks: [{ type: 'text', text: 'Hello.' }] },
        );
        return engine;
      },
      env: { TMPDIR: `${agentTmp}/` },
      endpointModels: [endpointModel(MODEL, `/models/${MODEL}`), endpointModel(PLAIN, `/models/${PLAIN}`)],
      spawn: stdoutLines(out),
    });
    try {
      const cwd = { cwd: real.project };
      const { sessionId } = await real.host.createSession({ modelId: MODEL, ...cwd });
      const replyOf = (id: string) => out.find((m) => (m.result as Json | undefined)?.sessionId === id)!.result as Json;
      const options = replyOf(sessionId).configOptions as Json[];
      const effort = options.find((option) => option.id === 'reasoning_effort')!;
      expect(effort).toMatchObject({ category: 'thought_level', type: 'select', currentValue: 'high' });
      expect((effort.options as Json[]).map((option) => [option.value, option.name])).toEqual([
        ['none', 'Off'],
        ['low', 'Low'],
        ['medium', 'Medium'],
        ['high', 'High'],
      ]);

      expect((await real.host.startTurn(sessionId, PROMPT, { modelId: MODEL, ...cwd }).done).outcome).toBe('completed');
      const mainOf = (model: string) =>
        engine!.modelRequests.filter(
          (q) => q.body.model === model && !(q.tools.length === 1 && q.tools[0] === 'session_title'),
        );
      const main = mainOf(MODEL);
      expect(main.length).toBeGreaterThan(0);
      for (const q of main) {
        expect(q.body.output_config).toEqual({ effort: 'high' });
        expect(q.body.thinking).toEqual({ type: 'adaptive', display: 'summarized' });
        expect(q.body.extra_body).toEqual({ reasoning_effort: 'high' });
      }

      const plain = await real.host.createSession({ modelId: PLAIN, ...cwd });
      const plainOptions = replyOf(plain.sessionId).configOptions as Json[];
      expect(plainOptions.map((option) => option.id)).not.toContain('reasoning_effort');
      expect((await real.host.startTurn(plain.sessionId, PROMPT, { modelId: PLAIN, ...cwd }).done).outcome).toBe(
        'completed',
      );
      const plainMain = mainOf(PLAIN);
      expect(plainMain).toHaveLength(main.length);
      /** The body as sent, minus the model name and with the session id (a path in one tool's text) made equal. */
      const sameShape = (body: Json, id: string): Json => {
        const { model: _model, ...rest } = JSON.parse(JSON.stringify(body).split(id).join('<session>')) as Json;
        return rest;
      };
      main.forEach((q, i) => {
        const {
          output_config: _effort,
          thinking: _thinking,
          extra_body: _extra,
          ...levelled
        } = sameShape(q.body, sessionId);
        const bare = sameShape(plainMain[i]!.body, plain.sessionId);
        expect('output_config' in bare || 'thinking' in bare || 'extra_body' in bare).toBe(false);
        expect(levelled).toEqual(bare);
      });
    } finally {
      await real.close();
      rmSync(outer, { recursive: true, force: true });
    }
  });
});
