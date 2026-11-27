/**
 * Hook results on the real agent binary (`MLX_AGENT_E2E_BIN`) with a scripted engine and no model: a hooks file
 * in the agent home holds a `PreToolUse` hook that denies `rm`, a `Stop` hook that fails and a `UserPromptSubmit` hook
 * that blocks a prompt. The lines v0.3.4 sends have the fields of the fake agent's lines; the host takes
 * `hook_run_started`; the reducer gives the same failed row, Blocked call, note and stored end live and from
 * `updates.jsonl`. Skipped without the variable.
 */

import { mkdirSync, mkdtempSync, readdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { hookSource, parseUpdates, TranscriptReducer } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { spawnAgent, type SpawnAgent } from '../src/chat/acp/process.js';
import { AGENT_ARGS } from '../src/chat/agent/profiles.js';
import { fakeHost, noProviders, until, type Json } from './helpers/chat-fake.js';
import { hasToolResult, lastMessage, messageText, startScriptedEngine } from './helpers/scripted-engine.js';

const BIN = process.env.MLX_AGENT_E2E_BIN;
const MODEL = 'qwen3.5-4b';
const AGENT_WAIT_MS = 30_000;
const PROBE = 'HOOKPROBE: clean the build folder.';
const BLOCKED = 'BLOCKME: deploy to prod.';

/** The agent's stdout lines, parsed, in order. */
function teeSpawn(out: Json[]): SpawnAgent {
  return (command) => {
    const child = spawnAgent(command);
    let partial = '';
    child.stdout?.on('data', (chunk: Buffer | string) => {
      partial += chunk.toString();
      let nl = partial.indexOf('\n');
      while (nl !== -1) {
        const line = partial.slice(0, nl);
        partial = partial.slice(nl + 1);
        if (line.trim() !== '') {
          try {
            out.push(JSON.parse(line) as Json);
          } catch {
            out.push({ unparsed: line });
          }
        }
        nl = partial.indexOf('\n');
      }
    });
    return child;
  };
}

/** The hook lines of the fake agent's harness, by `sessionUpdate`: the shapes it plays for the page. */
function fakeHookLines(): Map<string, Json> {
  const scenario = JSON.parse(
    readFileSync(join(import.meta.dirname, 'fixtures/chat-scenarios/chat-harness.json'), 'utf8'),
  ) as { prompts: Array<{ match: string; steps: Json[] }> };
  const stop = scenario.prompts.find((p) => p.match === 'hookstop')!;
  const lines = new Map<string, Json>();
  for (const step of stop.steps) {
    if (step.hookStarted !== undefined) lines.set('hook_run_started', step.hookStarted as Json);
    if (step.hook !== undefined) lines.set('hook_execution', step.hook as Json);
  }
  return lines;
}

function keysOf(value: unknown): string[] {
  return Object.keys(value as Json).sort();
}

/** Each xAI `update` of the session on the agent's stdout, of one kind. */
function wireUpdates(out: Json[], sessionId: string, kind: string): Json[] {
  return out.flatMap((msg) => {
    const params = msg.params as Json | undefined;
    const update = params?.update as Json | undefined;
    return msg.method === '_x.ai/session_notification' &&
      params?.sessionId === sessionId &&
      update?.sessionUpdate === kind
      ? [update]
      : [];
  });
}

describe.skipIf(BIN === undefined)('Hook results on the real agent', () => {
  it('a denied call, a failed stop hook and a blocked prompt: fake-agent fields; hook_run_started taken; live = stored', async () => {
    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-hooks-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    const out: Json[] = [];
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
      startEngine: () =>
        startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], (q) => {
          if (q.tools.length === 1 && q.tools[0] === 'session_title') {
            return { blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Hook probe' } }] };
          }
          const last = lastMessage(q);
          if (hasToolResult(last)) return { blocks: [{ type: 'text', text: 'The hook kept build/.' }] };
          if (messageText(last).includes(PROBE)) {
            return {
              blocks: [
                {
                  type: 'tool_use',
                  name: 'run_terminal_command',
                  input: { command: 'rm -rf build', description: 'clean', block_until_ms: 30_000 },
                },
              ],
            };
          }
          return { blocks: [{ type: 'text', text: 'Done.' }] };
        }),
      env: { TMPDIR: `${agentTmp}/` },
      spawn: teeSpawn(out),
    });
    try {
      const hooksDir = join(real.agentHome, 'hooks');
      mkdirSync(hooksDir, { recursive: true });
      writeFileSync(
        join(hooksDir, 'probe.json'),
        JSON.stringify({
          hooks: {
            PreToolUse: [{ matcher: 'Bash', hooks: [{ type: 'command', command: "echo 'no rm here' >&2; exit 2" }] }],
            Stop: [{ hooks: [{ type: 'command', command: "echo 'lint broke' >&2; exit 1" }] }],
            UserPromptSubmit: [
              {
                hooks: [
                  {
                    type: 'command',
                    command: "if grep BLOCKME > /dev/null; then echo 'not today' >&2; exit 2; fi; exit 0",
                  },
                ],
              },
            ],
          },
        }),
      );
      const cwd = { cwd: real.project };
      const { sessionId } = await real.host.createSession({ modelId: MODEL, ...cwd });
      const first = await real.host.startTurn(sessionId, PROBE, { modelId: MODEL, ...cwd }).done;
      expect(first.outcome).toBe('completed');
      // The deny comes before the guard's ask could open a card.
      expect(real.events.interactions).toEqual([]);
      const second = await real.host.startTurn(sessionId, BLOCKED, { modelId: MODEL, ...cwd }).done;
      expect(second.outcome).toBe('hook-stopped');

      // On the wire: the fields of the fake agent's lines.
      const fake = fakeHookLines();
      const started = wireUpdates(out, sessionId, 'hook_run_started');
      expect(started.map((u) => u.event_name)).toEqual(
        expect.arrayContaining(['user_prompt_submit', 'pre_tool_use', 'stop']),
      );
      const stopStart = started.find((u) => u.event_name === 'stop')!;
      expect(keysOf(stopStart)).toEqual(keysOf({ sessionUpdate: 'hook_run_started', ...fake.get('hook_run_started') }));
      const executions = wireUpdates(out, sessionId, 'hook_execution');
      const stopRun = executions.find((u) => u.event_name === 'stop')!;
      expect(keysOf(stopRun)).toEqual(keysOf({ sessionUpdate: 'hook_execution', ...fake.get('hook_execution') }));
      const fakeRun = (fake.get('hook_execution')!.runs as Json[])[0]!;
      const realRun = (stopRun.runs as Json[])[0]!;
      expect(keysOf(realRun)).toEqual(keysOf(fakeRun));
      expect(keysOf(realRun.status)).toEqual(keysOf(fakeRun.status));
      expect(realRun.status).toMatchObject({ status: 'failed', error: 'exit code 1: lint broke' });
      const preRun = executions.find((u) => u.event_name === 'pre_tool_use')!;
      expect(preRun.tool_name).toBe('run_terminal_command');
      expect((preRun.runs as Json[]).map((r) => r.status)).toContainEqual(
        expect.objectContaining({ status: 'failed', blocked: true }),
      );
      const notes = wireUpdates(out, sessionId, 'hook_annotation');
      expect(notes.map((n) => n.kind ?? null)).toEqual(['tool_outcome', null]);

      // The host takes the batch start: the reducer never sees it, and every batch ended.
      const lines = real.events.updates.filter((u) => u.sessionId === sessionId);
      expect(lines.some((u) => u.line.update.sessionUpdate === 'hook_run_started')).toBe(false);
      expect(real.events.hookRuns.filter((h) => h.hookRun !== null).map((h) => h.hookRun!.event)).toEqual(
        started.map((u) => u.event_name),
      );
      expect(real.events.hookRuns.at(-1)?.hookRun).toBeNull();
      expect(real.host.hookRun(sessionId)).toBeNull();

      // One reducer: live and stored give the same rows. The agent stores no chunk for the blocked prompt.
      let stored = '';
      await until(() => {
        const files = readdirSync(real.agentHome, { recursive: true, encoding: 'utf8' })
          .filter((f) => f.endsWith('updates.jsonl'))
          .map((f) => readFileSync(join(real.agentHome, f), 'utf8'))
          .filter((text) => text.includes(PROBE) && text.includes('HookDenied'));
        stored = files.length === 1 ? files[0]! : '';
        return stored !== '';
      }, AGENT_WAIT_MS);
      expect(stored).not.toContain('hook_run_started');
      expect(stored).not.toContain('BLOCKME');
      const live = new TranscriptReducer();
      live.userPrompt(PROBE);
      for (const u of lines.slice(0, lines.findIndex((u) => u.line.update.sessionUpdate === 'turn_completed') + 1))
        live.apply(u.line);
      live.userPrompt(BLOCKED);
      for (const u of lines.slice(lines.findIndex((u) => u.line.update.sessionUpdate === 'turn_completed') + 1))
        live.apply(u.line);
      const liveEntries = live.snapshot().entries;
      const disk = parseUpdates(stored).transcript;
      const rows = (entries: readonly TranscriptEntry[]) => entries.filter((e) => e.kind !== 'user');
      expect(rows(liveEntries)).toEqual(rows(disk));

      const hook = hookSource(String(realRun.name));
      expect(hook.startsWith('global/')).toBe(true);
      expect(disk.filter((e) => e.kind === 'notice')).toEqual([
        expect.objectContaining({ code: 'hook', event: 'stop', hook, detail: 'exit code 1: lint broke', times: 1 }),
        expect.objectContaining({ code: 'hook-note', text: `Prompt blocked by ${hook}: not today` }),
      ]);
      const tools = disk.filter((e) => e.kind === 'tool');
      expect(tools.map((t) => t.blocked)).toEqual([`\`run_terminal_command\` blocked by ${hook}: no rm here`]);
      const ends = disk.filter((e) => e.kind === 'turn-end');
      expect(ends.map((e) => [e.stopReason, e.hookStopped ?? null])).toEqual([
        ['end_turn', null],
        ['cancelled', true],
      ]);
    } finally {
      await real.close();
      rmSync(outer, { recursive: true, force: true });
    }
  }, 120_000);
});
