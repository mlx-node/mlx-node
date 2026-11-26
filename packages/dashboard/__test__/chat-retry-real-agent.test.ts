/**
 * Retry rows on the real agent binary (`MLX_AGENT_E2E_BIN`) with a scripted engine and no model: the engine answers
 * a model call with a 503, the agent tries again and sends `retry_state`: once on the turn's first call, once on the
 * call after a call a hook denied. Its store writes that xAI line before the ACP line it still holds, so the reducer
 * places the row at the turn's end (or before the next call), the same live and from `updates.jsonl`. Skipped without
 * the variable.
 */

import { mkdirSync, mkdtempSync, readdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { parseUpdates, TranscriptReducer } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { AGENT_ARGS } from '../src/chat/agent/profiles.js';
import { fakeHost, noProviders, until } from './helpers/chat-fake.js';
import {
  hasToolResult,
  lastMessage,
  messageText,
  startScriptedEngine,
  type ModelReply,
} from './helpers/scripted-engine.js';

const BIN = process.env.MLX_AGENT_E2E_BIN;
const MODEL = 'qwen3.5-4b';
const AGENT_WAIT_MS = 30_000;
const PROMPT = 'Retry probe: say hello.';
const AFTER_CALL = 'Retry probe two: clean the build folder.';

const comparable = (entries: readonly TranscriptEntry[]): unknown[] =>
  entries.map((entry) => {
    if (entry.kind === 'user') return { kind: 'user', text: entry.text };
    if (entry.kind === 'assistant') {
      const { ts: _ts, ...rest } = entry;
      return rest;
    }
    return entry;
  });

describe.skipIf(BIN === undefined)('Retry rows on the real agent', () => {
  it('a 503 on the first call and one after a call: retry_state on the wire; each row at its end, live = stored', async () => {
    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-retry-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    let main = 0;
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
      startEngine: () =>
        startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], (q): ModelReply => {
          if (q.tools.length === 1 && q.tools[0] === 'session_title') {
            return { blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Retry probe' } }] };
          }
          main += 1;
          const last = lastMessage(q);
          if (hasToolResult(last)) {
            return main % 2 === 0 ? { status: 503, blocks: [] } : { blocks: [{ type: 'text', text: 'Kept it.' }] };
          }
          if (messageText(last).includes(AFTER_CALL)) {
            return {
              blocks: [
                { type: 'text', text: 'Cleaning.' },
                {
                  type: 'tool_use',
                  name: 'run_terminal_command',
                  input: { command: 'rm -rf build', description: 'clean', block_until_ms: 30_000 },
                },
              ],
            };
          }
          if (main === 1) return { status: 503, blocks: [] };
          return { blocks: [{ type: 'text', text: 'Hello.' }] };
        }),
      env: { TMPDIR: `${agentTmp}/` },
    });
    try {
      const hooksDir = join(real.agentHome, 'hooks');
      mkdirSync(hooksDir, { recursive: true });
      writeFileSync(
        join(hooksDir, 'probe.json'),
        JSON.stringify({
          hooks: {
            PreToolUse: [{ matcher: 'Bash', hooks: [{ type: 'command', command: "echo 'no rm here' >&2; exit 2" }] }],
          },
        }),
      );
      const cwd = { cwd: real.project };
      const { sessionId } = await real.host.createSession({ modelId: MODEL, ...cwd });
      expect((await real.host.startTurn(sessionId, PROMPT, { modelId: MODEL, ...cwd }).done).outcome).toBe('completed');
      const firstLines = real.events.updates.filter((u) => u.sessionId === sessionId).map((u) => u.line);
      expect((await real.host.startTurn(sessionId, AFTER_CALL, { modelId: MODEL, ...cwd }).done).outcome).toBe(
        'completed',
      );
      const lines = real.events.updates.filter((u) => u.sessionId === sessionId).map((u) => u.line);
      const retries = lines.filter((l) => l.xai && l.update.sessionUpdate === 'retry_state');
      expect(retries.map((l) => l.update.type)).toEqual(['retrying', 'retrying']);

      let stored = '';
      await until(() => {
        const files = readdirSync(real.agentHome, { recursive: true, encoding: 'utf8' })
          .filter((f) => f.endsWith('updates.jsonl'))
          .map((f) => readFileSync(join(real.agentHome, f), 'utf8'))
          .filter((text) => text.includes(AFTER_CALL) && text.split('turn_completed').length === 3);
        stored = files.length === 1 ? files[0]! : '';
        return stored !== '';
      }, AGENT_WAIT_MS);
      const kinds = stored
        .split('\n')
        .filter((line) => line !== '')
        .map((line) => {
          const update = (JSON.parse(line) as { params?: { update?: { sessionUpdate?: string } } }).params?.update;
          return String(update?.sessionUpdate);
        });
      // The store held the call's update when the second retry came, so it wrote the retry first.
      const call = kinds.indexOf('tool_call');
      expect(kinds.slice(call, call + 3)).toEqual(['tool_call', 'retry_state', 'tool_call_update']);

      const live = new TranscriptReducer();
      live.userPrompt(PROMPT);
      for (const line of firstLines) live.apply(line);
      live.userPrompt(AFTER_CALL);
      for (const line of lines.slice(firstLines.length)) live.apply(line);
      const shown = live.snapshot().entries;
      const disk = parseUpdates(stored).transcript;
      expect(comparable(shown)).toEqual(comparable(disk));
      expect(disk.map((e) => e.kind)).toEqual([
        'user',
        'assistant',
        'notice',
        'turn-end',
        'user',
        'assistant',
        'tool',
        'assistant',
        'notice',
        'turn-end',
      ]);
      expect(disk.filter((e) => e.kind === 'notice')).toEqual([
        expect.objectContaining({ code: 'retry', attempts: 1, outcome: 'recovered' }),
        expect.objectContaining({ code: 'retry', attempts: 1, outcome: 'recovered' }),
      ]);
    } finally {
      await real.close();
      rmSync(outer, { recursive: true, force: true });
    }
  }, 120_000);
});
