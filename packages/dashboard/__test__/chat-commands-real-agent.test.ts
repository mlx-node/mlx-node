/**
 * Slash commands on the real agent binary (`MLX_AGENT_E2E_BIN`) with a scripted engine and no model: the list the host
 * pulls after the load and the push after an answer, read into the rows a Work page may send (the allowed built-ins
 * and a skill from the Work folder); the fake agent plays the same rows; a refused name never reaches the agent; a
 * skill goes as typed and the agent expands it; a built-in is a host turn with no model call; and the live rows equal
 * the rows read back from disk. Skipped without the variable.
 */

import { mkdirSync, mkdtempSync, readdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { parseUpdates, TranscriptReducer } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import type { AcpClient } from '../src/chat/acp/client.js';
import { AgentHostError, type AgentHost } from '../src/chat/agent/agent-host.js';
import { AGENT_ARGS } from '../src/chat/agent/profiles.js';
import { AGENT_BUILTINS } from '../src/chat/commands.js';
import { fakeHost, noProviders, until, type FakeHost, type Json } from './helpers/chat-fake.js';
import { lastMessage, messageText, startScriptedEngine, type ModelReply } from './helpers/scripted-engine.js';

const BIN = process.env.MLX_AGENT_E2E_BIN;
const MODEL = 'qwen3.5-4b';
const AGENT_WAIT_MS = 30_000;
const SKILL = 'probe-skill';
const SKILL_BODY = 'SKILL BODY MARKER';

const VIEW = [
  {
    name: 'deep-research',
    kind: 'builtin',
    description: 'Research with bounded parallel agents, cross-check evidence, and write a cited report',
    hint: '<query>',
  },
  {
    name: 'workflow',
    kind: 'builtin',
    description: 'Launch a saved workflow, list runs, or manage a run (pause, resume, stop, save)',
    hint: '<name> [--agent-budget N] [--effort LEVEL] [args] | runs | pause|resume|stop|save [name]',
  },
  {
    name: 'goal',
    kind: 'builtin',
    description: 'Set, manage, or check an autonomous goal',
    hint: '<objective> [--budget <tokens>] | status | pause | resume | clear',
  },
  { name: 'loop', kind: 'builtin', description: 'Run a prompt on a recurring interval', hint: '[interval] <prompt>' },
  { name: SKILL, kind: 'skill', description: 'Probe the folder', hint: '[path]' },
];

function writeSkill(project: string): void {
  const dir = join(project, '.mlx-agent', 'skills', SKILL);
  mkdirSync(dir, { recursive: true });
  writeFileSync(
    join(dir, 'SKILL.md'),
    `---\nname: ${SKILL}\ndescription: Probe the folder\nargument-hint: "[path]"\n---\n${SKILL_BODY}\n`,
  );
}

function clientOf(host: AgentHost): AcpClient {
  return (Reflect.get(host, 'running') as { process: { client: AcpClient } }).process.client;
}

async function rowsOf(host: AgentHost, sessionId: string): Promise<Json[]> {
  const reply = (await clientOf(host).request('_x.ai/commands/list', { sessionId }, null)) as { commands: Json[] };
  return reply.commands;
}

function codeOf(run: () => unknown): string | null {
  try {
    run();
  } catch (error) {
    return error instanceof AgentHostError ? error.code : String(error);
  }
  return null;
}

function storedText(agentHome: string, marker: string): string {
  const files = readdirSync(agentHome, { recursive: true, encoding: 'utf8' })
    .filter((f) => f.endsWith('updates.jsonl'))
    .map((f) => readFileSync(join(agentHome, f), 'utf8'))
    .filter((text) => text.includes(marker));
  return files.length === 1 ? files[0]! : '';
}

/** The user and answer rows, as a reopened or a live thread draws them. */
function bubbles(entries: readonly TranscriptEntry[]): Array<{ kind: string; text: string }> {
  return entries.flatMap((e) => (e.kind === 'user' || e.kind === 'assistant' ? [{ kind: e.kind, text: e.text }] : []));
}

describe.skipIf(BIN === undefined)('slash commands on the real agent', () => {
  it('the list read from pull and push, the fake’s rows, a refused name, a skill, a host turn, live rows equal disk rows', async () => {
    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-commands-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    /** The text of the last user message of each model call that was not a title call. */
    const asked: string[] = [];
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
      startEngine: () =>
        startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], (q): ModelReply => {
          if (q.tools.length === 1 && q.tools[0] === 'session_title') {
            return {
              blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Commands probe' } }],
            };
          }
          asked.push(messageText(lastMessage(q)));
          return { blocks: [{ type: 'text', text: 'Read it.' }] };
        }),
      env: { TMPDIR: `${agentTmp}/` },
    });
    let fake: FakeHost | null = null;
    try {
      writeSkill(real.project);
      const { sessionId } = await real.host.createSession({ modelId: MODEL, cwd: real.project });
      const load = { modelId: MODEL, cwd: real.project };

      // 1. The pull after the load: no turn ran yet, and the push before `session/new` answered was dropped.
      await until(() => real.events.commands.length > 0, AGENT_WAIT_MS);
      expect(real.events.commands).toEqual([{ sessionId, commands: VIEW }]);
      const rows = await rowsOf(real.host, sessionId);
      const builtinNames = new Set(AGENT_BUILTINS.flatMap((b) => [b.name, ...b.aliases]));
      const skillless = (list: Json[]) => list.filter((r) => (r._meta as Json | undefined)?.bareName === undefined);
      // The agent lists the app's own `read-pdf` as a user workflow; the host drops that row from the view.
      const appRows = rows.filter((r) => r.name === 'read-pdf').map((r) => (r._meta as Json).workflowPath);
      expect(appRows).toEqual([join(real.agentHome, 'workflows', 'read-pdf.rhai')]);
      const agentBuiltins = skillless(rows).filter((r) => r.name !== 'read-pdf');
      expect(agentBuiltins.map((r) => r.name).filter((name) => !builtinNames.has(name as string))).toEqual([]);

      // 2. The fake agent plays the same rows for a Work session with the same skill.
      fake = await fakeHost('chat-harness', {
        mode: 'work',
        mutate: (scenario) => {
          const work = scenario.work as Json;
          work.session = {
            ...(work.session as Json),
            commands: { skills: [{ name: SKILL, description: 'Probe the folder', hint: '[path]' }] },
          };
        },
      });
      const fakeSession = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      await until(() => fake!.events.commands.length > 0);
      expect(fake.events.commands.at(-1)!.commands).toEqual(VIEW);
      const fakeRows = await rowsOf(fake.host, fakeSession.sessionId);
      expect(skillless(fakeRows)).toEqual(agentBuiltins);
      const skillRow = (list: Json[]) => {
        const row = list.find((r) => r.name === SKILL)!;
        return { ...row, _meta: Object.keys(row._meta as Json).sort() };
      };
      expect(skillRow(fakeRows)).toEqual(skillRow(rows));

      // 3. Refused on the host before anything is sent.
      for (const text of ['/yolo on', '/compact', '/hooks-trust', '/status', '/nope', '/read-pdf {"pdf":"p1"}'])
        expect([text, codeOf(() => real.host.startTurn(sessionId, text, load))]).toEqual([text, 'slash-text']);

      // 4. The push after an answer: the list the host holds is replaced by the agent's.
      const session = (Reflect.get(real.host, 'sessions') as Map<string, { commands: unknown }>).get(sessionId)!;
      session.commands = [];
      const marks: Array<[number, string]> = [];
      const turn = async (text: string) => {
        marks.push([real.events.updates.length, text]);
        const started = real.host.startTurn(sessionId, text, load);
        expect([text, (await started.done).outcome]).toEqual([text, 'completed']);
      };
      await turn('hello');
      await until(() => real.events.commands.length === 2, AGENT_WAIT_MS);
      expect(real.events.commands.at(-1)).toEqual({ sessionId, commands: VIEW });

      // 5. A skill goes as typed; the agent adds its body for the model.
      const before = asked.length;
      await turn(`/${SKILL} src`);
      expect(asked.length).toBe(before + 1);
      expect(asked.at(-1)).toContain(`/${SKILL} src`);
      expect(asked.at(-1)).toContain(SKILL_BODY);

      // 6. A built-in is the agent's host turn: no model call, its answer on the stream.
      await turn('/workflow runs');
      await turn('/goal');
      expect(asked.length).toBe(before + 1);

      // 7. Live rows (the host's typed row, then the agent's lines) equal the rows read back from disk.
      const live = new TranscriptReducer();
      const lines = real.events.updates;
      let at = 0;
      for (const [mark, text] of marks) {
        for (; at < mark; at++) if (lines[at]!.sessionId === sessionId) live.apply(lines[at]!.line);
        live.userPrompt(text);
      }
      for (; at < lines.length; at++) if (lines[at]!.sessionId === sessionId) live.apply(lines[at]!.line);
      const liveRows = bubbles(live.snapshot().entries);
      expect(liveRows.map((r) => r.text)).toEqual([
        'hello',
        'Read it.',
        `/${SKILL} src`,
        'Read it.',
        '/workflow runs',
        expect.stringContaining('No workflow runs'),
        '/goal',
        expect.stringContaining('No goal'),
      ]);
      const disk = () => bubbles(parseUpdates(storedText(real.agentHome, '/workflow runs')).transcript);
      await until(() => disk().length === liveRows.length, AGENT_WAIT_MS);
      expect(disk()).toEqual(liveRows);
    } finally {
      await fake?.close();
      await real.close();
      rmSync(outer, { recursive: true, force: true });
    }
  }, 120_000);
});
