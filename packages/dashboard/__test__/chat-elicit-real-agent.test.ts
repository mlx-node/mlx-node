/**
 * MCP input on the real agent binary (`MLX_AGENT_E2E_BIN`) with a scripted engine and no model: a stdio MCP
 * server in the agent home's config asks for input while its tool runs. The request and its live signals have the
 * shape the fake agent plays; a refused Send keeps the card; the server gets the exact content, a decline and a
 * cancel. A link request: the card holds the address, Open page reaches the server as a bare accept, and the server's
 * "done" comes back as `_x.ai/mcp/elicit_complete`, counted only for the page this app opened. Skipped without the
 * variable.
 */

import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { spawnAgent, type SpawnAgent } from '../src/chat/acp/process.js';
import { AGENT_ARGS } from '../src/chat/agent/profiles.js';
import type { McpInteraction } from '../src/chat/types.js';
import { fakeHost, noProviders, until, type Json } from './helpers/chat-fake.js';
import { hasToolResult, lastMessage, messageText, startScriptedEngine } from './helpers/scripted-engine.js';

const BIN = process.env.MLX_AGENT_E2E_BIN;
const MODEL = 'qwen3.5-4b';
const AGENT_WAIT_MS = 30_000;
const SERVER = join(import.meta.dirname, 'helpers/elicit-mcp-server.mjs');
const MESSAGE = 'Which repository should the issue go to?';
/** The keys of `_x.ai/mcp/elicit` in form mode, as the fake agent sends them. */
const FORM_KEYS = ['sessionId', 'toolCallId', 'serverName', 'message', 'mode', 'requestedSchema'];
/** The keys of `_x.ai/mcp/elicit` in link mode, as the fake agent sends them. */
const LINK_KEYS = ['sessionId', 'toolCallId', 'serverName', 'message', 'mode', 'url', 'elicitationId'];

/** Every JSON line the agent writes to the host, in order. */
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
        nl = partial.indexOf('\n');
        if (line.trim() === '') continue;
        try {
          out.push(JSON.parse(line) as Json);
        } catch {
          out.push({ unparsed: line });
        }
      }
    });
    return child;
  };
}

function lines(path: string): Json[] {
  if (!existsSync(path)) return [];
  return readFileSync(path, 'utf8')
    .split('\n')
    .filter((line) => line !== '')
    .map((line) => JSON.parse(line) as Json);
}

describe.skipIf(BIN === undefined)('MCP input on the real agent', () => {
  it('form: the card, a refused then a good Send, Decline and Stop; the server gets each answer', async () => {
    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-elicit-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    const wire: Json[] = [];
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
      startEngine: () =>
        startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], (q) => {
          if (q.tools.length === 1 && q.tools[0] === 'session_title') {
            return { blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Elicit' } }] };
          }
          const last = lastMessage(q);
          if (hasToolResult(last) && messageText(last).includes('elicit:')) {
            return { blocks: [{ type: 'text', text: 'Done.' }] };
          }
          if (hasToolResult(last)) {
            return {
              blocks: [
                { type: 'tool_use', name: 'use_tool', input: { tool_name: 'probe__file_issue', tool_input: {} } },
              ],
            };
          }
          return { blocks: [{ type: 'tool_use', name: 'search_tool', input: { query: 'probe file issue' } }] };
        }),
      env: { TMPDIR: `${agentTmp}/` },
      spawn: teeSpawn(wire),
    });
    const record = join(outer, 'server.jsonl');
    const config = join(outer, 'server.json');
    writeFileSync(
      config,
      JSON.stringify({
        message: MESSAGE,
        record,
        requestedSchema: {
          type: 'object',
          properties: {
            repo: { type: 'string', title: 'Repository', enum: ['shop-api', 'shop-web'] },
            title: { type: 'string', title: 'Title', minLength: 3 },
            labels: { type: 'array', title: 'Labels', items: { type: 'string', enum: ['bug', 'feature'] } },
            priority: { type: 'integer', title: 'Priority', minimum: 1, maximum: 5 },
            mine: { type: 'boolean', title: 'Assign to me', default: true },
          },
          required: ['repo', 'title'],
        },
      }),
    );
    writeFileSync(
      join(real.agentHome, 'config.toml'),
      `[mcp_servers.probe]\ncommand = ${JSON.stringify(process.execPath)}\nargs = [${JSON.stringify(SERVER)}, ${JSON.stringify(config)}]\n`,
    );
    const answers = (): Json[] => lines(record).filter((line) => line.kind === 'elicit-answer');
    let seen = 0;
    /** Allows each tool call once until the MCP card opens. */
    const nextCard = async (sessionId: string): Promise<McpInteraction> => {
      for (;;) {
        await until(() => real.events.interactions.length > seen, AGENT_WAIT_MS);
        const next = real.events.interactions[seen]!.interaction;
        seen += 1;
        if (next.kind === 'mcp-input') return next;
        expect(next.kind).toBe('permission');
        real.host.answer(sessionId, next.id, 'allow-once');
      }
    };
    try {
      const { sessionId } = await real.host.createSession({ modelId: MODEL, cwd: real.project });
      const where = { modelId: MODEL, cwd: real.project };

      const first = real.host.startTurn(sessionId, 'File an issue.', where);
      const card = await nextCard(sessionId);
      expect(card).toMatchObject({ kind: 'mcp-input', mode: 'form', server: 'probe', message: MESSAGE });
      expect(card.toolCallId.startsWith('mcp-elicit-')).toBe(true);
      expect(card.unusable).toBeUndefined();
      expect(card.form?.leftOut).toBe(0);
      expect(card.form?.fields.map((f) => [f.name, f.title, f.type, f.required])).toEqual([
        ['repo', 'Repository', 'choice', true],
        ['title', 'Title', 'text', true],
        ['labels', 'Labels', 'choices', false],
        ['priority', 'Priority', 'integer', false],
        ['mine', 'Assign to me', 'boolean', false],
      ]);
      expect(card.form?.fields[4]).toMatchObject({ initial: true });

      const request = wire.find((m) => m.method === '_x.ai/mcp/elicit')!;
      const params = request.params as Json;
      expect(Object.keys(params)).toEqual(FORM_KEYS);
      expect(params).toMatchObject({ sessionId, toolCallId: card.toolCallId, serverName: 'probe', mode: 'form' });
      const signals = (): Json[] =>
        wire
          .filter((m) => m.method === '_x.ai/session_notification')
          .map((m) => (m.params as Json).update as Json)
          .filter((u) => u.tool_call_id === card.toolCallId);
      expect(signals()).toEqual([
        { sessionUpdate: 'pending_interaction', tool_call_id: card.toolCallId, kind: 'mcp_elicitation' },
      ]);
      expect(wire.indexOf(request)).toBeGreaterThan(
        wire.findIndex((m) => JSON.stringify(m).includes('"pending_interaction"')),
      );

      expect(real.host.answerElicit(sessionId, card.id, { mcp: 'accept', values: [1, 'Ca', [0], '9', false] })).toEqual(
        [null, { code: 'minLength', n: 3 }, null, { code: 'max', n: 5 }, null],
      );
      expect(real.host.openInteractions(sessionId).map((i) => i.id)).toEqual([card.id]);
      expect(answers()).toEqual([]);
      expect(
        real.host.answerElicit(sessionId, card.id, { mcp: 'accept', values: [1, 'Cart', [0], '3', false] }),
      ).toBeNull();
      expect((await first.done).outcome).toBe('completed');
      expect(answers()).toEqual([
        {
          kind: 'elicit-answer',
          answer: {
            action: 'accept',
            content: { repo: 'shop-web', title: 'Cart', labels: ['bug'], priority: 3, mine: false },
          },
        },
      ]);
      // The agent's end of the wait comes after the answer and closes nothing more.
      await until(() => signals().length === 2, AGENT_WAIT_MS);
      expect(signals()[1]).toEqual({ sessionUpdate: 'interaction_resolved', tool_call_id: card.toolCallId });
      expect(real.events.closed.filter((c) => c.id === card.id)).toEqual([
        { sessionId, id: card.id, reason: 'answered' },
      ]);
      expect(real.events.mcp).toEqual([{ sessionId, closed: { server: 'probe', mode: 'form', outcome: 'accept' } }]);

      const second = real.host.startTurn(sessionId, 'File another.', where);
      const declined = await nextCard(sessionId);
      expect(declined.id).not.toBe(card.id);
      expect(real.host.answerElicit(sessionId, declined.id, { mcp: 'decline' })).toBeNull();
      expect((await second.done).outcome).toBe('completed');
      expect(answers()[1]).toEqual({ kind: 'elicit-answer', answer: { action: 'decline' } });
      expect(real.events.mcp.map((m) => m.closed.outcome)).toEqual(['accept', 'decline']);

      const third = real.host.startTurn(sessionId, 'And one more.', where);
      const stopped = await nextCard(sessionId);
      expect(real.host.cancel(sessionId)).toBe(true);
      expect((await third.done).outcome).toBe('cancelled');
      expect(real.events.closed.filter((c) => c.id === stopped.id)).toEqual([
        { sessionId, id: stopped.id, reason: 'cancelled' },
      ]);
      expect(real.events.mcp.map((m) => m.closed.outcome)).toEqual(['accept', 'decline', 'cancel']);
      await until(() => answers().length === 3, AGENT_WAIT_MS);
      expect(answers()[2]).toEqual({ kind: 'elicit-answer', answer: { action: 'cancel' } });

      // The live signals are never update lines, and no typed value reaches the cards or their close lines.
      expect(JSON.stringify(real.events.updates)).not.toContain('mcp_elicitation');
      const shown = JSON.stringify([real.events.interactions, real.events.closed, real.events.mcp]);
      expect(shown).not.toContain('Cart');
    } finally {
      await real.close();
      rmSync(outer, { recursive: true, force: true });
    }
  }, 120_000);

  it('link: the card, Open page as a bare accept, and the server done only for the page this app opened', async () => {
    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-elicit-link-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    const wire: Json[] = [];
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
      startEngine: () =>
        startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], (q) => {
          if (q.tools.length === 1 && q.tools[0] === 'session_title') {
            return { blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Link' } }] };
          }
          const last = lastMessage(q);
          if (hasToolResult(last) && messageText(last).includes('elicit:')) {
            return { blocks: [{ type: 'text', text: 'Done.' }] };
          }
          if (hasToolResult(last)) {
            return {
              blocks: [
                { type: 'tool_use', name: 'use_tool', input: { tool_name: 'probe__file_issue', tool_input: {} } },
              ],
            };
          }
          return { blocks: [{ type: 'tool_use', name: 'search_tool', input: { query: 'probe file issue' } }] };
        }),
      env: { TMPDIR: `${agentTmp}/` },
      spawn: teeSpawn(wire),
    });
    const record = join(outer, 'server.jsonl');
    const config = join(outer, 'server.json');
    const url = 'https://github.com/login/oauth/authorize?client_id=abc&state=xyz';
    writeFileSync(config, JSON.stringify({ mode: 'url', message: 'Sign in to GitHub.', url, record, complete: true }));
    writeFileSync(
      join(real.agentHome, 'config.toml'),
      `[mcp_servers.probe]\ncommand = ${JSON.stringify(process.execPath)}\nargs = [${JSON.stringify(SERVER)}, ${JSON.stringify(config)}]\n`,
    );
    const answers = (): Json[] => lines(record).filter((line) => line.kind === 'elicit-answer');
    const completes = (): Json[] => wire.filter((m) => m.method === '_x.ai/mcp/elicit_complete');
    let seen = 0;
    const nextCard = async (sessionId: string): Promise<McpInteraction> => {
      for (;;) {
        await until(() => real.events.interactions.length > seen, AGENT_WAIT_MS);
        const next = real.events.interactions[seen]!.interaction;
        seen += 1;
        if (next.kind === 'mcp-input') return next;
        expect(next.kind).toBe('permission');
        real.host.answer(sessionId, next.id, 'allow-once');
      }
    };
    try {
      const { sessionId } = await real.host.createSession({ modelId: MODEL, cwd: real.project });
      const where = { modelId: MODEL, cwd: real.project };

      const first = real.host.startTurn(sessionId, 'Sign in.', where);
      const card = await nextCard(sessionId);
      expect(card).toMatchObject({
        kind: 'mcp-input',
        mode: 'url',
        server: 'probe',
        message: 'Sign in to GitHub.',
        link: { url, host: 'github.com', secure: true },
      });
      expect(card.unusable).toBeUndefined();
      const request = wire.find((m) => m.method === '_x.ai/mcp/elicit')!;
      const params = request.params as Json;
      expect(Object.keys(params)).toEqual(LINK_KEYS);
      expect(params).toMatchObject({ sessionId, toolCallId: card.toolCallId, serverName: 'probe', mode: 'url', url });
      expect(params.elicitationId).toBe('link-1');

      expect(real.host.answerElicit(sessionId, card.id, { mcp: 'accept' })).toBeNull();
      expect((await first.done).outcome).toBe('completed');
      expect(answers()).toEqual([{ kind: 'elicit-answer', answer: { action: 'accept' } }]);
      expect(real.events.mcp).toEqual([{ sessionId, closed: { server: 'probe', mode: 'url', outcome: 'accept' } }]);
      await until(() => real.events.mcpDone.length === 1, AGENT_WAIT_MS);
      expect(real.events.mcpDone).toEqual([{ sessionId, server: 'probe' }]);
      expect(completes().map((m) => m.params)).toEqual([{ sessionId, elicitationId: 'link-1', serverName: 'probe' }]);

      // A declined link: the server still says done, and no line comes of it.
      const second = real.host.startTurn(sessionId, 'Sign in again.', where);
      const declined = await nextCard(sessionId);
      expect(real.host.answerElicit(sessionId, declined.id, { mcp: 'decline' })).toBeNull();
      expect((await second.done).outcome).toBe('completed');
      expect(answers()[1]).toEqual({ kind: 'elicit-answer', answer: { action: 'decline' } });
      await until(() => completes().length === 2, AGENT_WAIT_MS);
      expect((completes()[1]!.params as Json).elicitationId).toBe('link-2');
      await new Promise((resolve) => setTimeout(resolve, 200));
      expect(real.events.mcpDone).toHaveLength(1);
      expect(real.events.mcp.map((m) => m.closed.outcome)).toEqual(['accept', 'decline']);
      // The done line is taken by the host: never an update line.
      expect(JSON.stringify(real.events.updates)).not.toContain('elicit_complete');
    } finally {
      await real.close();
      rmSync(outer, { recursive: true, force: true });
    }
  }, 120_000);
});
