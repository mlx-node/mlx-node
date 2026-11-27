/**
 * The Context panel on the host and the service: `_x.ai/session/info` read only while the conversation is loaded, under
 * a soft deadline that never ends the agent, and never a start or a load for a panel.
 */

import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vite-plus/test';

import type { SessionInfoFacts } from '../src/chat/acp/messages.js';
import { conversationContext } from '../src/chat/service.js';
import { fakeHost, fakeService, until, type FakeHost, type FakeHostOptions, type Json } from './helpers/chat-fake.js';

const MODEL = 'qwen3.5-4b';
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const gates: string[] = [];
afterEach(() => {
  for (const dir of gates.splice(0)) rmSync(dir, { recursive: true, force: true });
});

function gate(): string {
  const dir = mkdtempSync(join(tmpdir(), 'mlx-context-gate-'));
  gates.push(dir);
  return join(dir, 'go');
}

async function withHost(name: string, run: (fake: FakeHost) => Promise<void>, options: FakeHostOptions = {}) {
  const fake = await fakeHost(name, options);
  try {
    await run(fake);
  } finally {
    await fake.close();
  }
}

const asked = (fake: { sent(): Json[] }, method: string): Json[] => fake.sent().filter((m) => m.method === method);
const starts = (fake: { records(): Json[] }): number => fake.records().filter((r) => r.kind === 'start').length;

/** One answer that leaves the context at `used` tokens. */
function counted(used: number, window = 262144): (scenario: Json) => void {
  return (scenario) => {
    scenario.initialize = { models: [{ modelId: MODEL, totalContextTokens: window }] };
    scenario.prompts = [
      {
        match: '',
        steps: [
          { text: 'Answer.', totalTokens: used },
          { turnCompleted: { inputTokens: 400, outputTokens: 30 } },
          { end: { stopReason: 'end_turn' } },
        ],
      },
    ];
  };
}

const facts: SessionInfoFacts = {
  agentName: 'mlx-chat',
  model: MODEL,
  apiBackend: 'Messages',
  toolDefinitionsCount: 2,
  systemPromptTokens: 432,
  toolDefinitionsTokens: 261,
  usageCategories: [],
  used: 5_000,
  total: 262144,
  messageTokens: 4_300,
  freeTokens: 257144,
  compactionCount: 1,
  autoCompactThresholdPercent: 85,
  turnIndex: 2,
};

describe('conversationContext', () => {
  it('the parts need every number and a window; the use is the agent’s when the parts are there', () => {
    expect(conversationContext(facts, 4_000)).toEqual({
      used: 5_000,
      parts: {
        systemTokens: 432,
        toolCount: 2,
        toolTokens: 261,
        messageTokens: 4_300,
        autoCompactPercent: 85,
        compactions: 1,
      },
    });
    expect(conversationContext(null, 4_000)).toEqual({ used: 4_000, parts: null });
    // The agent's answer when its session gave none: zeros and no window.
    expect(conversationContext({ ...facts, total: 0, used: 0 }, 4_000)).toEqual({ used: 4_000, parts: null });
    expect(conversationContext({ ...facts, messageTokens: null }, null)).toEqual({ used: null, parts: null });
    expect(conversationContext({ ...facts, autoCompactThresholdPercent: null }, 7)).toEqual({ used: 7, parts: null });
    expect(conversationContext({ ...facts, used: null }, 7)).toMatchObject({ used: 7, parts: { systemTokens: 432 } });
  });
});

describe('the Context panel on the host', () => {
  it('reads the loaded session, also while an answer runs; nothing for a session it does not hold', async () => {
    const go = gate();
    await withHost(
      'chat-basic',
      async (fake) => {
        expect(await fake.host.contextInfo('nobody')).toBeNull();
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        const before = asked(fake, '_x.ai/session/info').length;
        const turn = fake.host.startTurn(id, 'hello', { modelId: MODEL });
        await until(() => fake.host.isBusy(id) && asked(fake, 'session/prompt').length === 1, 5_000);
        await sleep(50);
        const info = await fake.host.contextInfo(id);
        expect(info).toMatchObject({
          used: 1_234,
          total: 262144,
          systemPromptTokens: 432,
          toolDefinitionsCount: 2,
          toolDefinitionsTokens: 261,
          messageTokens: 541,
          compactionCount: 0,
          autoCompactThresholdPercent: 85,
        });
        expect(asked(fake, '_x.ai/session/info').slice(before)).toEqual([
          expect.objectContaining({ params: { sessionId: id } }),
        ]);
        writeFileSync(go, '');
        expect((await turn.done).outcome).toBe('completed');
      },
      {
        mutate: (scenario) => {
          scenario.initialize = { models: [{ modelId: MODEL, totalContextTokens: 262144 }] };
          scenario.prompts = [
            {
              match: '',
              steps: [
                { text: 'Half', totalTokens: 1_234 },
                { holdOn: go },
                { text: ' done.' },
                { turnCompleted: { inputTokens: 400, outputTokens: 30 } },
                { end: { stopReason: 'end_turn' } },
              ],
            },
          ];
        },
      },
    );
  }, 20_000);

  it('a late answer misses the soft deadline: no parts, and the agent lives on', async () => {
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        const pid = fake.records().find((r) => r.kind === 'start')!.pid as number;
        const asked0 = Date.now();
        expect(await fake.host.contextInfo(id)).toBeNull();
        expect(Date.now() - asked0).toBeLessThan(250);
        await sleep(400);
        expect(starts(fake)).toBe(1);
        expect(() => process.kill(pid, 0)).not.toThrow();
        expect(fake.host.isHot(id)).toBe(true);
        expect((await fake.host.startTurn(id, 'hi', { modelId: MODEL }).done).outcome).toBe('completed');
      },
      {
        timings: { contextInfoMs: 100 },
        mutate: (scenario) => {
          (scenario.session as Json).infoLateMs = 300;
        },
      },
    );
  }, 20_000);

  it('nothing once the process stopped: the panel starts no agent', async () => {
    await withHost('chat-basic', async (fake) => {
      const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
      expect(await fake.host.contextInfo(id)).not.toBeNull();
      await fake.host.restart();
      expect(await fake.host.contextInfo(id)).toBeNull();
      await sleep(50);
      expect(starts(fake)).toBe(1);
    });
  }, 20_000);

  it('a session the new process has not loaded is not asked, while another one runs there', async () => {
    await withHost('chat-basic', async (fake) => {
      const { sessionId: old } = await fake.host.createSession({ modelId: MODEL });
      await fake.host.restart();
      const { sessionId: fresh } = await fake.host.createSession({ modelId: MODEL });
      expect(starts(fake)).toBe(2);
      const before = asked(fake, '_x.ai/session/info').length;
      expect(await fake.host.contextInfo(old)).toBeNull();
      expect(asked(fake, '_x.ai/session/info').slice(before)).toEqual([]);
      expect(await fake.host.contextInfo(fresh)).not.toBeNull();
    });
  }, 20_000);
});

describe('the Context panel through the service', () => {
  it('hot: the agent’s parts and use; cold after a restart: the stored use, no parts, no agent start', async () => {
    const fake = fakeService('chat-basic', { mutate: counted(4_321) });
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'hello there' });
      const id = conversation.id;
      await until(() => !fake.service.busyConversations().includes(id), 5_000);
      expect(await fake.service.context(id)).toEqual({
        used: 4_321,
        parts: {
          systemTokens: 432,
          toolCount: 2,
          toolTokens: 261,
          messageTokens: 4_321 - 693,
          autoCompactPercent: 85,
          compactions: 0,
        },
      });
      await fake.service.close();
      const next = fake.restart();
      const startsBefore = starts(fake);
      const infoBefore = asked(fake, '_x.ai/session/info').length;
      expect(await next.context(id)).toEqual({ used: 4_321, parts: null });
      await sleep(50);
      expect(starts(fake)).toBe(startsBefore);
      expect(asked(fake, '_x.ai/session/info').length).toBe(infoBefore);
      expect(asked(fake, 'session/load')).toEqual([]);
      await expect(next.context('no-such-id')).rejects.toMatchObject({ code: 'E_NOT_FOUND' });
    } finally {
      await fake.close();
    }
  }, 30_000);

  it('after Compact now the parts count the compaction and the use drops', async () => {
    const fake = fakeService('chat-basic', {
      mutate: (scenario) => {
        counted(41_000)(scenario);
        scenario.compactNow = { tokensBefore: 41_000, tokensAfter: 9_000 };
      },
    });
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'hello there' });
      const id = conversation.id;
      await until(() => !fake.service.busyConversations().includes(id), 5_000);
      expect(await fake.service.context(id)).toMatchObject({ used: 41_000, parts: { compactions: 0 } });
      expect(await fake.service.compact(id)).toEqual({});
      await until(() => !fake.service.busyConversations().includes(id), 5_000);
      expect(await fake.service.context(id)).toMatchObject({ used: 9_000, parts: { compactions: 1 } });
    } finally {
      await fake.close();
    }
  }, 30_000);
});
