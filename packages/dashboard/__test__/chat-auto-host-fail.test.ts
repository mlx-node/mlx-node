/**
 * Auto in the Work host when the check itself fails: a decision or a disk check that throws is a card
 * with the `unread` reason, never an allow.
 */

import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import type { PermissionInteraction } from '../src/chat/types.js';
import { fakeHost, noProviders, until, type Json } from './helpers/chat-fake.js';

const fail = vi.hoisted(() => ({ decide: false, facts: false }));

vi.mock('../src/chat/agent/auto.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/chat/agent/auto.js')>();
  return {
    ...actual,
    decideAuto: (...args: Parameters<typeof actual.decideAuto>) => {
      if (fail.decide) throw new Error('decide failed');
      return actual.decideAuto(...args);
    },
    autoFacts: (...args: Parameters<typeof actual.autoFacts>) => {
      if (fail.facts) return Promise.reject(new Error('facts failed'));
      return actual.autoFacts(...args);
    },
  };
});

const MODEL = 'qwen3.5-4b';

function bashStep(command: string): Json {
  return {
    permission: {
      toolName: 'run_terminal_command',
      kind: 'execute',
      title: `Execute \`${command}\``,
      rawInput: { variant: 'Bash', command, description: 'run', is_background: false },
      output: 'ok',
    },
  };
}

afterEach(() => {
  fail.decide = false;
  fail.facts = false;
});

describe('an Auto check that fails', () => {
  it.each([
    ['the decision throws', 'decide'],
    ['the disk check rejects', 'facts'],
  ] as const)('%s: the card with the unread reason, no allow and no history line', async (_name, which) => {
    fail[which] = true;
    let lines = 0;
    const fake = await fakeHost('work-permission', {
      mode: 'work',
      providers: noProviders(),
      mutate: (scenario) => {
        scenario.prompts = [
          { match: '', steps: [bashStep('ls'), { text: 'Done.', chunk: 5 }, { end: { stopReason: 'end_turn' } }] },
        ];
      },
      permission: { modeOf: () => 'auto', watched: () => true },
      onModeAllowed: async () => {
        lines += 1;
        return true;
      },
    });
    try {
      const id = (await fake.host.createSession({ modelId: MODEL, cwd: fake.project })).sessionId;
      const turn = fake.host.startTurn(id, 'go', { modelId: MODEL, cwd: fake.project });
      await until(() => fake.events.interactions.length === 1, 5_000);
      const card = fake.events.interactions[0]!.interaction as PermissionInteraction;
      expect([card.autoAsk, card.options.map((o) => o.optionId), lines]).toEqual([
        'unread',
        ['always-allow', 'allow-once', 'reject-once'],
        0,
      ]);
      fake.host.answer(id, card.id, 'reject-once');
      expect((await turn.done).outcome).toBe('denied');
    } finally {
      await fake.close();
    }
  });
});
