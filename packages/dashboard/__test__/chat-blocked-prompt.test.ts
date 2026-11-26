/**
 * A prompt a `UserPromptSubmit` hook blocked, through the service on the fake agent: the agent stores no chunk of it,
 * so the thread drops its bubble and keeps the hook's note and the end ("Stopped by a hook"), live and after a
 * Control Panel restart alike. The end carries the typed text for the page's message box; an automatic `continue`
 * carries none.
 */

import { mkdirSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import type { ChatService } from '../src/chat/service.js';
import type { ConversationEvent } from '../src/chat/types.js';
import { applyConversationEvent, type ConversationState } from '../ui/src/lib/chat-store.js';
import { fakeService, until, type Json } from './helpers/chat-fake.js';

const BLOCKED = 'hookblock: deploy to prod';

/** The harness waits seconds on a hook so its line can be seen; here the wait is short. A plain first answer. */
function quick(scenario: Json): void {
  for (const list of [scenario.prompts, (scenario.work as Json).prompts] as Json[][]) {
    for (const prompt of list) if (prompt.promptBlocked !== undefined) (prompt.promptBlocked as Json).waitMs = 20;
    list.unshift({ match: 'hello', steps: [{ text: 'Hi.' }, { end: { stopReason: 'end_turn' } }] });
  }
}

interface Page {
  state: () => ConversationState;
  current: () => ConversationState | null;
  events: ConversationEvent[];
  off: () => void;
}

function watch(service: ChatService, id: string): Page {
  const events: ConversationEvent[] = [];
  let state: ConversationState | null = null;
  const off = service.watchConversation(id, (event) => {
    events.push(event);
    const next = applyConversationEvent(state, event);
    if (next !== 'resync') state = next;
  });
  return { state: () => state!, current: () => state, events, off };
}

const ends = (page: Page): ConversationEvent[] => page.events.filter((e) => e.type === 'turn' && e.turn === null);

/** What the live and the stored thread share: a live prompt has no index and a live thought has its times. */
const shape = (entries: readonly TranscriptEntry[]): unknown[] =>
  entries.map((e) => {
    if (e.kind === 'user' || e.kind === 'assistant') return [e.kind, e.text];
    if (e.kind === 'notice') return [e.kind, e.code, e.code === 'hook-note' ? e.text : null];
    if (e.kind === 'turn-end') return [e.kind, e.stopReason, e.hookStopped ?? false];
    return [e.kind];
  });

describe('A prompt a hook blocked, through the service', () => {
  // Mutation: drop `dropTypedPrompt` → the live thread keeps the bubble the stored one has not; drop `blocked` from
  // the end → the message box gets nothing back.
  it.each(['chat', 'work'] as const)(
    '%s: only the hook row and the end stay, live and after a restart; the end gives the text back',
    async (mode) => {
      const fake = fakeService('chat-harness', { mutate: quick });
      try {
        const project = join(fake.root, 'home', 'code', 'app');
        mkdirSync(project, { recursive: true });
        const cwd = mode === 'work' ? { cwd: project } : {};
        const { conversation } = await fake.service.create({ mode, text: 'hello there', ...cwd });
        const id = conversation.id;
        const page = watch(fake.service, id);
        const answered = (): boolean =>
          page.current() !== null &&
          page.state().turn === null &&
          page.state().entries.some((e) => e.kind === 'assistant');
        await until(answered, 8_000);
        const before = page.state().entries.length;

        await fake.service.prompt(id, BLOCKED);
        const blocked = (): ConversationEvent[] =>
          ends(page).filter((e) => e.type === 'turn' && e.outcome === 'hook-stopped');
        await until(() => blocked().length === 1, 8_000);
        expect(blocked()[0]).toMatchObject({ outcome: 'hook-stopped', blocked: BLOCKED });
        await until(() => page.state().entries.every((e) => e.kind !== 'user' || e.text !== BLOCKED));
        expect(page.state().lastEnd).toMatchObject({ outcome: 'hook-stopped', blocked: BLOCKED });
        const live = page.state().entries;
        expect(live.filter((e) => e.kind === 'user').map((e) => e.text)).toEqual(['hello there']);
        expect(shape(live.slice(before))).toEqual([
          ['notice', 'hook-note', 'Prompt blocked by global/policy: no deploys on Friday'],
          ['turn-end', 'cancelled', true],
        ]);
        page.off();

        await fake.service.close();
        const after = watch(fake.restart(), id);
        await until(() => after.current() !== null && after.state().entries.length > 0, 8_000);
        expect(shape(after.state().entries)).toEqual(shape(live));
        after.off();
      } finally {
        await fake.close();
      }
    },
    30_000,
  );
});
