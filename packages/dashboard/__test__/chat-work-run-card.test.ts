/**
 * The run card on the fake agent's `workflow` step (`fixtures/chat-scenarios/work-workflow-run.json`): the service's
 * conversation stream, folded by the page's own reducer, drawn by the page's thread. Work in Ask: the start card, the
 * helper's request with "Asked by", the report, the wake.
 */

import { mkdirSync } from 'node:fs';
import { join } from 'node:path';

import { createElement, type ReactElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vite-plus/test';

import type { ChatService } from '../src/chat/service.js';
import type { ConversationEvent, PermissionInteraction } from '../src/chat/types.js';
import { ChatThread } from '../ui/src/components/chat-thread.js';
import { applyConversationEvent, type ConversationState } from '../ui/src/lib/chat-store.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import { PermissionCard } from '../ui/src/pages/chat/work/permission.js';
import { fakeService, until } from './helpers/chat-fake.js';

class Page {
  readonly events: ConversationEvent[] = [];
  state: ConversationState | null = null;
  readonly detach: () => void;

  constructor(service: ChatService, id: string) {
    this.detach = service.watchConversation(id, (event) => {
      this.events.push(event);
      const next = applyConversationEvent(this.state, event);
      if (next !== 'resync') this.state = next;
    });
  }

  open(): PermissionInteraction[] {
    return (this.state?.interactions ?? []).filter((i): i is PermissionInteraction => i.kind === 'permission');
  }

  wakeSeen(): boolean {
    return this.events.some((e) => e.type === 'turn' && e.turn?.origin === 'wake');
  }
}

/** The thread and the first open card as the page draws them, from the folded state, as markup. */
function draw(state: ConversationState): string {
  const first = state.interactions[0];
  const asks = new Map<string, string>();
  for (const item of state.interactions)
    if (item.kind === 'permission' && item.from !== undefined && item.from.run !== null && !asks.has(item.from.run))
      asks.set(item.from.run, item.from.label);
  const card =
    first !== undefined && first.kind === 'permission'
      ? createElement(PermissionCard, {
          interaction: first,
          permissionMode: 'ask',
          folder: state.conversation.cwd,
          temp: null,
          position: { index: 1, count: state.interactions.length },
          composerRef: { current: null },
          onAnswer: () => Promise.resolve(),
        })
      : null;
  const element: ReactElement = createElement(
    MemoryRouter,
    null,
    createElement(ChatThread, {
      conversationId: state.conversation.id,
      mode: 'work',
      entries: state.entries,
      turns: state.turns,
      turn: state.turn,
      lastEnd: state.lastEnd,
      runs: state.runs,
      runAsks: asks,
      onStopRun: () => Promise.resolve(),
      card,
      asking: first !== undefined && (first.kind !== 'permission' || first.from?.run == null),
    }),
  );
  return renderToStaticMarkup(element);
}

/** The text of some markup: tags dropped, the five entities React writes decoded. */
function textOf(html: string): string {
  let out = '';
  let inTag = false;
  for (const ch of html) {
    if (ch === '<') inTag = true;
    else if (ch === '>') inTag = false;
    else if (!inTag) out += ch;
  }
  return out
    .split('&quot;')
    .join('"')
    .split('&#x27;')
    .join("'")
    .split('&lt;')
    .join('<')
    .split('&gt;')
    .join('>')
    .split('&amp;')
    .join('&');
}

/** The text of the card's visible "Asked by" line (its screen-reader copy is a separate node); null when none. */
function askedBy(html: string): string | null {
  const mark = 'data-asked-by="">';
  const at = html.indexOf(mark);
  if (at === -1) return null;
  const from = at + mark.length;
  return textOf(html.slice(from, html.indexOf('</span>', from)));
}

/** The run card's markup: its header and its whole text. */
function runCard(html: string): { header: string; text: string } | null {
  const at = html.indexOf('data-run="wf_1"');
  if (at === -1) return null;
  const open = html.lastIndexOf('<section', at);
  const section = html.slice(open, html.indexOf('</section>', at));
  const header = section.slice(section.indexOf('<header'), section.indexOf('</header>'));
  return { header: textOf(header), text: textOf(section) };
}

describe('Work: the run card on the fake agent', () => {
  it('start card, a helper asks with its name, the run ends with its report, then the wake', async () => {
    const fake = fakeService('work-workflow-run');
    const project = join(fake.root, 'home', 'code', 'app');
    mkdirSync(project, { recursive: true });
    const window = fake.service.watchHost(() => undefined);
    try {
      const { conversation } = await fake.service.create({ mode: 'work', text: 'research', cwd: project });
      const page = new Page(fake.service, conversation.id);
      try {
        await until(() => page.open().some((i) => i.toolName === 'workflow'), 5_000);
        const start = page.open().find((i) => i.toolName === 'workflow')!;
        await fake.service.answer(conversation.id, start.id, { optionId: 'allow-once' });

        await until(() => page.open().some((i) => i.from !== undefined), 8_000);
        const ask = page.open().find((i) => i.from !== undefined)!;
        expect(ask.from).toEqual({ label: 'researcher-0', run: 'deep-research', cwd: null });
        let html = draw(page.state!);
        const title = fill(CHAT.run.title, { name: 'deep-research' });
        expect(runCard(html)!.header).toBe(`${title}${CHAT.run.waitingYou}${CHAT.run.stop}`);
        expect(runCard(html)!.text).toContain(fill(CHAT.run.asks, { label: 'researcher-0' }));
        expect(askedBy(html)).toBe(fill(CHAT.permFrom, { label: 'researcher-0', run: 'deep-research' }));
        expect(html).not.toContain('data-wait-line');

        await fake.service.answer(conversation.id, ask.id, { optionId: 'allow-once' });
        await until(
          () => page.state?.entries.some((e) => e.kind === 'workflow' && e.status === 'complete') === true,
          15_000,
        );
        await until(() => page.wakeSeen(), 8_000);
        await until(() => page.state?.turn === null, 8_000);
        html = draw(page.state!);
        expect(page.state!.runs).toEqual([]);
        expect(runCard(html)!.header).toBe(`${title}${CHAT.run.status.complete}`);
        expect(runCard(html)!.text).toContain('The probe found three files.');
        expect(html.split('data-run=').length).toBe(2);
      } finally {
        page.detach();
      }
    } finally {
      window();
      await fake.close();
    }
  }, 40_000);

  /** The steps as the card draws them: `done`, `current` or `open` per step, in order. */
  function steps(html: string): string[] {
    const out: string[] = [];
    let at = html.indexOf('<li');
    while (at !== -1) {
      const tag = html.slice(at, html.indexOf('>', at));
      out.push(tag.includes('data-complete') ? 'done' : tag.includes('aria-current="step"') ? 'current' : 'open');
      at = html.indexOf('<li', at + 1);
    }
    return out;
  }

  /** Work in Ask on the fake agent with `workflow` options merged in; resolves once the start card is allowed. */
  async function started(options: Record<string, unknown>) {
    const fake = fakeService('work-workflow-run', {
      mutate: (scenario) => {
        const steps = (scenario.prompts as { steps: { workflow?: Record<string, unknown> }[] }[])[0]!.steps;
        steps[1] = { workflow: { ...steps[1]!.workflow, ...options } };
      },
    });
    const project = join(fake.root, 'home', 'code', 'app');
    mkdirSync(project, { recursive: true });
    const window = fake.service.watchHost(() => undefined);
    const { conversation } = await fake.service.create({ mode: 'work', text: 'research', cwd: project });
    const page = new Page(fake.service, conversation.id);
    await until(() => page.open().some((i) => i.toolName === 'workflow'), 5_000);
    const start = page.open().find((i) => i.toolName === 'workflow')!;
    await fake.service.answer(conversation.id, start.id, { optionId: 'allow-once' });
    const close = async () => {
      page.detach();
      window();
      await fake.close();
    };
    return { fake, id: conversation.id, page, close };
  }

  const ended = (page: Page, status: string) => () =>
    page.state?.entries.some((e) => e.kind === 'workflow' && e.status === status) === true;

  it("Stop run, in the agent's shape: Stopped, the step it was on stays current, no step line, no report", async () => {
    const { fake, id, page, close } = await started({ ask: false, completeAfterMs: 60_000 });
    try {
      await until(() => page.state?.turn === null && page.state.runs.length === 1, 8_000);
      await fake.service.stopRun(id, 'wf_1');
      await until(ended(page, 'cancelled'), 8_000);
      const html = draw(page.state!);
      const title = fill(CHAT.run.title, { name: 'deep-research' });
      expect(runCard(html)!.header).toBe(`${title}${CHAT.run.status.cancelled}`);
      expect(steps(html)).toEqual(['done', 'current', 'open']);
      expect(runCard(html)!.text).not.toContain(fill(CHAT.run.phase, { i: 2, n: 3, phase: 'Research' }));
      expect(runCard(html)!.text).not.toContain('The probe found three files.');
      expect(page.wakeSeen()).toBe(false);
    } finally {
      await close();
    }
  }, 40_000);

  it("a failed run, in the agent's shape: Failed, its error, the step it failed on stays current", async () => {
    const { page, close } = await started({ ask: false, status: 'failed', error: 'The script broke.', wake: false });
    try {
      await until(ended(page, 'failed'), 8_000);
      const html = draw(page.state!);
      expect(runCard(html)!.header).toBe(`${fill(CHAT.run.title, { name: 'deep-research' })}${CHAT.run.status.failed}`);
      expect(steps(html)).toEqual(['done', 'current', 'open']);
      expect(runCard(html)!.text).toContain(fill(CHAT.run.error, { message: 'The script broke.' }));
      expect(runCard(html)!.text).not.toContain('Paused');
    } finally {
      await close();
    }
  }, 40_000);

  it("a complete run, in the agent's shape: every step done", async () => {
    const { page, close } = await started({ ask: false, wake: false });
    try {
      await until(ended(page, 'complete'), 8_000);
      expect(steps(draw(page.state!))).toEqual(['done', 'done', 'done']);
    } finally {
      await close();
    }
  }, 40_000);
});
