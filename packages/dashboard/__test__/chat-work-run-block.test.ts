/**
 * The run block on the fake agent's `workflow` step (`fixtures/chat-scenarios/work-workflow-run.json`): the service's
 * conversation stream, folded by the page's own reducer, drawn by the page's thread. Work in Ask: the start card, the
 * agent's request with "Asked by", the report, the wake.
 */

import { mkdirSync } from 'node:fs';
import { join } from 'node:path';

import { createElement, type ReactElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vite-plus/test';

import type { ChatService } from '../src/chat/service.js';
import type { ConversationEvent, PermissionInteraction } from '../src/chat/types.js';
import type { RunView } from '../src/chat/types.js';
import { ChatThread } from '../ui/src/components/chat-thread.js';
import { RunBlock } from '../ui/src/components/run-block.js';
import { type AgentAsk, runAgents } from '../ui/src/lib/agents.js';
import { applyConversationEvent, type ConversationState } from '../ui/src/lib/chat-store.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import { nowWords } from '../ui/src/lib/helper-words.js';
import type { RunEntry } from '../ui/src/lib/run-view.js';
import { PermissionCard } from '../ui/src/pages/chat/work/permission.js';
import { fakeService, until } from './helpers/chat-fake.js';
import { runView } from './helpers/run-views.js';

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
  const asks: AgentAsk[] = [];
  for (const item of state.interactions)
    if (item.kind === 'permission' && item.from !== undefined)
      asks.push({
        id: item.id,
        child: item.from.child,
        run: item.from.run,
        label: item.from.label,
        title: item.title,
        access: item.access,
        openedAt: item.openedAt,
      });
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
      agentAsks: asks,
      onStopRun: () => Promise.resolve(),
      live: state.live,
      agentLive: state.conversation.hot,
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

/** The run block's markup: its state, its header, its whole text and its markup. */
function runBlock(html: string): { state: string; header: string; text: string; html: string } | null {
  const at = html.indexOf('data-run="wf_1"');
  if (at === -1) return null;
  const open = html.lastIndexOf('<section', at);
  const section = html.slice(open, html.indexOf('</section>', at));
  const header = section.slice(section.indexOf('<header'), section.indexOf('</header>'));
  const mark = 'data-run-state="';
  const from = section.indexOf(mark) + mark.length;
  return {
    state: section.slice(from, section.indexOf('"', from)),
    header: textOf(header),
    text: textOf(section),
    html: section,
  };
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

        // The helper asks while the typed answer may still run; the thread is drawn once that answer has ended.
        await until(() => page.open().some((i) => i.from !== undefined) && page.state?.turn === null, 8_000);
        expect(page.open().find((i) => i.from !== undefined)!.from).toEqual({
          label: 'researcher-0',
          run: 'deep-research',
          cwd: null,
          child: expect.stringContaining('-child-'),
        });
        const ask = page.open().find((i) => i.from !== undefined)!;
        let html = draw(page.state!);
        const name = `${CHAT.run.namePrefix}deep-research`;
        expect(runBlock(html)!.header.startsWith(`${name}${CHAT.run.status.waitingYou}`)).toBe(true);
        expect(runBlock(html)!.header.endsWith(CHAT.run.buttons.stop)).toBe(true);
        expect(runBlock(html)!.text).toContain(fill(CHAT.run.asks[ask.access], { label: 'researcher-0' }));
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
        expect(page.state!.runs).toEqual([runView('wf_1', { live: false, status: 'complete' })]);
        expect(runBlock(html)!.state).toBe('done');
        expect(runBlock(html)!.header.startsWith(`${name}${CHAT.run.status.done}`)).toBe(true);
        expect(html.split('data-run=').length).toBe(2);
        // Mutation: draw the report in the block too → it shows twice. The wake row holds it, in a closed fold.
        expect(html).toContain('data-run-end="wf_1"');
        expect(textOf(html)).toContain(fill(CHAT.run.finished.complete, { name: 'deep-research' }));
        expect(runBlock(html)!.text).not.toContain('The probe found three files.');
      } finally {
        page.detach();
      }
    } finally {
      window();
      await fake.close();
    }
  }, 40_000);

  /** The step chips as the block draws them: each chip's mark, in order. */
  function steps(html: string): string[] {
    const out: string[] = [];
    let at = html.indexOf('<button');
    while (at !== -1) {
      const tag = html.slice(at, html.indexOf('>', at));
      const mark = tag.indexOf('data-mark="');
      if (mark !== -1 && tag.includes('aria-pressed')) {
        const from = mark + 'data-mark="'.length;
        out.push(tag.slice(from, tag.indexOf('"', from)));
      }
      at = html.indexOf('<button', at + 1);
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

  it("Stop run, in the agent's shape: Stopped, the step it was on takes the stop's mark, no step line, no report", async () => {
    const { fake, id, page, close } = await started({ ask: false, completeAfterMs: 60_000 });
    try {
      await until(() => page.state?.turn === null && page.state.runs.length === 1, 8_000);
      await fake.service.stopRun(id, 'wf_1');
      await until(ended(page, 'cancelled'), 8_000);
      // The host holds Stopping until the run's agents drained.
      await until(() => page.state?.runs.some((run) => run.runId === 'wf_1' && !run.stopping) === true, 8_000);
      const html = draw(page.state!);
      const block = runBlock(html)!;
      expect(block.header).toContain(fill(CHAT.run.status.stoppedByYouStep, { i: 2, phase: 'Research' }));
      expect(steps(block.html)).toEqual(['done', 'stopped', 'pending']);
      expect(block.text).not.toContain(fill(CHAT.run.step, { i: 2, n: 3, phase: 'Research' }));
      expect(block.text).not.toContain('The probe found three files.');
      expect(page.wakeSeen()).toBe(false);
    } finally {
      await close();
    }
  }, 40_000);

  it("a failed run, in the agent's shape: Failed, its error, the step it failed on takes the failed mark", async () => {
    const { page, close } = await started({ ask: false, status: 'failed', error: 'The script broke.', wake: false });
    try {
      await until(ended(page, 'failed'), 8_000);
      const block = runBlock(draw(page.state!))!;
      expect(block.header).toContain(fill(CHAT.run.status.failedStep, { i: 2, phase: 'Research' }));
      expect(steps(block.html)).toEqual(['done', 'failed', 'pending']);
      expect(block.text).toContain('The script broke.');
      expect(block.text).not.toContain('Paused');
    } finally {
      await close();
    }
  }, 40_000);

  it("a complete run, in the agent's shape: every step done", async () => {
    const { page, close } = await started({ ask: false, wake: false });
    try {
      await until(ended(page, 'complete'), 8_000);
      expect(steps(runBlock(draw(page.state!))!.html)).toEqual(['done', 'done', 'done']);
    } finally {
      await close();
    }
  }, 40_000);
});

describe('Work: the run block shows its agents live (#10)', () => {
  // Mutation: match a tick to an agent row by its label → the row shows no live numbers.
  it('a running agent shows its live tick: its time and context, then its turn and tool calls on a second line', async () => {
    const fake = fakeService('work-workflow-run', {
      timings: { activityEveryMs: 50 },
      mutate: (scenario) => {
        const steps = (scenario.prompts as { steps: Record<string, unknown>[] }[])[0]!.steps;
        const workflow = steps[1]!.workflow as Record<string, unknown>;
        steps[1] = {
          workflow: {
            ...workflow,
            ask: false,
            wake: false,
            childSessionId: 'researcher-child',
            completeAfterMs: 60_000,
          },
        };
        steps.splice(2, 0, {
          helperProgress: {
            child: 'researcher-child',
            ticks: [{ turn_count: 1, tool_call_count: 9, context_usage_pct: 33, tokens_used: 12_400 }],
          },
        });
      },
    });
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
        await until(() => page.state?.live.helpers.some((h) => h.childId === 'researcher-child') === true, 8_000);
        await until(() => page.state?.runs.length === 1, 8_000);
        const text = runBlock(draw(page.state!))!.text;
        const doing = page.state!.live.helpers.find((h) => h.childId === 'researcher-child')!.now;
        expect(text).toContain(`researcher-0${nowWords(doing)}`);
        expect(text).toContain(fill(CHAT.agent.facts.contextPct, { pct: 33 }));
        expect(text).toContain(
          [fill(CHAT.activity.progressTurn, { n: 1 }), fill(CHAT.subagent.toolCalls, { n: 9 })].join(' · '),
        );
      } finally {
        page.detach();
      }
    } finally {
      window();
      await fake.close();
    }
  }, 40_000);

  const agent = (agentId: string, state: string, tokens: number, durationMs: number) => ({
    agentId,
    label: agentId,
    phase: 'Research',
    state,
    tokens,
    durationMs,
  });
  const ENTRY: RunEntry = {
    kind: 'workflow',
    ts: 1,
    updatedTs: 2,
    runId: 'wf_1',
    revision: 3,
    name: 'deep-research',
    status: 'active',
    elapsedMs: 0,
    phases: [{ title: 'Research', state: 'active' }],
    currentPhase: 'Research',
    agentBudget: null,
    agentsUsed: 2,
    agents: [agent('writer', 'done', 8_140, 62_000), agent('researcher-1', 'running', 0, 0)],
    phasesEntered: [],
  };
  const tick = (childId: string, toolCalls: number) => ({
    childId,
    turns: 1,
    toolCalls,
    tokens: 12_400,
    contextPct: 33,
    errors: 0,
    failedCalls: 0,
    durationMs: 42_000,
    contextWindow: null,
    at: Date.now(),
    now: null,
  });
  const block = (view: RunView | undefined): string =>
    textOf(
      renderToStaticMarkup(
        createElement(
          MemoryRouter,
          null,
          createElement(RunBlock, {
            runId: 'wf_1',
            entry: ENTRY,
            live: true,
            view,
            by: null,
            wakes: [],
            resultAt: null,
            progress: [tick('writer', 77), tick('researcher-1', 9)],
            agents: runAgents([ENTRY]).get('wf_1')!,
          }),
        ),
      ),
    );

  it('a done agent shows no tick; a run the host no longer holds shows no live numbers', () => {
    const held = block(runView('wf_1'));
    expect(held).not.toContain(fill(CHAT.subagent.toolCalls, { n: 77 }));
    expect(held).toContain(
      `researcher-1${CHAT.agent.now.starting}0:42 · ${fill(CHAT.agent.facts.contextPct, { pct: 33 })}`,
    );
    expect(held).toContain(fill(CHAT.subagent.toolCalls, { n: 9 }));
    const gone = block(undefined);
    expect(gone).not.toContain(fill(CHAT.subagent.toolCalls, { n: 9 }));
  });
});
