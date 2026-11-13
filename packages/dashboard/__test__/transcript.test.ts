/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { parseUpdates } from '../src/agent-sessions/transcript.js';
import { Transcript, type TranscriptProps } from '../ui/src/components/transcript.js';
import { LIVE_FRESH_MS, useFreshness } from '../ui/src/lib/sessions.js';
import type { TranscriptEntry, TurnMetric } from '../ui/src/lib/types.js';
import { waitFor } from './render.js';

let mounted: { root: Root; container: HTMLElement } | undefined;

function mount(element: ReactElement): HTMLElement {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(element));
  mounted = { root, container };
  return container;
}

function rerender(element: ReactElement): void {
  act(() => mounted!.root.render(element));
}

afterEach(() => {
  if (mounted === undefined) return;
  act(() => mounted!.root.unmount());
  mounted.container.remove();
  mounted = undefined;
});

const T0 = 1_700_000_000_000;

function metric(ts: number): TurnMetric {
  return {
    promptId: `p${ts}`,
    ts,
    inputTokens: 100,
    outputTokens: 20,
    cachedTokens: 0,
    reasoningTokens: 0,
    elapsedMs: 4000,
    ttftMs: 100,
    prefillMs: null,
    prefillTokens: null,
    prefillCalls: null,
    modelCalls: null,
    decodeTps: 50,
    model: 'qwen3.8-27b',
  };
}

/** `count` closed turns: a prompt, an answer, one tool call and a turn end each. */
function closedTurns(count: number): { entries: TranscriptEntry[]; turns: TurnMetric[] } {
  const entries: TranscriptEntry[] = [];
  const turns: TurnMetric[] = [];
  for (let n = 1; n <= count; n += 1) {
    const ts = T0 + n * 60_000;
    entries.push(
      { kind: 'user', ts, text: `Prompt number ${n}`, promptIndex: n },
      {
        kind: 'tool',
        ts: ts + 500,
        id: `tool-${n}`,
        name: `tool_${n}`,
        title: `Tool ${n}`,
        status: 'completed',
        endTs: ts + 900,
      },
      { kind: 'assistant', ts: ts + 1000, text: `Answer body ${n}`, endTs: ts + 3000 },
      { kind: 'turn-end', ts: ts + 4000, stopReason: 'end_turn', elapsedMs: 4000 },
    );
    turns.push(metric(ts));
  }
  return { entries, turns };
}

function transcript(props: Partial<TranscriptProps> & Pick<TranscriptProps, 'entries' | 'turns'>): ReactElement {
  return createElement(
    MemoryRouter,
    null,
    createElement(Transcript, { model: 'qwen3.8-27b', sessionLive: false, onInspectTurn: () => undefined, ...props }),
  );
}

function typeInto(input: HTMLInputElement, value: string): void {
  act(() => {
    Reflect.set(HTMLInputElement.prototype, 'value', value, input);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

/**
 * A live body: one closed turn, then an open turn. Text entries carry endTs (the last chunk seen), as the reader serves
 * them. 'answer': thinking, a finished tool, a running tool, then the arriving answer. 'thinking': the thinking is last.
 */
function liveBody(ending: 'answer' | 'thinking' = 'answer'): { entries: TranscriptEntry[]; turns: TurnMetric[] } {
  const { entries, turns } = closedTurns(1);
  const ts = T0 + 10 * 60_000;
  entries.push(
    { kind: 'tool', ts: ts - 100, id: 'stale', name: 'stale_tool', title: 'never finished', status: 'in_progress' },
    { kind: 'turn-end', ts: ts - 50, stopReason: 'cancelled' },
    { kind: 'user', ts, text: 'Keep going', promptIndex: 9 },
    { kind: 'thinking', ts: ts + 100, text: 'Weighing the options', endTs: ts + 180 },
  );
  if (ending === 'answer') {
    entries.push(
      {
        kind: 'tool',
        ts: ts + 200,
        id: 'a',
        name: 'read_file',
        title: 'planner.rs',
        status: 'completed',
        endTs: ts + 300,
      },
      { kind: 'tool', ts: ts + 400, id: 'b', name: 'run_terminal_command', title: 'cargo test', status: 'in_progress' },
      { kind: 'assistant', ts: ts + 500, text: 'Working on it', endTs: ts + 600 },
    );
  }
  return { entries, turns: [...turns, metric(ts - 50)] };
}

describe('Transcript', () => {
  it("a closed group's text is not in the DOM; the last one is open", () => {
    const container = mount(transcript(closedTurns(5)));
    const text = container.textContent ?? '';
    expect(text).toContain('Turn 1');
    expect(text).toContain('Prompt number 1');
    expect(text).not.toContain('Answer body 1');
    expect(text).not.toContain('tool_1');
    expect(text).toContain('Answer body 5');
    expect(text).toContain('Turn ended · end_turn · 4.00 s');
  });

  it('counts the tokens of a subagent run in agreement with the number', () => {
    const { entries, turns } = closedTurns(1);
    const sub = (childId: string, tokens: number): TranscriptEntry => ({
      kind: 'subagent',
      ts: T0 + 60_500,
      childId,
      type: 'explore',
      description: `Run ${childId}`,
      status: 'completed',
      tokens,
    });
    entries.splice(2, 0, sub('one', 1), sub('two', 2));
    const container = mount(transcript({ entries, turns }));
    const text = container.textContent ?? '';
    expect(text).toContain('1 token');
    expect(text).not.toContain('1 tokens');
    expect(text).toContain('2 tokens');
  });

  it('opens every group when there are three or fewer', () => {
    const container = mount(transcript(closedTurns(3)));
    expect(container.textContent).toContain('Answer body 1');
    expect(container.textContent).toContain('Answer body 3');
  });

  it('find keeps only the matching turns, opens them and says how many match', async () => {
    const container = mount(transcript(closedTurns(6)));
    const input = container.querySelector<HTMLInputElement>('input[type="search"]')!;
    typeInto(input, 'ANSWER BODY 2');
    await waitFor(() => (container.textContent ?? '').includes('1 of 6 turns match'));
    const text = container.textContent ?? '';
    expect(text).toContain('Answer body 2');
    expect(text).not.toContain('Prompt number 6');
    expect(container.querySelector('[aria-live="polite"]')?.textContent).toBe('1 of 6 turns match');
  });

  it('Messages hides tools', () => {
    const container = mount(transcript(closedTurns(2)));
    expect(container.textContent).toContain('tool_1');
    const messages = [...container.querySelectorAll('button')].find((button) => button.textContent === 'Messages')!;
    act(() => messages.click());
    expect(container.textContent).not.toContain('tool_1');
    expect(container.textContent).toContain('Answer body 1');
  });

  it('Inspect hands the turn number to the page', () => {
    const inspected: number[] = [];
    const container = mount(transcript({ ...closedTurns(2), onInspectTurn: (turn) => inspected.push(turn) }));
    const links = [...container.querySelectorAll('a')].filter((link) => link.textContent === 'Inspect');
    act(() => links[1]!.click());
    expect(inspected).toEqual([2]);
  });

  it('with live, the answer ending the open turn is written and the running tool is live; without it nothing is', () => {
    const body = liveBody('answer');
    const container = mount(transcript({ ...body, sessionLive: true, live: true, asOfMs: T0 + 10 * 60_000 + 7100 }));
    const thinking = [...container.querySelectorAll('[data-live]')].find((node) =>
      (node.textContent ?? '').includes('Thinking'),
    );
    expect(thinking).toBeUndefined();
    expect(container.textContent).toContain('Thinking for 80 ms');
    expect(container.querySelector('[data-status="running"][data-live]')).not.toBeNull();
    expect(container.querySelector('article[data-role="assistant"][data-live]')?.textContent).toContain('Writing');
    expect(container.textContent).toContain('Using 2 tools');
    expect(container.textContent).toContain('in progress');

    rerender(transcript({ ...body, sessionLive: true, live: false }));
    expect(container.querySelectorAll('[data-live]')).toHaveLength(0);
    expect(container.textContent).toContain('Thinking');
    expect(container.textContent).not.toContain('Writing');
    expect(container.textContent).toContain('running');
  });

  it('with live, thinking that ends the open turn is live as of the body and shows no done time', () => {
    const body = liveBody('thinking');
    const container = mount(transcript({ ...body, sessionLive: true, live: true, asOfMs: T0 + 10 * 60_000 + 7100 }));
    const thinking = [...container.querySelectorAll('[data-live]')].find((node) =>
      (node.textContent ?? '').includes('Thinking'),
    );
    expect(thinking).toBeDefined();
    expect(thinking!.textContent).toContain('for 0:07');

    rerender(transcript({ ...body, sessionLive: true, live: false }));
    expect(container.querySelectorAll('[data-live]')).toHaveLength(0);
    expect(container.textContent).toContain('Thinking');
    expect(container.textContent).not.toContain('80 ms');
  });

  it('reads live text from real reader output: the last text of an open turn is live until turn_completed', () => {
    const SID = 'session-1';
    const envelope = (method: string, update: object, n: number): string =>
      JSON.stringify({
        timestamp: 1791643279,
        method,
        params: { sessionId: SID, update, _meta: { eventId: `${SID}-${n}`, agentTimestampMs: T0 + n } },
      });
    const acp = (update: object, n: number): string => envelope('session/update', update, n);
    const chunk = (sessionUpdate: string, text: string, n: number): string =>
      acp({ sessionUpdate, content: { type: 'text', text } }, n);
    const prompt = chunk('user_message_chunk', 'Plan the release', 1);
    const thinking = [chunk('agent_thought_chunk', 'Weighing ', 2), chunk('agent_thought_chunk', 'options', 3)];
    const answer = [chunk('agent_message_chunk', 'Here is ', 4), chunk('agent_message_chunk', 'the plan', 5)];
    const completed = envelope(
      '_x.ai/session/update',
      { sessionUpdate: 'turn_completed', prompt_id: 'p1', stop_reason: 'end_turn' },
      6,
    );
    const render = (lines: string[]): HTMLElement => {
      const parsed = parseUpdates(`${lines.join('\n')}\n`);
      const props = {
        entries: parsed.transcript,
        turns: parsed.turns,
        sessionLive: true,
        live: true,
        asOfMs: T0 + 4002,
      };
      if (mounted === undefined) return mount(transcript(props));
      rerender(transcript(props));
      return mounted.container;
    };
    const liveThinking = (container: HTMLElement): Element | undefined =>
      [...container.querySelectorAll('[data-live]')].find((node) => (node.textContent ?? '').includes('Thinking'));
    const writing = (container: HTMLElement): Element | null =>
      container.querySelector('article[data-role="assistant"][data-live]');

    let container = render([prompt, ...thinking]);
    expect(liveThinking(container)?.textContent).toContain('for 0:04');
    expect(writing(container)).toBeNull();

    container = render([prompt, ...thinking, ...answer]);
    expect(liveThinking(container)).toBeUndefined();
    expect(writing(container)?.textContent).toContain('Writing');

    container = render([prompt, ...thinking, ...answer, completed]);
    expect(container.querySelectorAll('[data-live]')).toHaveLength(0);
    expect(container.textContent).not.toContain('Writing');
  });

  it('a running tool in a closed turn reads not finished', async () => {
    const container = mount(transcript({ ...liveBody(), sessionLive: true, live: true }));
    const header = [...container.querySelectorAll('button[aria-expanded]')].find((button) =>
      (button.textContent ?? '').includes('Turn 2'),
    ) as HTMLButtonElement;
    act(() => header.click());
    await waitFor(() => (container.textContent ?? '').includes('stale_tool'));
    const stale = [...container.querySelectorAll('[data-status]')].find((node) =>
      (node.textContent ?? '').includes('stale_tool'),
    )!;
    expect(stale.getAttribute('data-status')).toBe('other');
    expect(stale.textContent).toContain('not finished');
    expect(stale.hasAttribute('data-live')).toBe(false);
  });

  it('the unclosed last turn of a session that no longer runs reads as cut off, not in progress', () => {
    const container = mount(transcript(liveBody()));
    const text = container.textContent ?? '';
    expect(text).not.toContain('in progress');
    expect(text).toContain('Used 2 tools');
    expect(text).not.toContain('Using 2 tools');
    const cut = [...container.querySelectorAll('[data-status]')].find((node) =>
      (node.textContent ?? '').includes('cargo test'),
    )!;
    expect(cut.getAttribute('data-status')).toBe('other');
    expect(cut.textContent).toContain('not finished');
    expect(container.querySelector('[data-status="running"]')).toBeNull();
  });

  it('a new turn arriving opens when the turn before it was open', () => {
    const before = closedTurns(5);
    const container = mount(transcript({ ...before, sessionLive: true, live: true }));
    expect(container.textContent).toContain('Answer body 5');
    const ts = T0 + 10 * 60_000;
    const entries: TranscriptEntry[] = [
      ...before.entries,
      { kind: 'user', ts, text: 'Next prompt', promptIndex: 6 },
      { kind: 'assistant', ts: ts + 100, text: 'Fresh answer' },
    ];
    rerender(transcript({ entries, turns: before.turns, sessionLive: true, live: true }));
    expect(container.textContent).toContain('Fresh answer');
    expect(container.textContent).toContain('in progress');
  });

  it('shows the reader error and the empty state', () => {
    const container = mount(transcript({ entries: [], turns: [], transcriptError: 'bad line 3' }));
    expect(container.textContent).toContain('Transcript could not be read: bad line 3');
    rerender(transcript({ entries: [], turns: [] }));
    expect(container.textContent).toContain('No messages in this session.');
  });
});

function Probe({ data }: { data: object | undefined }): ReactElement {
  const { fresh, arrivedAt } = useFreshness(data, LIVE_FRESH_MS);
  return createElement('output', null, `${String(fresh)} ${String(arrivedAt)}`);
}

describe('useFreshness', () => {
  it('is fresh for LIVE_FRESH_MS after each new body, then stale until the next one', () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
    try {
      vi.setSystemTime(T0);
      const first = { n: 1 };
      const container = mount(createElement(Probe, { data: first }));
      const read = (): string[] => (container.textContent ?? '').split(' ');
      expect(read()).toEqual(['true', String(T0)]);

      act(() => {
        vi.advanceTimersByTime(LIVE_FRESH_MS);
      });
      expect(read()).toEqual(['false', String(T0)]);

      rerender(createElement(Probe, { data: first }));
      expect(read()[0]).toBe('false');

      act(() => {
        vi.advanceTimersByTime(500);
      });
      rerender(createElement(Probe, { data: { n: 2 } }));
      expect(read()).toEqual(['true', String(T0 + LIVE_FRESH_MS + 500)]);

      rerender(createElement(Probe, { data: undefined }));
      expect(read()).toEqual(['false', 'null']);
    } finally {
      vi.useRealTimers();
    }
  });
});
