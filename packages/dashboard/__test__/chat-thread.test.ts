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
import { formatClock } from '../ui/src/lib/format.js';
import type { TranscriptEntry } from '../ui/src/lib/types.js';
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
  metric,
  T0,
  thinking,
  liveOutcomeSample,
  longThread,
  outcomeSample,
  streamingSample,
  user,
  finishedRun,
  RUN_ID,
  runSample,
  workflowRun,
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

/** A wait line's words, without its clock. */
function waitText(line: Element | null | undefined): string | undefined {
  const copy = line?.cloneNode(true) as HTMLElement | undefined;
  copy?.querySelector('[data-wait-clock]')?.remove();
  return copy?.textContent ?? undefined;
}

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

describe('ChatThread: the automatic continue and the thought time', () => {
  const lengthEnd: TranscriptEntry = {
    kind: 'turn-end',
    ts: T0 + 2_000,
    stopReason: 'error',
    error: 'response truncated by max_tokens',
    lengthLimit: true,
  };
  const continued: TranscriptEntry = {
    kind: 'user',
    ts: T0 + 3_000,
    text: CHAT.continuePrompt,
    promptIndex: 1,
    continued: true,
  };

  // Mutation: draw the continued entry as a bubble → red.
  it('a continued entry draws a divider, not a bubble; a stored thought shows no time', () => {
    const sample: ThreadSample = {
      mode: 'work',
      entries: [
        user('fix it'),
        thinking('Let me look.'),
        lengthEnd,
        continued,
        answer('Done.', T0 + 4_000),
        end('end_turn', T0 + 5_000),
      ],
      turns: [metric(), metric()],
      turn: null,
      lastEnd: null,
    };
    const container = mount(threadOf(sample));
    const text = container.textContent ?? '';
    expect(text).toContain(CHAT.continued);
    expect(container.querySelectorAll('[data-role="user"]')).toHaveLength(1);
    expect(text).toContain(CHAT.outcomes['output-limit']);
    expect(text).not.toContain('response truncated');
    expect(text).toContain(CHAT.thought);
    expect(text).not.toContain(`${CHAT.thought} for`);
  });

  it('a thought timed live shows its time', () => {
    const timed: TranscriptEntry = { kind: 'thinking', ts: T0 + 100, endTs: T0 + 5_100, text: 'Hm.', timed: true };
    const container = mount(
      threadOf({ ...outcomeSample('end_turn'), entries: [user('hi'), timed, answer('ok'), end()] }),
    );
    expect(container.textContent).toContain(`${CHAT.thought} for 5`);
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
    const line = (): HTMLElement | null =>
      container.querySelector<HTMLElement>('article[aria-busy="true"][data-wait-line]');
    expect(line()?.getAttribute('data-wait-line')).toBe('think');
    expect(line()?.getAttribute('aria-label')).toBe('Assistant');
    expect(waitText(line())).toBe('Thinking…');
    const orb = line()?.querySelectorAll('canvas[data-wait-orb]');
    expect(orb).toHaveLength(1);
    expect(orb?.[0]?.getAttribute('data-wait-orb')).toBe('think');
    expect(orb?.[0]?.getAttribute('aria-hidden')).toBe('true');
    rerender(threadOf({ ...base, turn: null }, { creating: true }));
    expect(container.textContent).toContain('Starting…');
    expect(line()?.getAttribute('data-wait-line')).toBe('hold');
    expect(line()?.querySelector('canvas[data-wait-orb="hold"]')).toBe(orb?.[0]);
    rerender(threadOf(streamingSample('Hello')));
    expect(container.textContent).not.toContain('Thinking…');
    expect(container.querySelector('[data-wait-line]')).toBeNull();
    expect(container.querySelector('[data-wait-orb]')).toBeNull();
  });

  it('after 3 s with no stream event the line holds: same orb, the waiting text', () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    try {
      const base = outcomeSample('end_turn');
      const container = mount(
        threadOf({ ...base, turn: { turnId: 't2', phase: 'running', startedAt: 0, wait: null } }),
      );
      const orb = container.querySelector('[data-wait-line="think"] canvas[data-wait-orb="think"]');
      expect(orb).not.toBeNull();
      act(() => {
        vi.advanceTimersByTime(3_000);
      });
      const line = container.querySelector('article[aria-busy="true"][data-wait-line="hold"]');
      expect(waitText(line)).toBe(CHAT.pendingWaiting);
      expect(line?.querySelector('canvas[data-wait-orb="hold"]')).toBe(orb);
    } finally {
      vi.useRealTimers();
    }
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
  const orbsOf = (container: HTMLElement): string[] =>
    [...container.querySelectorAll('[data-wait-orb]')].map((node) =>
      node.closest('[data-load-card]') === null ? 'outside' : (node.getAttribute('data-wait-orb') ?? ''),
    );

  it('the load card names the whole model, its size and the time since the load started; no bar, no percent', () => {
    vi.useFakeTimers({ now: NOW, toFake: ['Date', 'setInterval', 'clearInterval'] });
    try {
      const container = mount(
        threadOf(waiting(NOW - 9_000), {
          model: LONG,
          host: {
            inference: 'loading',
            loading: [{ modelId: LONG, startedAt: NOW - 134_000, phase: 'weights', stepsDone: 0, steps: 0 }],
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
        loading: loading === null ? [] : [{ ...loading, phase: 'weights' as const, stepsDone: 0, steps: 0 }],
        resident: [],
        answers: [],
      });
      const container = mount(threadOf(waiting(NOW - 61_000), { model: 'qwen-a', host: host(null) }));
      expect(cardOf(container).getAttribute('data-variant')).toBe('busy');
      expect(orbsOf(container)).toEqual(['hold']);
      expect(container.textContent).toContain(CHAT.load.titleQueued);
      expect(container.textContent).toContain(CHAT.load.busyFirst);
      expect(container.textContent).toContain('1:01');
      expect(container.textContent).not.toContain(CHAT.load.size.slice(-7));
      rerender(threadOf(waiting(NOW - 61_000), { model: 'qwen-a', host: host({ modelId: 'qwen-b', startedAt: NOW }) }));
      expect(cardOf(container).getAttribute('data-variant')).toBe('other-first');
      expect(orbsOf(container)).toEqual(['hold']);
      expect(container.textContent).toContain(fill(CHAT.load.otherFirst, { other: 'qwen-b' }));
      expect(container.textContent).toContain('1:01');
      rerender(
        threadOf(waiting(NOW - 61_000), { model: 'qwen-a', host: host({ modelId: 'qwen-a', startedAt: NOW - 5_000 }) }),
      );
      expect(cardOf(container).getAttribute('data-variant')).toBe('loading');
      expect(orbsOf(container)).toEqual(['load']);
      expect(cardOf(container).querySelector('[data-wait-orb]')?.getAttribute('aria-hidden')).toBe('true');
      expect(container.textContent).toContain('0:05');
    } finally {
      vi.useRealTimers();
    }
  });

  it('while the engine prepares GPU kernels the card says so, counts the steps and fills a bar', () => {
    vi.useFakeTimers({ now: NOW, toFake: ['Date', 'setInterval', 'clearInterval'] });
    try {
      const container = mount(
        threadOf(waiting(NOW - 9_000), {
          model: 'qwen-a',
          host: {
            inference: 'loading',
            loading: [{ modelId: 'qwen-a', startedAt: NOW - 40_000, phase: 'kernels', stepsDone: 12, steps: 45 }],
            resident: [],
            answers: [],
          },
        }),
      );
      const card = cardOf(container);
      expect(card.getAttribute('data-phase')).toBe('kernels');
      expect(card.textContent).toContain(CHAT.load.titleKernels);
      expect(card.textContent).not.toContain(CHAT.load.title);
      expect(card.textContent).toContain('12 of 45');
      expect(card.textContent).toContain(CHAT.load.kernelNote);
      expect(card.textContent).toContain(CHAT.load.stopNote);
      const bar = byRole(container, 'meter', CHAT.load.titleKernels);
      expect(bar?.getAttribute('aria-valuenow')).toBe('27');
      expect(bar?.getAttribute('aria-valuetext')).toBe('27%, 12 of 45');
    } finally {
      vi.useRealTimers();
    }
  });

  it('the weight read, or a load with no steps, keeps the plain card', () => {
    vi.useFakeTimers({ now: NOW, toFake: ['Date', 'setInterval', 'clearInterval'] });
    try {
      const hostOf = (phase: 'weights' | 'kernels', stepsDone: number | null, steps: number | null) => ({
        inference: 'loading' as const,
        loading: [{ modelId: 'qwen-a', startedAt: NOW - 1_000, phase, stepsDone, steps }],
        resident: [],
        answers: [],
      });
      const container = mount(threadOf(waiting(NOW - 1_000), { model: 'qwen-a', host: hostOf('weights', 0, 0) }));
      for (const host of [hostOf('weights', 0, 0), hostOf('kernels', null, null), hostOf('kernels', 0, 0)]) {
        rerender(threadOf(waiting(NOW - 1_000), { model: 'qwen-a', host }));
        const card = cardOf(container);
        expect(card.textContent).toContain(CHAT.load.title);
        expect(card.textContent).toContain(CHAT.load.wait);
        expect(card.textContent).not.toContain(CHAT.load.titleKernels);
        expect(card.textContent).not.toContain(' of ');
        expect(container.querySelector('[role="meter"]')).toBeNull();
      }
    } finally {
      vi.useRealTimers();
    }
  });

  it('two loads at once: each conversation shows its own model and count', () => {
    vi.useFakeTimers({ now: NOW, toFake: ['Date', 'setInterval', 'clearInterval'] });
    try {
      const host = {
        inference: 'loading' as const,
        loading: [
          { modelId: 'qwen-a', startedAt: NOW - 60_000, phase: 'kernels' as const, stepsDone: 30, steps: 50 },
          { modelId: 'qwen-b', startedAt: NOW - 5_000, phase: 'kernels' as const, stepsDone: 3, steps: 45 },
        ],
        resident: [],
        answers: [],
      };
      const container = mount(threadOf(waiting(NOW - 1_000), { model: 'qwen-a', host }));
      expect(cardOf(container).textContent).toContain('30 of 50');
      expect(cardOf(container).textContent).toContain('1:00');
      rerender(threadOf(waiting(NOW - 1_000), { model: 'qwen-b', host }));
      expect(cardOf(container).getAttribute('data-variant')).toBe('loading');
      expect(cardOf(container).textContent).toContain('3 of 45');
      expect(cardOf(container).textContent).toContain('0:05');
      expect(cardOf(container).textContent).not.toContain('30 of 50');
      rerender(threadOf(waiting(NOW - 1_000), { model: 'qwen-c', host }));
      expect(cardOf(container).getAttribute('data-variant')).toBe('other-first');
      expect(cardOf(container).textContent).toContain(fill(CHAT.load.otherFirst, { other: 'qwen-a' }));
      expect(cardOf(container).textContent).not.toContain(' of 5');
    } finally {
      vi.useRealTimers();
    }
  });

  it('a model in memory: the card waits for the engine and never says the model loads', () => {
    const onStatus = vi.fn();
    const host = {
      inference: 'ready' as const,
      loading: [],
      resident: [{ modelId: 'qwen-a', sizeBytes: null }],
      answers: [],
    };
    const container = mount(threadOf(waiting(Date.now()), { model: 'qwen-a', host, onStatus }));
    const card = cardOf(container);
    expect(card.getAttribute('data-variant')).toBe('in-memory');
    expect(orbsOf(container)).toEqual(['hold']);
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
      loading: [],
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
    const host = { inference: 'ready' as const, loading: [], resident: [], answers: [] };
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
        host: {
          inference: 'loading',
          loading: [{ modelId: 'qwen-a', startedAt: 0, phase: 'weights', stepsDone: 0, steps: 0 }],
          resident: [],
          answers: [],
        },
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
          loading: [],
          resident: [{ modelId: 'qwen-a', sizeBytes: 2 * 1024 ** 3 }],
          answers: [],
        },
      }),
    );
    expect(container.textContent).not.toContain(fill(CHAT.engine.stillLoading, { model: 'qwen-a' }));
    expect(container.textContent).toContain(fill(CHAT.engine.resident, { model: 'qwen-a', size: '2.0 GB' }));
    expect(anchors(container)).not.toContain('/engine');
    rerender(
      threadOf(sample, { model: 'qwen-a', host: { inference: 'ready', loading: [], resident: [], answers: [] } }),
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
      loading: [],
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

describe('ChatThread: the wait line between the steps of a long turn', () => {
  const NOW = 1_794_411_600_000;
  const tool = (status: string, ts: number, endTs?: number, id = 'r1'): TranscriptEntry => ({
    kind: 'tool',
    ts,
    id,
    name: 'probe',
    title: 'Probe src/main.rs',
    status,
    ...(endTs === undefined ? {} : { endTs }),
  });
  const work = (entries: TranscriptEntry[]): ThreadSample => ({
    mode: 'work',
    entries: [user('Fix the build', NOW - 600_000), ...entries],
    turns: [],
    turn: { turnId: 't', phase: 'running', startedAt: NOW - 600_000, wait: null },
    lastEnd: null,
  });
  const afterRead = work([
    thinking('Look at main', NOW - 200_000, NOW - 190_000),
    tool('completed', NOW - 189_000, NOW - 65_000),
  ]);
  type Request = NonNullable<NonNullable<ChatThreadProps['host']>['requests']>[number];
  const request = (over: Partial<Request>): Request => ({
    session: 'conv-1',
    modelId: 'qwen-a',
    phase: 'decode',
    startedAt: NOW - 30_000,
    prefillDone: null,
    prefillTotal: null,
    generated: null,
    ...over,
  });
  const hostWith = (requests: Request[], queued = 0) => ({
    inference: 'ready' as const,
    loading: [],
    resident: [{ modelId: 'qwen-a', sizeBytes: null }],
    answers: [{ modelId: 'qwen-a', inFlight: requests.length, maxConcurrent: 4, queued, lastStart: NOW - 30_000 }],
    requests,
  });
  const props = (requests: Request[], queued = 0): Partial<ChatThreadProps> => ({
    conversationId: 'conv-1',
    model: 'qwen-a',
    host: hostWith(requests, queued),
  });
  const line = (container: HTMLElement): HTMLElement | null =>
    container.querySelector<HTMLElement>('article[aria-busy="true"][data-wait-line]');
  const clock = (container: HTMLElement): string | undefined =>
    line(container)?.querySelector('[data-wait-clock]')?.textContent ?? undefined;
  const timers = () =>
    vi.useFakeTimers({ now: NOW, toFake: ['Date', 'setInterval', 'clearInterval', 'setTimeout', 'clearTimeout'] });

  it('after a finished tool row the turn says it is working, with the time since that step ended', () => {
    timers();
    try {
      const container = mount(threadOf(afterRead, { model: 'qwen-a' }));
      expect(container.textContent).toContain('Thought');
      expect(waitText(line(container))).toBe(CHAT.pendingWorking);
      expect(line(container)?.getAttribute('data-wait-line')).toBe('think');
      expect(clock(container)).toBe('1:05');
      act(() => {
        vi.advanceTimersByTime(3_000);
      });
      expect(waitText(line(container))).toBe(CHAT.pendingWaiting);
      expect(clock(container)).toBe('1:08');
    } finally {
      vi.useRealTimers();
    }
  });

  it("names what the engine does with this conversation's request: queued, reading the prompt, generating", () => {
    const container = mount(threadOf(afterRead, props([request({ phase: 'queued' })])));
    expect(waitText(line(container))).toBe(CHAT.pendingQueued);
    expect(line(container)?.getAttribute('data-wait-line')).toBe('hold');
    rerender(
      threadOf(
        afterRead,
        props([
          request({ session: 'conv-2', phase: 'decode', startedAt: NOW - 400_000 }),
          request({ phase: 'prefill', prefillDone: 69_300, prefillTotal: 121_300 }),
        ]),
      ),
    );
    expect(waitText(line(container))).toBe(fill(CHAT.pendingReadingOf, { done: '69.3K', total: '121.3K' }));
    const bar = byRole(container, 'meter', CHAT.pendingReadMeter);
    expect(bar?.getAttribute('aria-valuenow')).toBe('57');
    expect(bar?.getAttribute('aria-valuetext')).toBe('57%, 69.3K of 121.3K');
    rerender(threadOf(afterRead, props([request({ phase: 'decode' })])));
    expect(waitText(line(container))).toBe(CHAT.pendingThinking);
    expect(container.querySelector('[role="meter"]')).toBeNull();
  });

  it("another conversation's request never reads as this one's: the model is busy only when ours may wait behind it", () => {
    const other = request({ session: 'conv-2', phase: 'prefill', prefillDone: 5_000, prefillTotal: 20_000 });
    const container = mount(threadOf(afterRead, props([other])));
    expect(waitText(line(container))).toBe(CHAT.pendingWorking);
    rerender(threadOf(afterRead, props([other], 1)));
    expect(waitText(line(container))).toBe(fill(CHAT.pendingBusyReading, { done: '5,000', total: '20K' }));
    expect(waitText(line(container))).not.toContain('conversation');
    rerender(threadOf(afterRead, props([request({ session: null, phase: 'decode', startedAt: NOW - 10_000 })])));
    expect(waitText(line(container))).toBe(CHAT.pendingBusy);
    rerender(threadOf(afterRead, props([request({ session: null, phase: 'decode', startedAt: NOW - 120_000 })])));
    expect(waitText(line(container))).toBe(CHAT.pendingWorking);
  });

  it('a model load or a stop keeps its own row between steps', () => {
    const loading = {
      ...hostWith([]),
      inference: 'loading' as const,
      loading: [{ modelId: 'qwen-a', startedAt: NOW - 5_000, phase: 'weights' as const, stepsDone: 0, steps: 0 }],
    };
    const container = mount(threadOf(afterRead, { conversationId: 'conv-1', model: 'qwen-a', host: loading }));
    expect(container.querySelector('[data-load-card]')).not.toBeNull();
    expect(line(container)).toBeNull();
    rerender(
      threadOf(
        { ...afterRead, turn: { turnId: 't', phase: 'cancelling', startedAt: NOW - 600_000, wait: null } },
        props([request({ phase: 'decode' })]),
      ),
    );
    expect(waitText(line(container))).toBe(CHAT.pendingStopping);
  });

  it('hides while a tool runs, while text streams, and while a permission card is open', () => {
    vi.useFakeTimers({ now: NOW, toFake: ['Date', 'setTimeout', 'clearTimeout'] });
    try {
      const container = mount(threadOf(work([tool('in_progress', NOW - 1_000)]), props([request({})])));
      expect(line(container)).toBeNull();
      rerender(threadOf(work([tool('pending', NOW - 1_000)]), props([request({})])));
      expect(line(container)).toBeNull();
      rerender(threadOf(afterRead, { ...props([request({})]), asking: true }));
      expect(line(container)).toBeNull();
      const streaming = work([tool('completed', NOW - 9_000, NOW - 8_000), answer('Now I', NOW - 500)]);
      rerender(threadOf(streaming, { ...props([request({})]), asOfMs: NOW - 500 }));
      expect(line(container)).toBeNull();
      expect(container.textContent).toContain('Now I');
      act(() => {
        vi.advanceTimersByTime(3_000);
      });
      expect(waitText(line(container))).toBe(CHAT.pendingThinking);
      rerender(threadOf(work([thinking('Hm', NOW, NOW + 100)]), { ...props([]), asOfMs: NOW + 3_100 }));
      expect(line(container)).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('a thought or text quiet for 3 s hands its live mark to the wait line: one Thinking, not two', () => {
    vi.useFakeTimers({ now: NOW, toFake: ['Date', 'setTimeout', 'clearTimeout'] });
    try {
      const live = (container: HTMLElement): Element[] => [...container.querySelectorAll('[data-live]')];
      const thought = work([thinking('Now call the build', NOW - 500, NOW - 400)]);
      const container = mount(threadOf(thought, { ...props([request({})]), asOfMs: NOW - 400 }));
      expect(line(container)).toBeNull();
      expect(live(container)).toHaveLength(1);
      expect(container.textContent).toContain(CHAT.thinking);
      act(() => {
        vi.advanceTimersByTime(3_000);
      });
      expect(waitText(line(container))).toBe(CHAT.pendingThinking);
      expect(live(container)).toHaveLength(0);
      expect(container.textContent?.split(CHAT.thinking)).toHaveLength(2);
      expect(container.textContent).toContain(CHAT.thought);
      rerender(threadOf(work([answer('Now I', NOW + 2_900)]), { ...props([request({})]), asOfMs: NOW + 3_000 }));
      expect(line(container)).toBeNull();
      expect(container.querySelector('[data-role="assistant"][data-live]')).not.toBeNull();
      act(() => {
        vi.advanceTimersByTime(3_000);
      });
      expect(waitText(line(container))).toBe(CHAT.pendingThinking);
      expect(container.querySelector('[data-role="assistant"][data-live]')).toBeNull();
      expect(container.textContent).not.toContain(CHAT.writing);
    } finally {
      vi.useRealTimers();
    }
  });

  it('the clock restarts at each new wait', () => {
    timers();
    try {
      const first = work([tool('completed', NOW - 70_000, NOW - 61_000)]);
      const container = mount(threadOf(first, props([])));
      expect(clock(container)).toBe('1:01');
      act(() => {
        vi.advanceTimersByTime(2_000);
      });
      expect(clock(container)).toBe('1:03');
      const running = work([
        tool('completed', NOW - 70_000, NOW - 61_000),
        tool('in_progress', NOW + 2_000, undefined, 'r2'),
      ]);
      rerender(threadOf(running, props([])));
      expect(line(container)).toBeNull();
      act(() => {
        vi.advanceTimersByTime(10_000);
      });
      const second = work([
        tool('completed', NOW - 70_000, NOW - 61_000),
        tool('completed', NOW + 2_000, NOW + 12_000, 'r2'),
      ]);
      rerender(threadOf(second, props([])));
      expect(clock(container)).toBe('0:00');
      act(() => {
        vi.advanceTimersByTime(4_000);
      });
      expect(clock(container)).toBe('0:04');
    } finally {
      vi.useRealTimers();
    }
  });
});

describe('ChatThread: the compaction line and its marker', () => {
  const NOW = 1_794_411_600_000;
  const tool = (ts: number, endTs: number): TranscriptEntry => ({
    kind: 'tool',
    ts,
    id: 'r1',
    name: 'probe',
    title: 'Probe src/main.rs',
    status: 'completed',
    endTs,
  });
  const work = (entries: TranscriptEntry[]): ThreadSample => ({
    mode: 'work',
    entries: [user('Fix the build', NOW - 600_000), ...entries],
    turns: [],
    turn: { turnId: 't', phase: 'running', startedAt: NOW - 600_000, wait: null },
    lastEnd: null,
  });
  const afterRead = work([thinking('Look at main', NOW - 200_000, NOW - 190_000), tool(NOW - 189_000, NOW - 65_000)]);
  const compaction = { phase: 'running' as const, startedAt: NOW - 42_000, tokensBefore: 222_822 };
  type Request = NonNullable<NonNullable<ChatThreadProps['host']>['requests']>[number];
  const request = (over: Partial<Request>): Request => ({
    session: 'conv-1',
    modelId: 'qwen-a',
    phase: 'decode',
    startedAt: NOW - 40_000,
    prefillDone: null,
    prefillTotal: null,
    generated: null,
    ...over,
  });
  const props = (requests: Request[], over: Partial<ChatThreadProps> = {}): Partial<ChatThreadProps> => ({
    conversationId: 'conv-1',
    model: 'qwen-a',
    host: {
      inference: 'ready',
      loading: [],
      resident: [{ modelId: 'qwen-a', sizeBytes: null }],
      answers: [{ modelId: 'qwen-a', inFlight: 1, maxConcurrent: 4, queued: 0, lastStart: NOW - 40_000 }],
      requests,
    },
    compaction,
    ...over,
  });
  const lines = (container: HTMLElement): HTMLElement[] => [
    ...container.querySelectorAll<HTMLElement>('article[aria-busy="true"][data-wait-line]'),
  ];
  const detail = (container: HTMLElement): string | null =>
    container.querySelector('[data-compaction-line] article .text-muted')?.textContent ?? null;
  const help = (container: HTMLElement): boolean =>
    container.querySelector('[data-compaction-line] > p')?.textContent === CHAT.compactingHelp;

  it('takes the place of the wait line: one line with the fold orb and the clock since the compaction started', () => {
    vi.useFakeTimers({ now: NOW, toFake: ['Date', 'setInterval', 'clearInterval', 'setTimeout', 'clearTimeout'] });
    try {
      const container = mount(threadOf(afterRead, props([])));
      expect(lines(container)).toHaveLength(1);
      const line = lines(container)[0]!;
      expect(line.getAttribute('data-wait-line')).toBe('fold');
      expect(line.querySelector('canvas')?.getAttribute('data-wait-orb')).toBe('fold');
      expect(waitText(line)).toBe(CHAT.compacting);
      expect(line.querySelector('[data-wait-clock]')?.textContent).toBe('0:42');
      expect(detail(container)).toBeNull();
      expect(help(container)).toBe(true);
      act(() => {
        vi.advanceTimersByTime(3_000);
      });
      expect(lines(container)).toHaveLength(1);
      expect(line.querySelector('[data-wait-clock]')?.textContent).toBe('0:45');
      rerender(threadOf(afterRead, props([], { compaction: null })));
      expect(lines(container)).toHaveLength(1);
      expect(lines(container)[0]!.getAttribute('data-wait-line')).not.toBe('fold');
      expect(container.querySelector('[data-compaction-line]')).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('shows while text would hide the wait line, and the tail hands it its live mark', () => {
    vi.useFakeTimers({ now: NOW, toFake: ['Date', 'setTimeout', 'clearTimeout'] });
    try {
      const streaming = work([answer('Now I', NOW - 100)]);
      const container = mount(threadOf(streaming, props([], { asOfMs: NOW - 100 })));
      expect(lines(container).map((l) => l.getAttribute('data-wait-line'))).toEqual(['fold']);
      expect(container.querySelectorAll('[data-live]')).toHaveLength(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it("says what the conversation's summary request does: reading with a bar, then writing with its tokens", () => {
    const reading = request({ phase: 'prefill', prefillDone: 113_200, prefillTotal: 172_200 });
    const container = mount(threadOf(afterRead, props([reading])));
    expect(detail(container)).toBe(fill(CHAT.compactingReading, { done: '113.2K', total: '172.2K' }));
    expect(byRole(container, 'meter', CHAT.pendingReadMeter)?.getAttribute('aria-valuenow')).toBe('66');
    rerender(threadOf(afterRead, props([request({ generated: 3_200 })])));
    expect(detail(container)).toBe(fill(CHAT.compactingWritingOf, { n: '3,200' }));
    expect(container.querySelector('[role="meter"]')).toBeNull();
    rerender(threadOf(afterRead, props([request({ generated: 12_400 })])));
    expect(detail(container)).toBe(fill(CHAT.compactingWritingOf, { n: '12.4K' }));
    rerender(threadOf(afterRead, props([request({})])));
    expect(detail(container)).toBe(CHAT.compactingWriting);
    rerender(threadOf(afterRead, props([request({ phase: 'queued' })])));
    expect(detail(container)).toBeNull();
    // A request from before the compaction, or another conversation's, says nothing about it.
    rerender(threadOf(afterRead, props([request({ startedAt: NOW - 50_000, generated: 900 })])));
    expect(detail(container)).toBeNull();
    rerender(threadOf(afterRead, props([request({ session: 'conv-2', generated: 900 })])));
    expect(detail(container)).toBeNull();
    expect(lines(container)).toHaveLength(1);
  });

  it('the help sentence shows only for the first compaction of a conversation', () => {
    const before: TranscriptEntry = { kind: 'compaction', ts: NOW - 300_000, outcome: 'done' };
    const sample = work([before, tool(NOW - 189_000, NOW - 65_000)]);
    const container = mount(threadOf(sample, props([])));
    expect(lines(container)).toHaveLength(1);
    expect(help(container)).toBe(false);
  });

  it('a marker at the place of each compaction: counts and time when the agent gave them, failed, stopped', () => {
    const at = NOW - 100_000;
    const entries: TranscriptEntry[] = [
      { kind: 'compaction', ts: at, outcome: 'done', tokensBefore: 222_822, tokensAfter: 31_204 },
      { kind: 'compaction', ts: at, outcome: 'done', tokensAfter: 9_000 },
      { kind: 'compaction', ts: at, outcome: 'failed', retry: true },
      { kind: 'compaction', ts: at, outcome: 'failed' },
      { kind: 'compaction', ts: at, outcome: 'stopped' },
      { kind: 'compaction', ts: 0, outcome: 'stopped' },
    ];
    const container = mount(
      threadOf({ ...work([...entries, answer('Done.', NOW - 1_000)]), turn: null, lastEnd: null }),
    );
    const clockAt = formatClock(at);
    expect([...container.querySelectorAll('[data-compaction-mark]')].map((m) => m.textContent)).toEqual([
      `${fill(CHAT.compactedTokens, { before: '222.8K', after: '31.2K' })} · ${clockAt}`,
      `${CHAT.compacted} · ${clockAt}`,
      `${CHAT.compactFailedRetry} · ${clockAt}`,
      `${CHAT.compactFailed} · ${clockAt}`,
      `${CHAT.compactStopped} · ${clockAt}`,
      CHAT.compactStopped,
    ]);
    expect(fill(CHAT.compactedTokens, { before: '222.8K', after: '31.2K' })).toBe(
      'Conversation compacted · 222.8K → 31.2K tokens',
    );
    expect(lines(container)).toHaveLength(0);
  });

  // Mutation: drop the `after * 100 > before * 99` check in compactionText → "236K → 235.9K tokens" and the rest.
  it('a compaction whose count did not fall (equal, higher, or under 1% lower) shows no numbers', () => {
    const at = NOW - 100_000;
    const entries: TranscriptEntry[] = [
      { kind: 'compaction', ts: at, outcome: 'done', tokensBefore: 235_952, tokensAfter: 235_939 },
      { kind: 'compaction', ts: at, outcome: 'done', tokensBefore: 235_952, tokensAfter: 235_952 },
      { kind: 'compaction', ts: at, outcome: 'done', tokensBefore: 235_952, tokensAfter: 236_100 },
      { kind: 'compaction', ts: at, outcome: 'done', tokensBefore: 10_000, tokensAfter: 9_901 },
      { kind: 'compaction', ts: at, outcome: 'done', tokensBefore: 10_000, tokensAfter: 9_900 },
    ];
    const container = mount(
      threadOf({ ...work([...entries, answer('Done.', NOW - 1_000)]), turn: null, lastEnd: null }),
    );
    const clockAt = formatClock(at);
    expect([...container.querySelectorAll('[data-compaction-mark]')].map((m) => m.textContent)).toEqual([
      `${CHAT.compacted} · ${clockAt}`,
      `${CHAT.compacted} · ${clockAt}`,
      `${CHAT.compacted} · ${clockAt}`,
      `${CHAT.compacted} · ${clockAt}`,
      `${fill(CHAT.compactedTokens, { before: '10K', after: '9,900' })} · ${clockAt}`,
    ]);
  });
});

describe('ChatThread: the workflow run card', () => {
  const card = (container: HTMLElement): HTMLElement => {
    const found = container.querySelector<HTMLElement>(`[data-run="${RUN_ID}"]`)?.closest('section');
    if (found == null) throw new Error('no run card');
    return found;
  };
  const heading = (container: HTMLElement): string => card(container).querySelector('h3')?.textContent ?? '';
  const stopNone = (): Promise<void> => Promise.resolve();
  const status = (container: HTMLElement): string =>
    card(container).querySelector('header [data-tone]')?.textContent ?? '';

  it('running: title, status, step line, helpers used of the budget, one row per helper, and Stop run', () => {
    const container = mount(
      threadOf(runSample(workflowRun()), { runs: [{ runId: RUN_ID, stopping: false }], onStopRun: stopNone }),
    );
    expect(heading(container)).toBe(fill(CHAT.run.title, { name: 'deep-research' }));
    const text = card(container).textContent ?? '';
    expect(status(container)).toBe(CHAT.run.status.active);
    expect(text).toContain(fill(CHAT.run.phase, { i: 2, n: 4, phase: 'Research' }));
    expect(text).toContain(fill(CHAT.run.helpersUsed, { used: 5, budget: 128 }));
    const steps = byRole(card(container), 'list', CHAT.run.steps)!;
    expect(steps.querySelector('[aria-current="step"]')?.textContent).toContain('Research');
    expect(byRole(card(container), 'meter', fill(CHAT.run.helpersUsed, { used: 5, budget: 128 }))).not.toBeNull();
    const helpers = byRole(card(container), 'button', fill(CHAT.run.helpers, { n: 3 }))!;
    expect(helpers.getAttribute('aria-expanded')).toBe('true');
    for (const label of ['research-planner', 'researcher-0', 'researcher-1']) expect(text).toContain(label);
    expect(text).toContain('11K tokens · 2m 47s');
    expect(text).toContain(CHAT.run.helperState.failed);
    expect(byRole(card(container), 'button', CHAT.run.stop)).not.toBeNull();
    expect(card(container).getAttribute('aria-busy')).toBe('true');
  });

  it('Stop run asks first, keeps running on the cancel button, and sends the run id on confirm', async () => {
    const onStopRun = vi.fn(() => Promise.resolve());
    const container = mount(
      threadOf(runSample(workflowRun()), { runs: [{ runId: RUN_ID, stopping: false }], onStopRun }),
    );
    act(() => byRole(card(container), 'button', CHAT.run.stop)!.click());
    await waitFor(() => byRole(document.body, 'dialog') !== null);
    const dialog = byRole(document.body, 'dialog')!;
    expect(dialog.textContent).toContain(CHAT.run.stopTitle);
    expect(dialog.textContent).toContain(CHAT.run.stopText);
    const keep = allByRole(dialog, 'button').find((b) => b.textContent === CHAT.run.stopKeep)!;
    act(() => keep.click());
    expect(onStopRun).not.toHaveBeenCalled();
    act(() => byRole(card(container), 'button', CHAT.run.stop)!.click());
    await waitFor(() => byRole(document.body, 'dialog') !== null);
    const confirm = allByRole(byRole(document.body, 'dialog')!, 'button').find((b) => b.textContent === CHAT.run.stop)!;
    await act(async () => {
      confirm.click();
      await Promise.resolve();
    });
    expect(onStopRun).toHaveBeenCalledWith(RUN_ID);
  });

  it('stopping: the status says so and Stop run is gone', () => {
    const container = mount(
      threadOf(runSample(workflowRun()), { runs: [{ runId: RUN_ID, stopping: true }], onStopRun: stopNone }),
    );
    expect(status(container)).toBe(CHAT.run.stopping);
    expect(byRole(card(container), 'button', CHAT.run.stop)).toBeNull();
  });

  it('waiting: a helper of this run asks, the card says who, and the request card below is not the turn waiting', () => {
    const asks = new Map([['deep-research', 'researcher-0']]);
    const container = mount(
      threadOf(runSample(workflowRun()), {
        runs: [{ runId: RUN_ID, stopping: false }],
        runAsks: asks,
        onStopRun: stopNone,
        card: createElement('div', { 'data-interaction-card': '' }, 'Allow this command?'),
      }),
    );
    const text = card(container).textContent ?? '';
    expect(status(container)).toBe(CHAT.run.waitingYou);
    expect(text).toContain(fill(CHAT.run.asks, { label: 'researcher-0' }));
    expect(byRole(card(container), 'button', CHAT.run.stop)).not.toBeNull();
    // The request card is drawn at the thread's end, after the run card.
    const request = container.querySelector('[data-interaction-card]')!;
    expect(card(container).compareDocumentPosition(request) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it('a run with no turn: no wait line says the agent works; the live run card is the only sign', () => {
    const container = mount(
      threadOf(runSample(workflowRun()), {
        runs: [{ runId: RUN_ID, stopping: false }],
        conversationId: 'conv-1',
        host: { inference: 'ready', loading: [], resident: [], answers: [], requests: [] },
      }),
    );
    expect(container.querySelector('[data-wait-line]')).toBeNull();
    expect(container.querySelector('[data-compaction-line]')).toBeNull();
    expect(status(container)).toBe(CHAT.run.status.active);
  });

  it('a wake turn while a run is live shows the wait line as for any turn', () => {
    const sample = runSample(workflowRun(), {
      turnId: 'w1',
      phase: 'running',
      startedAt: T0 + 800,
      wait: null,
      origin: 'wake',
    });
    const container = mount(threadOf(sample, { runs: [{ runId: RUN_ID, stopping: false }] }));
    expect(container.querySelector('[data-wait-line]')).not.toBeNull();
  });

  it('done: Done, no step line, no Stop run, the report open as Markdown, the helpers closed', () => {
    const container = mount(threadOf(runSample(finishedRun())));
    const text = card(container).textContent ?? '';
    expect(status(container)).toBe(CHAT.run.status.complete);
    expect(text).not.toContain('Step ');
    expect(text).toContain(fill(CHAT.run.helpersUsed, { used: 8, budget: 128 }));
    expect(byRole(card(container), 'button', CHAT.run.stop)).toBeNull();
    const report = byRole(card(container), 'button', CHAT.run.report)!;
    expect(report.getAttribute('aria-expanded')).toBe('true');
    expect(text).toContain('Findings');
    expect(card(container).querySelector('strong')?.textContent).toBe('4 lanes');
    expect(byRole(card(container), 'button', fill(CHAT.run.helpers, { n: 3 }))?.getAttribute('aria-expanded')).toBe(
      'false',
    );
    expect(byRole(card(container), 'list', CHAT.run.steps)?.querySelector('[aria-current="step"]')).toBeNull();
  });

  it('a cut report says so', () => {
    const container = mount(threadOf(runSample(finishedRun({ truncated: true }))));
    expect(card(container).textContent).toContain(CHAT.run.reportCut);
  });

  it('ended early: active on disk but not held by the host reads "Ended early", with no Stop run', () => {
    const container = mount(threadOf(runSample(workflowRun()), { runs: [], onStopRun: stopNone }));
    const text = card(container).textContent ?? '';
    expect(status(container)).toBe(CHAT.run.status.interrupted);
    expect(text).not.toContain('Step ');
    expect(byRole(card(container), 'button', CHAT.run.stop)).toBeNull();
  });

  it('paused, out of helpers, failed and stopped: their words, a pause message, and no Stop run', () => {
    const cases: Array<[NonNullable<Parameters<typeof workflowRun>[0]>, string]> = [
      [{ status: 'user_paused', pauseMessage: 'Waiting for a key' }, CHAT.run.status.paused],
      [{ status: 'budget_limited' }, CHAT.run.status.budget_limited],
      [{ status: 'failed' }, CHAT.run.status.failed],
      [{ status: 'cancelled' }, CHAT.run.status.cancelled],
      [{ status: 'blocked' }, CHAT.run.status.blocked],
    ];
    for (const [over, word] of cases) {
      const container = mount(
        threadOf(runSample(workflowRun(over)), { runs: [{ runId: RUN_ID, stopping: false }], onStopRun: stopNone }),
      );
      expect(status(container), word).toBe(word);
      expect(byRole(card(container), 'button', CHAT.run.stop), word).toBeNull();
      if (over.pauseMessage !== undefined)
        expect(card(container).textContent).toContain(fill(CHAT.run.paused, { message: 'Waiting for a key' }));
      act(() => mounted!.root.unmount());
      container.remove();
      mounted = undefined;
    }
  });

  it('a run with no budget says how many helpers it used', () => {
    const container = mount(threadOf(runSample(workflowRun({ agentBudget: null }))));
    expect(card(container).textContent).toContain(fill(CHAT.run.helpersUsedNoBudget, { used: 5 }));
    expect(byRole(card(container), 'meter')).toBeNull();
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
