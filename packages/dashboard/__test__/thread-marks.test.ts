/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { EventRow } from '../ui/src/components/arc-x/event-row/event-row.js';
import { ExactText, InteractionCard } from '../ui/src/components/arc-x/interaction-card/interaction-card.js';
import { AllowedLine, ModeLine } from '../ui/src/components/chat-permission-marks.js';
import { CHAT } from '../ui/src/lib/chat-strings.js';
import type { PermissionInteraction, PermissionMark } from '../ui/src/lib/chat-types.js';
import { formatClock } from '../ui/src/lib/format.js';
import {
  cardDescription,
  changedText,
  type LineMark,
  PATH_SHOWN_MAX,
  settingsToast,
  shortPath,
} from '../ui/src/lib/permission-text.js';
import { cssRules, cssValue } from './helpers/css-rules.js';

const P = CHAT.permission;
const AT = Date.UTC(2026, 10, 13, 1, 36, 5);

let mounted: { root: Root; container: HTMLElement } | undefined;

function mount(element: ReactElement): HTMLElement {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(createElement(MemoryRouter, null, element)));
  mounted = { root, container };
  return container;
}

afterEach(() => {
  if (mounted === undefined) return;
  act(() => mounted!.root.unmount());
  mounted.container.remove();
  mounted = undefined;
});

/** The CSS module class of a name, as the test build writes it (`_line_hash`). */
function hasClass(node: Element, name: string): boolean {
  for (const cls of node.classList) if (cls === name || cls.startsWith(`_${name}_`)) return true;
  return false;
}

function markOf(mark: LineMark): Element {
  const container = mount(createElement(ModeLine, { mark, temp: null }));
  expect(container.children).toHaveLength(1);
  return container.firstElementChild!;
}

describe('a thread mark', () => {
  it('is one note: icon, label, fact and time on one line; no tab stop, no control', () => {
    const root = markOf({ kind: 'mode', at: AT, mode: 'auto', from: 'ask', by: 'user' });
    expect(root.getAttribute('role')).toBe('note');
    expect(root.getAttribute('data-thread-mark')).toBe('mode');
    expect(root.getAttribute('data-mode')).toBe('auto');
    expect(root.getAttribute('data-tone')).toBe('neutral');
    expect(root.hasAttribute('aria-label')).toBe(false);
    expect(root.children).toHaveLength(1);
    const line = root.firstElementChild!;
    expect(hasClass(line, 'line')).toBe(true);
    const parts = [...line.children].filter((n) => !hasClass(n, 'srOnly'));
    expect(parts.map((n) => n.tagName)).toEqual(['SPAN', 'SPAN', 'SPAN', 'TIME']);
    expect(parts[0]!.getAttribute('aria-hidden')).toBe('true');
    expect(parts[0]!.querySelector('svg')?.getAttribute('width')).toBe('12');
    expect(parts[1]!.textContent).toBe('Mode: Auto');
    expect(parts[2]!.textContent).toBe(P.lineAuto);
    expect(parts[3]!.getAttribute('datetime')).toBe(new Date(AT).toISOString());
    expect(root.textContent).toBe(`Mode: Auto. ${P.lineAuto} ${formatClock(AT)}`);
    expect(root.querySelector('button, a, [tabindex], p')).toBeNull();
    expect(root.querySelector('[data-wrap]')).toBeNull();
  });

  it('has a second line only for what the owner sees nowhere else, outside the line', () => {
    const cases: Array<[LineMark, string | null, string]> = [
      [{ kind: 'mode', at: AT, mode: 'auto', from: 'ask', by: 'user' }, null, 'neutral'],
      [
        { kind: 'mode', at: AT, mode: 'auto', from: 'ask', by: 'user', editsUnasked: true, userHooks: true },
        `${P.lineEditsUnasked} ${P.lineHooks}`,
        'neutral',
      ],
      [{ kind: 'mode', at: AT, mode: 'auto', from: 'bypass', by: 'user' }, P.lineAutoDown, 'neutral'],
      [{ kind: 'mode', at: AT, mode: 'bypass', from: 'ask', by: 'user' }, null, 'danger'],
      [
        { kind: 'mode', at: AT, mode: 'bypass', from: 'ask', by: 'user', userHooks: true },
        P.bypass.lineHooks,
        'danger',
      ],
      [{ kind: 'mode', at: AT, mode: 'ask', from: 'bypass', by: 'user' }, null, 'neutral'],
      [{ kind: 'mode', at: AT, mode: 'ask', from: 'bypass', by: 'restart' }, null, 'neutral'],
      [{ kind: 'mode', at: AT, mode: 'ask', from: 'auto', by: 'record-failed' }, null, 'neutral'],
    ];
    for (const [mark, detail, tone] of cases) {
      const root = markOf(mark);
      expect(root.getAttribute('data-tone'), JSON.stringify(mark)).toBe(tone);
      const second = root.querySelector('p');
      expect(second?.textContent ?? null, JSON.stringify(mark)).toBe(detail);
      if (second !== null) {
        expect(second.parentElement).toBe(root);
        expect(root.lastElementChild).toBe(second);
      }
      act(() => mounted!.root.unmount());
      mounted!.container.remove();
      mounted = undefined;
    }
  });

  it('a settings mark is a warning and names the changed files in code font, at most two', () => {
    const one = markOf({ kind: 'settings', at: AT, changed: ['~/.claude.json'] });
    expect(one.getAttribute('data-thread-mark')).toBe('settings');
    expect(one.getAttribute('data-tone')).toBe('warning');
    expect(one.textContent).toContain(`${P.settingsLabel}. ${P.settingsLine}`);
    expect(one.querySelector('p')?.textContent).toBe('Changed: ~/.claude.json.');
    expect([...one.querySelectorAll('p code')].map((c) => c.textContent)).toEqual(['~/.claude.json']);
    act(() => mounted!.root.unmount());
    mounted = undefined;

    const two = markOf({ kind: 'settings', at: AT, changed: ['/x/home/config.toml', '~/.claude.json'] });
    expect(two.querySelector('p')?.textContent).toBe('Changed: /x/home/config.toml and ~/.claude.json.');
    expect([...two.querySelectorAll('p code')].map((c) => c.textContent)).toEqual([
      '/x/home/config.toml',
      '~/.claude.json',
    ]);
    act(() => mounted!.root.unmount());
    mounted = undefined;

    const four = markOf({ kind: 'settings', at: AT, changed: ['~/a.toml', '~/b.json', '~/c.json', '~/d.json'] });
    expect(four.querySelector('p')?.textContent).toBe('Changed: ~/a.toml, ~/b.json and 2 more.');
    expect(four.querySelectorAll('p code')).toHaveLength(2);
  });

  it('a long path is shortened in the middle and breaks only as one word with the punctuation after it', () => {
    const long =
      '/private/tmp/claude-501/-Users-brooklyn-workspace-github-mlx-node/3f839611-f8c7-48e3-8900/home/folder/.mcp.json';
    const local = '~/workspace/github/mlx-node/.worktrees/app-redesign/.claude/settings.local.json';
    expect(shortPath(long)).toBe('/private/…/folder/.mcp.json');
    expect(shortPath(local)).toBe('~/workspace/…/.claude/settings.local.json');
    expect(shortPath('~/.claude/settings.json')).toBe('~/.claude/settings.json');
    expect(shortPath(`/${'n'.repeat(PATH_SHOWN_MAX + 10)}.json`)).toBe(`/${'n'.repeat(PATH_SHOWN_MAX + 10)}.json`);
    expect(changedText([local, long])).toBe(
      'Changed: ~/workspace/…/.claude/settings.local.json and /private/…/folder/.mcp.json.',
    );
    expect(settingsToast([long])).toBe(`${P.settingsText} Changed: /private/…/folder/.mcp.json.`);

    const cases: Array<[string[], string[], string]> = [
      [[long], ['.'], 'Changed: /private/…/folder/.mcp.json.'],
      [
        ['~/.claude/settings.json', long],
        ['', '.'],
        'Changed: ~/.claude/settings.json and /private/…/folder/.mcp.json.',
      ],
      [
        [local, long, '~/a.json'],
        [',', ''],
        `Changed: ~/workspace/…/.claude/settings.local.json, /private/…/folder/.mcp.json and 1 more.`,
      ],
    ];
    for (const [changed, after, text] of cases) {
      const root = markOf({ kind: 'settings', at: AT, changed });
      const second = root.querySelector('p')!;
      expect(second.textContent).toBe(text);
      const words = [...second.children];
      expect(words.every((word) => hasClass(word, 'path'))).toBe(true);
      expect(words.map((word) => word.querySelector('code')!.getAttribute('title'))).toEqual(changed.slice(0, 2));
      expect(words.map((word) => word.textContent!.slice(word.querySelector('code')!.textContent!.length))).toEqual(
        after,
      );
      for (const node of second.childNodes) {
        if (node.nodeType !== 3) continue;
        expect(['.', ','].includes(node.textContent!.trim()), JSON.stringify(node.textContent)).toBe(false);
      }
      act(() => mounted!.root.unmount());
      mounted!.container.remove();
      mounted = undefined;
    }
    expect(cssValue(cssRules('components/chat-permission-marks.module.css'), '.path', 'white-space')).toBe('nowrap');
  });

  it('is a rule, not a box: hairlines on both sides, no border, one line that never wraps, paths never broken', () => {
    const rules = cssRules('components/arc-x/thread-mark/thread-mark.module.css');
    for (const rule of rules) {
      for (const property of rule.decls.keys()) {
        expect(property.startsWith('border'), `${rule.selectors.join(', ')} ${property}`).toBe(false);
        expect(property, rule.selectors.join(', ')).not.toBe('outline');
      }
    }
    expect(cssValue(rules, '.mark::before', 'height')).toBe('1px');
    expect(cssValue(rules, '.mark::after', 'height')).toBe('1px');
    expect(cssValue(rules, '.mark', 'grid-template-columns')).toBe(
      'minmax(var(--space-4), 1fr) auto minmax(var(--space-4), 1fr)',
    );
    expect(cssValue(rules, '.line', 'white-space')).toBe('nowrap');
    expect(cssValue(rules, '.fact', 'text-overflow')).toBe('ellipsis');
    expect(cssValue(rules, '.detail code', 'white-space')).toBe('nowrap');
  });
});

describe('EventRow: one line, no wrap option', () => {
  it('the allowed line and the other rows keep their bordered one-line form', () => {
    const allowed: Extract<PermissionMark, { kind: 'allowed' }> = {
      kind: 'allowed',
      at: AT,
      mode: 'bypass',
      toolCallId: 'h1',
      access: 'mcp',
      card: null,
    };
    const shapes: ReactElement[] = [
      createElement(AllowedLine, { mark: allowed, temp: null }),
      createElement(EventRow, { icon: createElement('svg'), label: 'No prompt typed', text: '' }),
      createElement(EventRow, {
        icon: createElement('svg'),
        label: 'Subagent · explore',
        text: 'Explore the cache',
        to: '/sessions/child-1',
        facts: ['completed', null, '42.1 s'],
        tone: 'danger',
        error: 'linker not found',
        ts: AT,
      }),
      createElement(EventRow, { label: 'unknown', text: 'a kind the page does not know' }),
    ];
    for (const shape of shapes) {
      const container = mount(shape);
      const row = container.firstElementChild!;
      expect(hasClass(row, 'row')).toBe(true);
      expect(row.hasAttribute('data-wrap')).toBe(false);
      expect(row.getAttribute('role')).toBeNull();
      const line = row.firstElementChild!;
      expect(hasClass(line, 'line')).toBe(true);
      const text = line.querySelector('[class*="_text_"]');
      expect(text?.getAttribute('title')).toBe(text?.textContent);
      act(() => mounted!.root.unmount());
      mounted!.container.remove();
      mounted = undefined;
    }
    const rules = cssRules('components/arc-x/event-row/event-row.module.css');
    expect(rules.some((rule) => rule.selectors.some((s) => s.includes('data-wrap')))).toBe(false);
    expect(cssValue(rules, '.text', 'white-space')).toBe('nowrap');
    expect(cssValue(rules, '.row', 'border')).toBe('1px solid var(--border)');
  });
});

/** The server's cap of a card title. */
const TITLE_MAX = 200;

const bash = (
  title: string,
  command: string,
  titleCut?: true,
): Pick<PermissionInteraction, 'access' | 'title' | 'command' | 'path' | 'titleCut'> => ({
  access: 'bash',
  title,
  command,
  path: null,
  ...(titleCut === undefined ? {} : { titleCut }),
});

const LONG_COMMAND =
  'ps aux | grep -E "cargo|rustc|lldb" | grep -v grep | head -10; echo "---"; ls /tmp/napi-rs-a6ih 2>/dev/null | head; echo "---"; cd /tmp/napi-rs-a6ih 2>/dev/null && git status --short | head -10';

describe('cardDescription', () => {
  it('draws nothing when the title only names the command or the file of the well', () => {
    expect(cardDescription(bash('Execute `ls`', 'ls'))).toBeNull();
    expect(cardDescription(bash(`Execute \`${LONG_COMMAND}\``.slice(0, TITLE_MAX), LONG_COMMAND, true))).toBeNull();
    expect(cardDescription(bash('実行\u3000`ls`', 'ls'))).toBeNull();
    expect(cardDescription(bash('Execute', 'ls'))).toBeNull();
    expect(cardDescription(bash('run_terminal_command', 'npm test'))).toBeNull();
    expect(cardDescription(bash('Execute `printf a\nb`', 'printf a\nb'))).toBeNull();
    expect(
      cardDescription({ access: 'edit', title: 'Write ~/notes/out.txt', command: null, path: '~/notes/out.txt' }),
    ).toBeNull();
    expect(cardDescription({ access: 'edit', title: 'Edit n b.txt', command: null, path: 'n b.txt' })).toBeNull();
    expect(cardDescription(bash('Execute `ls` — ', 'ls'))).toBeNull();
  });

  it('draws only the words that follow the exact text, else the whole title', () => {
    expect(cardDescription(bash("Execute `ls` — hook 'mine' asks for confirmation", 'ls'))).toBe(
      "hook 'mine' asks for confirmation",
    );
    expect(cardDescription(bash("Execute `ls` — hook 'mine' asks: why", 'ls'))).toBe("hook 'mine' asks: why");
    expect(cardDescription(bash('Run the tests', 'npm test'))).toBe('Run the tests');
    expect(cardDescription({ access: 'mcp', title: 'github__create_issue', command: null, path: null })).toBe(
      'github__create_issue',
    );
    expect(cardDescription({ access: 'other', title: 'Ask the helper', command: null, path: null })).toBe(
      'Ask the helper',
    );
    expect(cardDescription(bash('   ', 'ls'))).toBeNull();
  });

  it("names the command or file only where it ends at a word edge; a longer word is the agent's own text", () => {
    expect(cardDescription(bash('Execute `ls -la`', 'ls'))).toBe('Execute `ls -la`');
    expect(cardDescription(bash('Execute `lsof -i`', 'ls'))).toBe('Execute `lsof -i`');
    expect(cardDescription(bash('Run gitk', 'git'))).toBe('Run gitk');
    expect(cardDescription(bash('Run npm-check-updates', 'npm'))).toBe('Run npm-check-updates');
    expect(cardDescription(bash('Run n', 'npm test'))).toBe('Run n');
    expect(cardDescription(bash('Run git status now', 'git status'))).toBe('now');
    expect(cardDescription({ access: 'edit', title: 'Edit src/app.tsx', command: null, path: 'src/app.ts' })).toBe(
      'Edit src/app.tsx',
    );
  });

  it('a one-word title is hidden only when it is the target, a verb or a tool name of the agent', () => {
    expect(cardDescription(bash('运行测试', 'npm test'))).toBe('运行测试');
    expect(cardDescription(bash('ls', 'ls'))).toBeNull();
    expect(cardDescription(bash('Write', 'ls'))).toBeNull();
    expect(cardDescription({ access: 'edit', title: 'search_replace', command: null, path: 'a.ts' })).toBeNull();
  });

  it('a title the server cut ends at a whole word with an ellipsis; one it did not cut is kept whole', () => {
    const words = 'Collect the logs of every service that ran in the last hour and summarize them for the report';
    let long = `${words} ${words} ${words}`;
    while (long[TITLE_MAX - 1] === ' ' || long[TITLE_MAX] === ' ') long = `x${long}`;
    const under = long.slice(0, TITLE_MAX - 1);
    expect(cardDescription(bash(under, 'npm test'))).toBe(under);
    const exact = long.slice(0, TITLE_MAX);
    expect(cardDescription(bash(exact, 'npm test'))).toBe(exact);
    const cut = long.slice(0, TITLE_MAX);
    const shown = cardDescription(bash(cut, 'npm test', true))!;
    expect(shown.endsWith('…')).toBe(true);
    const kept = shown.slice(0, -1);
    expect(kept.endsWith(' ') || kept.endsWith('-')).toBe(false);
    expect(cut.startsWith(kept)).toBe(true);
    expect(cut[kept.length]).toBe(' ');
    expect(kept.length).toBeLessThan(cut.length);
  });

  it('a cut title whose last word is long keeps its text and adds the ellipsis; a cut emoji title says so', () => {
    const token = `Run ${'x'.repeat(196)}`;
    expect(cardDescription(bash(token, 'npm test', true))).toBe(`${token}…`);
    const emoji = `Run ${'😀'.repeat(97)}`;
    expect(emoji.length).toBe(TITLE_MAX - 2);
    expect(cardDescription(bash(emoji, 'npm test', true))).toBe(`${emoji}…`);
  });
});

describe('InteractionCard rhythm', () => {
  it('head and description, then the wells, the notices, then the answer: buttons, then notes', () => {
    const container = mount(
      createElement(
        InteractionCard,
        {
          title: 'Run this command?',
          subtitle: "hook 'mine' asks for confirmation",
          subtitleTitle: "Execute `ls` — hook 'mine' asks for confirmation",
          warnings: ['Deletes files.'],
          actions: createElement('button', { type: 'button' }, 'Deny'),
          notes: ['Deny stops this answer.'],
        },
        createElement(ExactText, {
          text: 'ls',
          label: 'Command',
          overflowNote: () => '',
          gapWords: { spaces: '{n} spaces', emptyLines: '{n} empty lines' },
        }),
      ),
    );
    const group = container.querySelector('[role="group"]')!;
    const kids = [...group.children];
    expect(kids).toHaveLength(4);
    expect(hasClass(kids[0]!, 'top')).toBe(true);
    expect(kids[1]!.querySelector('pre')?.textContent).toBe('ls');
    expect(kids[2]!.tagName).toBe('UL');
    expect(hasClass(kids[3]!, 'foot')).toBe(true);
    const top = [...kids[0]!.children];
    expect(hasClass(top[0]!, 'head')).toBe(true);
    expect(top[1]!.tagName).toBe('P');
    expect(hasClass(top[1]!, 'subtitle')).toBe(true);
    expect(top[1]!.getAttribute('title')).toBe("Execute `ls` — hook 'mine' asks for confirmation");
    const foot = [...kids[3]!.children];
    expect(hasClass(foot[0]!, 'actions')).toBe(true);
    expect(hasClass(foot[1]!, 'notes')).toBe(true);
    const rules = cssRules('components/arc-x/interaction-card/interaction-card.module.css');
    expect(cssValue(rules, '.subtitle', 'white-space')).toBe('nowrap');
    expect(cssValue(rules, '.subtitle', 'text-overflow')).toBe('ellipsis');
    expect(cssValue(rules, '.subtitle', '-webkit-line-clamp')).toBeUndefined();
    expect(cssValue(rules, '.foot', 'margin-top')).toBe('var(--space-1)');
  });
});
