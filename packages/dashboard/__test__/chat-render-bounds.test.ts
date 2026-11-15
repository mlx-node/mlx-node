/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { ChatMarkdown } from '../ui/src/components/arc-x/chat-markdown/chat-markdown.js';
import { ReasoningBlock } from '../ui/src/components/arc-x/reasoning-block/reasoning-block.js';
import { ChatThread } from '../ui/src/components/chat-thread.js';
import { useStickToBottom, type StickToBottom } from '../ui/src/lib/stick-to-bottom.js';
import type { TranscriptEntry } from '../ui/src/lib/types.js';
import { formatSearchText } from '../ui/src/lib/web-result-text.js';
import { conversationHosts } from '../ui/src/lib/web-tools.js';
import { answer, directSearch, end, user } from './helpers/chat-thread-fixtures.js';
import { skipAnimations } from './render.js';

const parsed = vi.hoisted(() => ({ chars: 0 }));
vi.mock('../ui/src/lib/markdown-parse.js', async (importOriginal) => {
  const real = await importOriginal<typeof import('../ui/src/lib/markdown-parse.js')>();
  return {
    ...real,
    parseMarkdown: (text: string, base?: number) => {
      parsed.chars += text.length;
      return real.parseMarkdown(text, base);
    },
  };
});

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
  skipAnimations(true);
  vi.restoreAllMocks();
  if (mounted === undefined) return;
  const { root, container } = mounted;
  act(() => root.unmount());
  container.remove();
  mounted = undefined;
});

const RLO = String.fromCharCode(0x202e);
const BIDI = [0x202a, 0x202b, 0x202c, 0x202d, 0x202e, 0x2066, 0x2067, 0x2068, 0x2069, 0x200e, 0x200f].map((code) =>
  String.fromCharCode(code),
);
const FORBIDDEN_TAGS = new Set([
  'script',
  'img',
  'iframe',
  'object',
  'embed',
  'style',
  'link',
  'meta',
  'base',
  'form',
  'frame',
  'video',
  'audio',
  'source',
  'foreignobject',
  'math',
]);

/** A line of about 1 MB with a hostile piece every few KB. */
function megabyteLine(): string {
  const filler = 'lorem ipsum dolor sit amet '.repeat(150);
  const parts: string[] = [];
  let size = 0;
  for (let i = 0; size < 1_000_000; i += 1) {
    const piece = `${filler}**[x${i}](https://evil.example/${i} "t") <b onclick=alert(${i})>b</b> [y](javascript:alert(${i})) *${RLO}`;
    parts.push(piece);
    size += piece.length;
  }
  return parts.join('');
}

function hostileTurn(): TranscriptEntry[] {
  const results = formatSearchText('exa', [
    {
      title: `Release notes ${RLO}txt.exe`,
      url: 'https://ok.example/notes',
      date: null,
      snippet: '<script>alert(1)</script>',
    },
  ]);
  const tool: TranscriptEntry = {
    kind: 'tool',
    ts: 1,
    id: 'g1',
    name: 'other_tool',
    title: `Run ${RLO}gpj.exe <img src=x onerror=alert(1)>`,
    status: 'completed',
    input: '<script>alert(1)</script>',
    output: '<iframe src="https://evil.example/frame"></iframe>',
    endTs: 2,
  };
  const text = [
    'Here <script>alert(1)</script> and <img src=x onerror=alert(1)> and <a href="https://evil.example/raw">raw</a>.',
    `[click](javascript:alert(1)) [here](https://evil.example/?q=secret) ![pix](https://evil.example/p.png) <https://evil.example/auto> https://evil.example/bare [notes](https://ok.example/notes) [1] [2] file ${RLO}gpj.exe`,
    '',
    megabyteLine(),
    '',
    'Sources:',
    '[1] Release notes - https://ok.example/notes',
    '[2] Official guide - https://evil.example/guide',
  ].join('\n');
  return [user('Find the release notes'), directSearch(`notes ${RLO}gpj`, results), tool, answer(text)];
}

/** Every element and attribute came from the components, not from the text; anchors go only to allowed hosts. */
function expectInert(container: HTMLElement, allowed: ReadonlySet<string>): void {
  const wrong: string[] = [];
  for (const element of container.querySelectorAll('*')) {
    const tag = element.tagName.toLowerCase();
    if (FORBIDDEN_TAGS.has(tag)) wrong.push(`<${tag}>`);
    for (const attribute of element.attributes) {
      const { name, value } = attribute;
      if (name.startsWith('on') || name === 'src' || name === 'srcdoc') wrong.push(`${tag}[${name}]`);
      // A name or tooltip may quote the text; any other attribute holding it came from it.
      const quotes = name === 'aria-label' || name === 'title';
      if (
        (!quotes && (value.includes('javascript:') || value.includes('alert('))) ||
        BIDI.some((m) => value.includes(m))
      )
        wrong.push(`${tag}[${name}=${value.slice(0, 60)}]`);
    }
  }
  for (const anchor of container.querySelectorAll('a')) {
    const href = anchor.getAttribute('href') ?? '';
    if (href.startsWith('/') && !href.startsWith('//')) continue;
    const url = URL.canParse(href) ? new URL(href) : null;
    if (url === null || (url.protocol !== 'https:' && url.protocol !== 'http:') || !allowed.has(url.hostname))
      wrong.push(`a[href=${href.slice(0, 60)}]`);
  }
  expect(wrong).toEqual([]);
}

describe('a hostile turn', () => {
  it('renders as text: no element, attribute or anchor comes from it, and it renders in time', () => {
    const entries = hostileTurn();
    const allowed = conversationHosts(entries);
    expect([...allowed]).toEqual(['ok.example']);
    const thread = (finished: boolean): ReactElement =>
      createElement(
        MemoryRouter,
        null,
        createElement(ChatThread, {
          mode: 'chat',
          entries: finished ? [...entries, end()] : entries,
          turns: [],
          turn: finished ? null : { turnId: 't1', phase: 'running', startedAt: 0, wait: null },
          lastEnd: null,
        }),
      );

    let started = performance.now();
    const container = mount(thread(false));
    expect(performance.now() - started).toBeLessThan(5_000);
    expectInert(container, allowed);

    started = performance.now();
    rerender(thread(true));
    expect(performance.now() - started).toBeLessThan(5_000);
    expectInert(container, allowed);

    const text = container.textContent ?? '';
    expect(text.length).toBeGreaterThan(900_000);
    expect(text).toContain('<script>alert(1)</script>');
    expect(text).toContain('here (evil.example)');
    for (const mark of BIDI) expect(text.includes(mark)).toBe(false);
    const sourceButtons = [...container.querySelectorAll('button')].filter((b) =>
      (b.getAttribute('aria-label') ?? '').startsWith('Source '),
    );
    expect(sourceButtons.map((b) => b.getAttribute('aria-label'))).toEqual([
      'Source 1: Release notes txt.exe, ok.example',
    ]);
  }, 60_000);
});

describe('streaming cost', () => {
  it('each settled block of a streaming answer is parsed once', () => {
    const paragraphs = Array.from({ length: 300 }, (_, i) => `Paragraph ${i} ${'word '.repeat(40)}`);
    const element = (text: string): ReactElement =>
      createElement(ChatMarkdown, { text, streaming: true, links: { mode: 'work' } });
    let text = '';
    mount(element(text));
    parsed.chars = 0;
    for (const paragraph of paragraphs) {
      rerender(element(text + paragraph.slice(0, 100)));
      text += `${paragraph}\n\n`;
      rerender(element(text));
    }
    expect(parsed.chars).toBeLessThan(text.length * 8);
  });

  it('a live thinking tail draws only the end of the text', () => {
    skipAnimations(false);
    const text = 'thought '.repeat(125_000);
    const container = mount(
      createElement(ReasoningBlock, { text, label: 'Thinking', live: true, startTs: 0, asOfMs: 1_000 }),
    );
    const longest = Math.max(...[...container.querySelectorAll('div')].map((node) => node.textContent?.length ?? 0));
    expect(longest).toBeGreaterThan(1_000);
    expect(longest).toBeLessThan(10_000);
  });
});

describe('jump to latest', () => {
  it('a wider or narrower column is not new content', () => {
    const observers: Array<() => void> = [];
    vi.stubGlobal(
      'ResizeObserver',
      class {
        constructor(callback: () => void) {
          observers.push(callback);
        }
        observe(): void {}
        disconnect(): void {}
      },
    );
    let box = { width: 600, height: 400 };
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(
      () => ({ ...box, top: 0, left: 0, right: box.width, bottom: box.height, x: 0, y: 0 }) as DOMRect,
    );
    const root = document.createElement('div');
    document.body.appendChild(root);
    let stick: StickToBottom | undefined;
    function Probe(): ReactElement {
      stick = useStickToBottom(root, 'resize');
      return createElement('div', { ref: stick.contentRef });
    }
    try {
      mount(createElement(Probe));
      expect(observers).toHaveLength(1);
      act(() => {
        root.dispatchEvent(new WheelEvent('wheel', { deltaY: -40 }));
      });
      expect(stick!.pinned).toBe(false);
      box = { width: 400, height: 700 };
      act(() => observers[0]!());
      expect(stick!.fresh).toBe(false);
      box = { width: 400, height: 900 };
      act(() => observers[0]!());
      expect(stick!.fresh).toBe(true);
    } finally {
      vi.unstubAllGlobals();
      root.remove();
    }
  });
});
