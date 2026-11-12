/**
 * @vitest-environment happy-dom
 */

import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeAll, describe, expect, it } from 'vite-plus/test';

import { ChatMarkdown, visibleText, type LinkRule } from '../ui/src/components/arc-x/chat-markdown/chat-markdown.js';
import { parseMarkdown, plainText, type MdBlock, type MdInline } from '../ui/src/lib/markdown-parse.js';
import { repairTail } from '../ui/src/lib/stream-repair.js';
import { settledSplit } from '../ui/src/lib/stream-split.js';
import { skipAnimations } from './render.js';

function only<T extends MdBlock['type']>(blocks: MdBlock[], type: T): Extract<MdBlock, { type: T }> {
  expect(blocks).toHaveLength(1);
  expect(blocks[0]!.type).toBe(type);
  return blocks[0] as Extract<MdBlock, { type: T }>;
}

function inlinesOf(text: string): MdInline[] {
  return only(parseMarkdown(text), 'paragraph').children;
}

function kinds(nodes: readonly MdInline[]): string[] {
  return nodes.map((node) => node.type);
}

type Method = (this: unknown, ...args: unknown[]) => unknown;

/**
 * The work `run` does, counted through the built-ins it may call: an array shift or splice costs the elements it
 * moves, a push costs one, and a string trim, slice or join costs the characters it reads or makes.
 */
function countWork(run: () => void): number {
  let steps = 0;
  const undo: Array<() => void> = [];
  const wrap = (target: object, name: string, cost: (self: unknown, out: unknown) => number): void => {
    const methods = target as Record<string, Method>;
    const original = methods[name]!;
    methods[name] = function (this: unknown, ...args: unknown[]): unknown {
      const out = original.apply(this, args);
      steps += cost(this, out);
      return out;
    };
    undo.push(() => {
      methods[name] = original;
    });
  };
  const length = (value: unknown): number => (value as { length: number }).length;
  wrap(Array.prototype, 'shift', (self) => length(self) + 1);
  wrap(Array.prototype, 'splice', (self) => length(self) + 1);
  wrap(Array.prototype, 'push', () => 1);
  wrap(Array.prototype, 'join', (_self, out) => length(out));
  for (const name of ['trim', 'trimEnd', 'trimStart']) wrap(String.prototype, name, (self) => length(self));
  wrap(String.prototype, 'slice', (_self, out) => length(out));
  try {
    run();
  } finally {
    for (const restore of undo.reverse()) restore();
  }
  return steps;
}

/** How the work of `run` grows from size `n` to `4n`: about 4 when linear, about 16 when quadratic. */
function growth(run: (n: number) => void, n: number): number {
  return countWork(() => run(4 * n)) / countWork(() => run(n));
}

describe('parseMarkdown: blocks', () => {
  it('headings, rule, quote, lists, table and fenced code', () => {
    const blocks = parseMarkdown(
      [
        '# Title',
        '',
        '---',
        '',
        '> quoted **bold**',
        '',
        '- one',
        '- two',
        '',
        '3. three',
        '4. four',
        '',
        '| a | b |',
        '|:--|--:|',
        '| 1 | 2 |',
        '',
        '```ts',
        'const x = 1;',
        '```',
      ].join('\n'),
    );
    expect(blocks.map((block) => block.type)).toEqual(['heading', 'rule', 'quote', 'list', 'list', 'table', 'code']);
    const ordered = blocks[4] as Extract<MdBlock, { type: 'list' }>;
    expect(ordered.ordered).toBe(true);
    expect(ordered.start).toBe(3);
    const table = blocks[5] as Extract<MdBlock, { type: 'table' }>;
    expect(table.align).toEqual(['left', 'right']);
    expect(table.rows).toHaveLength(1);
    const code = blocks[6] as Extract<MdBlock, { type: 'code' }>;
    expect(code).toMatchObject({ lang: 'ts', value: 'const x = 1;', closed: true });
  });

  it('an unclosed fence runs to the end and says so; markers inside it are text', () => {
    const code = only(parseMarkdown('```py\nprint("**no**")\n[1](javascript:x)'), 'code');
    expect(code.closed).toBe(false);
    expect(code.value).toBe('print("**no**")\n[1](javascript:x)');
  });

  it('deep nesting stops at a fixed depth instead of recursing without end', () => {
    const deep = '>'.repeat(5_000) + ' x';
    const started = performance.now();
    const blocks = parseMarkdown(deep);
    expect(performance.now() - started).toBeLessThan(1_000);
    let depth = 0;
    let block: MdBlock | undefined = blocks[0];
    while (block?.type === 'quote') {
      depth += 1;
      block = block.children[0];
    }
    expect(depth).toBeLessThanOrEqual(24);
    const nested = parseMarkdown(Array.from({ length: 200 }, (_, i) => `${'  '.repeat(i)}- item`).join('\n'));
    expect(nested.length).toBeGreaterThan(0);
  });

  it('a table is bounded: no wide header, and a short row is never padded past twice its cells', () => {
    const wide = `|${'h|'.repeat(300)}\n|${'-|'.repeat(300)}\n${'a\n'.repeat(300)}`;
    expect(parseMarkdown(wide).some((block) => block.type === 'table')).toBe(false);
    const short = `|${'h|'.repeat(60)}\n|${'-|'.repeat(60)}\n${'a\n'.repeat(5_000)}`;
    const started = performance.now();
    const blocks = parseMarkdown(short);
    expect(performance.now() - started).toBeLessThan(2_000);
    const table = blocks.find((block): block is Extract<MdBlock, { type: 'table' }> => block.type === 'table');
    expect(table).toBeDefined();
    expect(table!.rows.every((row) => row.length <= 2)).toBe(true);
    const cells = table!.rows.reduce((n, row) => n + row.length, table!.head.length);
    expect(cells).toBeLessThanOrEqual(20_000);
    const normal = only(parseMarkdown('| a | b | c |\n|---|---|---|\n| 1 | 2 |'), 'table');
    expect(normal.rows[0]).toHaveLength(3);
  });
});

describe('parseMarkdown: inlines', () => {
  it('nested and crossed markers resolve like CommonMark', () => {
    expect(kinds(inlinesOf('***both***'))).toEqual(['em']);
    expect(plainText(inlinesOf('**a *b* c**'))).toBe('a b c');
    expect(kinds(inlinesOf('**a *b* c**'))).toEqual(['strong']);
    expect(plainText(inlinesOf('*a **b c'))).toBe('*a **b c');
    expect(kinds(inlinesOf('~~gone~~'))).toEqual(['del']);
    expect(kinds(inlinesOf('`**not bold**`'))).toEqual(['code']);
  });

  it('HTML stays text', () => {
    const nodes = inlinesOf('<img src=x onerror=alert(1)> and <script>alert(1)</script>');
    expect(kinds(nodes).every((kind) => kind === 'text')).toBe(true);
    expect(plainText(nodes)).toBe('<img src=x onerror=alert(1)> and <script>alert(1)</script>');
  });

  it('links keep their written address; the parser decides nothing about safety', () => {
    const nodes = inlinesOf('[a](javascript:alert(1)) [b](data:text/html,x) [c](https://ok.example/p "t")');
    const links = nodes.filter((node): node is Extract<MdInline, { type: 'link' }> => node.type === 'link');
    expect(links.map((link) => link.href)).toEqual(['javascript:alert(1)', 'data:text/html,x', 'https://ok.example/p']);
  });

  it('a link inside link text is not a second link; bare addresses are links', () => {
    const nodes = inlinesOf('[[x](https://a.example)](https://b.example) see https://c.example/x.');
    const links = nodes.filter((node): node is Extract<MdInline, { type: 'link' }> => node.type === 'link');
    expect(links.map((link) => link.href)).toContain('https://c.example/x');
    for (const link of links) expect(link.children.some((child) => child.type === 'link')).toBe(false);
  });

  it('[n] is a mark only for 1..99 and not before a "("', () => {
    const nodes = inlinesOf('a [1] b [100] c [0] d [2](https://x.example) e [99]');
    const cites = nodes.filter((node): node is Extract<MdInline, { type: 'cite' }> => node.type === 'cite');
    expect(cites.map((cite) => cite.n)).toEqual([1, 99]);
  });

  it('a huge line parses in time', () => {
    const huge = `${'*a '.repeat(20_000)}${'[x]('.repeat(5_000)}${'`'.repeat(10_000)} https://h.example/${'('.repeat(10_000)}`;
    const started = performance.now();
    const blocks = parseMarkdown(huge);
    expect(performance.now() - started).toBeLessThan(2_000);
    expect(blocks.length).toBeGreaterThan(0);
  });

  it('emphasis nests at most 32 deep; deeper markers stay text', () => {
    const n = 4_000;
    const paragraph = only(parseMarkdown(`${'*'.repeat(n)}x${'*'.repeat(n)}`), 'paragraph');
    let depth = 0;
    let level: readonly MdInline[] = paragraph.children;
    for (;;) {
      const wrapper = level.find((node) => node.type === 'em' || node.type === 'strong');
      if (wrapper === undefined || !('children' in wrapper)) break;
      depth += 1;
      level = wrapper.children;
    }
    expect(depth).toBe(32);
    expect(plainText(paragraph.children)).toContain('x');
  });

  it('link forms that used to cost the square of their length parse in time', () => {
    for (const text of [
      '[a](<'.repeat(320_000),
      `${'['.repeat(40_000)}${'[a](x)'.repeat(40_000)}`,
      '[a](b ('.repeat(20_000),
    ]) {
      const started = performance.now();
      parseMarkdown(text);
      expect(performance.now() - started).toBeLessThan(1_500);
    }
  });

  it('code spans pair each backtick run with the next run of the same length', () => {
    const shown = (text: string): string[] =>
      inlinesOf(text).map((node) => `${node.type}:${'value' in node ? node.value : ''}`);
    expect(shown('``a` b``')).toEqual(['code:a` b']);
    expect(shown('`a`` `')).toEqual(['code:a`` ']);
    expect(shown('x ` y')).toEqual(['text:x ` y']);
    expect(shown('`a` and `b`')).toEqual(['code:a', 'text: and ', 'code:b']);
    expect(shown('x ``` a ``` y')).toEqual(['text:x ', 'code:a', 'text: y']);
    expect(shown('`` ` ``')).toEqual(['code:`']);
    expect(shown('\\``a` b`')).toEqual(['text:`', 'code:a', 'text: b`']);
    expect(shown('`a` ``b`` `c')).toEqual(['code:a', 'text: ', 'code:b', 'text: `c']);
    expect(shown('```a`` b')).toEqual(['text:```a`` b']);
    expect(shown('`` a `` ` b `')).toEqual(['code:a', 'text: ', 'code:b']);
    expect(shown('`a` `b ``c`` d`')).toEqual(['code:a', 'text: ', 'code:b ``c`` d']);
    expect(shown('` `` `')).toEqual(['code:``']);
  });

  it('many code spans in one paragraph cost work in proportion to their number', () => {
    expect(growth((n) => parseMarkdown('`a` '.repeat(n)), 1_000)).toBeLessThan(6);
  });

  it('text nodes keep their source offsets', () => {
    const nodes = parseMarkdown('ab **cd**', 100);
    const paragraph = nodes[0] as Extract<MdBlock, { type: 'paragraph' }>;
    const first = paragraph.children[0] as Extract<MdInline, { type: 'text' }>;
    expect(first.at).toBe(100);
    const strong = paragraph.children[1] as Extract<MdInline, { type: 'em' | 'strong' | 'del' }>;
    expect(strong.type).toBe('strong');
    expect((strong.children[0] as Extract<MdInline, { type: 'text' }>).at).toBe(105);
  });
});

describe('streaming helpers', () => {
  it('settledSplit cuts only at a blank line outside a fence and outside an open container', () => {
    expect(settledSplit('one\n\ntwo')).toBe(5);
    expect(settledSplit('```\na\n\nb')).toBe(0);
    expect(settledSplit('- a\n\n- b')).toBe(0);
    expect(settledSplit('one')).toBe(0);
  });

  it('settledSplit resumed from an earlier cut finds the cut a full scan finds', () => {
    const text = 'one\n\ntwo\n\n```\na\n\nb\n```\n\nthree\n\n- x\n\nfour\n\nfive';
    const cuts = new Set<number>([0]);
    for (let end = 1; end <= text.length; end += 1) cuts.add(settledSplit(text.slice(0, end)));
    for (const cut of cuts) expect(settledSplit(text, cut)).toBe(settledSplit(text));
  });

  it('repairTail closes open marks, drops a half link and holds back a bare marker line', () => {
    expect(repairTail('Hello **wor')).toBe('Hello **wor**');
    expect(repairTail('Hello **')).toBe('Hello');
    expect(repairTail('see [the docs](https://ex')).toBe('see the docs');
    expect(repairTail('x\n```py')).toBe('x\n');
    expect(repairTail('| a | b')).toBe('');
  });

  it('the first half of a closing ** or ~~ is not drawn as a stray mark', () => {
    expect(repairTail('This is **bold*')).toBe('This is **bold**');
    expect(repairTail('A ~~struck~')).toBe('A ~~struck~~');
    expect(repairTail('**a *b*')).toBe('**a *b***');
  });

  it('repairTail drops a mark with only space after it and closes every other open mark', () => {
    const cases: Array<[string, string]> = [
      ['*a *b', '*a *b**'],
      ['x *a', 'x *a*'],
      ['*a ', '*a*'],
      ['*a\u00a0', '*a*'],
      ['a **b ~~c', 'a **b ~~c~~**'],
      ['see `code', 'see `code`'],
      ['tail `', 'tail'],
      ['a `  ', 'a'],
      ['**a** *b', '**a** *b*'],
      ['_a b', '_a b_'],
      ['word_ x', 'word_ x'],
      ['* item *a', '* item *a*'],
      ['~~a ~~b ', '~~a ~~b~~~~'],
      ['*a *b *', '*a *b'],
      ['x **a *b _c', 'x **a *b _c_***'],
      ['*a  *  ', '*a'],
      ['p **q** r *s', 'p **q** r *s*'],
      ['**a ~~b** c', '**a ~~b** c~~**'],
    ];
    for (const [tail, shown] of cases) expect(repairTail(tail), tail).toBe(shown);
  });

  it('many open marks on one line cost work in proportion to their number', () => {
    expect(growth((n) => repairTail('*a '.repeat(n)), 1_000)).toBeLessThan(6);
  });
});

describe('ChatMarkdown', () => {
  let mounted: { root: Root; container: HTMLElement } | undefined;
  beforeAll(() => {
    skipAnimations(true);
  });
  afterEach(() => {
    if (mounted === undefined) return;
    act(() => mounted!.root.unmount());
    mounted.container.remove();
    mounted = undefined;
  });

  function render(text: string, links: LinkRule, streaming = false): HTMLElement {
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    const container = document.createElement('div');
    document.body.appendChild(container);
    const root = createRoot(container);
    act(() => root.render(createElement(ChatMarkdown, { text, links, streaming })));
    mounted = { root, container };
    return container;
  }

  const CHAT_LINKS: LinkRule = { mode: 'chat', hosts: new Set(['ok.example']) };

  it('only http(s) links to conversation hosts are anchors', () => {
    const container = render(
      '[a](javascript:alert(1)) [b](data:text/html,x) [c](https://ok.example/p) [d](https://other.example/q) ![pic](https://other.example/i.png)',
      CHAT_LINKS,
    );
    const hrefs = [...container.querySelectorAll('a')].map((a) => a.getAttribute('href'));
    expect(hrefs).toEqual(['https://ok.example/p']);
    expect(container.querySelector('img')).toBeNull();
    expect(container.textContent).toContain('d (other.example)');
    expect(container.textContent).toContain('Image: pic');
  });

  it('Work links are active and say their host', () => {
    const container = render('[docs](https://other.example/q)', { mode: 'work' });
    expect(container.querySelector('a')?.getAttribute('href')).toBe('https://other.example/q');
    expect(container.textContent).toContain('docs (other.example)');
  });

  it('bidi override characters are dropped from what is shown', () => {
    const rlo = String.fromCharCode(0x202e);
    const container = render(`file ${rlo}gpj.exe and [x](https://ok.example/${rlo}a)`, CHAT_LINKS);
    expect(container.textContent?.includes(rlo)).toBe(false);
    expect(visibleText(`a${rlo}b${String.fromCharCode(0x2066)}c`)).toBe('abc');
  });

  it('link words that name another host show the host the link opens; the title is the parsed address', () => {
    const container = render(
      '[https://bank.example/login](https://ok.example/phish) [ok.example/a](https://ok.example/a) [guide](https://ok.example/g)',
      CHAT_LINKS,
    );
    expect(container.textContent).toContain('https://bank.example/login (ok.example)');
    expect(container.textContent).not.toContain('ok.example/a (ok.example)');
    expect(container.textContent).not.toContain('guide (ok.example)');
    expect(container.querySelector('a')?.getAttribute('title')).toBe('https://ok.example/phish');
  });

  it('an address of another host anywhere in the link words shows the host the link opens', () => {
    const container = render(
      '[Log in at bank.example now](https://ok.example/p2) and [see ok.example.](https://ok.example/s)',
      CHAT_LINKS,
    );
    expect(container.textContent).toContain('Log in at bank.example now (ok.example)');
    expect(container.textContent).not.toContain('see ok.example. (ok.example)');
  });

  it('a deeply nested answer renders instead of overflowing the stack', () => {
    const n = 20_000;
    const container = render(`${'*'.repeat(n)}x${'*'.repeat(n)}`, CHAT_LINKS);
    expect(container.textContent).toContain('x');
    expect(container.querySelectorAll('em, strong').length).toBeLessThanOrEqual(32);
  });

  it('a forged [n] without a citation map stays text', () => {
    const container = render('as said [1] and [2]', CHAT_LINKS);
    expect(container.querySelector('button')).toBeNull();
    expect(container.textContent).toBe('as said [1] and [2]');
  });

  it('a streaming tail with an open mark renders closed; code blocks carry Copy', () => {
    const container = render('```js\nlet a = 1;\n```\n\nNow **bold', CHAT_LINKS, true);
    expect(container.querySelector('strong')?.textContent).toBe('bold');
    expect(container.textContent).not.toContain('**');
    expect(container.querySelector('button[aria-label*="Copy"], button')).not.toBeNull();
  });
});
