import { getMarkdown, parseMarkdownToStructure, type MarkdownIt, type ParsedNode } from 'stream-markdown-parser';
import { describe, expect, it } from 'vite-plus/test';

import { citeNumber, linkView, visibleText, type LinkRule } from '../ui/src/lib/link-rule.js';
import {
  CHAT_PARSE_OPTIONS,
  chatCiteMarkdownIt,
  chatMarkdownIt,
  nodeWords,
  PACE_FRAME_MS,
  paceHoldMs,
} from '../ui/src/lib/markstream-config.js';

/** The node types ChatMarkdown draws itself or keeps from markstream; anything else would be drawn with its defaults. */
const ALLOWED = new Set([
  'paragraph',
  'heading',
  'list',
  'list_item',
  'blockquote',
  'table',
  'table_row',
  'table_cell',
  'code_block',
  'thematic_break',
  'text',
  'strong',
  'emphasis',
  'strikethrough',
  'inline_code',
  'link',
  'image',
  'hardbreak',
]);

const CORPUS = [
  '# Title',
  '',
  'Intro with **bold**, *em*, ~~gone~~, `code`, a [doc](https://ok.example/doc "t") and https://ok.example/bare.',
  'Next line  ',
  'after a hard break; cites [1] and [12], a link [2](https://ok.example/two) and <https://ok.example/auto>.',
  '',
  'Raw <script>alert(1)</script> <b onclick=alert(1)>b</b> <img src=x onerror=alert(1)> <br> <div>d</div>',
  '<a href="https://evil.example/raw">raw</a> &amp; ![pic](https://evil.example/p.png)',
  '',
  '<iframe src="https://evil.example/frame"></iframe>',
  '',
  '> quoted **text**',
  '> - inside',
  '',
  '- one',
  '  - nested',
  '- two',
  '',
  '3. three',
  '4. four',
  '',
  '| a | b |',
  '|:--|--:|',
  '| 1 | [x](https://ok.example/x) |',
  '',
  '```ts',
  'const x = 1;',
  '```',
  '',
  '```diff',
  '-old',
  '+new',
  '```',
  '',
  '```mermaid',
  'graph TD; A-->B',
  '```',
  '',
  '---',
  '',
  'main.py README.md lib.rs a@b.co www.x.com',
  '',
  '$x$ ==m== ++i++ ~s~ ^s^ :::note [^1]',
  '',
  '- [ ] t',
  '',
  '"q" -- (c)',
  '',
  '[1]: https://x',
].join('\n');

function chatMd(): MarkdownIt {
  return chatMarkdownIt(getMarkdown('test'));
}

function parse(text: string, md: MarkdownIt = chatMd(), final = true): ParsedNode[] {
  return parseMarkdownToStructure(text, md, { ...CHAT_PARSE_OPTIONS, final });
}

/** Every node of the tree: children, list items and table cells. */
function each(nodes: readonly ParsedNode[], visit: (node: ParsedNode) => void): void {
  for (const node of nodes) {
    visit(node);
    const any = node as ParsedNode & {
      children?: ParsedNode[];
      items?: ParsedNode[];
      header?: unknown;
      rows?: ParsedNode[];
      cells?: ParsedNode[];
    };
    for (const list of [any.children, any.items, any.rows, any.cells]) if (Array.isArray(list)) each(list, visit);
    // A table's header is its first row; a cell's `header` is a flag.
    if (typeof any.header === 'object' && any.header !== null) each([any.header as ParsedNode], visit);
  }
}

/** What would make the renderer draw markup, a link or a mark the chat rule did not decide. */
function violations(nodes: readonly ParsedNode[]): string[] {
  const out: string[] = [];
  each(nodes, (node) => {
    if (!ALLOWED.has(node.type)) out.push(`type ${node.type}`);
    if (node.type === 'link' && (node as { markup?: string }).markup === 'html') out.push('html link');
  });
  return out;
}

function texts(nodes: readonly ParsedNode[]): string {
  let out = '';
  each(nodes, (node) => {
    if (node.type === 'text') out += String((node as { content?: unknown }).content);
  });
  return out;
}

function only(nodes: readonly ParsedNode[], type: string): ParsedNode {
  expect(nodes.map((node) => node.type)).toEqual([type]);
  return nodes[0]!;
}

/** The checks the corpus must pass under a parser setup; the mutations below must each break one. */
function corpusFindings(make: () => MarkdownIt): string[] {
  const findings: string[] = [];
  const md = make();
  for (let end = 1; end <= CORPUS.length; end += 7)
    findings.push(...violations(parse(CORPUS.slice(0, end), md, false)));
  const whole = parse(CORPUS, make());
  findings.push(...violations(whole));
  const shown = texts(whole);
  if (!shown.includes('<script>alert(1)</script>')) findings.push('script tag not shown as text');
  const files = whole.find((node) => nodeWords([node]).startsWith('main.py'));
  if (files === undefined || nodeWords([files]) !== 'main.py README.md lib.rs a@b.co www.x.com')
    findings.push('file names');
  else each([files], (node) => node.type === 'link' && findings.push(`file name linked: ${nodeWords([node])}`));
  if (!shown.includes('[1]: https://x')) findings.push('link definition hidden');
  return findings;
}

describe('the chat parser setup', () => {
  it('parses the corpus at every prefix and whole into the drawn node types only', () => {
    expect(corpusFindings(chatMd)).toEqual([]);
  });

  it('named mutations each break the corpus check', () => {
    const mutations: Record<string, (md: MarkdownIt) => void> = {
      'fix_html_inline_streaming back on': (md) => md.core.ruler.enable(['fix_html_inline_streaming']),
      'html on': (md) => md.set({ html: true }),
      'fuzzyLink back on': (md) => (md.linkify as { set: (o: object) => void }).set({ fuzzyLink: true }),
      'block reference back on': (md) => md.block.ruler.enable(['reference']),
      'inline [n] rule back on': (md) => md.inline.ruler.enable(['reference']),
    };
    for (const [name, mutate] of Object.entries(mutations)) {
      const findings = corpusFindings(() => {
        const md = chatMd();
        mutate(md);
        return md;
      });
      expect(findings.length, name).toBeGreaterThan(0);
    }
  });

  it('raw HTML is text', () => {
    const text = '<b onclick=alert(1)>b</b> <img src=x onerror=alert(1)> <a href="https://e.example">a</a> <br>';
    const paragraph = only(parse(text), 'paragraph');
    expect(violations([paragraph])).toEqual([]);
    expect(texts([paragraph])).toBe(text);
  });

  it('file names, e-mails, www, math, marks, task boxes, containers and footnotes stay text', () => {
    for (const text of ['main.py README.md lib.rs a@b.co www.x.com', '$x$ ==m== ++i++ ~s~ ^s^ :::note [^1]']) {
      const paragraph = only(parse(text), 'paragraph');
      expect(violations([paragraph])).toEqual([]);
      expect(nodeWords([paragraph]), text).toBe(text);
    }
    const list = only(parse('- [ ] t'), 'list');
    expect(violations([list])).toEqual([]);
    expect(texts([list])).toBe('[ ] t');
  });

  it('no smart quotes or dashes; a link definition line stays visible; [n] stays text', () => {
    expect(nodeWords(parse('"q" -- (c)'))).toBe('"q" -- (c)');
    expect(texts(parse('[1]: https://x'))).toContain('[1]: ');
    const cites = only(parse('see [1] and [12], not [2](https://ok.example/two)'), 'paragraph');
    const kinds: string[] = [];
    each([cites], (node) => kinds.push(node.type));
    expect(kinds).not.toContain('reference');
    expect(texts([cites])).toContain('see [1] and [12], not ');
  });

  it('with sources, a written [n] is a mark and an escaped one stays text', () => {
    const cites = only(parse('see \\[1\\] and [2]', chatCiteMarkdownIt(getMarkdown('test'))), 'paragraph');
    const ids: string[] = [];
    each([cites], (node) => node.type === 'reference' && ids.push(String((node as { id?: unknown }).id)));
    expect(ids).toEqual(['2']);
    expect(texts([cites])).toBe('see [1] and ');
  });

  it('a diff fence keeps its whole text; an open fence says it is loading', () => {
    const diff = only(parse('```diff\n-old\n+new\n```'), 'code_block') as ParsedNode & { diff?: boolean };
    expect(diff.diff).toBe(true);
    expect(diff.raw).toContain('-old\n+new');
    const open = only(parse('```py\nprint(1)', chatMd(), false), 'code_block');
    expect(open.loading).toBe(true);
  });

  it('bare and written links keep their address; javascript: is not a link', () => {
    const hrefs: string[] = [];
    each(parse('[a](javascript:alert(1)) [c](https://ok.example/p "t") https://ok.example/bare'), (node) => {
      if (node.type === 'link') hrefs.push(String((node as { href?: unknown }).href));
    });
    expect(hrefs).toEqual(['https://ok.example/p', 'https://ok.example/bare']);
  });
});

describe('hostile input', () => {
  it('long and nested forms finish in time, parsed or refused', () => {
    const cases = [
      `${'*'.repeat(20_000)}x${'*'.repeat(20_000)}`,
      `${'>'.repeat(5_000)} x`,
      Array.from({ length: 200 }, (_, i) => `${'  '.repeat(i)}- item`).join('\n'),
      `${'*a '.repeat(20_000)}${'[x]('.repeat(5_000)}${'`'.repeat(10_000)} https://h.example/${'('.repeat(10_000)}`,
      '[a](<'.repeat(320_000),
      '[a](b ('.repeat(20_000),
      '`a` '.repeat(50_000),
    ];
    for (const text of cases) {
      const started = performance.now();
      try {
        parse(text);
      } catch (error) {
        // Too deep for the parser's stack: ChatMarkdown's PlainOnError shows the text plain.
        expect(error).toBeInstanceOf(RangeError);
      }
      expect(performance.now() - started, text.slice(0, 20)).toBeLessThan(3_000);
    }
  });

  it('a table stops growing past its cell cap', () => {
    const table = parse(`|${'h|'.repeat(60)}\n|${'-|'.repeat(60)}\n${'a\n'.repeat(5_000)}`)[0] as ParsedNode & {
      rows: Array<{ cells: unknown[] }>;
    };
    expect(table.type).toBe('table');
    expect(table.rows.reduce((n, row) => n + row.cells.length, 0)).toBeLessThanOrEqual(70_000);
  });
});

describe('link rule', () => {
  const chat: LinkRule = { mode: 'chat', hosts: new Set(['ok.example']) };
  const work: LinkRule = { mode: 'work' };

  it('only http(s) opens; in Chat only hosts of the conversation; otherwise words plus the host', () => {
    const cases: Array<[string, string, LinkRule, string | null, string | null]> = [
      ['javascript:alert(1)', 'a', chat, null, null],
      ['data:text/html,x', 'b', chat, null, null],
      ['https://ok.example/p', 'c', chat, 'https://ok.example/p', null],
      ['https://other.example/q', 'd', chat, null, 'other.example'],
      ['https://other.example/q', 'https://other.example/q', chat, null, null],
      ['https://other.example/q', 'docs', work, 'https://other.example/q', 'other.example'],
      ['https://other.example/q', 'https://other.example/q', work, 'https://other.example/q', null],
      ['javascript:alert(1)', 'x', work, null, null],
    ];
    for (const [href, words, rule, opens, host] of cases) {
      const view = linkView(href, words, rule);
      expect([view.href, view.host], `${href} ${words}`).toEqual([opens, host]);
    }
  });

  it('words that name another host show the host the link opens', () => {
    expect(linkView('https://ok.example/phish', 'https://bank.example/login', chat).host).toBe('ok.example');
    expect(linkView('https://ok.example/p2', 'Log in at bank.example now', chat).host).toBe('ok.example');
    expect(linkView('https://ok.example/a', 'ok.example/a', chat).host).toBeNull();
    expect(linkView('https://ok.example/s', 'see ok.example.', chat).host).toBeNull();
    expect(linkView('https://ok.example/g', 'guide', chat).host).toBeNull();
  });

  it('the title is the parsed address, and no bidi control survives', () => {
    const rlo = String.fromCharCode(0x202e);
    expect(linkView('https://ok.example/phish', 'x', chat).title).toBe('https://ok.example/phish');
    expect(linkView(`https://ok.example/${rlo}a`, 'x', chat).title.includes(rlo)).toBe(false);
    expect(visibleText(`a${rlo}b${String.fromCharCode(0x2066)}c${String.fromCharCode(0x200f)}`)).toBe('abc');
  });
});

describe('[n] marks', () => {
  it('a mark is 1 to 99 without a leading 0', () => {
    expect(['1', '9', '10', '99'].map(citeNumber)).toEqual([1, 9, 10, 99]);
    expect(['0', '01', '100', '', 'a', '1a'].map(citeNumber)).toEqual([null, null, null, null, null, null]);
  });
});

describe('the streaming hold (paceHoldMs)', () => {
  it('lets the next text in at once after a parse and render that fit in a frame', () => {
    // Mutation: the bar at 0, or `>=` at one frame: every chunk of a normal answer would wait.
    for (const cost of [0, 0.2, 1, 8, PACE_FRAME_MS]) expect(paceHoldMs(cost)).toBe(0);
  });

  it('holds the next text four times the cost after a slower one', () => {
    // Mutation: a factor of 1, 2 or 3.
    expect(paceHoldMs(17)).toBe(68);
    expect(paceHoldMs(100)).toBe(400);
    expect(paceHoldMs(500)).toBe(2_000);
  });

  it('keeps at least 75 % of the time free for a parse of up to 3.3 s', () => {
    // Mutation: a cap under 3x the cost of the 280K bracket run (2.8 s).
    for (let cost = PACE_FRAME_MS + 0.5; cost <= 3_300; cost += 7) {
      const hold = paceHoldMs(cost);
      expect(hold / (cost + hold)).toBeGreaterThanOrEqual(0.75);
    }
  });

  it('never holds more than 10 s', () => {
    // Mutation: no cap, or the cap as the least hold (Math.max).
    expect(paceHoldMs(2_100)).toBe(8_400);
    expect(paceHoldMs(2_600)).toBe(10_000);
    expect(paceHoldMs(Number.POSITIVE_INFINITY)).toBe(10_000);
    expect(paceHoldMs(20)).toBe(80);
  });

  it('holds nothing for a cost that is not a positive number', () => {
    // Mutation: no guard: NaN would become the hold.
    expect(paceHoldMs(Number.NaN)).toBe(0);
    expect(paceHoldMs(-5)).toBe(0);
  });
});
