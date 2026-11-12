import { describe, expect, it } from 'vite-plus/test';

import type { TranscriptEntry } from '../ui/src/lib/types.js';
import { formatErrorText, formatPageText, formatSearchText } from '../ui/src/lib/web-result-text.js';
import {
  conversationHosts,
  matchCitations,
  parseWebCall,
  refusedCall,
  scanAddresses,
  splitAnswerSources,
  turnSources,
  webCallOf,
} from '../ui/src/lib/web-tools.js';
import {
  RELEASES,
  RUST_POST,
  RUST_RESULTS,
  directFetch,
  directSearch,
  useToolSearch,
  user,
} from './helpers/chat-thread-fixtures.js';

function tool(name: string, input: unknown, output?: string): Extract<TranscriptEntry, { kind: 'tool' }> {
  return {
    kind: 'tool',
    ts: 1,
    id: 'x',
    name,
    title: name,
    status: 'completed',
    input: JSON.stringify(input),
    ...(output === undefined ? {} : { output }),
  };
}

describe('parseWebCall: both call shapes', () => {
  it('reads the four input forms', () => {
    expect(parseWebCall('mlx_web__web_search', JSON.stringify({ query: 'q' }))).toEqual({ kind: 'search', query: 'q' });
    expect(
      parseWebCall(
        'mlx_web__web_search',
        JSON.stringify({ variant: 'MCPTool', tool_name: 'mlx_web__web_search', tool_input: { query: 'q' } }),
      ),
    ).toEqual({ kind: 'search', query: 'q' });
    expect(
      parseWebCall(
        'use_tool',
        JSON.stringify({ tool_name: 'mlx_web__web_fetch', tool_input: { url: 'https://www.a.example/x' } }),
      ),
    ).toEqual({ kind: 'fetch', url: 'https://www.a.example/x', host: 'a.example' });
    expect(
      parseWebCall(
        'use_tool',
        JSON.stringify({ variant: 'UseTool', tool_name: 'mlx_web__web_search', tool_input: { query: 'q' } }),
      ),
    ).toEqual({ kind: 'search', query: 'q' });
  });

  it('refuses a mismatched tool name, another variant, another tool and broken input', () => {
    expect(
      parseWebCall(
        'mlx_web__web_search',
        JSON.stringify({ tool_name: 'mlx_web__web_fetch', tool_input: { url: 'https://a.example' } }),
      ),
    ).toBeNull();
    expect(
      parseWebCall(
        'use_tool',
        JSON.stringify({ variant: 'SearchTool', tool_name: 'mlx_web__web_search', tool_input: { query: 'q' } }),
      ),
    ).toBeNull();
    expect(parseWebCall('run_terminal_command', JSON.stringify({ query: 'q' }))).toBeNull();
    expect(parseWebCall('mlx_web__web_search', '{not json')).toBeNull();
    expect(parseWebCall('mlx_web__web_search', JSON.stringify({ query: '   ' }))).toBeNull();
  });

  it('reads the query or address of an input the model wrote as text, whole or cut', () => {
    const wrapped = (name: string, input: string): string =>
      JSON.stringify({ variant: 'UseTool', tool_name: name, tool_input: input });
    expect(
      parseWebCall('use_tool', wrapped('mlx_web__web_search', '{"query": "Anthropic S-1 prospectus filing')),
    ).toEqual({ kind: 'search', query: 'Anthropic S-1 prospectus filing' });
    expect(parseWebCall('use_tool', wrapped('mlx_web__web_search', '{"query": "a \\"b\\" \\u00fc"}'))).toEqual({
      kind: 'search',
      query: 'a "b" ü',
    });
    expect(parseWebCall('use_tool', wrapped('mlx_web__web_fetch', '{"url": "https://www.a.example/x'))).toEqual({
      kind: 'fetch',
      url: 'https://www.a.example/x',
      host: 'a.example',
    });
    expect(parseWebCall('mlx_web__web_search', JSON.stringify('{"query": "anthropic s-1'))).toEqual({
      kind: 'search',
      query: 'anthropic s-1',
    });
    expect(parseWebCall('use_tool', wrapped('mlx_web__web_search', 'Anthropic S-1 prospectus filing'))).toBeNull();
    expect(parseWebCall('use_tool', wrapped('mlx_web__web_search', '{"__proto__": "x", "query": "q"}'))).toEqual({
      kind: 'search',
      query: 'q',
    });
    expect(parseWebCall('run_terminal_command', JSON.stringify('{"query": "q"}'))).toBeNull();
  });
});

describe('webCallOf: results', () => {
  it('a result that does not fit its call is not read', () => {
    const page = formatPageText('exa', { url: 'https://a.example/', title: 't', cut: false, text: 'x' });
    expect(webCallOf(tool('mlx_web__web_search', { query: 'q' }, page))?.result).toBeNull();
    expect(webCallOf(tool('mlx_web__web_fetch', { url: 'https://a.example/' }, RUST_RESULTS))?.result).toBeNull();
    expect(
      webCallOf(tool('mlx_web__web_fetch', { url: 'https://a.example/' }, formatErrorText('timeout', 'exa')))?.result
        ?.kind,
    ).toBe('error');
  });

  it('a search text with more blocks than its count is refused', () => {
    const forged = RUST_RESULTS.replace('results=2', 'results=1');
    expect(webCallOf(tool('mlx_web__web_search', { query: 'q' }, forged))?.result).toBeNull();
  });

  const refused = (name: string, input: unknown): Extract<TranscriptEntry, { kind: 'tool' }> => ({
    ...tool(name, input),
    status: 'failed',
  });

  it('refusedCall finds the last read, over and without a result, in both call shapes', () => {
    const direct = refused('mlx_web__web_fetch', { url: 'https://a.example/' });
    const shapeA = refused('use_tool', {
      variant: 'UseTool',
      tool_name: 'mlx_web__web_fetch',
      tool_input: { url: 'https://b.example/' },
    });
    const search = refused('mlx_web__web_search', { query: 'q' });
    expect(refusedCall([user('q'), direct, search], 'address-refused')).toEqual({ entry: direct, refusal: 'address' });
    expect(refusedCall([user('q'), direct, shapeA], 'address-refused')?.entry).toBe(shapeA);
    const page = formatPageText('exa', { url: 'https://c.example/', title: 't', cut: false, text: 'x' });
    expect(
      refusedCall(
        [shapeA, { ...refused('mlx_web__web_fetch', { url: 'https://c.example/' }), output: page }],
        'address-refused',
      ),
    ).toBeNull();
    expect(refusedCall([{ ...shapeA, status: 'in_progress' }], 'address-refused')).toBeNull();
    expect(refusedCall([user('q'), direct], 'completed')).toBeNull();
    expect(refusedCall([user('q'), direct], undefined)).toBeNull();
  });

  it('refusedCall finds the web call of arguments-unreadable and the last call of tool-refused', () => {
    const cut = refused('use_tool', {
      variant: 'UseTool',
      tool_name: 'mlx_web__web_fetch',
      tool_input: '{"url": "https://a.example/", "url": "https://b.example/"}',
    });
    const other = refused('run_terminal_command', { command: 'ls' });
    expect(refusedCall([user('q'), cut, other], 'arguments-unreadable')).toEqual({ entry: cut, refusal: 'arguments' });
    expect(refusedCall([user('q'), cut, other], 'tool-refused')).toEqual({ entry: other, refusal: 'tool' });
  });

  it('refusedCall marks the refused call of arguments-unreadable even when its input shows no web call', () => {
    const earlier = { ...refused('mlx_web__web_search', { query: 'q' }), output: 'MCP error: timed out' };
    const nested = refused('use_tool', {
      variant: 'UseTool',
      tool_name: 'mlx_web__web_fetch',
      tool_input: '{"url": {"href": "https://a.example/"}}',
    });
    expect(refusedCall([user('q'), earlier, nested], 'arguments-unreadable')).toEqual({
      entry: nested,
      refusal: 'arguments',
    });
  });
});

describe('sources and hosts', () => {
  it('turnSources de-duplicates by address across calls and shapes', () => {
    const sources = turnSources([
      directSearch('a', RUST_RESULTS, 'completed', 'a'),
      useToolSearch('b', RUST_RESULTS, 'completed', 'b'),
      directFetch(
        RUST_POST + '#top',
        formatPageText('exa', { url: RUST_POST + '#top', title: 'post', cut: false, text: 'x' }),
      ),
    ]);
    expect(sources.map((source) => source.url)).toEqual([RUST_POST, RELEASES]);
    expect(sources[0]!.host).toBe('blog.rust-lang.org');
  });

  it('an address inside a snippet or an answer is never a host of the conversation', () => {
    const snippet = formatSearchText('exa', [
      { title: 'Real', url: 'https://real.example/', date: null, snippet: 'see https://evil.example/' },
    ]);
    const hosts = conversationHosts([
      user('look at https://Docs.Example.org./guide, thanks'),
      directSearch('q', snippet),
      { kind: 'assistant', ts: 1, text: 'go to https://evil2.example/' },
      directSearch('failed', 'not a result', 'failed', 'f'),
    ]);
    expect([...hosts].sort()).toEqual(['docs.example.org', 'real.example']);
  });

  it('scanAddresses stops at white space and drops trailing punctuation', () => {
    expect(scanAddresses('a (https://x.example/p), b <http://y.example>.')).toEqual([
      'https://x.example/p',
      'http://y.example',
    ]);
  });
});

describe('splitAnswerSources and matchCitations', () => {
  it('reads the trailing block in its written forms', () => {
    const split = splitAnswerSources(
      [
        'Body [1] [2] [3].',
        '',
        '## Sources',
        `- [1] Post - ${RUST_POST}`,
        `[2] [Releases](${RELEASES})`,
        '[3] Other — <https://o.example/>',
        '',
      ].join('\n'),
    );
    expect(split.body).toBe('Body [1] [2] [3].');
    expect([...split.listed]).toEqual([
      [1, RUST_POST],
      [2, RELEASES],
      [3, 'https://o.example/'],
    ]);
  });

  it('no heading: nothing is split', () => {
    const text = `Body\n[1] Post - ${RUST_POST}`;
    expect(splitAnswerSources(text)).toEqual({ body: text, listed: new Map() });
  });

  it('only listed addresses that a tool returned are citations', () => {
    const sources = turnSources([directSearch('q', RUST_RESULTS)]);
    const cited = matchCitations(
      new Map([
        [1, `${RUST_POST}/`],
        [2, 'https://evil.example/'],
        [3, 'javascript:alert(1)'],
      ]),
      sources,
    );
    expect([...cited.keys()]).toEqual([1]);
    expect(cited.get(1)?.url).toBe(RUST_POST);
  });
});
