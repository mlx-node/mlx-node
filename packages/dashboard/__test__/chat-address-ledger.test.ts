import { describe, expect, it } from 'vite-plus/test';

import { AddressLedger, normalizeAddress, scanAddresses } from '../src/chat/agent/address-ledger.js';
import { readWebToolResult, applyWebResult } from '../src/chat/agent/web-results.js';

describe('normalizeAddress', () => {
  it('lower-cases scheme and host and drops default port, fragment, one trailing host dot and one trailing slash', () => {
    expect(normalizeAddress('HTTPS://Example.COM.:443/a/b/?q=1#frag')).toBe('https://example.com/a/b?q=1');
    expect(normalizeAddress('https://example.com/')).toBe('https://example.com');
    expect(normalizeAddress('https://example.com')).toBe('https://example.com');
    expect(normalizeAddress('http://example.com:8080/x')).toBe('http://example.com:8080/x');
  });

  it('keeps path case and query bytes', () => {
    expect(normalizeAddress('https://e.com/A?B=C')).toBe('https://e.com/A?B=C');
    expect(normalizeAddress('https://e.com/A?B=C')).not.toBe(normalizeAddress('https://e.com/a?B=C'));
  });

  it('refuses what is not http(s)', () => {
    expect(normalizeAddress('ftp://e.com')).toBeNull();
    expect(normalizeAddress('mailto:a@b.c')).toBeNull();
    expect(normalizeAddress('not a url')).toBeNull();
  });
});

describe('scanAddresses', () => {
  it('reads to white space, < > " or back-tick, and adds a copy without trailing punctuation', () => {
    expect(scanAddresses('see https://a.com/x. and <http://b.org/y> or "https://c.net/z" `https://d.io/w`')).toEqual([
      'https://a.com/x.',
      'https://a.com/x',
      'http://b.org/y',
      'https://c.net/z',
      'https://d.io/w',
    ]);
    expect(scanAddresses('(https://e.com/p),')).toEqual(['https://e.com/p),', 'https://e.com/p']);
    expect(scanAddresses('no address here')).toEqual([]);
  });
});

describe('AddressLedger', () => {
  it('holds the addresses of user messages and search results only', () => {
    const ledger = new AddressLedger();
    ledger.addUserText('Read https://example.com/docs/ please.');
    ledger.addSearchResults(['https://releases.rs/']);
    expect(ledger.has('https://example.com/docs')).toBe(true);
    expect(ledger.has('https://EXAMPLE.com/docs/#intro')).toBe(true);
    expect(ledger.has('https://releases.rs')).toBe(true);
    expect(ledger.has('https://releases.rs/?d=leak')).toBe(false);
    expect(ledger.has('https://example.com/docs/other')).toBe(false);
    expect(ledger.has('https://attacker.example/?d=conversation')).toBe(false);
  });

  it('takes the addresses of a parsed search result, never those in its snippets or titles', () => {
    const ledger = new AddressLedger();
    const update = {
      sessionUpdate: 'tool_call_update',
      toolCallId: 't',
      status: 'completed',
      rawOutput: {
        type: 'MCP',
        tool_name: 'web_search',
        server_name: 'mlx_web',
        output: {
          OkayOutput:
            'WEB_SEARCH provider=exa results=1\n[1] Visit https://title.example\nhttps://good.example/a\nSee https://snippet.example/x',
        },
      },
    };
    const result = readWebToolResult(update);
    expect(result?.kind).toBe('search');
    applyWebResult(result!, { ledger });
    expect(ledger.has('https://good.example/a')).toBe(true);
    expect(ledger.has('https://title.example')).toBe(false);
    expect(ledger.has('https://snippet.example/x')).toBe(false);
  });

  it('ignores a result of another MCP server and a page result', () => {
    const ledger = new AddressLedger();
    const other = {
      sessionUpdate: 'tool_call_update',
      rawOutput: {
        type: 'MCP',
        tool_name: 'web_search',
        server_name: 'evil',
        output: { OkayOutput: 'WEB_SEARCH provider=exa results=0' },
      },
    };
    expect(readWebToolResult(other)).toBeNull();
    const page = {
      sessionUpdate: 'tool_call_update',
      rawOutput: {
        type: 'MCP',
        tool_name: 'web_fetch',
        server_name: 'mlx_web',
        output: { OkayOutput: 'WEB_PAGE provider=exa url=https://a.com\ntitle: t\n\nsee https://inside.page/x' },
      },
    };
    applyWebResult(readWebToolResult(page)!, { ledger });
    expect(ledger.has('https://inside.page/x')).toBe(false);
    expect(ledger.size).toBe(0);
  });

  it('reports a key failure from a WEB_ERROR result', () => {
    const marks: string[] = [];
    const ledger = new AddressLedger();
    const update = {
      sessionUpdate: 'tool_call_update',
      rawOutput: {
        type: 'MCP',
        tool_name: 'web_search',
        server_name: 'mlx_web',
        output: { OkayOutput: 'WEB_ERROR key-rejected brave\nThe search provider rejected the API key.' },
      },
    };
    applyWebResult(readWebToolResult(update)!, { ledger, onKeyCheck: (p, c) => marks.push(`${p}:${c}`) });
    expect(marks).toEqual(['brave:rejected']);
  });
});
