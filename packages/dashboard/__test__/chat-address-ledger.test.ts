import { mkdtempSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { AddressLedger, scanAddresses, writeLedgerFile } from '../src/chat/agent/address-ledger.js';
import { readWebToolResult, applyWebResult } from '../src/chat/agent/web-results.js';
import { normalizeAddress, readLedgerFile } from '../src/chat/web/ledger-file.js';

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

  it('keeps a ) or ] that closes a bracket inside the address, beside the old forms', () => {
    expect(scanAddresses('[Mercury](https://en.wikipedia.org/wiki/Mercury_(planet))')).toEqual([
      'https://en.wikipedia.org/wiki/Mercury_(planet))',
      'https://en.wikipedia.org/wiki/Mercury_(planet',
      'https://en.wikipedia.org/wiki/Mercury_(planet)',
    ]);
    expect(scanAddresses('[x](https://a.com/b[1]).')).toEqual([
      'https://a.com/b[1]).',
      'https://a.com/b[1',
      'https://a.com/b[1]',
    ]);
    const ledger = new AddressLedger();
    ledger.addPageText('See [Mercury](https://en.wikipedia.org/wiki/Mercury_(planet)).');
    expect(ledger.has('https://en.wikipedia.org/wiki/Mercury_(planet)')).toBe(true);
  });

  it('a bare address in parentheses, or followed by )., gives the same forms as before', () => {
    expect(scanAddresses('(see https://x.com/a)')).toEqual(['https://x.com/a)', 'https://x.com/a']);
    expect(scanAddresses('(see https://x.com/a).')).toEqual(['https://x.com/a).', 'https://x.com/a']);
    expect(scanAddresses('read https://x.com/a). next')).toEqual(['https://x.com/a).', 'https://x.com/a']);
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

  it('ignores a result of another MCP server', () => {
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
    expect(ledger.size).toBe(0);
  });

  it('takes every address in the text of a parsed page result, never its title', () => {
    const ledger = new AddressLedger();
    const page = {
      sessionUpdate: 'tool_call_update',
      rawOutput: {
        type: 'MCP',
        tool_name: 'web_fetch',
        server_name: 'mlx_web',
        output: {
          OkayOutput:
            'WEB_PAGE provider=exa url=https://sebastianraschka.com/blog\ntitle: see https://title.example\n\n' +
            'The report [Inkling 2 mini](https://thinkingmachines.ai/news/inkling-2-mini/). Also <http://b.org/y>.',
        },
      },
    };
    applyWebResult(readWebToolResult(page)!, { ledger });
    expect(ledger.has('https://thinkingmachines.ai/news/inkling-2-mini')).toBe(true);
    expect(ledger.has('http://b.org/y')).toBe(true);
    expect(ledger.has('https://title.example')).toBe(false);
    expect(ledger.has('https://sebastianraschka.com/blog')).toBe(false);
  });

  it('a page result that does not parse whole adds nothing', () => {
    const ledger = new AddressLedger();
    const update = {
      sessionUpdate: 'tool_call_update',
      rawOutput: {
        type: 'MCP',
        tool_name: 'web_fetch',
        server_name: 'mlx_web',
        output: { OkayOutput: 'WEB_PAGE provider=exa url=https://a.com\nno title line\n\nsee https://inside.page/x' },
      },
    };
    expect(readWebToolResult(update)).toBeNull();
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

describe('ledger file', () => {
  it('holds the normal forms, mode 0600, and reads back as the same set', () => {
    const dir = mkdtempSync(join(tmpdir(), 'mlx-ledger-'));
    const path = join(dir, 'l.json');
    const ledger = new AddressLedger();
    ledger.addUserText('Read https://Example.com/docs/ and https://releases.rs.');
    writeLedgerFile(path, ledger.members());
    expect(JSON.parse(readFileSync(path, 'utf8'))).toEqual(['https://example.com/docs', 'https://releases.rs']);
    expect(statSync(path).mode & 0o777).toBe(0o600);
    expect([...readLedgerFile(path)!]).toEqual(ledger.members());
    ledger.addPageText('https://next.example/x');
    writeLedgerFile(path, ledger.members());
    expect(readLedgerFile(path)?.has('https://next.example/x')).toBe(true);
    expect(readdirSync(dir)).toEqual(['l.json']);
  });

  it('reads as null when missing, broken or of another shape', () => {
    const dir = mkdtempSync(join(tmpdir(), 'mlx-ledger-'));
    expect(readLedgerFile(join(dir, 'missing.json'))).toBeNull();
    expect(readLedgerFile('')).toBeNull();
    expect(readLedgerFile(dir)).toBeNull();
  });
});
