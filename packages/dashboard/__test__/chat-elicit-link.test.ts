/**
 * MCP input, link mode: the address check of the agent's own client (`check_elicit_url`), the card the host
 * builds from a link request (the shown address is always what opens), and the reader of the server's "done".
 */

import { describe, expect, it } from 'vite-plus/test';

import {
  linkCheck,
  linkInteraction,
  readElicitComplete,
  readElicitRequest,
  URL_MAX_CHARS,
} from '../src/chat/agent/elicit.js';
import { REDACTED } from '../src/chat/redact.js';

const KEY = 'exa-KEY-0123456789abcdef';

describe('linkCheck', () => {
  it('takes https and http, and gives the parsed address, its site and whether it is encrypted', () => {
    expect(linkCheck('https://github.com/login/oauth/authorize?client_id=a&state=b')).toEqual({
      url: 'https://github.com/login/oauth/authorize?client_id=a&state=b',
      host: 'github.com',
      secure: true,
    });
    expect(linkCheck('http://intranet.local:8080/sign-in')).toEqual({
      url: 'http://intranet.local:8080/sign-in',
      host: 'intranet.local',
      secure: false,
    });
  });

  it('shows what opens: the parser form, spaces trimmed, the site in lower case, an international site as xn--', () => {
    expect(linkCheck('  HTTPS://GitHub.COM/a b?q=x y#f g  ')).toEqual({
      url: 'https://github.com/a%20b?q=x%20y#f%20g',
      host: 'github.com',
      secure: true,
    });
    expect(linkCheck('https://gіthub.com/login')).toEqual({
      url: 'https://xn--gthub-n2e.com/login',
      host: 'xn--gthub-n2e.com',
      secure: true,
      lookalike: true,
    });
    expect(linkCheck('https://[::1]:3000/cb')).toEqual({ url: 'https://[::1]:3000/cb', host: '[::1]', secure: true });
    // Hidden characters never reach the shown address.
    const hidden = linkCheck('https://example.com/a‮b\u0007c');
    expect(hidden?.url).toBe('https://example.com/a%E2%80%AEb%07c');
  });

  // Mutation: look for `xn--` in the whole site name as one → `xn--` only in a later part goes unflagged.
  it('a site name with any part in its xn-- form is a lookalike; a plain one is not', () => {
    expect(linkCheck('https://login.gіthub.com/')?.lookalike).toBe(true);
    expect(linkCheck('https://xn--80ak6aa92e.com/')).toEqual({
      url: 'https://xn--80ak6aa92e.com/',
      host: 'xn--80ak6aa92e.com',
      secure: true,
      lookalike: true,
    });
    expect(linkCheck('https://example.xn--p1ai/')?.lookalike).toBe(true);
    expect(linkCheck('http://XN--gthub-n2e.com/')).toMatchObject({ secure: false, lookalike: true });
    expect(linkCheck('https://xnn--example.com/')?.lookalike).toBeUndefined();
    expect(linkCheck('https://example.com/xn--path')?.lookalike).toBeUndefined();
    expect(linkCheck('https://github.com/')?.lookalike).toBeUndefined();
  });

  it('refuses every other scheme, a user name or password, an address that does not parse, and a long one', () => {
    for (const raw of [
      'javascript:alert(1)',
      'file:///etc/passwd',
      'data:text/html,<b>x</b>',
      'mailto:me@example.com',
      'app://mlx/settings',
      'ftp://example.com/x',
      'https://user:pass@example.com/',
      'https://user@example.com/',
      'https://:pass@example.com/',
      'example.com/login',
      '/login',
      '',
      'https://',
      'http://exa mple.com/',
    ]) {
      expect(linkCheck(raw), raw).toBeNull();
    }
    const base = 'https://example.com/';
    expect(linkCheck(base + 'a'.repeat(URL_MAX_CHARS - base.length))).not.toBeNull();
    expect(linkCheck(base + 'a'.repeat(URL_MAX_CHARS - base.length + 1))).toBeNull();
  });
});

function urlRequest(url: string, extra: Record<string, unknown> = {}) {
  const req = readElicitRequest({
    sessionId: 's1',
    toolCallId: 'mcp-elicit-1',
    serverName: 'github',
    message: 'Sign in.',
    mode: 'url',
    url,
    elicitationId: 'e-1',
    ...extra,
  });
  if (req === null || req.mode !== 'url') throw new Error('not a link request');
  return req;
}

describe('the link card', () => {
  it('reads the agent request: a url and an id of at most 128 characters', () => {
    expect(urlRequest('https://example.com')).toMatchObject({ mode: 'url', elicitationId: 'e-1' });
    const read = (extra: Record<string, unknown>) =>
      readElicitRequest({
        sessionId: 's1',
        toolCallId: 'mcp-elicit-1',
        serverName: 'github',
        message: 'Sign in.',
        mode: 'url',
        url: 'https://example.com',
        elicitationId: 'e-1',
        ...extra,
      });
    expect(read({ elicitationId: 'e'.repeat(128) })).not.toBeNull();
    expect(read({ elicitationId: 'e'.repeat(129) })).toBeNull();
    expect(read({ elicitationId: '' })).toMatchObject({ mode: 'url', elicitationId: '' });
    expect(read({ elicitationId: 7 })).toBeNull();
    expect(read({ elicitationId: undefined })).toBeNull();
    expect(read({ url: undefined })).toBeNull();
  });

  it('holds the whole address, its site and whether it is encrypted; the texts redacted', () => {
    const card = linkInteraction(
      'i1',
      urlRequest('https://github.com/login', { message: `Sign in. ${KEY}`, serverName: `git${KEY}` }),
      [KEY],
      5,
    );
    expect(card).toEqual({
      id: 'i1',
      kind: 'mcp-input',
      toolCallId: 'mcp-elicit-1',
      openedAt: 5,
      mode: 'url',
      server: `git${REDACTED}`,
      message: `Sign in. ${REDACTED}`,
      link: { url: 'https://github.com/login', host: 'github.com', secure: true },
    });
  });

  // Mutation: drop the empty id from the unusable check → the card offers Open page for a request the server never names.
  it('an empty elicitation id is a card that can only be declined, never one dropped unseen', () => {
    const card = linkInteraction('i1', urlRequest('https://github.com/login', { elicitationId: '' }), [], 5);
    expect(card.unusable).toBe('address');
    expect(card.link).toBeUndefined();
    expect(card.server).toBe('github');
  });

  it('an address that is not safe, or holds a key, offers Decline only and shows no address', () => {
    for (const url of ['javascript:alert(1)', `https://example.com/cb?k=${KEY}`, `https://${KEY}.example.com/`]) {
      const card = linkInteraction('i1', urlRequest(url), [KEY], 5);
      expect(card.unusable, url).toBe('address');
      expect(card.link).toBeUndefined();
      expect(JSON.stringify(card)).not.toContain(KEY);
    }
  });
});

describe("the server's done", () => {
  it('reads sessionId, elicitationId and the optional serverName', () => {
    expect(readElicitComplete({ sessionId: 's1', elicitationId: 'e-1', serverName: 'github' })).toEqual({
      sessionId: 's1',
      elicitationId: 'e-1',
      server: 'github',
    });
    expect(readElicitComplete({ sessionId: 's1', elicitationId: 'e-1' })).toEqual({
      sessionId: 's1',
      elicitationId: 'e-1',
      server: null,
    });
    for (const bad of [
      null,
      [],
      { elicitationId: 'e-1' },
      { sessionId: '', elicitationId: 'e-1' },
      { sessionId: 's1' },
      { sessionId: 's1', elicitationId: 'e'.repeat(129) },
      { sessionId: 's1', elicitationId: 'e-1', serverName: 3 },
    ]) {
      expect(readElicitComplete(bad)).toBeNull();
    }
  });
});
