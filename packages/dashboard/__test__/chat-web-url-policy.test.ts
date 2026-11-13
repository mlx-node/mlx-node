/**
 * The `web_fetch` address policy: local names, private and special IPv4 ranges in every spelling `new URL` accepts,
 * IPv6 literals, user info and a trailing dot are refused; ordinary names that look like hex or digits are not.
 */

import { describe, expect, it } from 'vite-plus/test';

import { fetchAllowed, forwardedAddress, MAX_FETCH_URL_CHARS } from '../src/chat/web/url-policy.js';

describe('web_fetch address policy', () => {
  it('allows public names and public IPv4, also names made of hex letters', () => {
    for (const url of [
      'https://bbc.de/news',
      'https://cafe.de/',
      'https://dead.be',
      'https://1.1.1.1/dns',
      'http://example.com/a?b=c',
      'https://example.com./trailing-dot-of-a-public-name',
      'https://100.63.255.255/',
      'https://100.128.0.1/',
      'https://172.15.0.1/',
      'https://172.32.0.1/',
      'https://223.255.255.255/',
      'https://xn--mller-kva.de/',
    ]) {
      expect([url, fetchAllowed(url)]).toEqual([url, true]);
    }
  });

  it('refuses local names, private ranges in any spelling, IPv6 literals and user info', () => {
    for (const url of [
      'https://localhost./',
      'https://localhost/',
      'https://api.localhost/',
      'https://nas.local./share',
      'https://nas.local/share',
      'https://x.internal./',
      'https://printer.lan/',
      'https://router.home.arpa/',
      'https://intranet/',
      'https://0x7f.1/',
      'https://2130706433/',
      'https://127.1/',
      'https://017700000001/',
      'https://[::1]/',
      'https://[fe80::1]/',
      'https://[::ffff:127.0.0.1]/',
      'https://0.0.0.0/',
      'https://10.1.2.3/',
      'https://100.64.0.1/',
      'https://100.127.255.255/',
      'https://169.254.169.254/latest/meta-data',
      'https://172.16.0.1/',
      'https://172.31.255.255/',
      'https://192.168.1.1/',
      'https://224.0.0.1/',
      'https://255.255.255.255/',
      'https://user:pass@example.com/',
      'https://user@example.com/',
      'ftp://example.com/',
      'file:///etc/hosts',
      'javascript:alert(1)',
      'not a url',
      '',
      `https://example.com/${'a'.repeat(MAX_FETCH_URL_CHARS)}`,
    ]) {
      expect([url, fetchAllowed(url)]).toEqual([url, false]);
    }
  });

  it('refuses what another URL parser could read as a different host: empty labels, backslashes, white space', () => {
    for (const url of [
      'http://localhost../',
      'http://intranet../',
      'http://nas.local../',
      'http://x.internal../',
      'http://router.home.arpa../',
      'http://192.168.1.1../',
      'https://a..example.com/',
      'https://example.com\\@169.254.169.254/latest/meta-data/',
      'https://example.com\\path',
      'http://127.0.0.1\t.nip.io/',
      'https://exa\nmple.com/',
      'https://example.com/a b',
      'https://ex\u00e4mple.com/',
      'https://example.com\uff0f.evil.example/',
    ]) {
      expect([url, fetchAllowed(url)]).toEqual([url, false]);
    }
    expect(fetchAllowed('https://example.com/caf\u00e9?q=\u00fc')).toBe(true);
  });

  it('forwards an address trimmed and without its fragment', () => {
    expect(forwardedAddress('  https://example.com/a?b#frag  ')).toBe('https://example.com/a?b');
    expect(forwardedAddress('https://example.com/#')).toBe('https://example.com/');
    expect(forwardedAddress('https://example.com/a')).toBe('https://example.com/a');
  });
});
