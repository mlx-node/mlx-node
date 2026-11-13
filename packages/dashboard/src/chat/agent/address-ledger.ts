/**
 * The addresses Chat's `web_fetch` may read: those a user message or a parsed search result put into this
 * conversation. The provider fetches a page from its own servers, so an address chosen by injected text would carry
 * conversation text to a stranger's server. Addresses in page text, snippets, titles or the model's own text never
 * count. In memory only.
 */

import { forwardedAddress } from '../web/url-policy.js';

const TRAILING_PUNCTUATION = '.,;:!?)]';

/**
 * `new URL`; scheme and host lower-cased; default port, fragment, one trailing host dot and one trailing `/` of the
 * path dropped; path and query otherwise byte-equal. null for anything that is not http(s).
 */
export function normalizeAddress(text: string): string | null {
  let url: URL;
  try {
    url = new URL(text);
  } catch {
    return null;
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
  let host = url.hostname.toLowerCase();
  if (host.endsWith('.')) host = host.slice(0, -1);
  if (host === '') return null;
  const port = url.port === '' ? '' : `:${url.port}`;
  const auth = url.username !== '' || url.password !== '' ? `${url.username}:${url.password}@` : '';
  let path = url.pathname;
  if (path.endsWith('/')) path = path.slice(0, -1);
  return `${url.protocol}//${auth}${host}${port}${path}${url.search}`;
}

/**
 * True when the address the web server forwards for this text (`forwardedAddress`: trimmed, fragment cut off) is what
 * a parser would send, up to scheme and host case, a missing final `/` and UTF-8 escapes of non-ASCII. The ledger
 * compares normal forms, but the forwarded text goes to the provider, so any part the normal form drops or the parser
 * rewrites (dot segments, stripped tabs and newlines, back-slashes, default port, escaped host bytes) could carry
 * conversation text past the ledger. The fragment is the one part never forwarded.
 */
export function forwardsAsWritten(text: string): boolean {
  if (text.trim() === '') return false;
  const trimmed = forwardedAddress(text);
  for (let i = 0; i < trimmed.length; i++) {
    const c = trimmed.charCodeAt(i);
    if (c <= 32 || c === 127 || c === 92) return false;
  }
  let href: string;
  try {
    href = new URL(trimmed).href;
  } catch {
    return false;
  }
  const expected = escapeNonAscii(lowerAuthority(trimmed));
  return href === expected || href === `${expected}/`;
}

/** Lower-cases everything before the first `/`, `?` after `scheme://` (scheme, user info, host, port). */
function lowerAuthority(text: string): string {
  const sep = text.indexOf('://');
  if (sep === -1) return text;
  let end = sep + 3;
  while (end < text.length && text[end] !== '/' && text[end] !== '?') end++;
  return text.slice(0, end).toLowerCase() + text.slice(end);
}

function escapeNonAscii(text: string): string {
  let out = '';
  for (const ch of text) {
    if (ch.charCodeAt(0) < 128) {
      out += ch;
      continue;
    }
    for (const byte of Buffer.from(ch, 'utf8')) out += `%${byte.toString(16).toUpperCase().padStart(2, '0')}`;
  }
  return out;
}

function isAddressEnd(ch: string): boolean {
  const c = ch.charCodeAt(0);
  if (ch === '<' || ch === '>' || ch === '"' || ch === '`') return true;
  return (
    c <= 32 || c === 0x85 || c === 0xa0 || c === 0x2028 || c === 0x2029 || c === 0x3000 || (c >= 0x2000 && c <= 0x200a)
  );
}

/**
 * Every `http://` or `https://` address in a text, read to the next white space, `<`, `>`, `"` or back-tick; each one
 * also without trailing `.,;:!?)]`.
 */
export function scanAddresses(text: string): string[] {
  const found: string[] = [];
  let i = 0;
  while (i < text.length) {
    const http = text.indexOf('http://', i);
    const https = text.indexOf('https://', i);
    const start = http === -1 ? https : https === -1 ? http : Math.min(http, https);
    if (start === -1) break;
    let end = start;
    while (end < text.length && !isAddressEnd(text[end]!)) end++;
    const raw = text.slice(start, end);
    found.push(raw);
    let trimmed = raw;
    while (trimmed.length > 0 && TRAILING_PUNCTUATION.includes(trimmed[trimmed.length - 1]!))
      trimmed = trimmed.slice(0, -1);
    if (trimmed !== raw && trimmed !== '') found.push(trimmed);
    i = end > start ? end : start + 1;
  }
  return found;
}

export class AddressLedger {
  private readonly members = new Set<string>();

  /** The addresses of one user message. */
  addUserText(text: string): void {
    for (const address of scanAddresses(text)) this.add(address);
  }

  /** The `url` of every result of one parsed search result. */
  addSearchResults(urls: Iterable<string>): void {
    for (const url of urls) this.add(url);
  }

  has(address: string): boolean {
    const normal = normalizeAddress(address);
    return normal !== null && this.members.has(normal);
  }

  /** Hosts of every member, for link rendering. */
  hosts(): Set<string> {
    const out = new Set<string>();
    for (const m of this.members) {
      try {
        out.add(new URL(m).hostname);
      } catch {
        // Members are normal forms of parsed URLs.
      }
    }
    return out;
  }

  get size(): number {
    return this.members.size;
  }

  private add(address: string): void {
    const normal = normalizeAddress(address);
    if (normal !== null) this.members.add(normal);
  }
}
