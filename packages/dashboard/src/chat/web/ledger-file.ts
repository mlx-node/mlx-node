/**
 * The Chat address ledger as one file: the host writes the normal forms of a session's ledger as a JSON array of
 * strings, and `web_fetch` reads it at every call. Anything but such an array (no file, a broken file, another shape)
 * reads as null, which allows no address.
 */

import { readFileSync } from 'node:fs';

import { forwardedAddress } from './url-policy.js';

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

export function ledgerFileText(members: Iterable<string>): string {
  return JSON.stringify([...members]);
}

export function parseLedgerFile(text: string): ReadonlySet<string> | null {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return null;
  }
  if (!Array.isArray(raw)) return null;
  const members = new Set<string>();
  for (const item of raw as unknown[]) {
    if (typeof item !== 'string') return null;
    members.add(item);
  }
  return members;
}

export function readLedgerFile(path: string): ReadonlySet<string> | null {
  if (path === '') return null;
  try {
    return parseLedgerFile(readFileSync(path, 'utf8'));
  } catch {
    return null;
  }
}

/** Whether `web_fetch` may read this address: it forwards as written and its normal form is in the ledger. */
export function ledgerAllows(members: ReadonlySet<string> | null, text: string): boolean {
  if (members === null || !forwardsAsWritten(text)) return false;
  const normal = normalizeAddress(text);
  return normal !== null && members.has(normal);
}
