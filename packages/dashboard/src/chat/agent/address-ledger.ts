/**
 * The addresses Chat's `web_fetch` may read: those a user message, a parsed search result or the text of a page read
 * put into this conversation. The provider fetches a page from its own servers, so an address chosen by injected text
 * would carry conversation text to a stranger's server; an address taken whole from a page the provider returned
 * carries only what that page already held. Addresses in snippets, titles or the model's own text never count. The
 * host writes the members to the session's ledger file at every change, and the `mlx_web` server checks each read
 * against that file.
 */

import { randomUUID } from 'node:crypto';
import { renameSync, unlinkSync, writeFileSync } from 'node:fs';

import { ledgerFileText, normalizeAddress } from '../web/ledger-file.js';

const TRAILING_PUNCTUATION = '.,;:!?)]';

function isAddressEnd(ch: string): boolean {
  const c = ch.charCodeAt(0);
  if (ch === '<' || ch === '>' || ch === '"' || ch === '`') return true;
  return (
    c <= 32 || c === 0x85 || c === 0xa0 || c === 0x2028 || c === 0x2029 || c === 0x3000 || (c >= 0x2000 && c <= 0x200a)
  );
}

function count(text: string, ch: string): number {
  let n = 0;
  for (const c of text) if (c === ch) n++;
  return n;
}

/** Without trailing `.,;:!?)]`, except a `)` or `]` that closes a `(` or `[` inside the address. */
function trimUnbalanced(raw: string): string {
  let out = raw;
  while (out.length > 0) {
    const last = out[out.length - 1]!;
    if (!TRAILING_PUNCTUATION.includes(last)) break;
    if (last === ')' && count(out, ')') <= count(out, '(')) break;
    if (last === ']' && count(out, ']') <= count(out, '[')) break;
    out = out.slice(0, -1);
  }
  return out;
}

/**
 * Every `http://` or `https://` address in a text, read to the next white space, `<`, `>`, `"` or back-tick; each one
 * also without trailing `.,;:!?)]`, and without those but a `)` or `]` that closes a bracket inside it (a Markdown link
 * to `.../Mercury_(planet)`).
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
    const balanced = trimUnbalanced(raw);
    if (balanced !== raw && balanced !== trimmed) found.push(balanced);
    i = end > start ? end : start + 1;
  }
  return found;
}

export class AddressLedger {
  private readonly set = new Set<string>();

  /** The addresses of one user message. */
  addUserText(text: string): void {
    for (const address of scanAddresses(text)) this.add(address);
  }

  /** The `url` of every result of one parsed search result. */
  addSearchResults(urls: Iterable<string>): void {
    for (const url of urls) this.add(url);
  }

  /** The addresses in the text of one parsed page result. */
  addPageText(text: string): void {
    for (const address of scanAddresses(text)) this.add(address);
  }

  /** The normal forms, in the order they were added. */
  members(): string[] {
    return [...this.set];
  }

  has(address: string): boolean {
    const normal = normalizeAddress(address);
    return normal !== null && this.set.has(normal);
  }

  /** Hosts of every member, for link rendering. */
  hosts(): Set<string> {
    const out = new Set<string>();
    for (const m of this.set) {
      try {
        out.add(new URL(m).hostname);
      } catch {
        // Members are normal forms of parsed URLs.
      }
    }
    return out;
  }

  get size(): number {
    return this.set.size;
  }

  private add(address: string): void {
    const normal = normalizeAddress(address);
    if (normal !== null) this.set.add(normal);
  }
}

/** Temp file and rename in the same folder, mode 0600, synchronous: the next `web_fetch` may read it right after. */
export function writeLedgerFile(path: string, members: Iterable<string>): void {
  const temp = `${path}.${randomUUID()}.tmp`;
  try {
    writeFileSync(temp, ledgerFileText(members), { mode: 0o600, flag: 'wx' });
    renameSync(temp, path);
  } finally {
    try {
      unlinkSync(temp);
    } catch {
      // Renamed: nothing left.
    }
  }
}
