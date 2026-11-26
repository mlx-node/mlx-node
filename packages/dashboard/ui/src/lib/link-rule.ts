/**
 * What a Markdown link or `[n]` mark in model text may become. Pure: the renderer draws what these return.
 */

import { displayHost, hostKey, httpUrl } from './web-tools';

/**
 * 'chat': a link is a link only when its host is one of `hosts`; any other link is its words plus the host as text.
 * 'work': every http(s) link is a link, followed by its host when the words are not the address.
 */
export type LinkRule = { mode: 'chat'; hosts: ReadonlySet<string> } | { mode: 'work' };

/** How one link is drawn: its words, then ` (host)` when `host` is set; an anchor only when `href` is set. */
export interface LinkView {
  /** The parsed http(s) address the anchor opens; null: the words are text. */
  href: string | null;
  /** The anchor's tooltip: the parsed address without bidi controls. */
  title: string;
  /** The host shown after the words; null: none. */
  host: string | null;
}

/** Bidi embedding, override and isolate controls and direction marks: they could make a host or a word read in another order. */
function isBidiControl(code: number): boolean {
  return (
    (code >= 0x202a && code <= 0x202e) ||
    (code >= 0x2066 && code <= 0x2069) ||
    code === 0x200e ||
    code === 0x200f ||
    code === 0x061c
  );
}

/** The text without bidi controls. */
export function visibleText(text: string): string {
  let out = '';
  let start = 0;
  for (let i = 0; i < text.length; i += 1) {
    if (isBidiControl(text.charCodeAt(i))) {
      out += text.slice(start, i);
      start = i + 1;
    }
  }
  return start === 0 ? text : out + text.slice(start);
}

const MAX_NAMED_WORDS = 64;

/** A word of link text that reads as an address of another host than the link opens (or as one that does not parse). */
export function wordNamesOtherHost(raw: string, url: URL): boolean {
  let from = 0;
  let to = raw.length;
  while (from < to && '("\'<'.includes(raw[from]!)) from += 1;
  while (to > from && '.,;:!?)"\'>'.includes(raw[to - 1]!)) to -= 1;
  const word = raw.slice(from, to);
  const scheme = word.includes('://');
  if (!scheme && !word.includes('.')) return false;
  const named = httpUrl(scheme ? word : `https://${word}`);
  if (named === null) return scheme;
  return displayHost(named) !== displayHost(url);
}

/** Any word of the link text names another host; past a fixed number of address-like words, assume one does. */
export function namesOtherHost(words: string, url: URL): boolean {
  const text = visibleText(words);
  let seen = 0;
  let start = 0;
  for (let i = 0; i <= text.length; i += 1) {
    const ch = text[i];
    if (ch !== undefined && ch !== ' ' && ch !== '\t' && ch !== '\n') continue;
    const word = text.slice(start, i);
    start = i + 1;
    if (!word.includes('.') && !word.includes('://')) continue;
    seen += 1;
    if (seen > MAX_NAMED_WORDS || wordNamesOtherHost(word, url)) return true;
  }
  return false;
}

/**
 * The view of a link to `href` with the visible `words` under `rule`: only http(s) can open; in Chat only a host of
 * the conversation opens, and words that name another host show the host the link really opens.
 */
export function linkView(href: string, words: string, rule: LinkRule): LinkView {
  const url = httpUrl(href.trim());
  if (url === null) return { href: null, title: '', host: null };
  const isAddress = words.trim() === href.trim();
  const host = displayHost(url);
  const active = rule.mode === 'work' || rule.hosts.has(hostKey(url));
  if (!active) return { href: null, title: '', host: isAddress ? null : host };
  const showHost = !isAddress && (rule.mode === 'work' || namesOtherHost(words, url));
  return { href: url.href, title: visibleText(url.href), host: showHost ? host : null };
}

/** The number of a `[n]` mark: 1 to 99, written without a leading 0; anything else is not a mark. */
export function citeNumber(text: string): number | null {
  if (text.length === 0 || text.length > 2) return null;
  for (const ch of text) if (ch < '0' || ch > '9') return null;
  if (text[0] === '0') return null;
  return Number(text);
}
