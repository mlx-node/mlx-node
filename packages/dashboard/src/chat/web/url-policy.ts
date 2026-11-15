/**
 * Which addresses `web_fetch` sends to a provider. The provider reads the page from its own servers, so this is not
 * the leak control (that is the Chat address ledger); it refuses addresses that have no business leaving the Mac:
 * local names, private and special IPv4 ranges, IPv6 literals, user info. `new URL` already writes every IPv4
 * spelling as four decimal octets, so the scan reads parts of the parsed URL and never guesses at spellings.
 */

export const MAX_FETCH_URL_CHARS = 2048;

const LOCAL_SUFFIXES = ['.localhost', '.local', '.internal', '.lan', '.home.arpa'];

/** The address the server forwards: trimmed, without its fragment (a fragment never reaches a server). */
export function forwardedAddress(text: string): string {
  const trimmed = text.trim();
  const hash = trimmed.indexOf('#');
  return hash === -1 ? trimmed : trimmed.slice(0, hash);
}

function allDigits(text: string): boolean {
  if (text === '') return false;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c < 0x30 || c > 0x39) return false;
  }
  return true;
}

function privateIpv4(host: string): boolean {
  const parts = host.split('.');
  if (parts.length !== 4 || !parts.every(allDigits)) return true;
  const [a, b] = parts.map(Number) as [number, number, number, number];
  if (a === 0 || a === 10 || a === 127 || a >= 224) return true;
  if (a === 100 && b >= 64 && b <= 127) return true;
  if (a === 169 && b === 254) return true;
  if (a === 172 && b >= 16 && b <= 31) return true;
  return a === 192 && b === 168;
}

/**
 * The text is forwarded as written, so it must name the same host in every URL parser, not only in WHATWG `new URL`
 * (which drops tabs and line breaks, reads a backslash as `/` and maps non-ASCII host letters): no white space, control
 * character or backslash anywhere, and only ASCII up to the end of the host.
 */
function parsesAlike(text: string): boolean {
  const scheme = text.indexOf('://');
  let inHost = true;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c <= 0x20 || c === 0x7f || c === 0x5c) return false;
    if (i > scheme + 2 && (c === 0x2f || c === 0x3f || c === 0x23)) inHost = false;
    if (inHost && c >= 0x80) return false;
  }
  return true;
}

/** True when the address may be sent to a provider. */
export function fetchAllowed(text: string): boolean {
  if (text.length > MAX_FETCH_URL_CHARS || !parsesAlike(text)) return false;
  let url: URL;
  try {
    url = new URL(text);
  } catch {
    return false;
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return false;
  if (url.username !== '' || url.password !== '') return false;
  let host = url.hostname;
  if (host.endsWith('.')) host = host.slice(0, -1);
  if (host === '' || host.startsWith('[')) return false;
  const labels = host.split('.');
  if (labels.includes('')) return false;
  if (allDigits(labels[labels.length - 1]!)) return !privateIpv4(host);
  if (host === 'localhost' || !host.includes('.')) return false;
  for (const suffix of LOCAL_SUFFIXES) if (host.endsWith(suffix)) return false;
  return true;
}
