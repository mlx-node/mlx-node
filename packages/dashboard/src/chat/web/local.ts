/**
 * Firecrawl on this Mac: a `firecrawl-local serve` the user installs and starts; the app only talks to it.
 *
 * The address is loopback only and saved as an IP literal, so no name lookup can move it and no page address crosses
 * the network in clear text. The page itself is then fetched from inside the user's network, so before each use the
 * check proves that the server is Firecrawl Local and that it refuses a private target on its own (its rule also runs
 * after DNS and on every redirect). Paths are fixed here. No request carries an Authorization header: the server
 * would pass a caller's key on to Firecrawl Cloud.
 */

import { httpJson, record } from './http.js';

export const LOCAL_DEFAULT_URL = 'http://127.0.0.1:3002';
export const LOCAL_HEALTH_TIMEOUT_MS = 2_000;
export const LOCAL_PROBE_TIMEOUT_MS = 5_000;
export const LOCAL_CLOUD_TIMEOUT_MS = 2_000;
/** The `timeout` the probe asks the server to give its own scrape. */
export const LOCAL_PROBE_SCRAPE_MS = 3_000;
export const LOCAL_MIN_PORT = 1024;
export const LOCAL_MAX_PORT = 65_535;
export const LOCAL_ENGINE = 'firecrawl-local';
const MAX_VERSION_CHARS = 16;

/**
 * `ok`: Firecrawl Local, refusing private targets. `not-running`: no answer. `not-firecrawl`: something else answers.
 * `open`: it reads private addresses (`--allow-private`). `token`: it asks for a token (`--token`).
 */
export type LocalCheck = 'ok' | 'not-running' | 'not-firecrawl' | 'open' | 'token';

/** `off`: no page leaves this Mac. `fallback`: a page the server cannot load is sent to Firecrawl Cloud. */
export type LocalCloud = 'off' | 'fallback';

export interface LocalCheckResult {
  check: LocalCheck;
  /** The server's version, only when it is digits and dots. */
  version: string | null;
  /** The server's Cloud setting; null when it was not read or the answer was not one of the two known refusals. */
  cloud: LocalCloud | null;
}

export interface LocalCheckOptions {
  fetch: typeof fetch;
  /** Also read the Cloud setting. The web tool's check before a page read leaves it out. */
  readCloud: boolean;
  healthMs?: number;
  probeMs?: number;
  cloudMs?: number;
}

/**
 * The saved form of a typed address, or null. Accepted: `http://` then `127.0.0.1`, `localhost` or `[::1]` with an
 * explicit port from 1024 to 65535, nothing after the port but one optional `/`. Saved as `http://127.0.0.1:<port>`
 * (`localhost` is rewritten, so no name is ever looked up) or `http://[::1]:<port>`.
 */
export function parseLocalAddress(raw: string): string | null {
  const text = raw.trim();
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    // White space, controls, non-ASCII, `\`, `@` (user info), `?` and `#`.
    if (c <= 0x20 || c >= 0x7f || c === 0x5c || c === 0x40 || c === 0x3f || c === 0x23) return null;
  }
  if (text.slice(0, 7).toLowerCase() !== 'http://') return null;
  const rest = text.slice(7);
  const slash = rest.indexOf('/');
  const authority = slash === -1 ? rest : rest.slice(0, slash);
  const tail = slash === -1 ? '' : rest.slice(slash);
  if (authority === '' || (tail !== '' && tail !== '/')) return null;
  let url: URL;
  try {
    url = new URL(`http://${authority}/`);
  } catch {
    return null;
  }
  if (url.port === '') return null;
  const port = Number(url.port);
  if (!Number.isInteger(port) || port < LOCAL_MIN_PORT || port > LOCAL_MAX_PORT) return null;
  if (url.hostname === '[::1]') return `http://[::1]:${port}`;
  if (url.hostname === '127.0.0.1' || url.hostname === 'localhost') return `http://127.0.0.1:${port}`;
  return null;
}

/** The text when it already is a saved form, else null. */
export function localOrigin(text: string): string | null {
  return parseLocalAddress(text) === text ? text : null;
}

function versionOf(value: unknown): string | null {
  if (typeof value !== 'string' || value === '' || value.length > MAX_VERSION_CHARS) return null;
  for (let i = 0; i < value.length; i++) {
    const c = value.charCodeAt(i);
    if (c !== 0x2e && (c < 0x30 || c > 0x39)) return null;
  }
  return value;
}

function codeOf(json: unknown): unknown {
  return record(json)?.code;
}

/**
 * Three requests to the saved address, in order: `GET /health` (must name the engine `firecrawl-local`); a scrape of
 * the server's own `/health`, a loopback target it must refuse with 403 `BLOCKED_TARGET` (a refused target is never
 * handed to Firecrawl Cloud); then, when asked and only after both passed, the Cloud setting, read from a forwarded
 * route the server refuses itself (501 `CLOUD_ONLY` when the Cloud is off, 403 `CROSS_SITE_REQUEST` for a request
 * marked as coming from a web page), so this request does not leave the Mac either. Never throws.
 */
export async function checkLocal(url: string, opts: LocalCheckOptions): Promise<LocalCheckResult> {
  const base = localOrigin(url);
  // Not a saved form: nothing is sent.
  if (base === null) return { check: 'not-running', version: null, cloud: null };
  const health = await httpJson({
    fetch: opts.fetch,
    baseUrl: base,
    method: 'GET',
    path: '/health',
    auth: {},
    timeoutMs: opts.healthMs ?? LOCAL_HEALTH_TIMEOUT_MS,
  });
  if (health.kind === 'failed') {
    return { check: health.code === 'provider-error' ? 'not-firecrawl' : 'not-running', version: null, cloud: null };
  }
  if (health.status === 401) return { check: 'token', version: null, cloud: null };
  const body = record(health.json);
  if (health.status !== 200 || !health.parsed || body === null || body.engine !== LOCAL_ENGINE) {
    return { check: 'not-firecrawl', version: null, cloud: null };
  }
  const version = versionOf(body.version);
  const probe = await httpJson({
    fetch: opts.fetch,
    baseUrl: base,
    method: 'POST',
    path: '/v2/scrape',
    auth: {},
    body: { url: `${base}/health`, formats: ['markdown'], timeout: LOCAL_PROBE_SCRAPE_MS },
    timeoutMs: opts.probeMs ?? LOCAL_PROBE_TIMEOUT_MS,
  });
  if (probe.kind === 'failed' && probe.code !== 'provider-error') return { check: 'not-running', version, cloud: null };
  if (probe.kind === 'response' && probe.status === 401) return { check: 'token', version, cloud: null };
  if (probe.kind !== 'response' || probe.status !== 403 || codeOf(probe.json) !== 'BLOCKED_TARGET') {
    return { check: 'open', version, cloud: null };
  }
  if (!opts.readCloud) return { check: 'ok', version, cloud: null };
  const cloud = await httpJson({
    fetch: opts.fetch,
    baseUrl: base,
    method: 'GET',
    path: '/v2/team/credit-usage',
    auth: {},
    headers: { 'Sec-Fetch-Site': 'cross-site' },
    timeoutMs: opts.cloudMs ?? LOCAL_CLOUD_TIMEOUT_MS,
  });
  if (cloud.kind === 'response' && cloud.status === 501 && codeOf(cloud.json) === 'CLOUD_ONLY') {
    return { check: 'ok', version, cloud: 'off' };
  }
  if (cloud.kind === 'response' && cloud.status === 403 && codeOf(cloud.json) === 'CROSS_SITE_REQUEST') {
    return { check: 'ok', version, cloud: 'fallback' };
  }
  return { check: 'ok', version, cloud: null };
}
