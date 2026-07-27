/**
 * What `app://` should serve for a given request path — the whole decision,
 * with no Electron in it.
 *
 * This is a separate module from `protocol.ts` for a concrete reason, not tidiness:
 * `import { net, protocol } from 'electron'` cannot be resolved outside an Electron
 * process, so anything in that file is unreachable from a plain Node test. The
 * traversal guard and the asset-404 rule are exactly the parts that must be
 * provable, so they live here and `protocol.ts` is reduced to I/O.
 */

import { existsSync, statSync } from 'node:fs';
import { extname, join, normalize, resolve, sep } from 'node:path';

const MIME: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.map': 'application/json; charset=utf-8',
};

export function mimeFor(file: string): string {
  return MIME[extname(file).toLowerCase()] ?? 'application/octet-stream';
}

/**
 * Everything is same-origin and bundled, so the policy is almost entirely
 * `'self'`. Two deliberate loosenings:
 *
 * - `style-src 'unsafe-inline'`: React sets element `style` attributes and the
 *   chart layer injects style tags. There is no nonce path for either, and inline
 *   *style* is not a code-execution vector the way inline script is.
 * - `img-src data: blob:`: icons and generated chart bitmaps.
 *
 * `script-src` stays strict — no `unsafe-inline`, no `unsafe-eval`. `connect-src`
 * is `'self'` only: the SPA talks over a MessagePort, not HTTP, so it has no
 * legitimate reason to reach the network at all.
 */
export const CSP = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  "connect-src 'self'",
  "object-src 'none'",
  "frame-ancestors 'none'",
  "base-uri 'none'",
  "form-action 'none'",
].join('; ');

/**
 * Resolve a URL pathname to a file inside `root`, or `null` if the input is not a
 * pathname we will serve.
 *
 * **`..` cannot escape here, and not because of the containment check below.**
 * Two layers collapse it before that line runs, which is worth stating because it
 * is easy to write a "blocks traversal" test that passes for the wrong reason:
 *
 *   1. the URL layer — `new URL('app://h/../x').pathname` is already `/x`;
 *   2. `path.normalize` on an ABSOLUTE path drops leading `..` (`/../x` → `/x`).
 *
 * So `/assets/../../../etc/passwd` lands on `<root>/etc/passwd` — contained, and
 * simply missing. The containment check is genuine defense in depth: it is what
 * holds if this is ever called with a relative path from some other transport,
 * which is also why the leading-slash contract is enforced explicitly rather than
 * left implicit. The guards that actually fire on real input are the malformed
 * escape and the NUL.
 */
export function resolveWithinRoot(root: string, pathname: string): string | null {
  // A URL pathname for a standard scheme is always absolute. Anything else is not
  // from the protocol handler, and is refused rather than silently resolved
  // relative to the root.
  if (!pathname.startsWith('/')) return null;

  let decoded: string;
  try {
    decoded = decodeURIComponent(pathname);
  } catch {
    // A malformed `%` escape is a bad request, not a path.
    return null;
  }
  // `%00` decodes to a real NUL, which truncates a path in some syscalls. Refuse
  // rather than normalise: `/index.html%00.png` must not become `/index.html`.
  if (decoded.includes('\0')) return null;

  const base = resolve(root);
  const candidate = resolve(join(base, normalize(decoded)));
  // The trailing separator matters: without it a sibling `/www-evil` would pass a
  // bare `startsWith('/www')`.
  if (candidate !== base && !candidate.startsWith(base + sep)) return null;
  return candidate;
}

export type WwwDecision =
  | { kind: 'file'; file: string; contentType: string; immutable: boolean }
  | { kind: 'notFound'; reason: 'escapes-root' | 'missing-asset' };

/**
 * Decide what to serve for `pathname`.
 *
 * The rule that is easy to get wrong: a miss under the hashed-asset directory must
 * 404, NOT fall through to `index.html`. With an index fallback a missing
 * `/assets/index-a1b2c3.js` returns HTML labelled `text/javascript`, and the app
 * fails as a blank window with a syntax error pointing at `<` — indistinguishable
 * from a dozen other packaging faults, and precisely what a stale or mis-copied
 * bundle produces. Everything else falls back to `index.html` so `BrowserRouter`
 * deep links (`/sessions/abc`) resolve.
 */
export function decideWww(root: string, pathname: string): WwwDecision {
  const base = resolve(root);
  const target = resolveWithinRoot(base, pathname);
  if (target === null) return { kind: 'notFound', reason: 'escapes-root' };

  const isAsset = pathname.startsWith('/assets/');
  const hit = existsSync(target) && statSync(target).isFile();

  if (hit) {
    return {
      kind: 'file',
      file: target,
      contentType: mimeFor(target),
      // Every asset filename is content-hashed, so it can be cached forever.
      // index.html is NOT hashed — it is the document that names the current
      // hashes, so caching it would pin the app to a stale build.
      immutable: isAsset,
    };
  }
  if (isAsset) return { kind: 'notFound', reason: 'missing-asset' };

  const index = join(base, 'index.html');
  return { kind: 'file', file: index, contentType: mimeFor(index), immutable: false };
}
