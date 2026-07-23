/**
 * Static SPA file serving for the dashboard. Streams files out of `webRoot`
 * with a small extension→content-type map, falls back to `index.html` for any
 * path that does not resolve to a real file (client-side routing), and marks
 * everything `Cache-Control: no-cache` so a rebuilt UI is never served stale.
 *
 * Pure `fs` — the dashboard is a viewer process that never links the native
 * addon.
 */

import { readFileSync, realpathSync, statSync } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { resolve, sep } from 'node:path';

const CONTENT_TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
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
  '.txt': 'text/plain; charset=utf-8',
  '.wasm': 'application/wasm',
};

function contentTypeFor(path: string): string {
  const dot = path.lastIndexOf('.');
  if (dot === -1) return 'application/octet-stream';
  return CONTENT_TYPES[path.slice(dot).toLowerCase()] ?? 'application/octet-stream';
}

/** The canonical (symlink-resolved) web root, or `null` if it does not exist. */
function canonicalRoot(webRoot: string): string | null {
  try {
    return realpathSync(resolve(webRoot));
  } catch {
    return null;
  }
}

/**
 * Resolve a request path to a real file inside the (already canonical) web
 * `root`, or `null` to fall back. Lexical `resolve` blocks `..` traversal; the
 * canonical re-check in {@link tryFile} additionally blocks a symlink *under* the
 * root that points outside it.
 */
function resolveFile(root: string, pathname: string): string | null {
  let rel: string;
  try {
    rel = decodeURIComponent(pathname);
  } catch {
    rel = pathname;
  }
  rel = rel.replace(/^\/+/, '');
  if (rel === '') return tryFile(resolve(root, 'index.html'), root);

  const target = resolve(root, rel);
  // Reject lexical traversal outside the web root; the SPA fallback handles it.
  if (target !== root && !target.startsWith(root + sep)) return null;
  return tryFile(target, root);
}

/**
 * Return `candidate` resolved to its real path only if it is a regular file whose
 * canonical location is contained in `canonRoot`. Canonicalizing both sides means
 * a legitimately symlinked root (e.g. macOS `/tmp`→`/private/tmp`) still serves,
 * while a symlink under the root escaping it is rejected before `readFileSync`
 * follows the link.
 */
function tryFile(candidate: string, canonRoot: string): string | null {
  try {
    const real = realpathSync(candidate);
    if (real !== canonRoot && !real.startsWith(canonRoot + sep)) return null;
    return statSync(real).isFile() ? real : null;
  } catch {
    return null;
  }
}

function send(res: ServerResponse, status: number, body: Buffer, contentType: string, head: boolean): void {
  res.writeHead(status, {
    'Content-Type': contentType,
    'Content-Length': body.byteLength,
    'Cache-Control': 'no-cache',
  });
  res.end(head ? undefined : body);
}

/**
 * Serve a GET/HEAD request from `webRoot`. Missing files fall back to
 * `index.html` (SPA client routing); a missing `index.html` is a 404.
 */
export function serveStatic(req: IncomingMessage, res: ServerResponse, webRoot: string, pathname: string): void {
  const method = req.method ?? 'GET';
  if (method !== 'GET' && method !== 'HEAD') {
    res.writeHead(405, { 'Content-Type': 'application/json; charset=utf-8', Allow: 'GET, HEAD' });
    res.end(JSON.stringify({ error: 'Method not allowed' }));
    return;
  }

  const head = method === 'HEAD';
  const root = canonicalRoot(webRoot);
  if (root === null) {
    res.writeHead(404, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-cache' });
    res.end(JSON.stringify({ error: 'Not found' }));
    return;
  }

  const file = resolveFile(root, pathname);
  if (file !== null) {
    send(res, 200, readFileSync(file), contentTypeFor(file), head);
    return;
  }

  // SPA fallback: serve index.html for any unmatched path.
  const index = tryFile(resolve(root, 'index.html'), root);
  if (index !== null) {
    send(res, 200, readFileSync(index), 'text/html; charset=utf-8', head);
    return;
  }

  res.writeHead(404, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-cache' });
  res.end(JSON.stringify({ error: 'Not found' }));
}
