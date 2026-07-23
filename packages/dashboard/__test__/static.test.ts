import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { serveStatic } from '../src/static.js';

/** Minimal ServerResponse capturing the status + written body. */
class MockRes {
  statusCode = 0;
  headers: Record<string, unknown> = {};
  body: Buffer | undefined;
  writeHead(status: number, headers: Record<string, unknown>): this {
    this.statusCode = status;
    this.headers = headers;
    return this;
  }
  end(chunk?: Buffer): void {
    this.body = chunk;
  }
}

function get(webRoot: string, pathname: string): MockRes {
  const res = new MockRes();
  serveStatic({ method: 'GET' } as IncomingMessage, res as unknown as ServerResponse, webRoot, pathname);
  return res;
}

let base: string;
let webRoot: string;

beforeEach(() => {
  base = mkdtempSync(join(tmpdir(), 'dash-static-'));
  webRoot = join(base, 'web');
  mkdirSync(webRoot, { recursive: true });
  writeFileSync(join(webRoot, 'index.html'), '<html>INDEX</html>');
  writeFileSync(join(webRoot, 'app.js'), 'APP');
});

afterEach(() => {
  rmSync(base, { recursive: true, force: true });
});

describe('serveStatic', () => {
  it('serves a real file inside the web root', () => {
    const res = get(webRoot, '/app.js');
    expect(res.statusCode).toBe(200);
    expect(res.body?.toString()).toBe('APP');
  });

  it('serves index.html at the root (SPA entry)', () => {
    const res = get(webRoot, '/');
    expect(res.statusCode).toBe(200);
    expect(res.body?.toString()).toContain('INDEX');
  });

  it('does NOT follow a symlink under the root that escapes it', () => {
    // The classic arbitrary-file-disclosure shape: a symlinked child pointing at a
    // secret outside the web root. A lexical prefix check would pass and then
    // readFileSync would follow the link; the canonical re-check must reject it and
    // fall back to index.html instead of leaking the target.
    const secret = join(base, 'secret.txt');
    writeFileSync(secret, 'SECRET-TOKEN');
    symlinkSync(secret, join(webRoot, 'evil.js'));

    const res = get(webRoot, '/evil.js');
    expect(res.body?.toString()).not.toContain('SECRET-TOKEN');
    // Falls back to the SPA entry rather than disclosing the escaping target.
    expect(res.statusCode).toBe(200);
    expect(res.body?.toString()).toContain('INDEX');
  });

  it('blocks .. traversal outside the root', () => {
    writeFileSync(join(base, 'outside.txt'), 'OUTSIDE');
    const res = get(webRoot, '/../outside.txt');
    expect(res.body?.toString()).not.toContain('OUTSIDE');
    expect(res.body?.toString()).toContain('INDEX');
  });

  it('still serves through a legitimately symlinked web root', () => {
    // Canonicalizing BOTH sides must not break a web root that is itself reached
    // through a symlink (e.g. macOS /tmp→/private/tmp, or a node_modules symlink).
    const realDir = join(base, 'real-web');
    mkdirSync(realDir, { recursive: true });
    writeFileSync(join(realDir, 'index.html'), '<html>LINKED-INDEX</html>');
    writeFileSync(join(realDir, 'app.js'), 'LINKED-APP');
    const linkRoot = join(base, 'link-web');
    mkdirSync(dirname(linkRoot), { recursive: true });
    symlinkSync(realDir, linkRoot);

    expect(get(linkRoot, '/app.js').body?.toString()).toBe('LINKED-APP');
    expect(get(linkRoot, '/').body?.toString()).toContain('LINKED-INDEX');
  });
});
