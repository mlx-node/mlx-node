import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

import { CSP, decideWww, resolveWithinRoot } from '../src/main/www.js';

let tmp: string;
let root: string;

beforeAll(() => {
  tmp = mkdtempSync(join(tmpdir(), 'mlx-www-'));
  root = join(tmp, 'www');
  mkdirSync(join(root, 'assets'), { recursive: true });
  writeFileSync(join(root, 'index.html'), '<!doctype html><div id="root"></div>');
  writeFileSync(join(root, 'assets', 'index-a1b2c3.js'), 'export const x = 1;');
  writeFileSync(join(root, 'assets', 'index-a1b2c3.css'), ':root{}');
  // A sibling directory sharing the root's name as a prefix. A containment check
  // written as a bare `startsWith(root)` would treat this as inside the root.
  mkdirSync(join(tmp, 'www-evil'), { recursive: true });
  writeFileSync(join(tmp, 'www-evil', 'stolen.txt'), 'secret');
  writeFileSync(join(tmp, 'outside.txt'), 'secret');
});

afterAll(() => {
  rmSync(tmp, { recursive: true, force: true });
});

describe('resolveWithinRoot', () => {
  it('resolves an ordinary path inside the root', () => {
    expect(resolveWithinRoot(root, '/assets/index-a1b2c3.js')).toBe(join(root, 'assets', 'index-a1b2c3.js'));
  });

  // `..` is collapsed to nothing BEFORE the containment check ever runs — the URL
  // layer does it once (`new URL('app://h/../x').pathname === '/x'`) and
  // `path.normalize` does it again for absolute paths. So a climb does not get
  // refused; it gets clamped INSIDE the root and then simply misses.
  //
  // This is asserted as containment rather than as a 404 on purpose. Writing it
  // as `toBeNull()` would pass — but for the wrong reason, and it would keep
  // passing if the containment check were deleted outright.
  it('clamps a .. climb inside the root instead of escaping', () => {
    expect(resolveWithinRoot(root, '/../outside.txt')).toBe(join(root, 'outside.txt'));
    expect(resolveWithinRoot(root, '/assets/../../outside.txt')).toBe(join(root, 'outside.txt'));
    expect(resolveWithinRoot(root, '/%2e%2e/outside.txt')).toBe(join(root, 'outside.txt'));
    expect(resolveWithinRoot(root, '//../../outside.txt')).toBe(join(root, 'outside.txt'));
    // The sibling-prefix directory is likewise reached as a CHILD, never as the
    // real /tmp/.../www-evil.
    expect(resolveWithinRoot(root, '/../www-evil/stolen.txt')).toBe(join(root, 'www-evil', 'stolen.txt'));
  });

  // The only way to reach the containment check, and the reason it is kept: a
  // relative path would otherwise resolve against the root's PARENT.
  it('refuses a non-absolute path outright', () => {
    expect(resolveWithinRoot(root, '../www-evil/stolen.txt')).toBeNull();
    expect(resolveWithinRoot(root, 'assets/index-a1b2c3.js')).toBeNull();
  });

  it('refuses a malformed percent escape rather than throwing', () => {
    expect(resolveWithinRoot(root, '/%zz')).toBeNull();
    expect(resolveWithinRoot(root, '/%')).toBeNull();
  });

  it('refuses an embedded NUL', () => {
    expect(resolveWithinRoot(root, '/index.html%00.png')).toBeNull();
  });

  it('allows the root itself', () => {
    expect(resolveWithinRoot(root, '/')).toBe(root);
  });
});

describe('decideWww', () => {
  it('serves a hashed asset with its real content type, cacheable forever', () => {
    const d = decideWww(root, '/assets/index-a1b2c3.js');
    expect(d).toMatchObject({
      kind: 'file',
      file: join(root, 'assets', 'index-a1b2c3.js'),
      contentType: 'text/javascript; charset=utf-8',
      immutable: true,
    });
    expect(decideWww(root, '/assets/index-a1b2c3.css')).toMatchObject({
      contentType: 'text/css; charset=utf-8',
    });
  });

  // BrowserRouter only works if an unknown path returns the document.
  it('falls back to index.html for a deep link', () => {
    for (const path of ['/', '/sessions', '/sessions/abc-123', '/metrics']) {
      expect(decideWww(root, path)).toMatchObject({
        kind: 'file',
        file: join(root, 'index.html'),
        contentType: 'text/html; charset=utf-8',
        // index.html names the current asset hashes, so caching it would pin the
        // app to a stale build.
        immutable: false,
      });
    }
  });

  // The rule this whole module exists for. An index fallback here returns HTML
  // labelled `text/javascript`; the app then dies as a blank window with a syntax
  // error pointing at `<`, which is what a stale or mis-copied bundle looks like
  // and is indistinguishable from a dozen unrelated packaging faults.
  it('404s a MISSING asset instead of falling back to index.html', () => {
    expect(decideWww(root, '/assets/index-deadbeef.js')).toEqual({
      kind: 'notFound',
      reason: 'missing-asset',
    });
    expect(decideWww(root, '/assets/nope.css')).toMatchObject({ kind: 'notFound' });
  });

  it('404s an input the resolver refuses', () => {
    expect(decideWww(root, 'assets/index-a1b2c3.js')).toEqual({ kind: 'notFound', reason: 'escapes-root' });
    expect(decideWww(root, '/%zz')).toEqual({ kind: 'notFound', reason: 'escapes-root' });
  });

  // `%00` must not truncate the name back to a servable file.
  it('does not let a NUL truncate a path into a real file', () => {
    expect(decideWww(root, '/index.html%00.png')).toMatchObject({ kind: 'notFound' });
  });

  // A directory is not a file. Serving it would hand the renderer a stream that
  // fails in a way that looks like a corrupt asset.
  it('does not serve a directory as a file', () => {
    expect(decideWww(root, '/assets')).toMatchObject({ kind: 'file', file: join(root, 'index.html') });
  });

  it('labels an unknown extension as a byte stream rather than guessing', () => {
    writeFileSync(join(root, 'weights.bin'), 'x');
    expect(decideWww(root, '/weights.bin')).toMatchObject({ contentType: 'application/octet-stream' });
  });
});

describe('CSP', () => {
  // Inline STYLE is allowed on purpose; inline SCRIPT is the code-execution
  // vector and must stay out, as must eval.
  it('keeps script-src strict', () => {
    const scriptSrc = CSP.split('; ').find((d) => d.startsWith('script-src'));
    expect(scriptSrc).toBe("script-src 'self'");
    expect(CSP).not.toContain("'unsafe-eval'");
  });

  it('confines the app to its own origin', () => {
    expect(CSP).toContain("default-src 'self'");
    // The SPA speaks over a MessagePort, never HTTP, so it has no reason to reach
    // the network at all.
    expect(CSP).toContain("connect-src 'self'");
    expect(CSP).toContain("object-src 'none'");
    expect(CSP).toContain("frame-ancestors 'none'");
  });
});
