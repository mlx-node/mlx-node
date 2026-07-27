/**
 * Direct unit coverage for `canonicalCacheRoot` — the JOIN KEY between the
 * agent that writes a trace and the dashboard that reads it back.
 *
 * Every test that reached this function before went through a directory that
 * EXISTS, so `realpath` succeeded on the first try and most of the function was
 * dead to the suite: both tilde spellings, the missing-root walk (which is the
 * NORMAL state of a tier nobody has opened yet — the F1 repro state), and the
 * no-resolvable-ancestor fallback. Two real defects lived in the unreached
 * part:
 *
 *  - root AND parent both missing returned a wholly UNcanonicalized path, so a
 *    symlinked ancestor (`/tmp` → `/private/tmp`) made the reader's key diverge
 *    from the writer's and rows from the same cache read as "a different cache
 *    directory";
 *  - the JS `realpathSync` does not fold filename case, so on case-insensitive
 *    APFS two spellings of one directory produced two keys — a populated disk
 *    scan beside a 0/0 trend.
 *
 * A wrong key here is always SILENT. It never throws; it just zeroes a number.
 */

import { existsSync, mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { canonicalCacheRoot } from '../src/cold-tier.js';

let base: string;

beforeEach(() => {
  base = mkdtempSync(join(tmpdir(), 'cold-root-'));
});

afterEach(() => {
  rmSync(base, { recursive: true, force: true });
});

/** Whether `dir`'s volume folds filename case (APFS default: yes). */
function volumeIsCaseInsensitive(dir: string): boolean {
  const probe = join(dir, 'CaseProbe');
  mkdirSync(probe, { recursive: true });
  return existsSync(join(dir, 'caseprobe'));
}

describe('canonicalCacheRoot — empty and disabled tiers', () => {
  it('keeps an empty root empty rather than inventing a cache identity', () => {
    // The native struct reports `root: ''` while the tier is disabled.
    expect(canonicalCacheRoot('')).toBe('');
  });

  it('treats a whitespace-only root as empty, not as a relative path', () => {
    // `resolve('  ')` would produce `<cwd>/  ` — a plausible-looking key that
    // matches nothing. Both the trim and the emptiness check are load-bearing.
    expect(canonicalCacheRoot('   ')).toBe('');
    expect(canonicalCacheRoot('\t\n')).toBe('');
  });
});

describe('canonicalCacheRoot — tilde expansion', () => {
  it('expands a bare `~` to the home directory', () => {
    expect(canonicalCacheRoot('~')).toBe(realpathSync.native(homedir()));
  });

  it('expands `~/…` against the home directory', () => {
    // A name that does not exist, so this also pins the missing-leaf walk.
    expect(canonicalCacheRoot('~/mlx-node-cold-root-probe')).toBe(
      join(realpathSync.native(homedir()), 'mlx-node-cold-root-probe'),
    );
  });

  it('leaves a `~name` prefix alone — only the shell spellings expand', () => {
    const result = canonicalCacheRoot('~cache');
    // Expanding this would drop the tilde and rename the directory.
    expect(basename(result)).toBe('~cache');
    expect(result).not.toBe(join(realpathSync.native(homedir()), 'ache'));
  });

  it('makes a relative root absolute', () => {
    const result = canonicalCacheRoot('relative-cold-root');
    expect(result).toBe(resolve('relative-cold-root'));
  });
});

describe('canonicalCacheRoot — symlink canonicalization', () => {
  it('resolves an existing root through its symlinks', () => {
    const real = join(base, 'real');
    mkdirSync(real, { recursive: true });
    const link = join(base, 'link');
    symlinkSync(real, link);
    expect(canonicalCacheRoot(link)).toBe(realpathSync.native(real));
  });

  it('canonicalizes a MISSING root through its existing parent', () => {
    // The normal state of a tier nobody has opened yet: the parent exists,
    // the managed child does not.
    const real = join(base, 'real');
    mkdirSync(real, { recursive: true });
    const link = join(base, 'link');
    symlinkSync(real, link);
    expect(canonicalCacheRoot(join(link, 'mlx-paged-v1'))).toBe(join(realpathSync.native(real), 'mlx-paged-v1'));
  });

  it('canonicalizes when the root AND its parent are both missing', () => {
    // The two-level case. Stopping after one level returned the raw lexical
    // path, so the writer's `/private/var/...` key and the reader's
    // `/var/...` key described the same directory and never joined.
    const real = join(base, 'real');
    mkdirSync(real, { recursive: true });
    const link = join(base, 'link');
    symlinkSync(real, link);
    const deep = join(link, 'never-created', 'mlx-paged-v1');
    expect(canonicalCacheRoot(deep)).toBe(join(realpathSync.native(real), 'never-created', 'mlx-paged-v1'));
    // The point of the assertion above: the lexical answer is a DIFFERENT
    // string, so agreeing with it would be the bug.
    expect(canonicalCacheRoot(deep)).not.toBe(deep);
  });

  it('agrees with itself no matter how deep the missing tail is', () => {
    // The invariant that actually matters: writer and reader may spell the
    // same location with different amounts of symlink in it.
    const real = join(base, 'real');
    mkdirSync(real, { recursive: true });
    const link = join(base, 'link');
    symlinkSync(real, link);
    const viaLink = canonicalCacheRoot(join(link, 'a', 'b', 'c', 'mlx-paged-v1'));
    const viaReal = canonicalCacheRoot(join(real, 'a', 'b', 'c', 'mlx-paged-v1'));
    expect(viaLink).toBe(viaReal);
  });

  it('falls back to the lexical absolute path when no ancestor resolves', () => {
    // Fails OPEN to a stable string: this is an identity key, not a security
    // boundary, and `undefined` would be worse than a key that matches nothing.
    const nowhere = '/mlx-node-no-such-volume-9f3a/cache/mlx-paged-v1';
    expect(canonicalCacheRoot(nowhere)).toBe(nowhere);
  });
});

describe('canonicalCacheRoot — case-insensitive volumes', () => {
  it('folds an existing directory to one key regardless of the spelling used', () => {
    const mixed = join(base, 'CacheDir');
    mkdirSync(mixed, { recursive: true });
    if (!volumeIsCaseInsensitive(base)) {
      // On a case-sensitive volume these really are two directories and must
      // NOT be folded together. Assert that instead of skipping silently.
      expect(canonicalCacheRoot(mixed)).not.toBe(canonicalCacheRoot(join(base, 'cachedir')));
      return;
    }
    // `MLX_COLD_CACHE_DIR=…/CacheDir` on one run and `…/cachedir` on the next
    // name ONE directory on APFS. Two keys would show blocks on disk beside a
    // 0/0 trend.
    const lower = join(base, 'cachedir');
    expect(canonicalCacheRoot(lower)).toBe(canonicalCacheRoot(mixed));
    // And the same must hold for the managed child hanging off it, which is
    // the path the tier actually reports.
    expect(canonicalCacheRoot(join(lower, 'mlx-paged-v1'))).toBe(canonicalCacheRoot(join(mixed, 'mlx-paged-v1')));
  });

  it('folds the existing ancestor of a missing root too', () => {
    const mixed = join(base, 'CacheDir');
    mkdirSync(mixed, { recursive: true });
    if (!volumeIsCaseInsensitive(base)) return;
    const child = canonicalCacheRoot(join(base, 'CACHEDIR', 'mlx-paged-v1'));
    expect(dirname(child)).toBe(canonicalCacheRoot(mixed));
    expect(basename(child)).toBe('mlx-paged-v1');
  });
});
