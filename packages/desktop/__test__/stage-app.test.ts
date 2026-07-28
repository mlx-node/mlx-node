/**
 * What the bundle is allowed to contain.
 *
 * `runtimeClosure` decides, package by package, what gets copied into the
 * shipped app. Two of its rules are not size tuning — they are the difference
 * between a bundle that notarizes and one that does not, and both were found by
 * running the release gate rather than by reading code:
 *
 *  - `@mlx-node/core-*` is napi's published prebuilt. Staging it shipped the
 *    239 MB native payload a second time (993 MB total) even though nothing
 *    loads it.
 *  - `@mariozechner/clipboard*` bakes upstream's CI home into its load commands,
 *    which `verify-bundle.sh` step [3/5] refuses outright.
 *
 * Both exclusions are silent by nature: the app still builds, still launches,
 * and still passes its own tests with either one wrong. Only a gate or a notary
 * says otherwise, and both of those are minutes-to-hours away from the edit. So
 * the rules are pinned here, at the point where they are cheap to check.
 *
 * These run against the REAL repo tree rather than a fixture. A fixture would
 * pin the shape of the walk while saying nothing about the dependency graph we
 * actually ship, and the graph is the part that moves under us — clipboard
 * arrived transitively, through a dependency nobody added for it.
 */

import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

import { runtimeClosure } from '../scripts/stage-app.js';

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');

// The same roots packaging uses: what the three entries import, not what
// packages/desktop/package.json happens to declare.
const ROOTS = ['@mlx-node/dashboard', '@mlx-node/server', '@mlx-node/lm'];

describe('runtimeClosure', () => {
  const closure = runtimeClosure(repoRoot, ROOTS);

  it('reaches the workspace packages the app actually runs', () => {
    expect(closure.workspace).toContain('@mlx-node/dashboard');
    expect(closure.workspace).toContain('@mlx-node/server');
    expect(closure.external.length).toBeGreaterThan(0);
  });

  it('excludes the clipboard prebuilts that leak a build path', () => {
    // The assertion is on `external` rather than on the excluded list, because
    // `external` is what gets COPIED. Reporting a package as excluded while
    // still staging it would satisfy a check on the excluded list alone.
    const staged = closure.external.filter((name) => name.startsWith('@mariozechner/clipboard'));
    expect(staged).toEqual([]);
  });

  it('reports the exclusion instead of dropping it silently', () => {
    // A silent exclusion reads as "this dependency was never here" to the next
    // person wondering why clipboard support does nothing.
    expect(closure.excludedThirdParty).toContain('@mariozechner/clipboard');
  });

  it('still reaches clipboard in the graph — the exclusion is a decision, not an accident', () => {
    // Guard-the-guard. If pi-coding-agent ever drops the dependency, the two
    // tests above start passing for a reason that has nothing to do with the
    // exclusion rule, and the rule could then be deleted without any test
    // noticing. This fails loudly when that day comes.
    expect(existsSync(join(repoRoot, 'node_modules', '@mariozechner', 'clipboard'))).toBe(true);
  });

  it('never stages the napi prebuilt that would duplicate the native payload', () => {
    expect(closure.external.filter((name) => name.startsWith('@mlx-node/core-'))).toEqual([]);
  });
});
