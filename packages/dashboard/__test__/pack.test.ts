import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { beforeAll, describe, expect, it } from 'vite-plus/test';

// Regression guard for whole-branch finding #5: the published tarball must ship
// the built Vite SPA under `web/`, not just the server `dist/`. `tsc -b` (what CI
// runs before publish) never builds `web/`, and `web/` is gitignored — so this
// asserts both that `build:ui` produces the SPA and that `npm publish`'s file
// list would include it. Fails loudly if `web/` is absent from the pack set.

const here = dirname(fileURLToPath(import.meta.url));
const dashboardDir = join(here, '..');
const webDir = join(dashboardDir, 'web');

describe('dashboard npm pack ships the SPA', () => {
  beforeAll(() => {
    // Build the SPA exactly as `prepack` / the CI publish step does.
    execFileSync('yarn', ['build:ui'], { cwd: dashboardDir, stdio: 'pipe' });
  }, 180_000);

  it('build:ui emits web/index.html and a hashed asset', () => {
    expect(existsSync(join(webDir, 'index.html'))).toBe(true);
    const assets = existsSync(join(webDir, 'assets')) ? readdirSync(join(webDir, 'assets')) : [];
    expect(assets.some((f) => /\.(js|css)$/.test(f))).toBe(true);
  });

  it('npm pack file list includes the SPA', () => {
    const stdout = execFileSync('npm', ['pack', '--dry-run', '--json', '--ignore-scripts'], {
      cwd: dashboardDir,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    const parsed = JSON.parse(stdout) as Array<{ files: Array<{ path: string }> }>;
    const packed = parsed[0].files.map((f) => f.path);

    expect(packed).toContain('web/index.html');
    expect(packed.some((p) => /^web\/assets\/.+/.test(p))).toBe(true);
  });
});

/**
 * Read the ONE script bundle `web/index.html` actually references.
 *
 * Resolving through the HTML rather than globbing `web/assets/*.js` is the
 * whole point: a stale sibling asset left behind by an earlier build would
 * satisfy a glob while the browser downloads something else entirely. The
 * browser follows this `src` and nothing else, so the assertions must too.
 */
function servedBundle(): string {
  const html = readFileSync(join(webDir, 'index.html'), 'utf8');
  const src = /<script[^>]+src="\/?(assets\/index-[^"]+\.js)"/.exec(html);
  if (src === null) throw new Error(`web/index.html references no hashed script bundle:\n${html}`);
  return readFileSync(join(webDir, src[1]), 'utf8');
}

/**
 * The Cache-page fixes must survive the BUILD, not merely exist in `ui/src`.
 *
 * This suite exists because of how the defects were found: the served bundle
 * was searched for `sidecarCount`, `sidecarBytes`, `coldHits`, `coldMisses` and
 * `coldBytesRestored` and contained ZERO occurrences of any of them, while the
 * API had been serving all five for months. Over half the bytes on the Usage
 * tile were unlabelled because the SPA never asked. `pages.test.ts` mounts the
 * components straight from `ui/src`, so it passes on source alone and cannot
 * notice a `web/` that was never rebuilt — which is precisely the state that
 * shipped.
 *
 * Assertions are POSITIVE first. An absence-only suite ("no longer says
 * qwen3 dense") passes against an empty file, a 404 page, or a build that
 * silently produced nothing.
 *
 * Every string below is a plain literal in `cache.tsx`, chosen so minification
 * cannot dissolve it: mangling renames bindings, never string contents, and
 * none of these sit inside a template hole. Model family names are deliberately
 * NOT asserted — the empty-state hint builds them from the API's
 * `restoreFamilies`, so they legitimately do not appear in the bundle at all.
 */
describe('the served bundle carries the Cache-page fixes', () => {
  it('names both cold-tier object kinds in the shipped JS', () => {
    const bundle = servedBundle();
    // F2 — the tile and the chart must count the same thing, and the chart says
    // so in its title. "Blocks by age" over a blocks+sidecars histogram is the
    // miscount.
    expect(bundle).toContain('Objects by age');
    // F3 — `sidecarCount` and `sidecarBytes` reaching the browser, as rendered
    // text rather than as an erased TypeScript type.
    expect(bundle).toContain('prefix block');
    expect(bundle).toContain('state sidecar');
    expect(bundle).toContain(' sidecars');
    // F3 — the cold counters the API serves and nothing shipped ever read.
    expect(bundle).toContain('Cold-tier health');
    expect(bundle).toContain('acceptance bar: 0');
  });

  it('ships the scoped hit rate and the saturating percent rules', () => {
    const bundle = servedBundle();
    // F1 — the hit rate is labelled with the scope it was computed over.
    expect(bundle).toContain('this cache only');
    expect(bundle).toContain('no lookups recorded for this cache');
    // F6 — the saturation literals only exist if the rounding fix is compiled
    // in. A reverted `Math.round(fraction * 100)` has no such string.
    expect(bundle).toContain('>99%');
    expect(bundle).toContain('<1%');
  });

  it('no longer hardcodes a single restore family in the empty state', () => {
    const bundle = servedBundle();
    // F5 — four families are allowlisted; the hint named one, and the list is
    // now served over the wire so it cannot drift from the native gate.
    expect(bundle).not.toContain('qwen3 dense');
    expect(bundle).toContain('restore-eligible family');
  });
});
