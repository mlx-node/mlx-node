import { execFileSync } from 'node:child_process';
import { existsSync, readdirSync } from 'node:fs';
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
