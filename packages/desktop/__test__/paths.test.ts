/**
 * A path that is right in a dev checkout and wrong inside the .app produces a
 * blank window or a sidecar that will not fork, with nothing in any log that
 * names the path — and it is only observable after a packaging run, which is the
 * slowest feedback loop this project has.
 */

import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { bundledAddonPath, NATIVE_FILES } from '../scripts/payload.js';
import { resolveAppPaths, type AppLayout } from '../src/main/paths.js';

const DEV: AppLayout = {
  appPath: '/repo/packages/desktop',
  resourcesPath: '/repo/packages/desktop/node_modules/electron/dist/Electron.app/Contents/Resources',
  packaged: false,
  userData: '/Users/x/Library/Application Support/Electron',
};

const PACKAGED: AppLayout = {
  appPath: '/Applications/mlx-node.app/Contents/Resources/app',
  resourcesPath: '/Applications/mlx-node.app/Contents/Resources',
  packaged: true,
  userData: '/Users/x/Library/Application Support/mlx-node',
};

describe('resolveAppPaths', () => {
  it('serves the dashboard workspace build in dev', () => {
    // `vp build` in packages/dashboard/ui writes here, so a UI rebuild is live on
    // the next window load with no copy step.
    expect(resolveAppPaths(DEV).wwwRoot).toBe(join('/repo/packages/dashboard/web'));
  });

  it('serves a copied www root when packaged', () => {
    const { wwwRoot } = resolveAppPaths(PACKAGED);
    expect(wwwRoot).toBe('/Applications/mlx-node.app/Contents/Resources/www');
    // The dev layout climbs out of the desktop package; inside a bundle there is
    // no sibling workspace to climb to.
    expect(wwwRoot).not.toContain('dashboard');
  });

  it('keeps every user-writable path under userData', () => {
    for (const layout of [DEV, PACKAGED]) {
      const paths = resolveAppPaths(layout);
      expect(paths.settingsFile.startsWith(layout.userData)).toBe(true);
      expect(paths.traceDir.startsWith(layout.userData)).toBe(true);
    }
    // Never inside the bundle: a signed .app is read-only, and a write there
    // would break the code signature even if it succeeded.
    expect(resolveAppPaths(PACKAGED).settingsFile).not.toContain('.app/');
  });

  it('finds the sidecar entry beside the compiled main process', () => {
    for (const layout of [DEV, PACKAGED]) {
      expect(resolveAppPaths(layout).sidecarEntry).toBe(join(layout.appPath, 'dist/inference/index.js'));
    }
  });

  it('finds the control panel entry beside it', () => {
    for (const layout of [DEV, PACKAGED]) {
      expect(resolveAppPaths(layout).controlPanelEntry).toBe(join(layout.appPath, 'dist/control-panel/index.js'));
    }
    // Two entries, never one file doing both jobs: the whole point of CONTROL PANEL is
    // that it does not load what INFERENCE loads.
    expect(resolveAppPaths(DEV).controlPanelEntry).not.toBe(resolveAppPaths(DEV).sidecarEntry);
  });

  // `packages/core/index.cjs` checks `NAPI_RS_NATIVE_LIBRARY_PATH` first in
  // `requireNative()` and passes it straight to `require()`. Naming the
  // containing directory instead resolves back to `index.cjs` itself — a
  // circular self-require that returns half-built exports with no error at all.
  it('names the addon FILE, not the directory holding it', () => {
    for (const layout of [DEV, PACKAGED]) {
      const { nativeAddon } = resolveAppPaths(layout);
      expect(nativeAddon.endsWith(`/${NATIVE_FILES[0]}`), nativeAddon).toBe(true);
    }
  });

  // The runtime path and the packaging path are decided in two files that cannot
  // import each other — `src` must not depend on `scripts`. This is the only
  // thing keeping them together, and a mismatch is an app that packages cleanly
  // and cannot find its addon on first launch.
  it('agrees with where packaging puts the addon', () => {
    expect(resolveAppPaths(PACKAGED).nativeAddon).toBe(bundledAddonPath(PACKAGED.resourcesPath));
  });

  it('reads the addon out of the workspace in dev', () => {
    // The same file napi's own platform lookup would have found, so dev and the
    // bundle take the same branch of `requireNative()` rather than the packaged
    // path being exercised only by a packaged build.
    expect(resolveAppPaths(DEV).nativeAddon).toBe(join('/repo/packages/core', NATIVE_FILES[0]));
  });

  // A sandboxed preload must be CommonJS, which is why the source is `.cts` and
  // the build output is `.cjs`. Pointing `webPreferences.preload` at a `.js`
  // that does not exist fails silently: the window renders and never receives
  // its MessagePort.
  it('points at the CommonJS preload', () => {
    const { controlPanelPreload } = resolveAppPaths(DEV);
    expect(controlPanelPreload.endsWith('.cjs')).toBe(true);
    expect(controlPanelPreload).toBe(join(DEV.appPath, 'dist/preload/index.cjs'));
  });

  // `nativeImage` finds `iconTemplate@2x.png` by name, and the `Template` suffix
  // is what makes macOS tint it for light/dark. Both live in the filename.
  it('points at the template icon by its template name', () => {
    expect(resolveAppPaths(DEV).trayIcon).toBe(join(DEV.appPath, 'build/iconTemplate.png'));
  });
});
