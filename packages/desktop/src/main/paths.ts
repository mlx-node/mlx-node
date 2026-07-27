/**
 * Where the app's four moving parts live, in a dev checkout and inside a signed
 * `.app`.
 *
 * A pure function of the four Electron values that differ between the two, so
 * the layout is one table instead of a `app.isPackaged` ternary at each use
 * site. The failure this prevents is specific: a path that is right in dev and
 * wrong in the bundle produces a blank window or a sidecar that will not fork,
 * with no error that names the path — and it is only observable after a
 * packaging run.
 */

import { join } from 'node:path';

/** The Electron-supplied facts this depends on. See `app.getPath` / `app.getAppPath`. */
export interface AppLayout {
  /**
   * `app.getAppPath()`. In dev, the directory holding the package.json Electron
   * was launched with (`packages/desktop`). Packaged, the app root inside
   * `Contents/Resources`.
   */
  appPath: string;
  /** `process.resourcesPath`. Meaningless in dev; only read on the packaged branch. */
  resourcesPath: string;
  /** `app.isPackaged`. */
  packaged: boolean;
  /** `app.getPath('userData')` — `~/Library/Application Support/<productName>`. */
  userData: string;
}

export interface AppPaths {
  /** Root the `app://` scheme serves. The dashboard SPA build output. */
  wwwRoot: string;
  /**
   * The INFERENCE sidecar's built JS entry.
   *
   * **Must not be inside an asar archive.** It loads `@mlx-node/lm`, which
   * `dlopen`s `mlx-core.darwin-arm64.node`, and `dlopen` takes a real path on a
   * real filesystem — an asar-resident addon fails at load with an error that
   * names the archive, not the addon. Packaging must therefore either build with
   * `asar: false` or list this subtree in `asarUnpack`.
   */
  sidecarEntry: string;
  /** Per-generation `MLX_INFERENCE_TRACE_FILE` directory. Created by the supervisor. */
  traceDir: string;
  settingsFile: string;
  /** The menubar icon. `iconTemplate@2x.png` sits beside it and `nativeImage` finds it by name. */
  trayIcon: string;
  /** Preload for the Admin window. `.cjs`: a sandboxed preload cannot be ESM. */
  adminPreload: string;
}

export function resolveAppPaths(layout: AppLayout): AppPaths {
  const { appPath, resourcesPath, packaged, userData } = layout;
  return {
    // Packaged, the SPA is copied next to the app as an extra resource so it can
    // be replaced without rebuilding the archive. In dev it is read straight out
    // of the dashboard workspace, so `vp build` in `packages/dashboard/ui` is
    // visible on the next window load with no copy step.
    wwwRoot: packaged ? join(resourcesPath, 'www') : join(appPath, '..', 'dashboard', 'web'),
    sidecarEntry: join(appPath, 'dist', 'inference', 'index.js'),
    traceDir: join(userData, 'traces'),
    settingsFile: join(userData, 'settings.json'),
    trayIcon: join(appPath, 'build', 'iconTemplate.png'),
    adminPreload: join(appPath, 'dist', 'preload', 'index.cjs'),
  };
}
