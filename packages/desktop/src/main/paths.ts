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

/**
 * The addon's filename, duplicated from `scripts/payload.ts`'s `NATIVE_FILES[0]`.
 *
 * `src` may not import from `scripts` — the app would then carry its packaging
 * tool around at runtime — so the two copies are kept honest by a test that reads
 * both. A rename that only lands in one of them produces an app that packages
 * fine and cannot find its addon.
 */
const ADDON_FILE = 'mlx-core.darwin-arm64.node';

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
  /**
   * The CONTROL PANEL utilityProcess's built JS entry — the dashboard runtime and its
   * SQLite worker thread.
   *
   * Beside the sidecar entry and under the same rule about asar, for a different
   * reason: this one spawns a `node:worker_threads` worker, and `new Worker()`
   * takes a real path too.
   */
  controlPanelEntry: string;
  /**
   * The native addon, as a path to the **`.node` file itself**.
   *
   * Handed to the INFERENCE child as `NAPI_RS_NATIVE_LIBRARY_PATH`, which
   * `packages/core/index.cjs` checks FIRST in `requireNative()` and passes
   * straight to `require()`. Naming the containing directory instead does not
   * fail cleanly — measured on this tree:
   *
   *   the .node file            -> loads, 68 exports
   *   `packages/core/`          -> LOADS, 67 exports, `loadModel` undefined
   *                                (`require(dir)` resolves back to index.cjs and
   *                                 the module circularly requires itself)
   *   any other directory       -> throws "Cannot find native binding"
   *
   * The middle row is why this is a file path and why a test pins it: a sidecar
   * that starts, serves `/health`, and has no `loadModel` is the silent failure
   * this whole app is built to avoid.
   */
  nativeAddon: string;
  /** Per-generation `MLX_INFERENCE_TRACE_FILE` directory. Created by the supervisor. */
  traceDir: string;
  settingsFile: string;
  /** The menubar icon. `iconTemplate@2x.png` sits beside it and `nativeImage` finds it by name. */
  trayIcon: string;
  /** Preload for the Control Panel window. `.cjs`: a sandboxed preload cannot be ESM. */
  controlPanelPreload: string;
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
    controlPanelEntry: join(appPath, 'dist', 'control-panel', 'index.js'),
    // Packaged, `scripts/package.ts` stages the addon and both metallibs into
    // `Contents/Resources/native` — they must stay in ONE directory, since MLX
    // loads mlx.metallib from beside the binary and paged_attn.metallib
    // dladdr-relative from the .node. In dev it is read out of the workspace,
    // which is also where napi's own fallback would have found it: the variable
    // is set in both modes anyway so that dev and the bundle take the same code
    // path through `requireNative()`.
    nativeAddon: packaged ? join(resourcesPath, 'native', ADDON_FILE) : join(appPath, '..', 'core', ADDON_FILE),
    traceDir: join(userData, 'traces'),
    settingsFile: join(userData, 'settings.json'),
    trayIcon: join(appPath, 'build', 'iconTemplate.png'),
    controlPanelPreload: join(appPath, 'dist', 'preload', 'index.cjs'),
  };
}
