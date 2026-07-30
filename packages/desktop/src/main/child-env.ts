/**
 * What MAIN forces into each child's environment, on top of the inherited one.
 *
 * These are the `overrides` layer of `buildChildEnv` — above the user's shell,
 * below the trace variables — because they are the app's own configuration
 * rather than a preference. A stale `NAPI_RS_NATIVE_LIBRARY_PATH` exported in a
 * developer's shell, pointing at another checkout's addon, must not beat the one
 * that ships inside this bundle.
 *
 * Electron-free so both tables are testable; MAIN reads them and passes them on.
 */

/**
 * `packages/core/index.cjs`'s `requireNative()` checks this FIRST and passes it
 * straight to `require()`.
 *
 * It must name the `.node` FILE. Measured on this tree, with the addon built:
 *
 *   .../mlx-core.darwin-arm64.node  -> loads, 68 exports
 *   .../packages/core               -> LOADS, 67 exports, `loadModel` undefined
 *   any other directory             -> throws "Cannot find native binding"
 *
 * The middle row is the one to be afraid of. `require(<dir>)` resolves back to
 * `index.cjs`, which is the module already being evaluated, so Node hands back
 * its half-built exports — no error, no warning that matters, and a sidecar that
 * binds a port, answers `/health` with `ok`, and has no `loadModel`.
 */
const NATIVE_LIBRARY_PATH = 'NAPI_RS_NATIVE_LIBRARY_PATH';

/** Read by `resolveModelsDir` (INFERENCE) and `defaultModelsDir` (CONTROL PANEL) alike. */
const MODELS_DIR = 'MLX_MODELS_DIR';

export interface ChildEnvInput {
  /** `AppPaths.nativeAddon` — the `.node` file, never its directory. */
  nativeAddon: string;
  /** The models root chosen in the UI, or `null` to leave the default resolution alone. */
  modelsDir: string | null;
}

/**
 * INFERENCE's overrides.
 *
 * The addon path is set unconditionally, in dev as well as in the bundle, so
 * both take the same branch of `requireNative()`. The alternative — letting dev
 * fall through to napi's own platform lookup — means the packaged path is only
 * ever exercised by a packaged build, which is this project's slowest feedback
 * loop.
 */
export function sidecarEnvOverrides(input: ChildEnvInput): Record<string, string> {
  return {
    [NATIVE_LIBRARY_PATH]: input.nativeAddon,
    ...modelsDirEnv(input.modelsDir),
  };
}

/**
 * CONTROL PANEL's overrides: the models directory and nothing else.
 *
 * Deliberately without {@link NATIVE_LIBRARY_PATH}. CONTROL PANEL does not link the
 * addon — that is the point of the three-process split — and handing it a
 * working path would make an accidental `@mlx-node/lm` import load silently
 * instead of failing loudly.
 *
 * Both children get the SAME models dir. They disagree otherwise: INFERENCE
 * serves what `resolveModelsDir` finds and the dashboard lists what
 * `defaultModelsDir` finds, and a user who moved their models in the UI would
 * see a catalog that does not match what is servable.
 */
export function controlPanelEnvOverrides(input: Pick<ChildEnvInput, 'modelsDir'>): Record<string, string> {
  return modelsDirEnv(input.modelsDir);
}

function modelsDirEnv(modelsDir: string | null): Record<string, string> {
  // Absent rather than empty: `resolveModelsDir` and `defaultModelsDir` both
  // treat a zero-length value as unset, but an empty string in a child env is
  // indistinguishable from a bug at every point in between.
  return modelsDir === null ? {} : { [MODELS_DIR]: modelsDir };
}
