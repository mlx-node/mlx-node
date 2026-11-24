/**
 * What MAIN forces into each child's environment.
 *
 * Every one of these latches: `NAPI_RS_NATIVE_LIBRARY_PATH` is read inside
 * `requireNative()` while the sidecar's entry is still evaluating its imports,
 * and the engine's own knobs latch through a Rust `OnceLock` on first read. A
 * value that is not present at fork time can never be supplied afterwards, and
 * writing it later *succeeds silently* and changes nothing — which is why these
 * are tested as data rather than observed as behaviour.
 */

import { extname } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { controlPanelEnvOverrides, sidecarEnvOverrides, WINDOWS_STORE_ENV } from '../src/main/child-env.js';
import { resolveAppPaths } from '../src/main/paths.js';

const ADDON = '/Applications/mlx-node.app/Contents/Resources/native/mlx-core.darwin-arm64.node';

describe('sidecarEnvOverrides', () => {
  it('points NAPI_RS_NATIVE_LIBRARY_PATH at the .node FILE', () => {
    const env = sidecarEnvOverrides({ nativeAddon: ADDON, modelsDir: null });
    expect(env.NAPI_RS_NATIVE_LIBRARY_PATH).toBe(ADDON);
    // A directory here does not fail cleanly. `require(<dir with index.cjs>)`
    // resolves back to the module already being evaluated and Node hands back its
    // half-built exports: 67 keys, zero functions, no throw. See
    // `child-entries.test.ts`, which drives all three cases in a real process.
    expect(extname(env.NAPI_RS_NATIVE_LIBRARY_PATH)).toBe('.node');
  });

  it('carries the models dir only when one was chosen', () => {
    expect(sidecarEnvOverrides({ nativeAddon: ADDON, modelsDir: null })).not.toHaveProperty('MLX_MODELS_DIR');
    // Absent rather than empty: `resolveModelsDir` treats a zero-length value as
    // unset, but an empty string in a child env is indistinguishable from a bug
    // at every point in between.
    expect(sidecarEnvOverrides({ nativeAddon: ADDON, modelsDir: '/models' }).MLX_MODELS_DIR).toBe('/models');
  });

  it('points packaged Windows children at the bundled DLLs and NVRTC headers', () => {
    const env = sidecarEnvOverrides({
      nativeAddon: 'C:\\app\\resources\\native\\mlx-core.win32-arm64-msvc.node',
      nativeRuntimeDir: 'C:\\app\\resources\\native',
      modelsDir: null,
      baseEnv: { Path: 'C:\\Windows\\system32' },
    });
    expect(env.CUDA_HOME).toBe('C:\\app\\resources\\native');
    expect(env.CUDA_PATH).toBe('C:\\app\\resources\\native');
    expect(env.Path).toBe('C:\\app\\resources\\native;C:\\Windows\\system32');
    expect(env).not.toHaveProperty('PATH');
    expect(env.MLX_CUDA_USE_CUDNN_SDPA).toBe('0');
  });

  it('is exactly the set that has to beat the user’s shell', () => {
    // This is the `overrides` layer of `buildChildEnv` — above the inherited
    // environment. Anything added here silently wins over a value the developer
    // exported, so the set stays small and deliberate.
    expect(Object.keys(sidecarEnvOverrides({ nativeAddon: ADDON, modelsDir: '/m' })).sort()).toEqual([
      'MLX_MODELS_DIR',
      'NAPI_RS_NATIVE_LIBRARY_PATH',
    ]);
  });
});

describe('controlPanelEnvOverrides', () => {
  it('never hands CONTROL PANEL a working addon path', () => {
    const env = controlPanelEnvOverrides({ modelsDir: '/models' });
    // CONTROL PANEL does not link the addon — that is the point of the split. Handing it
    // a working path would make an accidental `@mlx-node/lm` import load
    // silently instead of failing loudly.
    expect(env).not.toHaveProperty('NAPI_RS_NATIVE_LIBRARY_PATH');
    expect(Object.keys(env)).toEqual(['MLX_MODELS_DIR']);
  });

  it('agrees with the sidecar about where models live', () => {
    // INFERENCE serves what `resolveModelsDir` finds and the dashboard lists what
    // `defaultModelsDir` finds. Both read `MLX_MODELS_DIR`; a user who moved
    // their models in the UI would otherwise see a catalog that does not match
    // what is servable.
    const modelsDir = '/Volumes/ssd/models';
    expect(controlPanelEnvOverrides({ modelsDir }).MLX_MODELS_DIR).toBe(
      sidecarEnvOverrides({ nativeAddon: ADDON, modelsDir }).MLX_MODELS_DIR,
    );
    expect(controlPanelEnvOverrides({ modelsDir: null })).toEqual({});
  });

  it('propagates the Microsoft Store flag only when enabled', () => {
    expect(controlPanelEnvOverrides({ modelsDir: null, windowsStore: true })).toEqual({
      [WINDOWS_STORE_ENV]: '1',
    });
    expect(controlPanelEnvOverrides({ modelsDir: null, windowsStore: false })).toEqual({});
    expect(controlPanelEnvOverrides({ modelsDir: '/models' })).not.toHaveProperty(WINDOWS_STORE_ENV);
  });
});

describe('the addon path the app will actually use', () => {
  it('is a file in both layouts', () => {
    for (const packaged of [true, false]) {
      const paths = resolveAppPaths({
        appPath: packaged ? '/Applications/mlx-node.app/Contents/Resources/app' : '/repo/packages/desktop',
        resourcesPath: '/Applications/mlx-node.app/Contents/Resources',
        packaged,
        userData: '/Users/x/Library/Application Support/mlx-node',
      });
      const env = sidecarEnvOverrides({ nativeAddon: paths.nativeAddon, modelsDir: null });
      expect(extname(env.NAPI_RS_NATIVE_LIBRARY_PATH), `packaged=${packaged}`).toBe('.node');
    }
  });
});
