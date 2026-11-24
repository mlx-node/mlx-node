import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import {
  buildPackagedRuntimeProbeScript,
  hasAgentVersionBanner,
  packagedRuntimeProbeEnv,
  verifyPackagedNativeRuntime,
} from '../scripts/verify-cli.js';

describe('hasAgentVersionBanner', () => {
  it.each(['1.0.45', '0.0.17', '1.2.3-rc.1', '1.2.3.4', '  1.0.45  ', '1.0.45 (d49757ec37ce)'])(
    'accepts the legacy bare version in %j',
    (banner) => {
      expect(hasAgentVersionBanner(banner)).toBe(true);
    },
  );

  it.each(['grok 1.0.45 (d49757ec37ce)', 'grok 1.0.45', 'grok 0.0.17 (hash)'])(
    'accepts the grok-branded banner in %j',
    (banner) => {
      expect(hasAgentVersionBanner(banner)).toBe(true);
    },
  );

  it.each([
    '',
    '   ',
    'grok',
    'grok ',
    'grok\n1.0.45',
    'grok 1.0',
    'grok 1.0.x',
    'mlx-agent 1.0.45',
    'v1.0.45',
    '1.0',
    '1.0.x',
    'x1.0.45',
  ])('rejects malformed output in %j', (banner) => {
    expect(hasAgentVersionBanner(banner)).toBe(false);
  });
});

describe('packagedRuntimeProbeEnv', () => {
  it('builds an isolated POSIX environment without native library overrides', () => {
    const env = packagedRuntimeProbeEnv('/tmp/probe-home', 'darwin', {
      PATH: '/opt/homebrew/bin:/usr/bin',
      NAPI_RS_NATIVE_LIBRARY_PATH: '/stale/mlx-core.darwin-arm64.node',
      MLX_CORE_NATIVE_LIBRARY_PATH: '/stale/core.node',
      UNRELATED_SECRET: 'should-not-leak',
    });

    expect(env).toEqual({
      ELECTRON_RUN_AS_NODE: '1',
      HOME: '/tmp/probe-home',
      PATH: '/usr/bin:/bin:/usr/sbin:/sbin',
    });
    expect(env).not.toHaveProperty('NAPI_RS_NATIVE_LIBRARY_PATH');
    expect(env).not.toHaveProperty('MLX_CORE_NATIVE_LIBRARY_PATH');
  });

  it('builds an isolated Windows environment preserving SystemRoot/WINDIR and omitting native overrides', () => {
    const env = packagedRuntimeProbeEnv('C:\\Users\\ci\\probe-home', 'win32', {
      systemroot: 'D:\\Windows',
      windir: 'D:\\Windows',
      NAPI_RS_NATIVE_LIBRARY_PATH: 'C:\\stale\\mlx-core.win32-arm64-msvc.node',
      MLX_CORE_NATIVE_LIBRARY_PATH: 'C:\\stale\\core.node',
      Path: 'C:\\Program Files\\nodejs',
    });

    expect(env.ELECTRON_RUN_AS_NODE).toBe('1');
    expect(env.HOME).toBe('C:\\Users\\ci\\probe-home');
    expect(env.USERPROFILE).toBe('C:\\Users\\ci\\probe-home');
    expect(env.TEMP).toBe('C:\\Users\\ci\\probe-home');
    expect(env.TMP).toBe('C:\\Users\\ci\\probe-home');
    expect(env.SystemRoot).toBe('D:\\Windows');
    expect(env.WINDIR).toBe('D:\\Windows');
    expect(env.PATH).toBe('D:\\Windows\\System32;D:\\Windows');
    expect(env).not.toHaveProperty('NAPI_RS_NATIVE_LIBRARY_PATH');
    expect(env).not.toHaveProperty('MLX_CORE_NATIVE_LIBRARY_PATH');
  });

  it('falls back to C:\\Windows on win32 when SystemRoot and WINDIR are unset', () => {
    const env = packagedRuntimeProbeEnv('C:\\Temp\\home', 'win32', {});
    expect(env.SystemRoot).toBe('C:\\Windows');
    expect(env.WINDIR).toBe('C:\\Windows');
    expect(env.PATH).toBe('C:\\Windows\\System32;C:\\Windows');
  });
});

describe('buildPackagedRuntimeProbeScript', () => {
  it('safely escapes Windows backslash paths via JSON.stringify and enforces appDir confinement', () => {
    const winAppDir = 'C:\\Users\\runner\\mlx-node\\resources\\app';
    const script = buildPackagedRuntimeProbeScript(winAppDir);
    expect(script).toContain(`const appDir = realpathSync(resolve(${JSON.stringify(winAppDir)}));`);
    expect(script).toContain('resolved outside packaged appDir');
  });
});

describe('verifyPackagedNativeRuntime', () => {
  const repoRoot = resolve(import.meta.dirname, '..', '..', '..');
  const clairicPdfDist = join(repoRoot, 'node_modules', '@mlx-node', 'lm', 'dist', 'clairic-pdf.js');
  const hasRuntimeDeps =
    existsSync(clairicPdfDist) &&
    existsSync(join(repoRoot, 'node_modules', '@napi-rs', 'keyring', 'package.json')) &&
    existsSync(join(repoRoot, 'node_modules', '@napi-rs', 'canvas', 'package.json')) &&
    existsSync(join(repoRoot, 'node_modules', 'pdfjs-dist', 'package.json'));

  it.skipIf(!hasRuntimeDeps)(
    'exercises Keyring, Canvas, pdfjs-dist assets, and Clairic PDF rasterization in a live subprocess',
    () => {
      const home = mkdtempSync(join(tmpdir(), 'mlx-runtime-probe-test-'));
      try {
        expect(() =>
          verifyPackagedNativeRuntime({
            executable: process.execPath,
            appDir: repoRoot,
            home,
          }),
        ).not.toThrow();

        // A subdirectory inside the checkout that has no packaged node_modules
        // must fail instead of silently resolving deps from the ancestor checkout.
        expect(() =>
          verifyPackagedNativeRuntime({
            executable: process.execPath,
            appDir: join(repoRoot, 'packages', 'desktop'),
            home,
          }),
        ).toThrow(/resolved outside packaged appDir/);
      } finally {
        rmSync(home, { recursive: true, force: true });
      }
    },
  );
});
