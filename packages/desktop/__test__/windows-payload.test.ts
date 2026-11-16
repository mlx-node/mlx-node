import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  missingWindowsZipEntries,
  readPackagedWindowsApp,
  selectWindowsRuntimeFiles,
  WINDOWS_ADDON,
  WINDOWS_RUNTIME_DLLS,
} from '../scripts/windows-payload.js';

const dynamicDlls = ['cudnn_graph64_9.dll', 'cudnn_ops64_9.dll', 'nvrtc-builtins64_134.dll'];
const runtimeNames = [WINDOWS_ADDON, ...WINDOWS_RUNTIME_DLLS, ...dynamicDlls];

describe('Windows runtime selection', () => {
  it('ships the addon, fixed CUDA/OpenBLAS libraries, split cuDNN, and NVRTC builtins once', () => {
    const selected = selectWindowsRuntimeFiles([...runtimeNames, 'unrelated.dll']);
    expect(selected[0]).toBe(WINDOWS_ADDON);
    expect(selected).toEqual(expect.arrayContaining(dynamicDlls));
    expect(selected).not.toContain('unrelated.dll');
    expect(selected.filter((name) => name === 'cudnn64_9.dll')).toHaveLength(1);
    expect(new Set(selected).size).toBe(selected.length);
  });

  it('fails closed on a missing fixed library or NVRTC builtins', () => {
    expect(() => selectWindowsRuntimeFiles(runtimeNames.filter((name) => name !== 'openblas.dll'))).toThrow(
      /openblas\.dll/,
    );
    expect(() =>
      selectWindowsRuntimeFiles(runtimeNames.filter((name) => !name.startsWith('nvrtc-builtins'))),
    ).toThrow(/NVRTC builtins/);
  });
});

describe('packaged Windows app', () => {
  let root: string;
  let app: string;
  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'mlx-windows-payload-'));
    app = join(root, 'mlx-node-win32-arm64');
    write('mlx-node.exe');
    write('resources/app/package.json', JSON.stringify({ version: '0.0.21' }));
    for (const name of runtimeNames) write(`resources/native/${name}`, name);
    write('resources/native/include/cuda_runtime.h');
    write('resources/native/include/cccl/cuda/__cccl_config');
    write('resources/native/include/cute/config.hpp');
    write('resources/native/include/cutlass/version.h');
  });
  afterEach(() => rmSync(root, { recursive: true, force: true }));

  function write(relative: string, contents = 'x'): void {
    const path = join(app, relative);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, contents);
  }

  it('reads the version back and verifies every native runtime file', () => {
    const packaged = readPackagedWindowsApp(app);
    expect(packaged.version).toBe('0.0.21');
    expect(packaged.runtimeFiles).toEqual(selectWindowsRuntimeFiles(runtimeNames));
  });

  it('rejects a package with a missing native DLL or header', () => {
    rmSync(join(app, 'resources/native/cudnn_ops64_9.dll'));
    expect(() => readPackagedWindowsApp(app)).toThrow(/cudnn_ops64_9\.dll/);
  });

  it('checks the final ZIP members, including the dashboard and NVRTC headers', () => {
    const required = [
      'mlx-node-win32-arm64/mlx-node.exe',
      'mlx-node-win32-arm64/resources/app/package.json',
      'mlx-node-win32-arm64/resources/www/index.html',
      'mlx-node-win32-arm64/resources/native/include/cuda_runtime.h',
      ...selectWindowsRuntimeFiles(runtimeNames).map(
        (name) => `mlx-node-win32-arm64/resources/native/${name}`,
      ),
    ];
    expect(missingWindowsZipEntries(required, 'mlx-node-win32-arm64', runtimeNames)).toEqual([]);
    expect(missingWindowsZipEntries(required.slice(1), 'mlx-node-win32-arm64', runtimeNames)).toEqual([
      'mlx-node.exe',
    ]);
  });
});
