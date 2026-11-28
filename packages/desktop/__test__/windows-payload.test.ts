import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { PayloadError } from '../scripts/payload.js';
import {
  assertWindowsAppRuntimeAssets,
  readPackagedWindowsApp,
  selectWindowsRuntimeFiles,
  WINDOWS_ADDON,
  WINDOWS_APP_RUNTIME_DIRS,
  WINDOWS_APP_RUNTIME_FILES,
  WINDOWS_RUNTIME_DLLS,
} from '../scripts/windows-payload.js';

const dynamicDlls = ['cudnn_graph64_9.dll', 'cudnn_ops64_9.dll', 'nvrtc-builtins64_134.dll'];
const runtimeNames = [WINDOWS_ADDON, ...WINDOWS_RUNTIME_DLLS, ...dynamicDlls];
const pdfjsSampleFiles = [
  'node_modules/pdfjs-dist/wasm/openjpeg.wasm',
  'node_modules/pdfjs-dist/cmaps/Adobe-GB1-UCS2.bcmap',
  'node_modules/pdfjs-dist/standard_fonts/LiberationSans-Regular.ttf',
] as const;

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
    expect(() => selectWindowsRuntimeFiles(runtimeNames.filter((name) => !name.startsWith('nvrtc-builtins')))).toThrow(
      /NVRTC builtins/,
    );
  });
});

describe('packaged Windows app', () => {
  let root: string;
  let app: string;
  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'mlx-windows-payload-'));
    app = join(root, 'Portato-win32-arm64');
    write('Portato.exe');
    write('resources/app/package.json', JSON.stringify({ version: '0.0.21' }));
    for (const relPath of WINDOWS_APP_RUNTIME_FILES) write(`resources/app/${relPath}`, relPath);
    for (const relPath of pdfjsSampleFiles) write(`resources/app/${relPath}`, relPath);
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

  it('rejects a package missing Portato.exe or a native DLL or header', () => {
    rmSync(join(app, 'Portato.exe'));
    expect(() => readPackagedWindowsApp(app)).toThrow(/Portato Windows app/);

    write('Portato.exe');
    rmSync(join(app, 'resources/native/cudnn_ops64_9.dll'));
    expect(() => readPackagedWindowsApp(app)).toThrow(/cudnn_ops64_9\.dll/);
  });

  it.each(WINDOWS_APP_RUNTIME_FILES)(
    'rejects a packaged Windows app when required app runtime file %s is missing or empty',
    (relPath) => {
      const fullPath = join(app, 'resources/app', relPath);
      writeFileSync(fullPath, '');
      expect(() => assertWindowsAppRuntimeAssets(join(app, 'resources/app'))).toThrow(PayloadError);
      expect(() => readPackagedWindowsApp(app)).toThrow(relPath);

      rmSync(fullPath);
      expect(() => readPackagedWindowsApp(app)).toThrow(PayloadError);
      expect(() => readPackagedWindowsApp(app)).toThrow(relPath);
    },
  );

  it.each(WINDOWS_APP_RUNTIME_DIRS)(
    'rejects a packaged Windows app when pdfjs-dist asset directory %s is missing or empty',
    (relDir) => {
      const fullDir = join(app, 'resources/app', relDir);
      rmSync(fullDir, { recursive: true, force: true });
      expect(() => readPackagedWindowsApp(app)).toThrow(PayloadError);
      expect(() => readPackagedWindowsApp(app)).toThrow(relDir);

      mkdirSync(fullDir, { recursive: true });
      writeFileSync(join(fullDir, 'empty.dat'), '');
      expect(() => readPackagedWindowsApp(app)).toThrow(PayloadError);
      expect(() => readPackagedWindowsApp(app)).toThrow(relDir);
    },
  );
});
