/** Windows ARM64 native runtime selection and packaged-layout checks. */

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

import { PayloadError } from './payload.js';

export const WINDOWS_ADDON = 'mlx-core.win32-arm64-msvc.node';

/** Keep the fixed names in sync with copyWindowsCudaRuntime in packages/core/build.ts. */
export const WINDOWS_RUNTIME_DLLS = [
  'cudart64_13.dll',
  'cublas64_13.dll',
  'cublasLt64_13.dll',
  'cufft64_12.dll',
  'cusolver64_12.dll',
  'cusparse64_12.dll',
  'nvJitLink_130_0.dll',
  'nvrtc64_130_0.dll',
  'cudnn64_9.dll',
  'cudnn_adv64_9.dll',
  'cudnn_cnn64_9.dll',
  'cudnn_engines_precompiled64_9.dll',
  'cudnn_engines_runtime_compiled64_9.dll',
  'cudnn_engines_tensor_ir64_9.dll',
  'cudnn_ext64_9.dll',
  'cudnn_graph64_9.dll',
  'cudnn_heuristic64_9.dll',
  'cudnn_ops64_9.dll',
  'openblas.dll',
] as const;

export interface WindowsPayloadSource {
  nativeDir: string;
  /** Addon followed by every CUDA/cuDNN/OpenBLAS DLL, all relative to nativeDir. */
  runtimeFiles: string[];
  /** CUDA toolkit headers; NVRTC needs these in addition to the MLX headers. */
  cudaIncludeDir: string;
  wwwRoot: string;
  appDir: string;
}

/** Select versioned NVRTC builtins and the split cuDNN libraries without shipping unrelated DLLs. */
export function selectWindowsRuntimeFiles(names: readonly string[]): string[] {
  const available = new Set(names);
  const dynamic = [...available]
    .filter((name) => /^nvrtc-builtins64_.*\.dll$/i.test(name) || /^cudnn.*\.dll$/i.test(name))
    .sort();
  if (!available.has(WINDOWS_ADDON)) {
    throw new PayloadError(`Missing Windows native addon: ${WINDOWS_ADDON}`, 'yarn build:native on Windows ARM64');
  }
  const missing = WINDOWS_RUNTIME_DLLS.filter((name) => !available.has(name));
  if (missing.length > 0) {
    throw new PayloadError(`Missing Windows runtime DLLs: ${missing.join(', ')}`, 'yarn build:native');
  }
  if (!dynamic.some((name) => /^nvrtc-builtins64_.*\.dll$/i.test(name))) {
    throw new PayloadError('Missing the NVRTC builtins DLL', 'yarn build:native with CUDA 13.4');
  }
  const fixed = new Set<string>(WINDOWS_RUNTIME_DLLS);
  return [WINDOWS_ADDON, ...WINDOWS_RUNTIME_DLLS, ...dynamic.filter((name) => !fixed.has(name))];
}

export function resolveWindowsPayload(repoRoot: string, env: NodeJS.ProcessEnv = process.env): WindowsPayloadSource {
  const nativeDir = join(repoRoot, 'packages', 'core');
  const addon = join(nativeDir, WINDOWS_ADDON);
  if (!existsSync(addon)) {
    throw new PayloadError(`Missing native artifact: ${addon}`, 'yarn build:native on Windows ARM64');
  }
  if (statSync(addon).size === 0) {
    throw new PayloadError(`Native artifact is empty: ${addon}`, 'yarn build:native on Windows ARM64');
  }

  let runtimeFiles: string[];
  try {
    runtimeFiles = selectWindowsRuntimeFiles(readdirSync(nativeDir));
  } catch (error) {
    if (error instanceof PayloadError) {
      throw new PayloadError(`${error.message} in ${nativeDir}`, error.remedy);
    }
    throw error;
  }
  for (const file of runtimeFiles) {
    const path = join(nativeDir, file);
    if (!existsSync(path) || statSync(path).size === 0) {
      throw new PayloadError(`Missing or empty Windows runtime file: ${path}`, 'yarn build:native');
    }
  }

  const cudaRoot = env.CUDA_PATH || env.CUDA_HOME;
  if (!cudaRoot) {
    throw new PayloadError(
      'CUDA_PATH is required to package the NVRTC runtime headers',
      'set CUDA_PATH to the Windows ARM64 CUDA 13.4 toolkit',
    );
  }
  const cudaIncludeDir = join(cudaRoot, 'include');
  if (!existsSync(join(cudaIncludeDir, 'cuda_runtime.h'))) {
    throw new PayloadError(
      `CUDA runtime headers not found: ${join(cudaIncludeDir, 'cuda_runtime.h')}`,
      'set CUDA_PATH to the Windows ARM64 CUDA 13.4 toolkit',
    );
  }
  for (const name of ['cccl', 'cute', 'cutlass']) {
    if (!existsSync(join(nativeDir, 'include', name))) {
      throw new PayloadError(
        `Missing NVRTC header tree: ${join(nativeDir, 'include', name)}`,
        'yarn build:native on Windows ARM64',
      );
    }
  }

  const wwwRoot = join(repoRoot, 'packages', 'dashboard', 'web');
  if (!existsSync(join(wwwRoot, 'index.html'))) {
    throw new PayloadError(
      `Dashboard SPA not built: ${join(wwwRoot, 'index.html')} is missing`,
      'yarn workspace @mlx-node/dashboard build:ui',
    );
  }

  const appDir = join(repoRoot, 'packages', 'desktop');
  if (!existsSync(join(appDir, 'dist', 'main', 'index.js'))) {
    throw new PayloadError(`Desktop main not built: ${join(appDir, 'dist', 'main', 'index.js')}`, 'yarn build:ts');
  }
  if (!existsSync(join(appDir, 'dist', 'preload', 'index.cjs'))) {
    throw new PayloadError(`Preload not built: ${join(appDir, 'dist', 'preload', 'index.cjs')}`, 'yarn build:ts');
  }

  return { nativeDir, runtimeFiles, cudaIncludeDir, wwwRoot, appDir };
}

/** Read back the version and native payload from the directory produced by Electron Packager. */
export function readPackagedWindowsApp(appPath: string): { version: string; runtimeFiles: string[] } {
  const manifestPath = join(appPath, 'resources', 'app', 'package.json');
  if (!existsSync(join(appPath, 'mlx-node.exe')) || !existsSync(manifestPath)) {
    throw new PayloadError(
      `${appPath} is not a packaged mlx-node Windows app`,
      'run package-windows.ts and pass its appPath output',
    );
  }
  const version = (JSON.parse(readFileSync(manifestPath, 'utf8')) as { version?: unknown }).version;
  if (typeof version !== 'string') {
    throw new PayloadError(`${manifestPath} has no string version`, 'package-windows.ts --app-version <version>');
  }

  const nativeDir = join(appPath, 'resources', 'native');
  if (!existsSync(nativeDir)) {
    throw new PayloadError(`Packaged native runtime is missing: ${nativeDir}`, 'repackage');
  }
  const runtimeFiles = selectWindowsRuntimeFiles(readdirSync(nativeDir));
  for (const file of runtimeFiles) {
    const path = join(nativeDir, file);
    if (statSync(path).size === 0) throw new PayloadError(`Packaged runtime file is empty: ${path}`, 'repackage');
  }
  for (const header of [
    'cuda_runtime.h',
    join('cccl', 'cuda', '__cccl_config'),
    join('cute', 'config.hpp'),
    join('cutlass', 'version.h'),
  ]) {
    if (!existsSync(join(nativeDir, 'include', header))) {
      throw new PayloadError(`Packaged NVRTC header is missing: ${join(nativeDir, 'include', header)}`, 'repackage');
    }
  }
  return { version, runtimeFiles };
}

/** Check the archive member list without extracting several GB of native runtime. */
export function missingWindowsZipEntries(
  entries: readonly string[],
  appDirectoryName: string,
  runtimeFiles: readonly string[],
): string[] {
  const normalized = new Set(entries.map((entry) => entry.replaceAll('\\', '/').replace(/^\.\//, '')));
  const required = [
    'mlx-node.exe',
    'resources/app/package.json',
    ...runtimeFiles.map((file) => `resources/native/${file}`),
    'resources/native/include/cuda_runtime.h',
    'resources/www/index.html',
  ];
  return required.filter((file) => !normalized.has(`${appDirectoryName}/${file}`));
}
