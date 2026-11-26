import { execFile } from 'node:child_process';
import { statfs } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { totalmem } from 'node:os';
import { isAbsolute } from 'node:path';

import type { CatalogGpuInfo } from '@mlx-node/agent/catalog';

import type { MainApiContext } from '../context.js';

const HOST_GPU_PROBE_TIMEOUT_MS = 10_000;
const HOST_GPU_PROBE_MAX_BUFFER = 64 * 1024;
const HOST_GPU_PROBE_SCRIPT = [
  'const mod = require(process.argv[1]);',
  'const caps = typeof mod.getRuntimeCapabilities === "function" ? mod.getRuntimeCapabilities() : null;',
  'if (caps && typeof caps.backend === "string" && caps.backend.length > 0 && typeof caps.totalMemoryBytes === "number" && Number.isFinite(caps.totalMemoryBytes) && caps.totalMemoryBytes >= 0) {',
  '  process.stdout.write(JSON.stringify({ backend: caps.backend, totalMemoryBytes: caps.totalMemoryBytes }));',
  '}',
].join(' ');

/** What the Models page needs to know about this machine: its memory, free space, OS platform, and optional GPU info. */
export interface HostInfo {
  totalMemBytes: number;
  /** `null` when the models folder cannot be read. */
  diskFreeBytes: number | null;
  platform: NodeJS.Platform;
  /** Host GPU backend and usable device memory (populated on non-darwin hosts when detected). */
  gpu?: CatalogGpuInfo;
}

export interface ProbeHostGpuOptions {
  signal?: AbortSignal;
  env?: NodeJS.ProcessEnv;
  platform?: NodeJS.Platform;
  execPath?: string;
}

interface ProbeInvocation {
  file: string;
  target: string;
  env: NodeJS.ProcessEnv;
}

function parseDesktopCliProbe(env: NodeJS.ProcessEnv, platform: NodeJS.Platform): ProbeInvocation | null {
  const raw = env.MLX_DESKTOP_CLI;
  if (raw === undefined || raw === '') return null;
  let parsed: { executable?: unknown; nativeAddon?: unknown; nativeRuntimeDir?: unknown };
  try {
    parsed = JSON.parse(raw) as { executable?: unknown; nativeAddon?: unknown; nativeRuntimeDir?: unknown };
  } catch {
    return null;
  }
  if (
    typeof parsed.executable !== 'string' ||
    !isAbsolute(parsed.executable) ||
    typeof parsed.nativeAddon !== 'string' ||
    !isAbsolute(parsed.nativeAddon)
  ) {
    return null;
  }
  const childEnv: NodeJS.ProcessEnv = {};
  let pathKey = 'PATH';
  for (const [key, value] of Object.entries(env)) {
    if (value === undefined) continue;
    const upper = key.toUpperCase();
    if (upper === 'NAPI_RS_NATIVE_LIBRARY_PATH' || upper === 'MLX_CORE_NATIVE_LIBRARY_PATH') continue;
    if (platform === 'win32' && upper === 'PATH') pathKey = key;
    childEnv[key] = value;
  }
  childEnv.ELECTRON_RUN_AS_NODE = '1';
  if (platform === 'win32' && typeof parsed.nativeRuntimeDir === 'string' && isAbsolute(parsed.nativeRuntimeDir)) {
    childEnv.CUDA_HOME = parsed.nativeRuntimeDir;
    childEnv.CUDA_PATH = parsed.nativeRuntimeDir;
    childEnv[pathKey] = childEnv[pathKey] ? `${parsed.nativeRuntimeDir};${childEnv[pathKey]}` : parsed.nativeRuntimeDir;
    childEnv.MLX_CUDA_USE_CUDNN_SDPA = '0';
  }
  return { file: parsed.executable, target: parsed.nativeAddon, env: childEnv };
}

function resolveProbeInvocation(options: ProbeHostGpuOptions): ProbeInvocation | null {
  const env = options.env ?? process.env;
  const platform = options.platform ?? process.platform;
  const fromDesktop = parseDesktopCliProbe(env, platform);
  if (fromDesktop !== null) return fromDesktop;
  try {
    const coreEntry = createRequire(import.meta.url).resolve('@mlx-node/core');
    return { file: options.execPath ?? process.execPath, target: coreEntry, env };
  } catch {
    return null;
  }
}

function parseGpuOutput(stdout: string): CatalogGpuInfo | undefined {
  const trimmed = stdout.trim();
  if (trimmed === '') return undefined;
  const candidates = [trimmed, trimmed.split(/\r?\n/).at(-1)?.trim() ?? ''];
  for (const candidate of candidates) {
    if (candidate === '') continue;
    try {
      const value = JSON.parse(candidate) as { backend?: unknown; totalMemoryBytes?: unknown };
      if (
        typeof value === 'object' &&
        value !== null &&
        typeof value.backend === 'string' &&
        value.backend.length > 0 &&
        typeof value.totalMemoryBytes === 'number' &&
        Number.isFinite(value.totalMemoryBytes) &&
        value.totalMemoryBytes >= 0
      ) {
        return { backend: value.backend, totalMemoryBytes: value.totalMemoryBytes };
      }
    } catch {
      // Try next candidate line.
    }
  }
  return undefined;
}

/**
 * Probe host GPU capabilities in a short-lived child process so the dashboard
 * process itself never maps the native addon (`packages/desktop/__test__/child-entries.test.ts`).
 *
 * Returns `undefined` immediately on macOS (Metal unified memory matches `totalmem()`)
 * and fails closed to `undefined` on any error or timeout.
 */
export async function probeHostGpu(options: ProbeHostGpuOptions = {}): Promise<CatalogGpuInfo | undefined> {
  const platform = options.platform ?? process.platform;
  if (platform === 'darwin') return undefined;
  const invocation = resolveProbeInvocation(options);
  if (invocation === null) return undefined;
  return await new Promise<CatalogGpuInfo | undefined>((resolve) => {
    execFile(
      invocation.file,
      ['-e', HOST_GPU_PROBE_SCRIPT, invocation.target],
      {
        env: invocation.env,
        timeout: HOST_GPU_PROBE_TIMEOUT_MS,
        maxBuffer: HOST_GPU_PROBE_MAX_BUFFER,
        windowsHide: true,
        ...(options.signal === undefined ? {} : { signal: options.signal }),
      },
      (error, stdout) => {
        if (error !== null) {
          resolve(undefined);
          return;
        }
        resolve(parseGpuOutput(stdout));
      },
    );
  });
}

export async function handleHost(ctx: Pick<MainApiContext, 'modelsDir' | 'hostGpu'>): Promise<HostInfo> {
  let diskFreeBytes: number | null;
  try {
    const fs = await statfs(ctx.modelsDir);
    diskFreeBytes = Number(fs.bavail) * Number(fs.bsize);
  } catch {
    diskFreeBytes = null;
  }
  let gpu: CatalogGpuInfo | undefined;
  if (ctx.hostGpu !== undefined) {
    try {
      gpu = await ctx.hostGpu();
    } catch {
      gpu = undefined;
    }
  }
  return {
    totalMemBytes: totalmem(),
    diskFreeBytes,
    platform: process.platform,
    ...(gpu !== undefined ? { gpu } : {}),
  };
}
