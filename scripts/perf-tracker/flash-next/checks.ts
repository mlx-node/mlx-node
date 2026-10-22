/// <reference types="node" />

import { execFileSync } from 'node:child_process';
import { lstat } from 'node:fs/promises';
import { isDeepStrictEqual } from 'node:util';

import { fileHashes, sh, sha256File, sleep, type FixtureCase } from '../common.js';
import { isLocalLlm, processTable } from '../qwen38/admission.js';
import type { Result } from '../qwen38/runtimes.js';
import { sampleAcceptance } from '../qwen38/suite.js';
import { ASSET_FILES, PROTOCOL, SPECULATION, type Loaded, type SplitIdentity, type Volume } from './suite.js';

export async function split(path: string, hash: boolean): Promise<SplitIdentity> {
  const info = await lstat(path);
  if (!info.isFile()) throw new Error(`${path}: not a regular file; copy the model files, not links`);
  const { size, mtimeMs, ctimeMs, ino, dev } = info;
  return { path, size, mtimeMs, ctimeMs, ino, dev, ...(hash ? { sha256: await sha256File(path) } : {}) };
}

export function volume(path: string): Volume {
  const device = sh('/usr/bin/stat', ['-f', '%Sd', path]);
  const plist = execFileSync('/usr/sbin/diskutil', ['info', '-plist', device]);
  const info = JSON.parse(
    execFileSync('/usr/bin/plutil', ['-convert', 'json', '-o', '-', '-'], { input: plist, encoding: 'utf8' }),
  ) as Record<string, unknown>;
  return {
    path,
    device,
    volumeName: String(info.VolumeName),
    volumeUuid: String(info.VolumeUUID),
    mountPoint: String(info.MountPoint),
    bus: String(info.BusProtocol),
    internal: info.Internal === true,
    solidState: info.SolidState === true,
  };
}

// The user's decision: both runtimes stream this model from the internal SSD, never a USB disk.
export function assertInternalSsd(paths: readonly string[]): Volume[] {
  const volumes = paths.map(volume);
  const outside = volumes.filter((item) => !item.internal || !item.solidState);
  if (outside.length) {
    const where = outside.map((item) => `${item.path} is on ${item.volumeName} (${item.bus})`).join('; ');
    throw new Error(`Copy the model files to the internal SSD first: ${where}`);
  }
  return volumes;
}

function runtimeProcesses(): { pid: number; command: string }[] {
  return processTable().filter((row) => row.pid !== process.pid && isLocalLlm(row));
}

// A model process left from an earlier block would shrink the next automatic budget.
export async function assertNoRuntimeAlive(label: string, waitMs = 60_000): Promise<void> {
  const deadline = performance.now() + waitMs;
  for (;;) {
    const alive = runtimeProcesses();
    if (!alive.length) return;
    if (performance.now() >= deadline) {
      const list = alive.map(({ pid, command }) => `${pid} ${command.slice(0, 160)}`).join('\n');
      throw new Error(`${label}: a model runtime is still running; stop it first:\n${list}`);
    }
    await sleep(1000);
  }
}

export function checkResult(result: Result, item: FixtureCase): void {
  if (
    result.numTokens !== PROTOCOL.outputTokens ||
    result.cachedTokens !== 0 ||
    result.promptTokens !== item.tokenIds.length
  ) {
    throw new Error(
      `${result.runtime} ${item.name}: ${result.numTokens} tokens, ${result.cachedTokens} cached, ${result.promptTokens} prompt`,
    );
  }
  sampleAcceptance(result, SPECULATION);
}

// The context cap and the extracted tokenizer assets are fixed per load, so check them per load.
export async function checkLoaded(loaded: Loaded, cases: readonly FixtureCase[], expected?: Record<string, string>) {
  if (loaded.hasMtp !== true) throw new Error('mlx-node loaded no MTP weights; not the recorded protocol');
  const needed = Math.max(...cases.map((item) => item.tokenIds.length)) + PROTOCOL.outputTokens;
  if (loaded.context.effectiveWindowTokens < needed) {
    throw new Error(
      `mlx-node admits ${loaded.context.effectiveWindowTokens} context tokens; the protocol needs ${needed}`,
    );
  }
  const assets = await fileHashes(loaded.assets, ASSET_FILES);
  if (Object.keys(assets).length !== ASSET_FILES.length) throw new Error(`${loaded.assets}: missing tokenizer assets`);
  if (expected && !isDeepStrictEqual(assets, expected)) {
    throw new Error(`${loaded.assets}: the extracted tokenizer assets changed since prewarm`);
  }
  return assets;
}
