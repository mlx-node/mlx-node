import { existsSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { tinyGgufBytes } from './tiny-gguf.js';

/**
 * Splash `gguf_target` + `model-check`: `mlx download` range-reads the header
 * of every selected target GGUF and checks it before any weight byte is
 * fetched. `downloadFile` is the fake range source; `downloadFileToCacheDir`
 * is the weight fetch that must not happen on a refusal.
 */
const REPO = 'unsloth/Tiny-GGUF';

interface ManifestEntry {
  type: 'file';
  path: string;
  size: number;
}

const hub = vi.hoisted(() => ({
  manifest: [] as ManifestEntry[],
  blobs: {} as Record<string, Uint8Array<ArrayBuffer>>,
  downloaded: [] as string[],
  snapshotDir: '',
  coreImported: false,
}));

vi.mock('@mlx-node/core', async (importActual) => {
  hub.coreImported = true;
  return await importActual();
});

vi.mock('@huggingface/hub', () => ({
  downloadFile: async (params: { path: string }) => {
    const bytes = hub.blobs[params.path];
    return bytes ? new Blob([bytes]) : new Blob(['{}']);
  },
  modelInfo: async () => ({ sha: 'a'.repeat(40) }),
  listFiles: async function* () {
    for (const entry of hub.manifest) yield entry;
  },
  downloadFileToCacheDir: async (params: { path: string }) => {
    hub.downloaded.push(params.path);
    const snapshot = join(hub.snapshotDir, params.path.replaceAll('/', '_'));
    const entry = hub.manifest.find((file) => file.path === params.path);
    writeFileSync(snapshot, hub.blobs[params.path] ?? 'x'.repeat(Math.max(1, entry?.size ?? 1)));
    return snapshot;
  },
}));

import { run } from '../../packages/cli/src/commands/download-model.js';

function serve(path: string, bytes: Uint8Array<ArrayBuffer>): void {
  hub.blobs[path] = bytes;
  hub.manifest.push({ type: 'file', path, size: bytes.length });
}

describe('download model GGUF header check', () => {
  let outputDir: string;
  let cacheDir: string;

  beforeEach(() => {
    outputDir = mkdtempSync(join(tmpdir(), 'mlx-gguf-check-out-'));
    cacheDir = mkdtempSync(join(tmpdir(), 'mlx-gguf-check-cache-'));
    hub.snapshotDir = mkdtempSync(join(tmpdir(), 'mlx-gguf-check-snap-'));
    hub.manifest = [];
    hub.blobs = {};
    hub.downloaded = [];
  });

  afterEach(() => {
    for (const dir of [outputDir, cacheDir, hub.snapshotDir]) rmSync(dir, { recursive: true, force: true });
  });

  const download = (glob: string) => run(['-m', REPO, '-o', outputDir, '-g', glob, '--cache-dir', cacheDir]);

  // First on purpose: no other case may have loaded the addon yet.
  it('never loads the native addon for a download without a GGUF', async () => {
    serve('model.safetensors', new Uint8Array(64));
    serve('config.json', new TextEncoder().encode('{}'));
    await download('*');
    expect(hub.downloaded).toContain('model.safetensors');
    expect(hub.coreImported).toBe(false);
  });

  it('downloads a target whose header passes', async () => {
    serve('Tiny-UD-Q4_K_XL.gguf', tinyGgufBytes({ arch: 'llama' }));
    await download('*UD-Q4_K_XL*');
    expect(hub.downloaded).toEqual(['Tiny-UD-Q4_K_XL.gguf']);
  });

  it('refuses a lone part of a split GGUF before any weight byte', async () => {
    serve(
      'Tiny-UD-Q4_K_XL.gguf',
      tinyGgufBytes({
        arch: 'qwen35',
        metadata: [
          ['split.no', { u16: 0 }],
          ['split.count', { u16: 3 }],
        ],
      }),
    );
    await expect(download('*UD-Q4_K_XL*')).rejects.toThrow(
      /Refusing to download Tiny-UD-Q4_K_XL\.gguf: .*split GGUF of 3 parts/,
    );
    expect(hub.downloaded).toEqual([]);
    expect(readdirSync(outputDir).filter((name) => name.endsWith('.gguf'))).toEqual([]);
  });

  it('refuses a selection that misses a part of a split GGUF', async () => {
    for (const k of [1, 3]) {
      serve(
        `Tiny-UD-Q4_K_XL-0000${k}-of-00003.gguf`,
        tinyGgufBytes({ arch: 'llama', tensors: [[`t${k}.weight`, [8]]] }),
      );
    }
    await expect(download('*UD-Q4_K_XL*')).rejects.toThrow(/parts 1, 3 of 3; select every part/);
    expect(hub.downloaded).toEqual([]);
    expect(existsSync(join(outputDir, 'Tiny-UD-Q4_K_XL-00001-of-00003.gguf'))).toBe(false);
  });
});
