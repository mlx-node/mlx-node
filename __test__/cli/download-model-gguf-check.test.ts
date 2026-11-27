import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
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
  noRevision: false,
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
  modelInfo: async () => (hub.noRevision ? {} : { sha: 'a'.repeat(40) }),
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

import { checkGgufTargets, planGgufProjectors, run } from '../../packages/cli/src/commands/download-model.js';

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
    hub.noRevision = false;
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

  it('returns the header architecture of each target', async () => {
    serve('Tiny-UD-Q4_K_M.gguf', tinyGgufBytes({ arch: 'inkling2' }));
    serve('sub/Other-Q8_0.gguf', tinyGgufBytes({ arch: 'llama' }));
    const checked = await checkGgufTargets({ repo: REPO, paths: ['Tiny-UD-Q4_K_M.gguf', 'sub/Other-Q8_0.gguf'] });
    expect(checked.targets).toEqual([['Tiny-UD-Q4_K_M.gguf'], ['sub/Other-Q8_0.gguf']]);
    expect(checked.architectures).toEqual(['inkling2', 'llama']);
  });

  it('downloads the vision projector the loader pairs beside a selected weight, and lists it in the marker', async () => {
    serve('Tiny-UD-Q4_K_M.gguf', tinyGgufBytes({ arch: 'inkling2' }));
    serve('Tiny-UD-Q4_K_XL.gguf', tinyGgufBytes({ arch: 'inkling2' }));
    serve('mmproj-BF16.gguf', tinyGgufBytes({ arch: 'clip' }));
    serve('mmproj-F16.gguf', tinyGgufBytes({ arch: 'clip' }));
    await download('*Q4_K_M*');
    expect([...hub.downloaded].sort()).toEqual(['Tiny-UD-Q4_K_M.gguf', 'mmproj-BF16.gguf']);
    expect(
      readdirSync(outputDir)
        .filter((name) => name.endsWith('.gguf'))
        .sort(),
    ).toEqual(['Tiny-UD-Q4_K_M.gguf', 'mmproj-BF16.gguf']);
    const marker = JSON.parse(readFileSync(join(outputDir, '.mlx-download-complete.json'), 'utf8')) as {
      files: string[];
    };
    expect(marker.files).toContain('mmproj-BF16.gguf');
  });

  it('says a weight with no projector beside it loads text-only, and installs it', async () => {
    serve('Tiny-UD-Q4_K_M.gguf', tinyGgufBytes({ arch: 'inkling2' }));
    const log = vi.spyOn(console, 'log');
    try {
      await download('*Q4_K_M*');
      expect(log).toHaveBeenCalledWith(
        `No vision projector beside Tiny-UD-Q4_K_M.gguf in ${REPO}: it loads text-only.`,
      );
    } finally {
      log.mockRestore();
    }
    expect(hub.downloaded).toEqual(['Tiny-UD-Q4_K_M.gguf']);
    expect(existsSync(join(outputDir, '.mlx-download-complete.json'))).toBe(true);
  });

  it('adds the projector to a folder that holds the matched weight when the revision cannot be resolved', async () => {
    serve('Tiny-UD-Q4_K_M.gguf', tinyGgufBytes({ arch: 'inkling2' }));
    serve('mmproj-BF16.gguf', tinyGgufBytes({ arch: 'clip' }));
    await download('*Q4_K_M*');
    rmSync(join(outputDir, 'mmproj-BF16.gguf'));
    hub.downloaded = [];
    hub.noRevision = true;
    await download('*Q4_K_M*');
    expect(hub.downloaded).toEqual(['mmproj-BF16.gguf']);
    expect(existsSync(join(outputDir, 'mmproj-BF16.gguf'))).toBe(true);
  });
});

describe('planGgufProjectors', () => {
  it('adds nothing when the selection already holds a projector in the weight folder', () => {
    const repo = ['M-UD-Q4_K_M.gguf', 'mmproj-BF16.gguf', 'mmproj-F16.gguf'];
    expect(
      planGgufProjectors([['M-UD-Q4_K_M.gguf']], ['qwen35'], repo, ['M-UD-Q4_K_M.gguf', 'mmproj-F16.gguf']),
    ).toEqual({ add: [], without: [] });
  });

  it('names the projectors of the folder when none names a precision the loader takes', () => {
    const repo = ['M-UD-Q4_K_M.gguf', 'mmproj-model.gguf', 'imatrix.gguf'];
    expect(planGgufProjectors([['M-UD-Q4_K_M.gguf']], ['qwen35'], repo, ['M-UD-Q4_K_M.gguf'])).toEqual({
      add: [],
      without: [{ weight: 'M-UD-Q4_K_M.gguf', tied: [], unranked: ['mmproj-model.gguf'] }],
    });
  });

  it('takes a projector only from the folder of the weight', () => {
    const target = ['BF16/M-BF16-00001-of-00002.gguf', 'BF16/M-BF16-00002-of-00002.gguf'];
    expect(planGgufProjectors([target], ['qwen35'], [...target, 'mmproj-BF16.gguf'], target)).toEqual({
      add: [],
      without: [{ weight: target[0], tied: [], unranked: [] }],
    });
    expect(planGgufProjectors([target], ['qwen35'], [...target, 'BF16/mmproj-BF16.gguf'], target)).toEqual({
      add: [{ path: 'BF16/mmproj-BF16.gguf', weight: target[0] }],
      without: [],
    });
  });
});
