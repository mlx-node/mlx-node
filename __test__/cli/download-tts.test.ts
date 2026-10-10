import { existsSync, mkdtempSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

const hub = vi.hoisted(() => ({
  sha: 'a'.repeat(40),
  cache: '',
  files: {} as Record<string, string>,
  revisions: [] as (string | undefined)[],
  failPath: undefined as string | undefined,
}));
vi.mock('@huggingface/hub', () => ({
  modelInfo: async () => ({ sha: hub.sha }),
  listFiles: async function* (p: { revision?: string }) {
    hub.revisions.push(p.revision);
    for (const [path, text] of Object.entries(hub.files)) yield { type: 'file', path, size: Buffer.byteLength(text) };
  },
  downloadFile: async (p: { path: string; revision?: string }) => {
    hub.revisions.push(p.revision);
    return new Blob([hub.files[p.path]]);
  },
  downloadFileToCacheDir: async (p: { path: string; revision?: string }) => {
    hub.revisions.push(p.revision);
    if (p.path === hub.failPath) throw Object.assign(new Error('Download refused'), { statusCode: 403 });
    const path = join(hub.cache, p.path.replaceAll('/', '_'));
    writeFileSync(path, hub.files[p.path]);
    return path;
  },
}));
import { run, isModelAlreadyDownloaded } from '../../packages/cli/src/commands/download-model.js';
const cleanup: string[] = [];
afterEach(() => {
  for (const path of cleanup.splice(0)) rmSync(path, { recursive: true, force: true });
});
function setup() {
  const path = mkdtempSync(join(tmpdir(), 'tts-download-'));
  cleanup.push(path);
  hub.cache = join(path, 'cache');
  mkdirSync(hub.cache);
  hub.revisions = [];
  hub.sha = 'a'.repeat(40);
  hub.failPath = undefined;
  hub.files = {
    'config.json': JSON.stringify({ model_type: 'qwen3_tts' }),
    'model.safetensors': 'root weights',
    'generation_config.json': '{}',
    'vocab.json': '{}',
    'merges.txt': '#version: 0.2',
    'speech_tokenizer/config.json': JSON.stringify({ model_type: 'qwen3_tts_tokenizer_12hz' }),
    'speech_tokenizer/model.safetensors': 'codec weights',
    'original/model.safetensors': 'unrelated',
  };
  return join(path, 'model');
}
function legacyCodec(output: string, filename = 'model.safetensors') {
  mkdirSync(join(output, 'speech_tokenizer'), { recursive: true });
  writeFileSync(join(output, 'config.json'), hub.files['config.json']);
  writeFileSync(join(output, 'model.safetensors'), hub.files['model.safetensors']);
  writeFileSync(join(output, 'speech_tokenizer/config.json'), hub.files['speech_tokenizer/config.json']);
  writeFileSync(join(output, 'speech_tokenizer', filename), 'old codec');
}
function shardedCodec() {
  delete hub.files['speech_tokenizer/model.safetensors'];
  hub.files['speech_tokenizer/model-00001-of-00002.safetensors'] = 'new codec shard one';
  hub.files['speech_tokenizer/model-00002-of-00002.safetensors'] = 'new codec shard two';
  hub.files['speech_tokenizer/model.safetensors.index.json'] = JSON.stringify({
    weight_map: { a: 'model-00001-of-00002.safetensors', b: 'model-00002-of-00002.safetensors' },
  });
}
describe('composite TTS resources', () => {
  it.each(['model.safetensors', 'weights.safetensors'])(
    'replaces a legacy child %s with the current indexed shards',
    async (single) => {
      const output = setup();
      legacyCodec(output, single);
      writeFileSync(join(output, 'speech_tokenizer/adapter.safetensors'), 'user adapter');
      shardedCodec();
      await run(['-m', 'example/model', '-o', output, '--cache-dir', hub.cache]);
      expect(existsSync(join(output, 'speech_tokenizer', single))).toBe(false);
      expect(readFileSync(join(output, 'speech_tokenizer/adapter.safetensors'), 'utf8')).toBe('user adapter');
      expect(readFileSync(join(output, 'speech_tokenizer/model-00002-of-00002.safetensors'), 'utf8')).toBe(
        'new codec shard two',
      );
      const marker = JSON.parse(readFileSync(join(output, '.mlx-download-complete.json'), 'utf8'));
      expect(marker.scope).toBe('full');
      expect(marker.files).toContain('speech_tokenizer/model-00002-of-00002.safetensors');
      expect(marker.files).not.toContain(`speech_tokenizer/${single}`);
      expect(isModelAlreadyDownloaded(output, readdirSync(output))).toBe(true);
    },
  );
  it('prunes a legacy codec with explicit component globs and --complete', async () => {
    const output = setup();
    legacyCodec(output);
    shardedCodec();
    await run([
      '-m',
      'example/model',
      '-o',
      output,
      '--cache-dir',
      hub.cache,
      '--glob',
      'model.safetensors',
      '--glob',
      'speech_tokenizer/*',
      '--complete',
    ]);
    expect(existsSync(join(output, 'speech_tokenizer/model.safetensors'))).toBe(false);
    expect(readFileSync(join(output, 'speech_tokenizer/model-00002-of-00002.safetensors'), 'utf8')).toBe(
      'new codec shard two',
    );
    const marker = JSON.parse(readFileSync(join(output, '.mlx-download-complete.json'), 'utf8'));
    expect(marker.scope).toBe('full');
    expect(marker.files).toContain('speech_tokenizer/model-00002-of-00002.safetensors');
  });
  it('removes legacy child shards and index when the revision switches to a single file', async () => {
    const output = setup();
    legacyCodec(output, 'model-00001-of-00001.safetensors');
    writeFileSync(
      join(output, 'speech_tokenizer/model.safetensors.index.json'),
      JSON.stringify({ weight_map: { a: 'model-00001-of-00001.safetensors' } }),
    );
    await run(['-m', 'example/model', '-o', output, '--cache-dir', hub.cache]);
    expect(readdirSync(join(output, 'speech_tokenizer')).sort()).toEqual(['config.json', 'model.safetensors']);
    expect(readFileSync(join(output, 'speech_tokenizer/model.safetensors'), 'utf8')).toBe('codec weights');
  });
  it('keeps legacy child weights when a replacement shard download fails', async () => {
    const output = setup();
    legacyCodec(output);
    shardedCodec();
    hub.failPath = 'speech_tokenizer/model-00002-of-00002.safetensors';
    await expect(run(['-m', 'example/model', '-o', output, '--cache-dir', hub.cache])).rejects.toThrow(
      'Download refused',
    );
    expect(readFileSync(join(output, 'speech_tokenizer/model.safetensors'), 'utf8')).toBe('old codec');
    expect(existsSync(join(output, '.mlx-download-complete.json'))).toBe(false);
  });
  it('does not prune or certify when an old single file would hide a missing indexed replacement shard', async () => {
    const output = setup();
    legacyCodec(output);
    shardedCodec();
    delete hub.files['speech_tokenizer/model-00002-of-00002.safetensors'];
    await expect(run(['-m', 'example/model', '-o', output, '--cache-dir', hub.cache])).rejects.toThrow(
      'Composite model is incomplete',
    );
    expect(readFileSync(join(output, 'speech_tokenizer/model.safetensors'), 'utf8')).toBe('old codec');
    expect(existsSync(join(output, '.mlx-download-complete.json'))).toBe(false);
  });
  it('does not certify a child whose shard index refers to absent weights', async () => {
    const output = setup();
    delete hub.files['speech_tokenizer/model.safetensors'];
    hub.files['speech_tokenizer/part-1.safetensors'] = 'partial codec';
    hub.files['speech_tokenizer/model.safetensors.index.json'] = JSON.stringify({
      weight_map: { a: 'part-1.safetensors', b: 'part-2.safetensors' },
    });
    await expect(run(['-m', 'example/model', '-o', output, '--cache-dir', hub.cache])).rejects.toThrow(
      'Composite model is incomplete',
    );
    expect(isModelAlreadyDownloaded(output, readdirSync(output))).toBe(false);
  });
  it('downloads the mandatory child at the root revision and repairs missing child weights', async () => {
    const output = setup();
    const args = ['-m', 'example/model', '-o', output, '--cache-dir', hub.cache];
    await run(args);
    expect(hub.revisions.length).toBeGreaterThan(2);
    expect(new Set(hub.revisions)).toEqual(new Set([hub.sha]));
    expect(readFileSync(join(output, 'speech_tokenizer/model.safetensors'), 'utf8')).toBe('codec weights');
    expect(readdirSync(output)).not.toContain('original');
    expect(isModelAlreadyDownloaded(output, readdirSync(output))).toBe(true);
    rmSync(join(output, 'speech_tokenizer/model.safetensors'));
    expect(isModelAlreadyDownloaded(output, readdirSync(output))).toBe(false);
    await run(args);
    expect(readFileSync(join(output, 'speech_tokenizer/model.safetensors'), 'utf8')).toBe('codec weights');
    const marker = JSON.parse(readFileSync(join(output, '.mlx-download-complete.json'), 'utf8'));
    expect(marker.files).toContain('speech_tokenizer/model.safetensors');
  });
  it('rejects a missing required component and an unpinned revision', async () => {
    const output = setup();
    delete hub.files['speech_tokenizer/model.safetensors'];
    await expect(run(['-m', 'example/model', '-o', output, '--cache-dir', hub.cache])).rejects.toThrow('incomplete');
    setup();
    hub.sha = '';
    await expect(run(['-m', 'example/model', '-o', output, '--cache-dir', hub.cache])).rejects.toThrow(
      'immutable revision',
    );
  });
});
