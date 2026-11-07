import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { ggufLoadEntries, ggufNextSteps, parseGgufFileName } from '../../packages/cli/src/commands/download-model.js';

/** A v3 GGUF header with no tensors; `architecture` undefined models shards 2..N of a llama.cpp split. */
function writeGguf(path: string, architecture?: string): void {
  const u32 = (value: number) => {
    const buffer = Buffer.alloc(4);
    buffer.writeUInt32LE(value);
    return buffer;
  };
  const u64 = (value: number) => {
    const buffer = Buffer.alloc(8);
    buffer.writeBigUInt64LE(BigInt(value));
    return buffer;
  };
  const str = (value: string) => Buffer.concat([u64(Buffer.byteLength(value)), Buffer.from(value)]);
  const kv = architecture === undefined ? [] : [str('general.architecture'), u32(8), str(architecture)];
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(
    path,
    Buffer.concat([Buffer.from('GGUF'), u32(3), u64(0), u64(architecture === undefined ? 0 : 1), ...kv]),
  );
}

const shards = (stem: string, count: number) =>
  Array.from(
    { length: count },
    (_, i) => `${stem}-${String(i + 1).padStart(5, '0')}-of-${String(count).padStart(5, '0')}.gguf`,
  );

describe('parseGgufFileName', () => {
  it('reads a llama.cpp split name, keeping the folder in the stem', () => {
    expect(parseGgufFileName('UD-Q4_K_XL/Flash-UD-Q4_K_XL-00002-of-00004.gguf')).toEqual({
      stem: 'UD-Q4_K_XL/Flash-UD-Q4_K_XL',
      ordinal: 2,
      shards: 4,
    });
  });

  it('accepts any case in the extension and the -of- separator', () => {
    expect(parseGgufFileName('M-00001-OF-00002.GGUF')).toEqual({ stem: 'M', ordinal: 1, shards: 2 });
  });

  it('treats a name without a full split suffix as a single file', () => {
    for (const path of [
      'M-Q4_K_M.gguf',
      'M-0001-of-00002.gguf',
      'M-00001-of-0002.gguf',
      'M-0000a-of-00002.gguf',
      'M00001-of-00002.gguf',
      '-00001-of-00002.gguf',
    ]) {
      expect(parseGgufFileName(path)).toEqual({ stem: path.slice(0, -'.gguf'.length), ordinal: 1, shards: 1 });
    }
  });

  it('rejects non-GGUF paths', () => {
    expect(parseGgufFileName('model.safetensors')).toBeUndefined();
    expect(parseGgufFileName('.gguf')).toBeUndefined();
  });
});

describe('ggufLoadEntries', () => {
  it('collapses a split set to its first shard, in any listing order', () => {
    const set = shards('UD-Q4_K_XL/Flash-UD-Q4_K_XL', 4);
    expect(ggufLoadEntries([set[2], set[0], set[3], set[1]])).toEqual([
      { path: set[0], shards: 4, present: 4, stem: 'UD-Q4_K_XL/Flash-UD-Q4_K_XL' },
    ]);
  });

  it('keeps single files and separate split sets apart', () => {
    const q8 = shards('Q8_0/M-Q8_0', 2);
    expect(ggufLoadEntries(['M-Q4_K_M.gguf', ...q8]).map((entry) => entry.path)).toEqual(['M-Q4_K_M.gguf', q8[0]]);
  });

  it('reports a partial split selection and still names shard 00001', () => {
    const set = shards('M', 3);
    expect(ggufLoadEntries([set[1]])).toEqual([{ path: set[0], shards: 3, present: 1, stem: 'M' }]);
  });

  it('ignores non-GGUF weights', () => {
    expect(ggufLoadEntries(['model.safetensors'])).toEqual([]);
  });
});

describe('ggufNextSteps', () => {
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'mlx-gguf-hint-'));
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  it('prints one direct-load path for a split set of a split-aware family', async () => {
    const set = shards('UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL', 4);
    set.forEach((path, i) => writeGguf(join(dir, path), i === 0 ? 'qwen4exp' : undefined));
    const lines = await ggufNextSteps(dir, set);
    expect(lines).toEqual([
      'Load directly (no conversion needed):',
      `  ${join(dir, set[0])} (reads the other 3 shard(s) beside it)`,
    ]);
    expect(lines.join('\n')).not.toContain('mlx convert');
  });

  it('tells a family without split support to merge first, never to convert each shard', async () => {
    const set = shards('Qwen3.6-27B-UD-Q4_K_XL', 2);
    writeGguf(join(dir, set[0]), 'qwen35');
    writeGguf(join(dir, set[1]));
    expect(await ggufNextSteps(dir, set)).toEqual([
      'Split GGUF with no split-aware loader: merge the shards, then convert:',
      `  llama-gguf-split --merge ${join(dir, set[0])} ${join(dir, 'Qwen3.6-27B-UD-Q4_K_XL.gguf')}`,
      `  mlx convert -i ${join(dir, 'Qwen3.6-27B-UD-Q4_K_XL.gguf')} -o ${dir}-mlx`,
    ]);
  });

  it('flags missing sidecars for families that need them', async () => {
    writeGguf(join(dir, 'gemma-4-12b-it-UD-Q4_K_XL.gguf'), 'gemma4');
    expect(await ggufNextSteps(dir, ['gemma-4-12b-it-UD-Q4_K_XL.gguf'])).toEqual([
      'Load directly (no conversion needed):',
      `  ${join(dir, 'gemma-4-12b-it-UD-Q4_K_XL.gguf')}`,
      `    needs config.json and tokenizer.json in ${dir}: run again with --assets-repo <base-model repo>`,
    ]);
    writeFileSync(join(dir, 'config.json'), '{}');
    writeFileSync(join(dir, 'tokenizer.json'), '{}');
    expect(await ggufNextSteps(dir, ['gemma-4-12b-it-UD-Q4_K_XL.gguf'])).toHaveLength(2);
  });

  it('does not offer --assets-repo for a nested GGUF, whose sidecars it would write to the wrong folder', async () => {
    const nested = 'UD-Q4_K_XL/gemma-4-12b-it-UD-Q4_K_XL.gguf';
    writeGguf(join(dir, nested), 'gemma4');
    writeFileSync(join(dir, 'config.json'), '{}');
    writeFileSync(join(dir, 'tokenizer.json'), '{}');
    expect(await ggufNextSteps(dir, [nested])).toEqual([
      'Load directly (no conversion needed):',
      `  ${join(dir, nested)}`,
      `    needs config.json and tokenizer.json in ${join(dir, 'UD-Q4_K_XL')}: copy them there from the base-model repo`,
    ]);
  });

  it('gives each converted variant its own output directory', async () => {
    writeGguf(join(dir, 'M-Q4_K_M.gguf'), 'llama');
    writeGguf(join(dir, 'M-Q8_0.gguf'), 'llama');
    expect(await ggufNextSteps(dir, ['M-Q4_K_M.gguf', 'M-Q8_0.gguf'])).toEqual([
      'To convert GGUF to MLX SafeTensors format:',
      `  mlx convert -i ${join(dir, 'M-Q4_K_M.gguf')} -o ${dir}-mlx-m-q4_k_m`,
      `  mlx convert -i ${join(dir, 'M-Q8_0.gguf')} -o ${dir}-mlx-m-q8_0`,
    ]);
  });

  it('falls back to the convert hint when the header cannot be read', async () => {
    writeFileSync(join(dir, 'broken.gguf'), 'not a gguf');
    expect(await ggufNextSteps(dir, ['broken.gguf'])).toEqual([
      'To convert GGUF to MLX SafeTensors format:',
      `  mlx convert -i ${join(dir, 'broken.gguf')} -o ${dir}-mlx`,
    ]);
  });

  it('names an incomplete split selection instead of a load or convert line', async () => {
    const set = shards('M', 3);
    writeGguf(join(dir, set[1]));
    expect(await ggufNextSteps(dir, [set[1]])).toEqual([
      'Incomplete split GGUF (select every shard to load or convert it):',
      `  ${join(dir, 'M')}: 1 of 3 shards selected`,
    ]);
  });
});
