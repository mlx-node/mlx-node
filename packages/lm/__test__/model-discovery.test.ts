import { chmodSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, relative } from 'node:path';

import { afterAll, describe, expect, it } from 'vite-plus/test';

import { discoverLocalChatModels, ggufSplitPart } from '../src/model-discovery.js';

const tmp = mkdtempSync(join(tmpdir(), 'mlx-discovery-'));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

// chmod 000 yields EACCES only on POSIX for a non-root user.
const canChmod = process.platform !== 'win32' && process.getuid?.() !== 0;

function minimalGguf(architecture: string): Buffer {
  const header = Buffer.alloc(24);
  header.write('GGUF');
  header.writeUInt32LE(3, 4);
  header.writeBigUInt64LE(1n, 16);
  const string = (value: string): Buffer => {
    const bytes = Buffer.from(value);
    const length = Buffer.alloc(8);
    length.writeBigUInt64LE(BigInt(bytes.length));
    return Buffer.concat([length, bytes]);
  };
  const stringType = Buffer.alloc(4);
  stringType.writeUInt32LE(8);
  return Buffer.concat([header, string('general.architecture'), stringType, string(architecture)]);
}

/** Write `files` (path → contents) under a fresh `<tmp>/<name>` models dir; returns the dir. */
function modelsDir(name: string, files: Record<string, string | Buffer>): string {
  const dir = join(tmp, name);
  for (const [path, contents] of Object.entries(files)) {
    mkdirSync(dirname(join(dir, path)), { recursive: true });
    writeFileSync(join(dir, path), contents);
  }
  return dir;
}

const FLASH_SHARDS = [1, 2, 3, 4].map((i) => `UD-Q4_K_XL/Flash-UD-Q4_K_XL-0000${i}-of-00004.gguf`);
/** `mlx download -g "*UD-Q4_K_XL*"` layout: README + marker at the repo root, shards one level down. */
const flashRepo = (extra: Record<string, string | Buffer> = {}): Record<string, string | Buffer> => ({
  'flash/README.md': '# Flash',
  'flash/.mlx-download-complete.json': '{}',
  ...Object.fromEntries(FLASH_SHARDS.map((path, i) => [`flash/${path}`, i === 0 ? minimalGguf('qwen4exp') : 'shard'])),
  ...Object.fromEntries(Object.entries(extra).map(([path, contents]) => [`flash/${path}`, contents])),
});

describe('discoverLocalChatModels — GGUF variant folders', () => {
  it('lists the first shard of a split set in a variant folder', async () => {
    const dir = modelsDir('variant-flash', flashRepo());
    const failures: string[] = [];
    expect(await discoverLocalChatModels(dir, { onEntryFailure: (_e, path) => failures.push(path) })).toEqual([
      expect.objectContaining({
        name: 'Flash-UD-Q4_K_XL',
        path: join(dir, 'flash', FLASH_SHARDS[0]),
        modelType: 'qwen4_exp',
        base: 'Flash',
        variant: 'UD-Q4_K_XL',
        format: 'gguf',
      }),
    ]);
    expect(failures).toEqual([]);
  });

  it('keeps the id of a single file, and a split set beside it with the same stem takes its folder', async () => {
    const dir = modelsDir('split-beside-single', {
      'direct/X.gguf': minimalGguf('qwen4exp'),
      'direct/X-00001-of-00002.gguf': minimalGguf('qwen4exp'),
      'direct/X-00002-of-00002.gguf': 'shard',
    });
    expect(
      (await discoverLocalChatModels(dir)).map(({ name, path, base, variant }) => ({ name, path, base, variant })),
    ).toEqual([
      { name: 'X', path: join(dir, 'direct', 'X.gguf'), base: 'X', variant: 'GGUF' },
      { name: 'direct-X', path: join(dir, 'direct', 'X-00001-of-00002.gguf'), base: 'direct-X', variant: 'GGUF' },
    ]);
  });

  describe('ids against the scan that named every GGUF by its file name', () => {
    const G = minimalGguf('qwen4exp');
    const MLX = JSON.stringify({ model_type: 'qwen3', max_position_embeddings: 4096 });
    const split = (stem: string): Record<string, string | Buffer> => ({
      [`${stem}-00001-of-00002.gguf`]: G,
      [`${stem}-00002-of-00002.gguf`]: 'shard',
    });
    /** `head`: id -> path as that scan gave them; `split`: path of a split set -> its id now. */
    const layouts: ReadonlyArray<{
      files: Record<string, string | Buffer>;
      head: Record<string, string>;
      split: Record<string, string>;
    }> = [
      {
        files: { 'direct/X.gguf': G, ...split('direct/X') },
        head: { X: 'direct/X.gguf', 'X-00001-of-00002': 'direct/X-00001-of-00002.gguf' },
        split: { 'direct/X-00001-of-00002.gguf': 'direct-X' },
      },
      {
        files: { ...split('a-repo/Model-Q4_K_M'), 'b-repo/Model-Q4_K_M.gguf': G },
        head: {
          'Model-Q4_K_M': 'b-repo/Model-Q4_K_M.gguf',
          'Model-Q4_K_M-00001-of-00002': 'a-repo/Model-Q4_K_M-00001-of-00002.gguf',
        },
        split: { 'a-repo/Model-Q4_K_M-00001-of-00002.gguf': 'a-repo-Model-Q4_K_M' },
      },
      {
        files: { ...split('A-repo/Model-Q4_K_M'), 'Model-Q4_K_M.gguf': G },
        head: {
          'Model-Q4_K_M': 'Model-Q4_K_M.gguf',
          'Model-Q4_K_M-00001-of-00002': 'A-repo/Model-Q4_K_M-00001-of-00002.gguf',
        },
        split: { 'A-repo/Model-Q4_K_M-00001-of-00002.gguf': 'A-repo-Model-Q4_K_M' },
      },
      { files: { 'Foo-GGUF/Q4_K_M.gguf': G }, head: { Q4_K_M: 'Foo-GGUF/Q4_K_M.gguf' }, split: {} },
      {
        files: { 'Foo-GGUF/Q4_K_M.gguf': G, 'Foo-Q4_K_M.gguf': G },
        head: { 'Foo-Q4_K_M': 'Foo-Q4_K_M.gguf', Q4_K_M: 'Foo-GGUF/Q4_K_M.gguf' },
        split: {},
      },
      {
        files: { 'A-GGUF/Q4_K_M.gguf': G, 'Q4_K_M.gguf': G },
        head: { Q4_K_M: 'A-GGUF/Q4_K_M.gguf', 'models5-Q4_K_M': 'Q4_K_M.gguf' },
        split: {},
      },
      {
        files: { 'a-GGUF/Q4_K_M.gguf': G, 'b-GGUF/Q4_K_M.gguf': G },
        head: { Q4_K_M: 'a-GGUF/Q4_K_M.gguf', 'b-GGUF-Q4_K_M': 'b-GGUF/Q4_K_M.gguf' },
        split: {},
      },
      { files: { 'repo/Q4_K_M/Q4_K_M.gguf': G }, head: { Q4_K_M: 'repo/Q4_K_M/Q4_K_M.gguf' }, split: {} },
      { files: { 'pack/Qwen4-8B/Q4_K_M.gguf': G }, head: { Q4_K_M: 'pack/Qwen4-8B/Q4_K_M.gguf' }, split: {} },
      {
        files: { ...split('Foo-GGUF/UD-Q4_K_XL/UD-Q4_K_XL'), 'Foo-UD-Q4_K_XL.gguf': G },
        head: {
          'Foo-UD-Q4_K_XL': 'Foo-UD-Q4_K_XL.gguf',
          'UD-Q4_K_XL-00001-of-00002': 'Foo-GGUF/UD-Q4_K_XL/UD-Q4_K_XL-00001-of-00002.gguf',
        },
        split: { 'Foo-GGUF/UD-Q4_K_XL/UD-Q4_K_XL-00001-of-00002.gguf': 'Foo-GGUF-Foo-UD-Q4_K_XL' },
      },
      {
        files: { ...split('A-repo/X'), 'X/config.json': MLX },
        head: { X: 'X', 'X-00001-of-00002': 'A-repo/X-00001-of-00002.gguf' },
        split: { 'A-repo/X-00001-of-00002.gguf': 'A-repo-X' },
      },
      {
        files: { 'Foo-GGUF/Q4_K_M.gguf': G, 'Foo-Q4_K_M/config.json': MLX },
        head: { 'Foo-Q4_K_M': 'Foo-Q4_K_M', Q4_K_M: 'Foo-GGUF/Q4_K_M.gguf' },
        split: {},
      },
      {
        files: { 'a-repo/Model-Q4_K_M.gguf': G, ...split('b-repo/Model-Q4_K_M') },
        head: {
          'Model-Q4_K_M': 'a-repo/Model-Q4_K_M.gguf',
          'Model-Q4_K_M-00001-of-00002': 'b-repo/Model-Q4_K_M-00001-of-00002.gguf',
        },
        split: { 'b-repo/Model-Q4_K_M-00001-of-00002.gguf': 'b-repo-Model-Q4_K_M' },
      },
      {
        files: { ...split('a/X'), 'b/X.gguf': G, 'b-X.gguf': G },
        head: { X: 'b/X.gguf', 'X-00001-of-00002': 'a/X-00001-of-00002.gguf', 'b-X': 'b-X.gguf' },
        split: { 'a/X-00001-of-00002.gguf': 'a-X' },
      },
      {
        files: { ...split('A/M'), 'M-00001-of-00002/config.json': MLX },
        head: {
          'M-00001-of-00002': 'A/M-00001-of-00002.gguf',
          'M-00001-of-00002-M-00001-of-00002': 'M-00001-of-00002',
        },
        split: { 'A/M-00001-of-00002.gguf': 'M' },
      },
      {
        files: split('qwen3.8-flash-next-gguf/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL'),
        head: {
          'Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00002':
            'qwen3.8-flash-next-gguf/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00002.gguf',
        },
        split: {
          'qwen3.8-flash-next-gguf/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00002.gguf':
            'Qwen3.8-Flash-Next-UD-Q4_K_XL',
        },
      },
    ];

    it.each(layouts.map((layout, index) => [index, layout] as const))(
      'layout %i: same id and path for every model that is not a split set; a split set takes its stem when free',
      async (index, { files, head, split: splitIds }) => {
        const dir = modelsDir(join('ids', `models${index}`), files);
        const now = (await discoverLocalChatModels(dir)).map((m) => `${m.name} -> ${relative(dir, m.path)}`);
        const expected = Object.entries(head).map(([id, path]) => `${splitIds[path] ?? id} -> ${path}`);
        expect(now.sort()).toEqual(expected.sort());
        for (const [id, path] of Object.entries(head)) {
          if (splitIds[path] === undefined) expect(now).toContain(`${id} -> ${path}`);
        }
      },
    );

    it('keeps the picker row of a file named only by its tag: the base comes from its folder', async () => {
      const dir = modelsDir(join('ids', 'tag-row'), { 'Foo-GGUF/Q4_K_M.gguf': G });
      expect((await discoverLocalChatModels(dir)).map(({ name, base, variant }) => ({ name, base, variant }))).toEqual([
        { name: 'Q4_K_M', base: 'Foo', variant: 'Q4_K_M' },
      ]);
    });
  });

  describe('a file whose name is a short form of its folder', () => {
    const G = minimalGguf('qwen4exp');
    const MLX = JSON.stringify({ model_type: 'qwen4_exp' });
    type Row = { name: string; base: string; variant: string };
    const suffixPair = (suffix: string): [string, Record<string, string | Buffer>, Row[]] => [
      `${suffix} repo, plain and ${suffix} files`,
      {
        [`Qwen4-8B-${suffix}-GGUF/Qwen4-8B-Q4_K_M.gguf`]: G,
        [`Qwen4-8B-${suffix}-GGUF/Qwen4-8B-${suffix}-Q8_0.gguf`]: G,
      },
      [
        { name: 'Qwen4-8B-Q4_K_M', base: 'Qwen4-8B-Q4_K_M', variant: 'Q4_K_M' },
        { name: `Qwen4-8B-${suffix}-Q8_0`, base: `Qwen4-8B-${suffix}`, variant: 'Q8_0' },
      ],
    ];
    const rows: ReadonlyArray<[string, Record<string, string | Buffer>, Row[]]> = [
      ...['Instruct', 'Thinking', 'Coder', 'Vision', 'Base', 'Thinking-2507'].map(suffixPair),
      [
        '-it repo',
        { 'gemma-5-4b-it-GGUF/gemma-5-4b-Q4_K_M.gguf': G, 'gemma-5-4b-it-GGUF/gemma-5-4b-it-Q8_0.gguf': G },
        [
          { name: 'gemma-5-4b-Q4_K_M', base: 'gemma-5-4b-Q4_K_M', variant: 'Q4_K_M' },
          { name: 'gemma-5-4b-it-Q8_0', base: 'gemma-5-4b-it', variant: 'Q8_0' },
        ],
      ],
      [
        'v2 repo beside a v1 MLX folder',
        { 'Qwen4-8B-v2-GGUF/Qwen4-8B-Q4_K_M.gguf': G, 'Qwen4-8B-4bit/config.json': MLX },
        [
          { name: 'Qwen4-8B-4bit', base: 'Qwen4-8B', variant: '4-bit' },
          { name: 'Qwen4-8B-Q4_K_M', base: 'Qwen4-8B-Q4_K_M', variant: 'Q4_K_M' },
        ],
      ],
      [
        'version word in the repo only',
        { 'Ornith-1.5-GGUF/Ornith-Q4_K_M.gguf': G, 'Ornith-GGUF/Ornith-Q8_0.gguf': G },
        [
          { name: 'Ornith-Q4_K_M', base: 'Ornith-Q4_K_M', variant: 'Q4_K_M' },
          { name: 'Ornith-Q8_0', base: 'Ornith', variant: 'Q8_0' },
        ],
      ],
      [
        '8B and 14B repos with size-less files',
        { 'Qwen4-8B-GGUF/Qwen4-Q4_K_M.gguf': G, 'Qwen4-14B-GGUF/Qwen4-Q8_0.gguf': G },
        [
          { name: 'Qwen4-Q4_K_M', base: 'Qwen4-Q4_K_M', variant: 'Q4_K_M' },
          { name: 'Qwen4-Q8_0', base: 'Qwen4-Q8_0', variant: 'Q8_0' },
        ],
      ],
      [
        'upper-case repo',
        { 'QWEN4-8B-INSTRUCT-GGUF/qwen4-8b-q4_k_m.gguf': G, 'QWEN4-8B-INSTRUCT-GGUF/qwen4-8b-instruct-q8_0.gguf': G },
        [
          { name: 'qwen4-8b-instruct-q8_0', base: 'qwen4-8b-instruct', variant: 'Q8_0' },
          { name: 'qwen4-8b-q4_k_m', base: 'qwen4-8b-q4_k_m', variant: 'Q4_K_M' },
        ],
      ],
      [
        'active-parameter word: joins the MLX folder',
        { 'Foo-35B-A3B-GGUF/Foo-35B-Q4_K_M.gguf': G, 'foo-35b-a3b/config.json': MLX },
        [
          { name: 'Foo-35B-Q4_K_M', base: 'Foo-35B-A3B', variant: 'Q4_K_M' },
          { name: 'foo-35b-a3b', base: 'Foo-35B-A3B', variant: 'MLX' },
        ],
      ],
      [
        'decimal active-parameter word, any case',
        { 'Foo-35B-a3.5b-GGUF/Foo-35B-Q4_K_M.gguf': G },
        [{ name: 'Foo-35B-Q4_K_M', base: 'Foo-35B-a3.5b', variant: 'Q4_K_M' }],
      ],
      [
        'active-parameter word and another word',
        { 'Foo-35B-A3B-Instruct-GGUF/Foo-35B-Q4_K_M.gguf': G, 'Foo-35B-Q8_0.gguf': G },
        [
          { name: 'Foo-35B-Q4_K_M', base: 'Foo-35B-Q4_K_M', variant: 'Q4_K_M' },
          { name: 'Foo-35B-Q8_0', base: 'Foo-35B', variant: 'Q8_0' },
        ],
      ],
      [
        'the same generic file name in repos of different models',
        {
          'Alpha-7B-GGUF/model-Q4_K_M.gguf': G,
          'Beta-9B-GGUF/model-Q8_0.gguf': G,
          'pack/Alpha/ggml-model-Q4_K_M.gguf': G,
          'pack/Beta/ggml-model-Q8_0.gguf': G,
        },
        [
          { name: 'ggml-model-Q4_K_M', base: 'ggml-model-Q4_K_M', variant: 'Q4_K_M' },
          { name: 'ggml-model-Q8_0', base: 'ggml-model-Q8_0', variant: 'Q8_0' },
          { name: 'model-Q4_K_M', base: 'model-Q4_K_M', variant: 'Q4_K_M' },
          { name: 'model-Q8_0', base: 'model-Q8_0', variant: 'Q8_0' },
        ],
      ],
      [
        'files of one model in a pack folder named after it',
        { 'pack/Alpha/Alpha-Q4_K_M.gguf': G, 'pack/Alpha/Alpha-Q8_0.gguf': G },
        [
          { name: 'Alpha-Q4_K_M', base: 'Alpha', variant: 'Q4_K_M' },
          { name: 'Alpha-Q8_0', base: 'Alpha', variant: 'Q8_0' },
        ],
      ],
      [
        'files named only by their tag, in folders that name no model',
        { 'Q4_K_M.gguf': G, 'GGUF/Q4_K_M.gguf': G, 'HF-GGUF/Q4_K_M.gguf': G },
        [
          { name: 'HF-GGUF-Q4_K_M', base: 'HF-GGUF-Q4_K_M', variant: 'Q4_K_M' },
          { name: 'Q4_K_M', base: 'Q4_K_M', variant: 'Q4_K_M' },
          { name: 'family-16-Q4_K_M', base: 'family-16-Q4_K_M', variant: 'Q4_K_M' },
        ],
      ],
    ];

    it.each(rows.map(([title, files, expected], index) => [title, index, files, expected] as const))(
      '%s',
      async (_title, index, files, expected) => {
        const dir = modelsDir(join('family', `family-${index}`), files);
        const byName = (a: Row, b: Row): number => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
        expect(
          (await discoverLocalChatModels(dir)).map(({ name, base, variant }) => ({ name, base, variant })),
        ).toEqual([...expected].sort(byName));
      },
    );
  });

  it('puts the repo folder base before a split stem that is only a quant tag, in any folder order', async () => {
    const dir = modelsDir('split-tag-only', {
      ...Object.fromEntries(
        [1, 2].flatMap((k) => [
          [`b-gguf/UD-Q4_K_XL/UD-Q4_K_XL-0000${k}-of-00002.gguf`, k === 1 ? minimalGguf('qwen4exp') : 'shard'],
          [`a-gguf/UD-Q4_K_XL-0000${k}-of-00002.gguf`, k === 1 ? minimalGguf('qwen4exp') : 'shard'],
        ]),
      ),
      'UD-Q8_0.gguf': minimalGguf('qwen4exp'),
    });
    expect(
      (await discoverLocalChatModels(dir)).map(({ name, path, base, variant }) => ({ name, path, base, variant })),
    ).toEqual([
      { name: 'UD-Q8_0', path: join(dir, 'UD-Q8_0.gguf'), base: 'UD-Q8_0', variant: 'UD-Q8_0' },
      {
        name: 'a-UD-Q4_K_XL',
        path: join(dir, 'a-gguf', 'UD-Q4_K_XL-00001-of-00002.gguf'),
        base: 'a',
        variant: 'UD-Q4_K_XL',
      },
      {
        name: 'b-UD-Q4_K_XL',
        path: join(dir, 'b-gguf', 'UD-Q4_K_XL', 'UD-Q4_K_XL-00001-of-00002.gguf'),
        base: 'b',
        variant: 'UD-Q4_K_XL',
      },
    ]);
  });

  it('never joins tag-only files of variant folders named after different models', async () => {
    const dir = modelsDir(
      'tag-only-model-folders',
      Object.fromEntries(
        ['Qwen4-8B', 'Qwen4-14B'].flatMap((model) =>
          [1, 2].map((k) => [
            `qwen-pack/${model}/Q4_K_M-0000${k}-of-00002.gguf`,
            k === 1 ? minimalGguf('qwen4exp') : 'shard',
          ]),
        ),
      ),
    );
    expect(
      (await discoverLocalChatModels(dir)).map(({ name, path, base, variant }) => ({ name, path, base, variant })),
    ).toEqual([
      {
        name: 'qwen-pack-Qwen4-14B-Q4_K_M',
        path: join(dir, 'qwen-pack', 'Qwen4-14B', 'Q4_K_M-00001-of-00002.gguf'),
        base: 'qwen-pack-Qwen4-14B',
        variant: 'Q4_K_M',
      },
      {
        name: 'qwen-pack-Qwen4-8B-Q4_K_M',
        path: join(dir, 'qwen-pack', 'Qwen4-8B', 'Q4_K_M-00001-of-00002.gguf'),
        base: 'qwen-pack-Qwen4-8B',
        variant: 'Q4_K_M',
      },
    ]);
  });

  it('names a split set behind a linked repo folder by its stem and keeps the link path', async () => {
    const external = modelsDir(
      'split-link-target',
      Object.fromEntries(
        [1, 2, 3, 4].map((k) => [
          `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000${k}-of-00004.gguf`,
          k === 1 ? minimalGguf('qwen4exp') : 'shard',
        ]),
      ),
    );
    const dir = modelsDir('split-link', {});
    mkdirSync(dir, { recursive: true });
    symlinkSync(external, join(dir, 'qwen3.8-flash-next-gguf'));
    expect(await discoverLocalChatModels(dir)).toEqual([
      expect.objectContaining({
        name: 'Qwen3.8-Flash-Next-UD-Q4_K_XL',
        path: join(dir, 'qwen3.8-flash-next-gguf', 'UD-Q4_K_XL', 'Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'),
        base: 'Qwen3.8-Flash-Next',
        variant: 'UD-Q4_K_XL',
        format: 'gguf',
      }),
    ]);
  });

  it('groups the builds of one repo and an MLX folder of the same model under one base', async () => {
    const dir = modelsDir('variant-group', {
      'qwen3.8-27b-gguf/Qwen3.8-27B-UD-Q4_K_M.gguf': minimalGguf('qwen4exp'),
      'qwen3.8-27b-gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf': minimalGguf('qwen4exp'),
      'qwen3.8-27b-nvfp4/config.json': JSON.stringify({
        model_type: 'qwen4_exp',
        quantization: { group_size: 16, bits: 4, mode: 'nvfp4', 'model.embed_tokens': { bits: 8, group_size: 64 } },
      }),
      'qwen3.8-27b-mlx/config.json': JSON.stringify({ model_type: 'qwen4_exp', torch_dtype: 'bfloat16' }),
    });
    expect(
      (await discoverLocalChatModels(dir)).map(({ name, base, variant, format }) => ({ name, base, variant, format })),
    ).toEqual([
      { name: 'Qwen3.8-27B-UD-Q4_K_M', base: 'Qwen3.8-27B', variant: 'UD-Q4_K_M', format: 'gguf' },
      { name: 'Qwen3.8-27B-UD-Q4_K_XL', base: 'Qwen3.8-27B', variant: 'UD-Q4_K_XL', format: 'gguf' },
      { name: 'qwen3.8-27b-mlx', base: 'Qwen3.8-27B', variant: 'BF16', format: 'mlx' },
      { name: 'qwen3.8-27b-nvfp4', base: 'Qwen3.8-27B', variant: 'NVFP4', format: 'mlx' },
    ]);
  });

  it('does not register the repo folder itself when it has a root config.json', async () => {
    const dir = modelsDir(
      'variant-root-config',
      flashRepo({ 'config.json': JSON.stringify({ model_type: 'qwen4_exp' }) }),
    );
    expect((await discoverLocalChatModels(dir)).map((model) => model.path)).toEqual([
      join(dir, 'flash', FLASH_SHARDS[0]),
    ]);
  });

  it('looks into variant folders when the root holds only a projector', async () => {
    const dir = modelsDir('variant-mmproj-root', flashRepo({ 'mmproj-F16.gguf': minimalGguf('clip') }));
    expect((await discoverLocalChatModels(dir)).map((model) => model.path)).toEqual([
      join(dir, 'flash', FLASH_SHARDS[0]),
    ]);
  });

  it('lists each variant, sorted, and scopes a clashing name by the repo folder', async () => {
    const dir = modelsDir('variant-two', {
      'repo/Q8_0/Model.gguf': minimalGguf('qwen4exp'),
      'repo/UD-Q4_K_XL/Model.gguf': minimalGguf('qwen4exp'),
    });
    expect((await discoverLocalChatModels(dir)).map(({ name, path }) => ({ name, path }))).toEqual([
      { name: 'Model', path: join(dir, 'repo', 'Q8_0', 'Model.gguf') },
      { name: 'repo-Model', path: join(dir, 'repo', 'UD-Q4_K_XL', 'Model.gguf') },
    ]);
  });

  it('keeps a top-level target authoritative and never looks deeper than one variant level', async () => {
    const dir = modelsDir('variant-depth', {
      'top/Model.gguf': minimalGguf('qwen4exp'),
      'top/UD-Q4_K_XL/Other.gguf': minimalGguf('qwen4exp'),
      'deep/a/b/Model.gguf': minimalGguf('qwen4exp'),
    });
    expect((await discoverLocalChatModels(dir)).map((model) => model.path)).toEqual([join(dir, 'top', 'Model.gguf')]);
  });

  it('skips companion-only, dot, asset, and safetensors-owned folders without reporting failures', async () => {
    const dir = modelsDir('variant-skips', {
      'mtp/MTP/mtp-Q8_0.gguf': minimalGguf('qwen4exp'),
      'mtp/MTP-Q8_0/mtp-Q4_K_XL.gguf': minimalGguf('qwen4exp'),
      'dot/.hidden/Model.gguf': minimalGguf('qwen4exp'),
      'assets/.mlx-qwen4-assets-v2-abc/Model.gguf': minimalGguf('qwen4exp'),
      'safe/config.json': JSON.stringify({ model_type: 'qwen4_exp' }),
      'safe/model.safetensors': 'x',
      'safe/sub/Model.gguf': minimalGguf('qwen4exp'),
      'owned/sub/model.safetensors': 'x',
      'owned/sub/Model.gguf': minimalGguf('qwen4exp'),
    });
    const failures: string[] = [];
    const models = await discoverLocalChatModels(dir, { onEntryFailure: (_e, path) => failures.push(path) });
    expect(models.map((model) => model.path)).toEqual([join(dir, 'safe')]);
    expect(failures).toEqual([]);
  });

  it('never lists dashboard staging jobs or publish backups, and keeps the installed ID', async () => {
    const dir = modelsDir('variant-staging', {
      '.staging/flash@0123456789abcdef0123456789abcdef01234567.42.uuid/Model-UD-Q4_K_XL.gguf': minimalGguf('qwen4exp'),
      '.staging/flash.backup-42.uuid/Model-UD-Q4_K_XL.gguf': minimalGguf('qwen4exp'),
      'flash/Model-UD-Q4_K_XL.gguf': minimalGguf('qwen4exp'),
    });
    expect((await discoverLocalChatModels(dir)).map(({ name, path }) => ({ name, path }))).toEqual([
      { name: 'Model-UD-Q4_K_XL', path: join(dir, 'flash', 'Model-UD-Q4_K_XL.gguf') },
    ]);
  });

  it('follows a symlinked variant folder and ignores a dangling one', async () => {
    const external = modelsDir('variant-link-target', { 'Model.gguf': minimalGguf('qwen4exp') });
    const dir = modelsDir('variant-link', { 'repo/README.md': '#' });
    symlinkSync(external, join(dir, 'repo', 'linked'));
    symlinkSync(join(tmp, 'variant-link-absent'), join(dir, 'repo', 'dangling'));
    expect((await discoverLocalChatModels(dir)).map((model) => model.path)).toEqual([
      join(dir, 'repo', 'linked', 'Model.gguf'),
    ]);
  });

  it('checks the sidecars of an asset-requiring family beside the variant GGUF', async () => {
    const config = JSON.stringify({ model_type: 'gemma4' });
    const dir = modelsDir('variant-gemma', {
      'gemma/config.json': config,
      'gemma/tokenizer.json': '{}',
      'gemma/UD-Q4_K_XL/gemma-UD-Q4_K_XL.gguf': minimalGguf('gemma4'),
    });
    // Root sidecars do not count: the loader reads them beside the GGUF.
    expect(await discoverLocalChatModels(dir)).toEqual([]);
    writeFileSync(join(dir, 'gemma', 'UD-Q4_K_XL', 'config.json'), config);
    writeFileSync(join(dir, 'gemma', 'UD-Q4_K_XL', 'tokenizer.json'), '{}');
    expect(await discoverLocalChatModels(dir)).toEqual([
      expect.objectContaining({ path: join(dir, 'gemma', 'UD-Q4_K_XL', 'gemma-UD-Q4_K_XL.gguf'), modelType: 'gemma4' }),
    ]);
  });

  it.skipIf(!canChmod)('reports an unreadable variant folder and keeps scanning its siblings', async () => {
    const dir = modelsDir('variant-unreadable', {
      'repo/A/Model-A.gguf': minimalGguf('qwen4exp'),
      'repo/B/Model-B.gguf': minimalGguf('qwen4exp'),
    });
    const locked = join(dir, 'repo', 'A');
    chmodSync(locked, 0o000);
    const failures: string[] = [];
    try {
      expect(
        (await discoverLocalChatModels(dir, { onEntryFailure: (_e, path) => failures.push(path) })).map((m) => m.path),
      ).toEqual([join(dir, 'repo', 'B', 'Model-B.gguf')]);
      expect(failures).toEqual([locked]);
    } finally {
      chmodSync(locked, 0o700);
    }
  });
});

describe('discoverLocalChatModels', () => {
  it('admits only shard 00001 of a split set, and only for families that allow splits', async () => {
    const dir = join(tmp, 'split-sets');
    const qwen4 = join(dir, 'qwen4');
    const clef = join(dir, 'clef');
    mkdirSync(qwen4, { recursive: true });
    mkdirSync(clef, { recursive: true });
    const first = join(qwen4, 'Flash-UD-Q4_K_XL-00001-of-00002.gguf');
    writeFileSync(first, minimalGguf('qwen4exp'));
    writeFileSync(join(clef, 'Clef-Q8_0-00001-of-00002.gguf'), minimalGguf('clef'));
    writeFileSync(join(clef, 'Clef-Q8_0-00002-of-00001.gguf'), minimalGguf('clef'));
    expect(await discoverLocalChatModels(dir)).toEqual([
      expect.objectContaining({ path: first, modelType: 'qwen4_exp' }),
    ]);
  });

  it('returns [] for a dir that does not exist — nothing is installed', async () => {
    await expect(discoverLocalChatModels(join(tmp, 'absent'))).resolves.toEqual([]);
  });

  it('throws when the dir exists but cannot be scanned — "scan failed" is not "empty"', async () => {
    // ENOTDIR from a regular file stands in for EACCES/EIO: any non-ENOENT
    // readdir failure must propagate. The desktop supervisor treats a
    // confirmed-empty library as permanent (no retries), so misreporting an
    // I/O error as empty would suppress recovery after the error clears.
    const file = join(tmp, 'not-a-dir');
    writeFileSync(file, 'x');
    await expect(discoverLocalChatModels(file)).rejects.toThrow();
  });

  it('treats a non-model directory as a clean skip, not a scan failure', async () => {
    // No config.json → definitively not a model: the entry is skipped and the
    // empty result means "nothing installed", so onEntryFailure stays quiet.
    const dir = join(tmp, 'only-junk');
    mkdirSync(join(dir, 'not-a-model'), { recursive: true });
    const failures: string[] = [];
    await expect(discoverLocalChatModels(dir, { onEntryFailure: (_e, path) => failures.push(path) })).resolves.toEqual(
      [],
    );
    expect(failures).toEqual([]);
  });

  it('ignores generated GGUF asset directories while discovering the source checkpoint', async () => {
    const dir = join(tmp, 'gguf-assets');
    mkdirSync(dir);
    const gguf = join(dir, 'Qwen3.8-Flash-Next.gguf');
    writeFileSync(gguf, minimalGguf('qwen4exp'));
    for (const name of ['.mlx-qwen4-assets-v2-0123456789abcdef', '.mlx-qwen4-assets-tmp-publishing']) {
      const assets = join(dir, name);
      mkdirSync(assets);
      writeFileSync(join(assets, 'config.json'), JSON.stringify({ model_type: 'qwen4_exp' }));
      writeFileSync(join(assets, 'tokenizer.json'), '{}');
      if (!name.includes('-tmp-')) writeFileSync(join(assets, 'complete'), '1');
    }
    const failures: string[] = [];
    expect(await discoverLocalChatModels(dir, { onEntryFailure: (_e, path) => failures.push(path) })).toEqual([
      expect.objectContaining({ path: gguf, modelType: 'qwen4_exp' }),
    ]);
    expect(failures).toEqual([]);
  });

  it.each(['direct', 'nested'] as const)(
    'discovers the exact dense Qwen3.5 Bonsai PQ2_0 GGUF in %s layout',
    async (layout) => {
      const dir = join(tmp, `bonsai-pq2-${layout}`);
      const repository = layout === 'direct' ? dir : join(dir, 'bonsai');
      mkdirSync(repository, { recursive: true });
      const gguf = join(repository, 'Ternary-Bonsai-2-27B-PQ2_0.gguf');
      writeFileSync(gguf, minimalGguf('qwen35'));
      const models = await discoverLocalChatModels(dir);
      expect(models).toEqual([expect.objectContaining({ path: gguf, modelType: 'qwen3_5' })]);
    },
  );

  it('matches the Bonsai PQ2_0 filename case-insensitively', async () => {
    const dir = join(tmp, 'bonsai-case');
    mkdirSync(dir);
    const gguf = join(dir, 'ternary-bonsai-2-27b-pq2_0.GGUF');
    writeFileSync(gguf, minimalGguf('qwen35'));
    const models = await discoverLocalChatModels(dir);
    expect(models).toEqual([expect.objectContaining({ path: gguf, modelType: 'qwen3_5' })]);
  });

  it.each([
    ['Other-Model-27B-PQ2_0.gguf', 'qwen35'],
    ['Ternary-Bonsai-2-27B-PTQ1_0.gguf', 'qwen35'],
    ['Ternary-Bonsai-2-27B-PQ2_0-extra.gguf', 'qwen35'],
    ['Ternary-Bonsai-2-27B-PQ2_0.gguf', 'qwen35moe'],
    ['Ternary-Bonsai-2-27B-PQ2_0-mmproj.gguf', 'qwen35'],
    ['Qwen3.5-27B-Q4_K_S.gguf', 'qwen35'],
    ['Qwen3.5-27B-Q4_0.gguf', 'qwen35'],
    ['Qwen3.5-27B-UD-IQ5_M.gguf', 'qwen35'],
    ['Qwen3.5-27B-UD-IQ2_L.gguf', 'qwen35'],
  ] as const)('does not widen the Bonsai gate to %s [%s]', async (name, architecture) => {
    const dir = join(tmp, `bonsai-reject-${name}`);
    mkdirSync(dir);
    writeFileSync(join(dir, name), minimalGguf(architecture));
    await expect(discoverLocalChatModels(dir)).resolves.toEqual([]);
  });

  it.each([
    'Qwen3.5-27B-Q4_K_XL.gguf',
    'Qwen3.8-27B-UD-Q4_K_M.gguf',
    'Qwen3.5-27B-Q4_K_M.gguf',
    // The Unsloth 2-bit mix: Q2_K joined the importable types.
    'Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf',
    // The Unsloth UD-IQ* mixes: every grid format imports now.
    'Qwen3.6-35B-A3B-UD-IQ1_S.gguf',
    'Qwen3.6-35B-A3B-UD-IQ1_M.gguf',
    'Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf',
    'Qwen3.6-35B-A3B-UD-IQ2_M.gguf',
    'Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf',
    'Qwen3.6-35B-A3B-UD-IQ3_S.gguf',
    'Qwen3.6-35B-A3B-UD-IQ4_XS.gguf',
    'Qwen3.6-35B-A3B-UD-IQ4_NL.gguf',
  ])('admits dense Qwen3.5 %s alongside the Bonsai name', async (name) => {
    const dir = join(tmp, `qwen35-admitted-${name}`);
    mkdirSync(dir);
    const gguf = join(dir, name);
    writeFileSync(gguf, minimalGguf('qwen35'));
    const models = await discoverLocalChatModels(dir);
    expect(models).toEqual([expect.objectContaining({ path: gguf, modelType: 'qwen3_5' })]);
  });

  it('finds part 1 of a split Qwen3.5 MoE XL GGUF and never its later parts', async () => {
    const dir = join(tmp, 'qwen35moe-split');
    mkdirSync(dir);
    const parts = [1, 2, 3].map((k) => join(dir, `Qwen3.5-122B-A10B-UD-Q4_K_XL-0000${k}-of-00003.gguf`));
    for (const part of parts) writeFileSync(part, minimalGguf('qwen35moe'));
    writeFileSync(join(dir, 'Model-Q4_0-00001-of-00002.gguf'), minimalGguf('qwen35moe'));
    writeFileSync(join(dir, 'Model-Q4_0-00002-of-00002.gguf'), minimalGguf('qwen35moe'));
    const models = await discoverLocalChatModels(dir);
    expect(models).toEqual([expect.objectContaining({ path: parts[0], modelType: 'qwen3_5_moe' })]);

    const single = join(tmp, 'qwen35moe-single');
    mkdirSync(single);
    const whole = join(single, 'Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf');
    writeFileSync(whole, minimalGguf('qwen35moe'));
    expect(await discoverLocalChatModels(single)).toEqual([
      expect.objectContaining({ path: whole, modelType: 'qwen3_5_moe' }),
    ]);
  });

  it('parses split GGUF part names without a pattern', () => {
    expect(ggufSplitPart('M-UD-Q4_K_XL-00002-of-00003.GGUF')).toEqual({ stem: 'M-UD-Q4_K_XL', part: 2, total: 3 });
    expect(ggufSplitPart('M-00004-of-00003.gguf')).toBeNull();
    expect(ggufSplitPart('M-00000-of-00003.gguf')).toBeNull();
    expect(ggufSplitPart('M-0001-of-00003.gguf')).toBeNull();
    expect(ggufSplitPart('M-00001-of-01025.gguf')).toBeNull();
    expect(ggufSplitPart('M.gguf')).toBeNull();
    expect(ggufSplitPart('M-00001-of-00002.safetensors')).toBeNull();
  });

  it.skipIf(!canChmod)('reports an entry it could not evaluate via onEntryFailure', async () => {
    // config.json exists but cannot be opened (EACCES): the entry may be a
    // model missing from the result, so the scan is incomplete — exactly the
    // signal that keeps an empty result from being read as permanent.
    const dir = join(tmp, 'unreadable-entry');
    mkdirSync(join(dir, 'model-a'), { recursive: true });
    const config = join(dir, 'model-a', 'config.json');
    writeFileSync(config, '{}');
    chmodSync(config, 0o000);
    const failures: string[] = [];
    await expect(discoverLocalChatModels(dir, { onEntryFailure: (_e, path) => failures.push(path) })).resolves.toEqual(
      [],
    );
    expect(failures).toEqual([join(dir, 'model-a')]);
    chmodSync(config, 0o600);
  });

  it.skipIf(!canChmod).each(['direct', 'nested'] as const)(
    'reports an unreadable %s Qwen4 GGUF and discovers it after access is restored',
    async (layout) => {
      const dir = join(tmp, `unreadable-gguf-${layout}`);
      const repository = layout === 'direct' ? dir : join(dir, 'qwen4');
      mkdirSync(repository, { recursive: true });
      const gguf = join(repository, 'Qwen3.8-Flash-Next-00001-of-00002.GGUF');
      writeFileSync(gguf, minimalGguf('qwen4exp'));
      chmodSync(gguf, 0o000);
      const failures: string[] = [];
      try {
        await expect(
          discoverLocalChatModels(dir, { onEntryFailure: (_e, path) => failures.push(path) }),
        ).resolves.toEqual([]);
        expect(failures).toEqual([gguf]);
      } finally {
        chmodSync(gguf, 0o600);
      }
      expect(await discoverLocalChatModels(dir)).toEqual([
        expect.objectContaining({ path: gguf, modelType: 'qwen4_exp' }),
      ]);
    },
  );

  it.skipIf(!canChmod)('reports unreadable inventories in config-less Qwen4 directories', async () => {
    const dir = join(tmp, 'unreadable-inventory');
    const repository = join(dir, 'qwen4');
    mkdirSync(repository, { recursive: true });
    writeFileSync(join(repository, 'model.gguf'), minimalGguf('qwen4exp'));
    // Search permission allows a config lookup to return ENOENT, but listing
    // the GGUF-only directory still fails and must mark the scan incomplete.
    chmodSync(repository, 0o111);
    const failures: string[] = [];
    try {
      await expect(
        discoverLocalChatModels(dir, { onEntryFailure: (_e, path) => failures.push(path) }),
      ).resolves.toEqual([]);
      expect(failures).toEqual([repository]);
    } finally {
      chmodSync(repository, 0o700);
    }
    expect(await discoverLocalChatModels(dir)).toHaveLength(1);
  });

  it('admits a symlinked Gemma4 GGUF whose required assets sit beside the TARGET', async () => {
    // The loader canonicalizes the GGUF path, so the config.json/tokenizer.json
    // it consumes live beside the link's TARGET, not the link. The assets gate
    // must check there too: `resolveDirentKind` admits the link as a file, and
    // requiring the assets beside the LINK would silently drop a checkpoint
    // that loads fine.
    const dir = join(tmp, 'gemma4-symlink-admitted');
    const external = join(tmp, 'gemma4-symlink-target');
    mkdirSync(dir, { recursive: true });
    mkdirSync(external);
    const target = join(external, 'Gemma-4-26B-real.gguf');
    writeFileSync(target, minimalGguf('gemma4'));
    writeFileSync(join(external, 'config.json'), JSON.stringify({ model_type: 'gemma4' }));
    writeFileSync(join(external, 'tokenizer.json'), '{}');
    const link = join(dir, 'Gemma-4-26B.gguf');
    symlinkSync(target, link);

    const models = await discoverLocalChatModels(dir);
    // Recorded path stays the lexical link — loading it canonicalizes anyway.
    expect(models).toEqual([expect.objectContaining({ path: link, modelType: 'gemma4' })]);
  });

  it('still skips a symlinked GGUF when the required assets are missing beside the TARGET', async () => {
    // The negative half: a symlink must not relax the requiresAssets gate.
    // Putting the assets beside the LINK (where the pre-fix check looked) is
    // the sharper case — the loader never sees them, so the entry must stay
    // skipped exactly like an asset-less real file.
    const dir = join(tmp, 'gemma4-symlink-skipped');
    const external = join(tmp, 'gemma4-symlink-no-assets');
    mkdirSync(dir, { recursive: true });
    mkdirSync(external);
    writeFileSync(join(external, 'Gemma-4-26B-real.gguf'), minimalGguf('gemma4'));
    writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: 'gemma4' }));
    writeFileSync(join(dir, 'tokenizer.json'), '{}');
    symlinkSync(join(external, 'Gemma-4-26B-real.gguf'), join(dir, 'Gemma-4-26B.gguf'));

    await expect(discoverLocalChatModels(dir)).resolves.toEqual([]);
  });
});
