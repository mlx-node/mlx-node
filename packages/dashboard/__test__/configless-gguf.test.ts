import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  renameSync,
  rmSync,
  statSync,
  symlinkSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { catalogBuildSlug } from '@mlx-node/agent/catalog';
import { discoverLocalChatModels } from '@mlx-node/lm/model-discovery';
import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { type CatalogBuildItem, catalogWithState, heldByOtherBuild } from '../src/catalog.js';
import {
  deleteLocalModel,
  discoverLocalModels,
  DOWNLOAD_COMPLETE_MARKER,
  isListedFolder,
  isModelInstalled,
  isModelPresent,
  summarizeForeignEntry,
} from '../src/models.js';
import { catalogActionState } from '../ui/src/pages/models/catalog.js';
import {
  catalogBuildOf,
  type ConfiglessLayout,
  FLASH_NEXT,
  ggufHeader,
  HY4,
} from './helpers/configless-gguf-fixtures.js';

let root: string;
let modelsDir: string;
/** Stands for the user's own folder outside the models folder, where a link at the slug points. */
let home: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'dash-configless-'));
  modelsDir = join(root, 'models');
  home = join(root, 'home');
  mkdirSync(modelsDir);
  mkdirSync(home);
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

function buildItem(slug: string): CatalogBuildItem {
  return catalogWithState(modelsDir)
    .flatMap((item) => item.builds)
    .find((build) => build.slug === slug)!;
}

/** Plants `layout` as a link at its slug to a folder in {@link home}, or as a real folder; returns that folder. */
function plant(layout: ConfiglessLayout, how: 'link' | 'folder', header?: Buffer): string {
  if (how === 'folder') {
    const dir = join(modelsDir, layout.slug);
    layout.plant(dir, header);
    return dir;
  }
  const dir = join(home, layout.slug);
  layout.plant(dir, header);
  symlinkSync(dir, join(modelsDir, layout.slug));
  return dir;
}

describe('a GGUF model that Chat loads with no config.json', () => {
  describe.each<[ConfiglessLayout, 'link' | 'folder']>([
    [HY4, 'link'],
    [FLASH_NEXT, 'link'],
    [HY4, 'folder'],
    [FLASH_NEXT, 'folder'],
  ])('%s as a %s', (layout, how) => {
    it('is present and installed on its card, with no blocker, no record claimed and no update', async () => {
      plant(layout, how);
      const slugDir = join(modelsDir, layout.slug);
      // The engine lists it: the premise of the rule.
      expect((await discoverLocalChatModels(modelsDir)).map((model) => model.modelType)).toEqual([layout.family]);
      expect(isModelPresent(slugDir)).toBe(true);
      expect(isModelInstalled(slugDir)).toBe(false);
      expect(buildItem(layout.slug)).toMatchObject({
        present: true,
        installed: false,
        heldBy: null,
        blockedByForeignDir: false,
        blocker: null,
        localRevision: null,
        localAssetsRepo: null,
        localAssetsRevision: null,
        listed: true,
      });
      expect(catalogActionState(buildItem(layout.slug), { job: undefined, updateAvailable: false })).toBe('installed');
    });

    it('lists in Local models with its family from the header and its build from the file name', () => {
      plant(layout, how);
      const { models, warnings } = discoverLocalModels(modelsDir);
      expect(warnings).toEqual([]);
      expect(models).toEqual([
        expect.objectContaining({
          name: layout.slug,
          path: join(modelsDir, layout.slug),
          modelType: layout.family,
          quant: layout.quant,
          contextWindow: null,
          link: how === 'link',
        }),
      ]);
      // A link row shows the size of the folder it points to.
      expect(models[0]!.sizeBytes).toBeGreaterThan(0);
      expect(isListedFolder(join(modelsDir, layout.slug))).toBe(true);
    });

    it('reads the same with and without the link to itself inside the folder', () => {
      const dir = plant(layout, how);
      const withLink = { row: discoverLocalModels(modelsDir).models, item: buildItem(layout.slug) };
      rmSync(join(dir, layout.slug));
      expect({ row: discoverLocalModels(modelsDir).models, item: buildItem(layout.slug) }).toEqual(withLink);
    });

    it('is not present with a header that cannot be read, and nothing throws', () => {
      const dir = plant(layout, how, ggufHeader(layout.architecture).subarray(0, 30));
      const slugDir = join(modelsDir, layout.slug);
      const check = (): void => {
        expect(isModelPresent(slugDir)).toBe(false);
        expect(isListedFolder(slugDir)).toBe(false);
        expect(discoverLocalModels(modelsDir).models).toEqual([]);
        expect(buildItem(layout.slug)).toMatchObject({ present: false, installed: false, listed: false });
        expect(summarizeForeignEntry(slugDir)).toMatchObject({ listed: false });
      };
      check();
      writeFileSync(join(dir, layout.firstPart), Buffer.from('GGUF'));
      check();
      writeFileSync(join(dir, layout.firstPart), ggufHeader(layout.architecture));
      chmodSync(join(dir, layout.firstPart), 0o000);
      try {
        check();
      } finally {
        chmodSync(join(dir, layout.firstPart), 0o644);
      }
      expect(isModelPresent(slugDir)).toBe(true);
    });

    it('Delete in Local models removes only the link, or the whole real folder', () => {
      const dir = plant(layout, how);
      deleteLocalModel(modelsDir, layout.slug);
      expect(existsSync(join(modelsDir, layout.slug))).toBe(false);
      expect(existsSync(join(dir, layout.firstPart))).toBe(how === 'link');
      expect(buildItem(layout.slug)).toMatchObject({ present: false, blocker: null });
    });
  });

  it.each<['link' | 'folder']>([['link'], ['folder']])(
    'keeps the card as it was for a family that needs asset files, as a %s: not present, not listed',
    async (how) => {
      const XL = catalogBuildOf('Qwen3.8-27B', 'UD-Q4_K_XL');
      const slug = catalogBuildSlug(XL);
      const dir = join(how === 'link' ? home : modelsDir, slug);
      const weight = join(dir, 'Qwen3.8-27B-UD-Q4_K_XL.gguf');
      mkdirSync(dir);
      if (how === 'link') symlinkSync(dir, join(modelsDir, slug));
      writeFileSync(weight, ggufHeader('gemma4'));
      expect(await discoverLocalChatModels(modelsDir)).toEqual([]);
      expect(isModelPresent(join(modelsDir, slug))).toBe(false);
      expect(buildItem(slug)).toMatchObject({
        present: false,
        listed: false,
        heldBy: null,
        blockedByForeignDir: true,
        blocker: { kind: how === 'link' ? 'symlink' : 'directory', listed: false, holdsModel: true },
      });
      expect(discoverLocalModels(modelsDir).warnings).toEqual([`${slug}: no readable config.json; skipped`]);
      expect(() => deleteLocalModel(modelsDir, slug)).toThrow('not a model checkpoint directory');
      // The same file with a header of a family that needs no asset files loads, so the card reads it as present.
      writeFileSync(weight, ggufHeader('qwen35'));
      expect(buildItem(slug)).toMatchObject({ present: true, listed: true, blocker: null });
    },
  );

  it('reads another build of the same repository as held, not installed and not a cleanup', () => {
    const XL = catalogBuildOf('Inkling-2-Mini', 'UD-Q4_K_XL');
    const M = catalogBuildOf('Inkling-2-Mini', 'UD-Q4_K_M');
    const dir = join(home, 'inkling');
    mkdirSync(dir);
    writeFileSync(join(dir, 'Inkling-2-Mini-UD-Q4_K_M.gguf'), ggufHeader('inkling2'));
    symlinkSync(dir, join(modelsDir, catalogBuildSlug(XL)));
    expect(buildItem(catalogBuildSlug(XL))).toMatchObject({
      present: false,
      installed: false,
      heldBy: 'UD-Q4_K_M',
      blockedByForeignDir: false,
      blocker: null,
    });
    expect(heldByOtherBuild(modelsDir, catalogBuildSlug(XL))).toBe('UD-Q4_K_M');
    expect(buildItem(catalogBuildSlug(M))).toMatchObject({ present: true, installed: false, heldBy: null });
  });

  it('counts only the GGUFs whose header loads, so a broken one names no build', () => {
    const XL = catalogBuildOf('Inkling-2-Mini', 'UD-Q4_K_XL');
    const M = catalogBuildOf('Inkling-2-Mini', 'UD-Q4_K_M');
    const dir = join(modelsDir, catalogBuildSlug(XL));
    mkdirSync(dir);
    writeFileSync(join(dir, 'Inkling-2-Mini-UD-Q4_K_XL.gguf'), ggufHeader('inkling2'));
    writeFileSync(join(dir, 'Inkling-2-Mini-UD-Q4_K_M.gguf'), Buffer.from('GGUF'));
    expect(buildItem(catalogBuildSlug(XL))).toMatchObject({ present: true, heldBy: null, blocker: null });
    expect(buildItem(catalogBuildSlug(M))).toMatchObject({ present: false });
    expect(discoverLocalModels(modelsDir).models).toEqual([
      expect.objectContaining({ name: catalogBuildSlug(XL), modelType: 'inkling2', quant: 'UD-Q4_K_XL' }),
    ]);
  });
});

describe('a GGUF with no config.json that is not the model of its catalog folder', () => {
  const XL = catalogBuildSlug(catalogBuildOf('Qwen3.8-27B', 'UD-Q4_K_XL'));
  const ORIGINAL = catalogBuildSlug(catalogBuildOf('Inkling-2-Mini', 'Original'));

  it.each<[string, string, string, string]>([
    ['a Hy4-flash folder holds a DeepSeek GGUF', HY4.slug, 'DeepSeek-V4.1-Flash-Q4_K_M.gguf', 'deepseek41'],
    ['a Hy4-flash folder holds a Qwen3.8-27B GGUF', HY4.slug, 'Qwen3.8-27B-UD-Q4_K_XL.gguf', 'qwen35'],
    ['a Hy4-flash folder holds a Hy4-flash GGUF of another build', HY4.slug, 'Hy4-flash-Q8_0.gguf', 'hy_v4_flash'],
    [
      'the Qwen3.8-27B XL folder holds a Hy4-flash header under the XL name',
      XL,
      'Qwen3.8-27B-UD-Q4_K_XL.gguf',
      'hy_v4_flash',
    ],
    [
      'the Inkling-2-Mini Original folder holds an Inkling GGUF',
      ORIGINAL,
      'Inkling-2-Mini-UD-Q4_K_XL.gguf',
      'inkling2',
    ],
  ])('%s: the card is not present and offers Update, as before', async (_name, slug, file, architecture) => {
    mkdirSync(join(modelsDir, slug));
    writeFileSync(join(modelsDir, slug, file), ggufHeader(architecture));
    // Chat loads it, as another model.
    expect((await discoverLocalChatModels(modelsDir)).map((model) => model.name)).toEqual([
      file.slice(0, -'.gguf'.length),
    ]);
    const item = buildItem(slug);
    expect(item).toMatchObject({
      present: false,
      installed: false,
      heldBy: null,
      blockedByForeignDir: true,
      blocker: { kind: 'directory', holdsModel: true },
    });
    expect(catalogActionState(item, { job: undefined, updateAvailable: false })).toBe('adopt');
    expect(heldByOtherBuild(modelsDir, slug)).toBeNull();
  });

  it('a full app record with no config.json of a GGUF of another family is not installed on the card', () => {
    const dir = join(modelsDir, HY4.slug);
    mkdirSync(dir);
    writeFileSync(join(dir, 'DeepSeek-V4.1-Flash-Q4_K_M.gguf'), ggufHeader('deepseek41'));
    writeFileSync(
      join(dir, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({
        repo: 'AngelSlim/Hy4-flash-GGUF',
        revision: '6'.repeat(40),
        files: ['DeepSeek-V4.1-Flash-Q4_K_M.gguf'],
        scope: 'full',
        completedAt: '2026-11-18T11:30:13.744Z',
      }),
    );
    expect(isModelInstalled(dir)).toBe(false);
    expect(buildItem(HY4.slug)).toMatchObject({ present: false, installed: false, localRevision: null });
  });

  it('a renamed folder with the Hy4-flash record and DeepSeek weights does not make the Hy4-flash card present', () => {
    const dir = join(modelsDir, 'my-folder');
    mkdirSync(dir);
    writeFileSync(join(dir, 'DeepSeek-V4.1-Flash-Q4_K_M.gguf'), ggufHeader('deepseek41'));
    writeFileSync(
      join(dir, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({
        repo: 'AngelSlim/Hy4-flash-GGUF',
        revision: '6'.repeat(40),
        files: [],
        scope: 'partial',
        completedAt: '2026-11-18T11:30:13.744Z',
      }),
    );
    expect(buildItem(HY4.slug)).toMatchObject({ present: false, installed: false, blocker: null });
    // The same record beside the Hy4-flash weights still finds the renamed folder.
    rmSync(join(dir, 'DeepSeek-V4.1-Flash-Q4_K_M.gguf'));
    writeFileSync(join(dir, HY4.firstPart), ggufHeader('hy_v4_flash'));
    writeFileSync(join(dir, 'Hy4-flash-Q4_K_M-00002-of-00002.gguf'), 'engram rows');
    expect(buildItem(HY4.slug)).toMatchObject({ present: true, installed: false, blocker: null });
  });
});

describe('the file rules of a GGUF with no config.json, as the engine reads them', () => {
  // Discovery lists a set with a missing part or a folder part (the load then refuses it); it skips a dead link.
  it.each<[string, (dir: string) => void, boolean]>([
    ['a missing part', () => undefined, true],
    [
      'a part that is a dead link',
      (dir) => symlinkSync(join(dir, 'gone.gguf'), join(dir, 'Hy4-flash-Q4_K_M-00002-of-00002.gguf')),
      false,
    ],
    ['a part that is a folder', (dir) => mkdirSync(join(dir, 'Hy4-flash-Q4_K_M-00002-of-00002.gguf')), true],
  ])('a split set with %s does not count', async (_name, plantPart, discoveryLists) => {
    const dir = join(modelsDir, HY4.slug);
    mkdirSync(dir);
    writeFileSync(join(dir, HY4.firstPart), ggufHeader('hy_v4_flash'));
    plantPart(dir);
    expect((await discoverLocalChatModels(modelsDir)).length).toBe(discoveryLists ? 1 : 0);
    expect(isModelPresent(dir)).toBe(false);
    expect(discoverLocalModels(modelsDir).models).toEqual([]);
    expect(buildItem(HY4.slug)).toMatchObject({ present: false, listed: false });
    rmSync(join(dir, 'Hy4-flash-Q4_K_M-00002-of-00002.gguf'), { recursive: true, force: true });
    writeFileSync(join(dir, 'Hy4-flash-Q4_K_M-00002-of-00002.gguf'), 'engram rows');
    expect(isModelPresent(dir)).toBe(true);
  });

  it.each<['root' | 'variant']>([['root'], ['variant']])(
    'a GGUF at the %s whose real folder holds a config.json does not count',
    async (where) => {
      // The engine reads the config.json beside the real file, and this one names no family it loads.
      const real = join(home, 'real');
      mkdirSync(real);
      writeFileSync(join(real, 'config.json'), JSON.stringify({ model_type: 'llama' }));
      writeFileSync(join(real, 'Hy4-flash-Q4_K_M.gguf'), ggufHeader('hy_v4_flash'));
      const dir = join(modelsDir, HY4.slug);
      const at = where === 'root' ? dir : join(dir, 'Q4_K_M');
      mkdirSync(at, { recursive: true });
      symlinkSync(join(real, 'Hy4-flash-Q4_K_M.gguf'), join(at, 'Hy4-flash-Q4_K_M.gguf'));
      expect(await discoverLocalChatModels(modelsDir)).toEqual([]);
      expect(isModelPresent(dir)).toBe(false);
      expect(discoverLocalModels(modelsDir).models).toEqual([]);
      expect(buildItem(HY4.slug)).toMatchObject({ present: false, listed: false });
      rmSync(join(real, 'config.json'));
      expect(isModelPresent(dir)).toBe(true);
    },
  );

  it('a GGUF whose name its family does not load does not count', async () => {
    // Qwen3.5 GGUF discovery loads only the XL-policy names, not `Q4_0`.
    const dir = join(modelsDir, 'qwen-q4-0');
    mkdirSync(dir);
    writeFileSync(join(dir, 'Qwen3.8-27B-Q4_0.gguf'), ggufHeader('qwen35'));
    expect(await discoverLocalChatModels(modelsDir)).toEqual([]);
    expect(isModelPresent(dir)).toBe(false);
    expect(discoverLocalModels(modelsDir).models).toEqual([]);
    rmSync(join(dir, 'Qwen3.8-27B-Q4_0.gguf'));
    writeFileSync(join(dir, 'Qwen3.8-27B-Q4_K_M.gguf'), ggufHeader('qwen35'));
    expect(discoverLocalModels(modelsDir).models).toEqual([
      expect.objectContaining({ name: 'qwen-q4-0', modelType: 'qwen3_5', quant: 'Q4_K_M' }),
    ]);
  });
});

describe('the header read, kept per file', () => {
  it('reads a file again when another file of the same size and mtime takes its name', () => {
    const dir = join(modelsDir, 'model');
    mkdirSync(dir);
    const weight = join(dir, 'Model-Q4_K_M.gguf');
    writeFileSync(weight, ggufHeader('qwen35'));
    const other = join(root, 'other.gguf');
    writeFileSync(other, ggufHeader('gemma4'));
    const time = 1_700_000_000;
    utimesSync(weight, time, time);
    utimesSync(other, time, time);
    const before = statSync(weight, { bigint: true });
    expect(discoverLocalModels(modelsDir).models.map((model) => model.modelType)).toEqual(['qwen3_5']);
    renameSync(other, weight);
    const after = statSync(weight, { bigint: true });
    expect([after.size, after.mtimeNs]).toEqual([before.size, before.mtimeNs]);
    expect(after.ino).not.toBe(before.ino);
    expect(discoverLocalModels(modelsDir).models).toEqual([]);
  });
});
