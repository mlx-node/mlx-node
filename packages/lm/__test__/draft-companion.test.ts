import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { findDFlash2Draft, isDFlash2DraftDirectory } from '../src/draft-companion.js';

describe('DFlash2 companion discovery', () => {
  let root: string;
  let target: string;
  let draft: string;
  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'mlx-draft-discovery-'));
    target = join(root, 'qwen3.8-27b-mxfp4-mlx');
    draft = join(root, 'Qwen3.8-27B-DFlash2');
    mkdirSync(target);
    writeFileSync(join(target, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }));
  });
  afterEach(() => rmSync(root, { recursive: true, force: true }));

  function writeDraft(path = draft): void {
    mkdirSync(path, { recursive: true });
    writeFileSync(join(path, 'config.json'), JSON.stringify({ architectures: ['DFlash2DraftModel'] }));
    writeFileSync(join(path, 'model.safetensors'), 'weights');
  }

  it('pairs a separately downloaded companion with safetensors and GGUF variants', () => {
    writeDraft();
    expect(findDFlash2Draft(target, 'qwen3_5')).toBe(draft);
    expect(findDFlash2Draft(join(target, 'Qwen3.8-27B-UD-Q4_K_XL.gguf'), 'qwen3_5')).toBe(draft);
    expect(findDFlash2Draft(join(root, 'Qwen3.8-27B-UD-Q3_K_XL.gguf'), 'qwen3_5')).toBe(draft);
  });

  it('keeps explicit embedded drafts ahead of the shared recommendation', () => {
    writeDraft();
    const embedded = join(target, 'draft');
    writeDraft(embedded);
    expect(findDFlash2Draft(target, 'qwen3_5')).toBe(embedded);
  });

  it('does not search above a standalone GGUF directory for a shared companion', () => {
    writeDraft();
    const store = join(root, 'models');
    mkdirSync(store);
    const gguf = join(store, 'Qwen3.8-27B-UD-Q4_K_XL.gguf');
    expect(findDFlash2Draft(gguf, 'qwen3_5')).toBeUndefined();
    const inside = join(store, 'qwen3.8-27b-dflash2');
    writeDraft(inside);
    expect(findDFlash2Draft(gguf, 'qwen3_5')).toBe(inside);
  });

  it('keeps a supplied model-store root authoritative for repository GGUFs', () => {
    writeDraft();
    const gguf = join(target, 'Qwen3.8-27B-UD-Q4_K_XL.gguf');
    expect(findDFlash2Draft(gguf, 'qwen3_5', target)).toBeUndefined();
    expect(findDFlash2Draft(gguf, 'qwen3_5', root)).toBe(draft);
  });

  it('never attaches the Qwen3.8 companion to a different model or family', () => {
    writeDraft();
    for (const name of ['qwen3.5-27b', 'qwen3.8-35b', 'qwen3.8-27bigger', 'other-model']) {
      expect(findDFlash2Draft(join(root, name), 'qwen3_5')).toBeUndefined();
    }
    expect(findDFlash2Draft(target, 'qwen3_5_moe')).toBeUndefined();
  });

  it('pairs a companion living behind a directory symlink', () => {
    // A Dirent reports the link itself — `isDirectory()` is false for a symlink —
    // so a store entry that merely LINKS the companion dir (e.g. a shared HF
    // checkout linked into the models dir) must still be found.
    const external = mkdtempSync(join(tmpdir(), 'mlx-draft-external-'));
    try {
      writeDraft(external);
      symlinkSync(external, join(root, 'Qwen3.8-27B-DFlash2'));
      expect(findDFlash2Draft(target, 'qwen3_5')).toBe(join(root, 'Qwen3.8-27B-DFlash2'));
    } finally {
      rmSync(external, { recursive: true, force: true });
    }
  });

  it('ignores a dangling or non-directory symlink at the companion slug', () => {
    // A broken link admits nothing, and a link to a FILE at the slug name is
    // not a companion directory.
    symlinkSync(join(root, 'no-such-target'), join(root, 'qwen3.8-27b-dflash2'));
    expect(findDFlash2Draft(target, 'qwen3_5')).toBeUndefined();
    writeFileSync(join(root, 'loose-file'), 'x');
    rmSync(join(root, 'qwen3.8-27b-dflash2'));
    symlinkSync(join(root, 'loose-file'), join(root, 'qwen3.8-27b-dflash2'));
    expect(findDFlash2Draft(target, 'qwen3_5')).toBeUndefined();
  });

  it('ignores incomplete, empty, and wrong-architecture companion weights', () => {
    writeDraft();
    rmSync(join(draft, 'model.safetensors'));
    expect(isDFlash2DraftDirectory(draft)).toBe(true);
    expect(findDFlash2Draft(target, 'qwen3_5')).toBeUndefined();
    writeFileSync(join(draft, 'model.safetensors'), '');
    expect(findDFlash2Draft(target, 'qwen3_5')).toBeUndefined();
    writeDraft();
    writeFileSync(join(draft, 'config.json'), JSON.stringify({ architectures: ['Qwen3ForCausalLM'] }));
    expect(findDFlash2Draft(target, 'qwen3_5')).toBeUndefined();
  });
});
