import { existsSync, mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  findDFlash2Draft,
  findInkling2DSparkDraft,
  isDFlash2DraftDirectory,
  isInkling2DSparkGguf,
} from '../src/draft-companion.js';

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

  function writeDraft(path = draft, config: Record<string, unknown> = {}): void {
    mkdirSync(path, { recursive: true });
    writeFileSync(join(path, 'config.json'), JSON.stringify({ architectures: ['DFlash2DraftModel'], ...config }));
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

  it('pairs the 35B MoE companion without borrowing the dense companion', () => {
    writeDraft();
    const moe = join(root, 'Qwen3.6-35B-A3B-Q4_K_M.gguf');
    expect(findDFlash2Draft(moe, 'qwen3_5_moe')).toBeUndefined();
    const moeDraft = join(root, 'Qwen3.6-35B-A3B-DFlash2');
    writeDraft(moeDraft);
    expect(findDFlash2Draft(moe, 'qwen3_5_moe')).toBe(moeDraft);
    expect(findDFlash2Draft(moe, 'qwen3_5')).toBeUndefined();
    expect(findDFlash2Draft(join(root, 'Ornith-1.5-35B-Q4_K_M.gguf'), 'qwen3_5_moe')).toBeUndefined();
  });

  it('does not borrow an embedded dense draft in a mixed standalone GGUF store', () => {
    const store = join(root, 'models');
    mkdirSync(store);
    const dense = join(store, 'Qwen3.8-27B-UD-Q4_K_XL.gguf');
    const moe = join(store, 'Qwen3.6-35B-A3B-Q4_K_M.gguf');
    writeFileSync(dense, 'dense');
    writeFileSync(moe, 'moe');
    const embedded = join(store, 'draft');
    writeDraft(embedded, { model_type: 'qwen3', hidden_size: 5120, num_target_layers: 64 });
    const moeDraft = join(store, 'Qwen3.6-35B-A3B-DFlash2');
    writeDraft(moeDraft, { model_type: 'qwen3', hidden_size: 2048, num_target_layers: 40 });

    expect(findDFlash2Draft(dense, 'qwen3_5')).toBe(embedded);
    expect(findDFlash2Draft(moe, 'qwen3_5_moe')).toBe(moeDraft);
    rmSync(moeDraft, { recursive: true });
    expect(findDFlash2Draft(moe, 'qwen3_5_moe')).toBeUndefined();
    writeDraft(embedded);
    expect(findDFlash2Draft(moe, 'qwen3_5_moe')).toBeUndefined();

    writeDraft(embedded, { model_type: 'qwen3', hidden_size: 2048, num_target_layers: 40 });
    expect(findDFlash2Draft(moe, 'qwen3_5_moe')).toBe(embedded);
  });

  it('allows a renamed MoE target to opt in with an embedded draft', () => {
    const renamed = join(root, 'ornith');
    mkdirSync(renamed);
    const embedded = join(renamed, 'draft');
    writeDraft(embedded);
    expect(findDFlash2Draft(renamed, 'qwen3_5_moe')).toBe(embedded);
    expect(findDFlash2Draft(renamed, 'qwen4_exp')).toBeUndefined();
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

  it('finds the embedded draft beside a symlinked GGUF target', () => {
    // Discovery records the lexical link path; the companion lives beside the
    // real target, so modelDir must derive from the resolved path.
    const external = mkdtempSync(join(tmpdir(), 'mlx-draft-external-'));
    try {
      const realGguf = join(external, 'weights.gguf');
      writeFileSync(realGguf, 'weights');
      const embedded = join(external, 'draft');
      writeDraft(embedded);
      const link = join(root, 'Qwen3.8-27B-UD-Q4_K_XL.gguf');
      symlinkSync(realGguf, link);
      // The draft is found through the target's directory, so the returned
      // path is the target side's canonical spelling.
      expect(findDFlash2Draft(link, 'qwen3_5')).toBe(realpathSync(embedded));
    } finally {
      rmSync(external, { recursive: true, force: true });
    }
  });

  it('searches beside a symlinked model directory even with an explicit models dir', () => {
    // Production callers pass modelsDir — the symlink's store — while the
    // real checkpoint dir can live outside it. The companion beside the real
    // target is found through dirname(resolvedDir), which the explicit
    // modelsDir must not suppress.
    const external = mkdtempSync(join(tmpdir(), 'mlx-draft-external-dir-'));
    try {
      const realTarget = join(external, 'real-model');
      mkdirSync(realTarget);
      writeFileSync(join(realTarget, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }));
      writeDraft(join(external, 'Qwen3.8-27B-DFlash2'));
      const link = join(root, 'qwen3.8-27b-link');
      symlinkSync(realTarget, link);
      expect(findDFlash2Draft(link, 'qwen3_5', root)).toBe(join(realpathSync(external), 'Qwen3.8-27B-DFlash2'));
    } finally {
      rmSync(external, { recursive: true, force: true });
    }
  });

  it('searches the resolved repo parent for a symlinked GGUF even with an explicit models dir', () => {
    // The GGUF symlink lives in the store, its target in an external
    // repository directory (config.json sibling), and the shared companion
    // sits beside that repository. An explicit modelsDir names the store —
    // it must not suppress the resolved repo parent, matching the directory
    // target behavior above.
    const external = mkdtempSync(join(tmpdir(), 'mlx-draft-external-repo-'));
    try {
      const repo = join(external, 'qwen-repo');
      mkdirSync(repo);
      writeFileSync(join(repo, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }));
      const realGguf = join(repo, 'weights.gguf');
      writeFileSync(realGguf, 'weights');
      writeDraft(join(external, 'Qwen3.8-27B-DFlash2'));
      const link = join(root, 'Qwen3.8-27B-UD-Q4_K_XL.gguf');
      symlinkSync(realGguf, link);
      expect(findDFlash2Draft(link, 'qwen3_5', root)).toBe(join(realpathSync(external), 'Qwen3.8-27B-DFlash2'));
    } finally {
      rmSync(external, { recursive: true, force: true });
    }
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

type GgufTestValue =
  | { kind: 'string'; value: string }
  | { kind: 'u32'; value: number }
  | { kind: 'u32Array'; value: readonly number[] };

function ggufKv(entries: ReadonlyArray<readonly [string, GgufTestValue]>): Buffer {
  const u32 = (v: number): Buffer => {
    const b = Buffer.alloc(4);
    b.writeUInt32LE(v);
    return b;
  };
  const u64 = (v: number): Buffer => {
    const b = Buffer.alloc(8);
    b.writeBigUInt64LE(BigInt(v));
    return b;
  };
  const str = (text: string): Buffer => {
    const bytes = Buffer.from(text, 'utf8');
    return Buffer.concat([u64(bytes.length), bytes]);
  };
  return Buffer.concat([
    Buffer.from('GGUF'),
    u32(3),
    u64(0),
    u64(entries.length),
    ...entries.flatMap(([key, entry]) => {
      if (entry.kind === 'string') return [str(key), u32(8), str(entry.value)];
      if (entry.kind === 'u32') return [str(key), u32(4), u32(entry.value)];
      return [str(key), u32(9), u32(4), u64(entry.value.length), ...entry.value.map(u32)];
    }),
  ]);
}

function ggufStringKv(entries: ReadonlyArray<readonly [string, string]>): Buffer {
  return ggufKv(entries.map(([k, v]) => [k, { kind: 'string', value: v }] as const));
}

function validInkling2DSparkHeader(): Buffer {
  return ggufKv([
    ['general.architecture', { kind: 'string', value: 'dflash' }],
    ['dflash.dspark.markov_rank', { kind: 'u32', value: 256 }],
    ['tokenizer.ggml.pre', { kind: 'string', value: 'inkling2' }],
    ['dflash.target_layers', { kind: 'u32Array', value: [5, 23, 41, 48] }],
  ]);
}

describe('Inkling2 DSpark GGUF companion discovery', () => {
  let root: string;
  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'mlx-inkling2-dspark-'));
  });
  afterEach(() => rmSync(root, { recursive: true, force: true }));

  it('discovers a colocated dspark-*.gguf carrying authoritative Inkling2 DSpark metadata', () => {
    const targetGguf = join(root, 'Inkling-2-Mini-UD-Q4_K_XL.gguf');
    const dsparkGguf = join(root, 'dspark-Inkling-2-Mini-Q8_0.gguf');
    writeFileSync(targetGguf, ggufStringKv([['general.architecture', 'inkling2']]));
    writeFileSync(dsparkGguf, validInkling2DSparkHeader());

    expect(isInkling2DSparkGguf(dsparkGguf)).toBe(true);
    expect(findInkling2DSparkDraft(targetGguf)).toBe(dsparkGguf);
    expect(findInkling2DSparkDraft(root)).toBe(dsparkGguf);
    expect(findDFlash2Draft(targetGguf, 'inkling2')).toBe(dsparkGguf);
  });

  it('rejects non-dflash, missing-DSpark-metadata, wrong-pretokenizer, or corrupt GGUFs', () => {
    const targetGguf = join(root, 'Inkling-2-Mini-UD-Q4_K_XL.gguf');
    writeFileSync(targetGguf, ggufStringKv([['general.architecture', 'inkling2']]));

    const wrongArch = join(root, 'dspark-a-wrong-arch.gguf');
    writeFileSync(
      wrongArch,
      ggufKv([
        ['general.architecture', { kind: 'string', value: 'llama' }],
        ['dflash.dspark.markov_rank', { kind: 'u32', value: 256 }],
        ['tokenizer.ggml.pre', { kind: 'string', value: 'inkling2' }],
        ['dflash.target_layers', { kind: 'u32Array', value: [5, 23, 41, 48] }],
      ]),
    );
    // Name-only match without authoritative DSpark keys must NOT pass (would fail native load).
    const nameOnlyDflash = join(root, 'dspark-b-name-only.gguf');
    writeFileSync(
      nameOnlyDflash,
      ggufStringKv([
        ['general.architecture', 'dflash'],
        ['general.name', 'Inkling-2-Mini'],
      ]),
    );
    const missingTargetLayers = join(root, 'dspark-c-no-layers.gguf');
    writeFileSync(
      missingTargetLayers,
      ggufKv([
        ['general.architecture', { kind: 'string', value: 'dflash' }],
        ['dflash.dspark.markov_rank', { kind: 'u32', value: 256 }],
        ['tokenizer.ggml.pre', { kind: 'string', value: 'inkling2' }],
      ]),
    );
    const wrongPre = join(root, 'dspark-d-gemma4.gguf');
    writeFileSync(
      wrongPre,
      ggufKv([
        ['general.architecture', { kind: 'string', value: 'dflash' }],
        ['dflash.dspark.markov_rank', { kind: 'u32', value: 256 }],
        ['tokenizer.ggml.pre', { kind: 'string', value: 'gemma4' }],
        ['dflash.target_layers', { kind: 'u32Array', value: [5, 23, 41, 48] }],
      ]),
    );
    const corrupt = join(root, 'dspark-e-corrupt.gguf');
    writeFileSync(corrupt, 'not-a-gguf');

    expect(isInkling2DSparkGguf(wrongArch)).toBe(false);
    expect(isInkling2DSparkGguf(nameOnlyDflash)).toBe(false);
    expect(isInkling2DSparkGguf(missingTargetLayers)).toBe(false);
    expect(isInkling2DSparkGguf(wrongPre)).toBe(false);
    expect(isInkling2DSparkGguf(corrupt)).toBe(false);
    expect(findInkling2DSparkDraft(targetGguf)).toBeUndefined();
  });

  it('discovers dspark-Inkling-2-Mini-Q8_0.gguf beside a symlinked GGUF', () => {
    const external = mkdtempSync(join(tmpdir(), 'mlx-inkling2-external-'));
    try {
      const realTarget = join(external, 'Inkling-2-Mini-UD-Q4_K_XL.gguf');
      const realDraft = join(external, 'dspark-Inkling-2-Mini-Q8_0.gguf');
      writeFileSync(realTarget, ggufStringKv([['general.architecture', 'inkling2']]));
      writeFileSync(realDraft, validInkling2DSparkHeader());
      const link = join(root, 'Inkling-2-Mini-UD-Q4_K_XL.gguf');
      symlinkSync(realTarget, link);
      expect(findDFlash2Draft(link, 'inkling2')).toBe(realpathSync(realDraft));
    } finally {
      rmSync(external, { recursive: true, force: true });
    }
  });

  it('discovers the real on-disk dspark-Inkling-2-Mini-Q8_0.gguf when present', () => {
    const modelDir = join(homedir(), '.mlx-node', 'models', 'inkling-2-mini-gguf');
    const targetGguf = join(modelDir, 'Inkling-2-Mini-UD-Q4_K_XL.gguf');
    const expectedDraft = join(modelDir, 'dspark-Inkling-2-Mini-Q8_0.gguf');
    if (!existsSync(targetGguf) || !existsSync(expectedDraft)) return;
    expect(isInkling2DSparkGguf(expectedDraft)).toBe(true);
    expect(findDFlash2Draft(targetGguf, 'inkling2')).toBe(expectedDraft);
    expect(findInkling2DSparkDraft(modelDir)).toBe(expectedDraft);
  });
});
