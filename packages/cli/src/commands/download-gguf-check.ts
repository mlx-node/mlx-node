import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, join } from 'node:path';

import { downloadFile } from '@huggingface/hub';
import { readGgufHeader } from '@mlx-node/lm/gguf-metadata';
import { ggufSplitPart, isGgufCompanionName } from '@mlx-node/lm/model-discovery';

/**
 * Splash `install/upstream.py` `gguf_target` + `model-check`: before any
 * weight byte is fetched, range-read the header of every selected target GGUF
 * (a split GGUF is one target of all its parts), write the headers to a
 * scratch target directory and run the native header check over it. Throws a
 * refusal naming the target; the native addon is loaded only when a GGUF
 * target is selected.
 */
export async function checkGgufTargets(opts: {
  repo: string;
  revision?: string;
  paths: readonly string[];
  accessToken?: string;
}): Promise<{ targets: string[][]; headerBytes: number }> {
  const groups = new Map<string, { stem: string; parts: Map<number, string>; totals: Set<number> }>();
  for (const path of opts.paths) {
    const name = basename(path);
    if (!name.toLowerCase().endsWith('.gguf') || isGgufCompanionName(name)) continue;
    const split = ggufSplitPart(name);
    const key = `${dirname(path)}\0${split ? `split\0${split.stem}` : `file\0${name}`}`;
    const group = groups.get(key) ?? { stem: split?.stem ?? name, parts: new Map(), totals: new Set() };
    group.parts.set(split?.part ?? 1, path);
    group.totals.add(split?.total ?? 1);
    groups.set(key, group);
  }
  const targets: string[][] = [];
  for (const { stem, parts, totals } of groups.values()) {
    const total = Math.max(...totals);
    const numbers = [...parts.keys()].sort((a, b) => a - b);
    if (totals.size !== 1 || numbers.length !== total || numbers.some((n, i) => n !== i + 1)) {
      throw new Error(
        `split GGUF ${stem}: the selection holds parts ${numbers.join(', ')} of ${total}; select every part`,
      );
    }
    targets.push(numbers.map((n) => parts.get(n) as string));
  }
  if (targets.length === 0) return { targets, headerBytes: 0 };

  const { checkGgufHeaders } = await import('@mlx-node/core');
  const repo = { type: 'model' as const, name: opts.repo };
  const fetch = async (path: string): Promise<Blob> => {
    const blob = await downloadFile({ repo, path, revision: opts.revision, accessToken: opts.accessToken });
    if (!blob) throw new Error(`${path} disappeared from ${opts.repo}`);
    return blob;
  };
  let headerBytes = 0;
  for (const target of targets) {
    const scratch = await mkdtemp(join(tmpdir(), 'mlx-gguf-check-'));
    try {
      const dir = join(scratch, 'target');
      await mkdir(dir);
      for (const path of target) {
        const blob = await fetch(path);
        const header = await readGgufHeader(
          async (position, length) => new Uint8Array(await blob.slice(position, position + length).arrayBuffer()),
          blob.size,
        );
        headerBytes += header.length;
        await writeFile(join(dir, basename(path)), header);
      }
      const folder = dirname(target[0]);
      const config = folder === '.' ? 'config.json' : `${folder}/config.json`;
      if (opts.paths.includes(config)) {
        await writeFile(join(dir, 'config.json'), new Uint8Array(await (await fetch(config)).arrayBuffer()));
      }
      try {
        checkGgufHeaders(dir);
      } catch (error) {
        throw new Error(
          `Refusing to download ${target.join(', ')}: ${error instanceof Error ? error.message : String(error)}. No weight was fetched.`,
        );
      }
    } finally {
      await rm(scratch, { recursive: true, force: true });
    }
  }
  return { targets, headerBytes };
}
