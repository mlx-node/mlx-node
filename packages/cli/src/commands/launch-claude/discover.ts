/** Discover locally-downloaded generative models under a given directory. */

import type { Dirent } from 'node:fs';
import { readdir } from 'node:fs/promises';
import { basename, join } from 'node:path';

import { detectModelType, type ModelType } from '@mlx-node/lm';
import { LAUNCH_PRESETS, type LaunchPreset } from '@mlx-node/server';

/** A locally-downloaded model paired with its sampling preset. */
export interface DiscoveredModel {
  name: string;
  path: string;
  modelType: ModelType;
  preset: LaunchPreset;
}

// Non-generative detection results that cannot back a chat endpoint.
const NON_GENERATIVE: ReadonlySet<ModelType> = new Set<ModelType>(['harrier', 'qianfan-ocr', 'internvl_chat']);

/**
 * Scan `dir` for model subdirectories. Each subdirectory with a recognized
 * `config.json` is returned with its inferred `ModelType` and `LaunchPreset`.
 * Non-generative types are silently skipped. Entries with no preset or an
 * undetectable config are skipped (warnings only emitted when `MLX_DEBUG`
 * is set). Must stay cheap — do not load weights here.
 */
export async function discoverModels(dir: string): Promise<DiscoveredModel[]> {
  const debug = Boolean(process.env.MLX_DEBUG);

  let entries: Dirent[];
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return [];
  }

  const out: DiscoveredModel[] = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const full = join(dir, entry.name);

    let modelType: ModelType;
    try {
      modelType = await detectModelType(full);
    } catch (err) {
      if (debug) console.warn(`[mlx] skip ${full}: ${(err as Error).message}`);
      continue;
    }

    if (NON_GENERATIVE.has(modelType)) continue;

    const preset = LAUNCH_PRESETS[modelType];
    if (!preset) {
      if (debug) console.warn(`[mlx] skip ${full}: no LAUNCH_PRESETS entry for ${modelType}`);
      continue;
    }

    out.push({ name: basename(full), path: full, modelType, preset });
  }

  out.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  return out;
}
