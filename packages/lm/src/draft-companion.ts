/** Native-free metadata and discovery for optional speculative draft checkpoints. */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { basename, dirname, join } from 'node:path';

export const QWEN38_DFLASH2 = {
  label: 'DFlash2',
  hfRepo: 'z-lab/Qwen3.8-27B-DFlash2',
  sizeGb: 3.85,
} as const;

function regularFile(path: string): boolean {
  try {
    const stat = statSync(path);
    return stat.isFile() && stat.size > 0;
  } catch {
    return false;
  }
}

/** Draft-only directories must never appear in a chat-model picker. */
export function isDFlash2DraftDirectory(path: string): boolean {
  const configPath = join(path, 'config.json');
  if (!regularFile(configPath)) return false;
  try {
    const config = JSON.parse(readFileSync(configPath, 'utf8')) as { architectures?: unknown } | null;
    return Array.isArray(config?.architectures) && config.architectures.includes('DFlash2DraftModel');
  } catch {
    return false;
  }
}

/** Cheap completeness check; the native loader validates tensor shapes and target compatibility. */
export function isDFlash2Companion(path: string): boolean {
  return isDFlash2DraftDirectory(path) && regularFile(join(path, 'model.safetensors'));
}

/**
 * Prefer a target's explicit `draft/`, then the shared Qwen3.8-27B companion in
 * its models directory. Shared pairing requires the Qwen3.8-27B target name;
 * Qwen3.5, MoE models and unrelated renamed checkpoints must not acquire it.
 */
export function findDFlash2Draft(modelPath: string, modelType: string, modelsDir?: string): string | undefined {
  if (modelType !== 'qwen3_5') return undefined;
  const isGguf = modelPath.toLowerCase().endsWith('.gguf');
  const modelDir = isGguf ? dirname(modelPath) : modelPath;
  const embedded = join(modelDir, 'draft');
  if (isDFlash2Companion(embedded)) return embedded;

  const targetName = /(?:^|[-_.])qwen3[._-]8[-_.]27b(?:[-_.]|$)/i;
  if (!targetName.test(basename(modelPath)) && !(isGguf && targetName.test(basename(modelDir)))) return undefined;
  const slug = QWEN38_DFLASH2.hfRepo.split('/')[1].toLowerCase();
  const roots = modelsDir === undefined ? [isGguf ? modelDir : dirname(modelDir)] : [modelsDir];
  // A sibling config identifies a downloaded model repository. Its parent
  // can hold the shared companion; a standalone GGUF's containing directory
  // is already the model store, so do not search above it. An explicit store
  // always takes precedence over this repository inference.
  if (modelsDir === undefined && isGguf && regularFile(join(modelDir, 'config.json'))) {
    roots.push(dirname(modelDir));
  }
  for (const root of roots) {
    try {
      const candidates = readdirSync(root, { withFileTypes: true })
        .filter((entry) => entry.isDirectory() && entry.name.toLowerCase() === slug)
        .sort((a, b) => a.name.localeCompare(b.name));
      for (const candidate of candidates) {
        const path = join(root, candidate.name);
        if (isDFlash2Companion(path)) return path;
      }
    } catch {
      // An absent/unreadable companion leaves ordinary target loading available.
    }
  }
  return undefined;
}
