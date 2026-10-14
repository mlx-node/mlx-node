/** Native-free metadata and discovery for optional speculative draft checkpoints. */
import { type Dirent, readdirSync, readFileSync, realpathSync, statSync } from 'node:fs';
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

/**
 * Sync counterpart of `resolveDirentKind` in `model-discovery.ts`: a Dirent
 * reports the link itself, never the target, so a symlinked companion directory
 * reads as neither file nor dir and would be silently skipped. `statSync`
 * follows the link and classifies what it points at; a dangling/unreadable link
 * resolves to `'other'` (a broken link admits nothing), and non-symlink entries
 * keep their `Dirent` answer unchanged.
 */
function direntKind(dir: string, entry: Dirent): 'file' | 'dir' | 'other' {
  if (!entry.isSymbolicLink()) {
    if (entry.isFile()) return 'file';
    if (entry.isDirectory()) return 'dir';
    return 'other';
  }
  try {
    const resolved = statSync(join(dir, entry.name));
    if (resolved.isFile()) return 'file';
    if (resolved.isDirectory()) return 'dir';
    return 'other';
  } catch {
    return 'other';
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
  // Discovery records the lexical symlink path, and the recorded model path
  // must stay lexical, so searches run against BOTH the lexical dir and the
  // symlink target's dir while return values keep the lexical spelling. A
  // companion beside the real GGUF is found either way; basename checks stay
  // lexical — a link named "qwen3.8-27b-*.gguf" is the caller's claim of
  // which family it wants, whatever the target file is called.
  let resolved = modelPath;
  try {
    resolved = realpathSync(modelPath);
  } catch {
    // Unresolvable path: fall back to lexical-only lookup rather than
    // disabling companion search.
  }
  const modelDir = isGguf ? dirname(modelPath) : modelPath;
  const resolvedDir = isGguf ? dirname(resolved) : resolved;
  const embedded = join(modelDir, 'draft');
  if (isDFlash2Companion(embedded)) return embedded;
  if (resolvedDir !== modelDir) {
    const resolvedEmbedded = join(resolvedDir, 'draft');
    if (isDFlash2Companion(resolvedEmbedded)) return resolvedEmbedded;
  }

  const targetName = /(?:^|[-_.])qwen3[._-]8[-_.]27b(?:[-_.]|$)/i;
  if (!targetName.test(basename(modelPath)) && !(isGguf && targetName.test(basename(modelDir)))) return undefined;
  const slug = QWEN38_DFLASH2.hfRepo.split('/')[1].toLowerCase();
  const roots = modelsDir === undefined ? [isGguf ? modelDir : dirname(modelDir)] : [modelsDir];
  // The linked target's own directory and its parent get the same searches —
  // a companion shipped beside the real file attaches identically to a
  // linked entry.
  if (resolvedDir !== modelDir) {
    roots.push(resolvedDir);
    if (!isGguf) {
      // A symlinked model DIRECTORY resolves to the target itself: the
      // companion beside the real checkpoint lives in the target's parent,
      // which an explicit modelsDir must not suppress — callers pass the
      // symlink's store while the real dir can live outside it.
      const resolvedParent = dirname(resolvedDir);
      if (!roots.includes(resolvedParent)) roots.push(resolvedParent);
    }
  }
  // A sibling config identifies a downloaded model repository. Its parent
  // can hold the shared companion; a standalone GGUF's containing directory
  // is already the model store, so do not search above it. An explicit store
  // always takes precedence over this repository inference.
  if (modelsDir === undefined && isGguf && regularFile(join(modelDir, 'config.json'))) {
    roots.push(dirname(modelDir));
  }
  if (modelsDir === undefined && isGguf && resolvedDir !== modelDir && regularFile(join(resolvedDir, 'config.json'))) {
    roots.push(dirname(resolvedDir));
  }
  for (const root of roots) {
    try {
      const candidates = readdirSync(root, { withFileTypes: true })
        .filter((entry) => direntKind(root, entry) === 'dir' && entry.name.toLowerCase() === slug)
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
