/** Native-free metadata and discovery for optional speculative draft checkpoints. */
import {
  closeSync,
  constants,
  type Dirent,
  fstatSync,
  openSync,
  readdirSync,
  readFileSync,
  readSync,
  realpathSync,
  statSync,
} from 'node:fs';
import { basename, dirname, join } from 'node:path';

export const QWEN38_DFLASH2 = {
  label: 'DFlash2',
  hfRepo: 'z-lab/Qwen3.8-27B-DFlash2',
  sizeGb: 3.85,
} as const;

export const QWEN36_MOE_DFLASH2 = {
  label: 'DFlash2',
  hfRepo: 'incoai/Qwen3.6-35B-A3B-DFlash2',
  sizeGb: 1.1,
} as const;

export const INKLING2_DFLASH2 = {
  label: 'Inkling-2-Mini DFlash2',
  hfRepo: 'incoai/Inkling-2-Mini-DFlash2',
  sizeGb: 1.0,
} as const;

export type DraftCompanion = typeof QWEN38_DFLASH2 | typeof QWEN36_MOE_DFLASH2 | typeof INKLING2_DFLASH2;

const GGUF_SCALAR_BYTES = new Map<number, number>([
  [0, 1],
  [1, 1],
  [2, 2],
  [3, 2],
  [4, 4],
  [5, 4],
  [6, 4],
  [7, 1],
  [10, 8],
  [11, 8],
  [12, 8],
]);
const MAX_GGUF_STRING_BYTES = 16 * 1024 * 1024;
const MAX_GGUF_HEADER_KEYS = 16384;
const GGUF_WINDOW_BYTES = 64 * 1024;

interface GgufDraftMetadata {
  architecture?: string;
  hasMarkovRank?: boolean;
  tokenizerPre?: string;
  hasTargetLayers?: boolean;
}

/**
 * Synchronous, native-free GGUF v3+ metadata probe for speculative draft files
 * (e.g. `dspark-Inkling-2-Mini-Q8_0.gguf`). Skips arrays without allocating
 * them and closes its descriptor even on malformed headers.
 */
function readGgufDraftMetadataSync(path: string): GgufDraftMetadata | undefined {
  let fd: number;
  try {
    fd = openSync(path, constants.O_RDONLY | constants.O_NONBLOCK);
  } catch {
    return undefined;
  }
  try {
    const info = fstatSync(fd);
    if (!info.isFile() || info.size < 24) return undefined;
    const size = info.size;
    let position = 0;
    let window = Buffer.alloc(0);
    let windowStart = 0;

    const load = (pos: number, len: number): Buffer => {
      const end = pos + len;
      if (end > size) throw new Error('truncated');
      if (pos < windowStart || end > windowStart + window.length) {
        const toRead = Math.min(Math.max(GGUF_WINDOW_BYTES, len), size - pos);
        const buf = Buffer.allocUnsafe(toRead);
        let filled = 0;
        while (filled < toRead) {
          const n = readSync(fd, buf, filled, toRead - filled, pos + filled);
          if (n <= 0) break;
          filled += n;
        }
        if (filled < len) throw new Error('truncated');
        window = buf.subarray(0, filled);
        windowStart = pos;
      }
      return window.subarray(pos - windowStart, end - windowStart);
    };

    const skip = (len: number): void => {
      if (!Number.isSafeInteger(len) || len < 0 || len > size - position) throw new Error('bounds');
      position += len;
    };

    const bytes = (len: number): Buffer => {
      const pos = position;
      skip(len);
      if (len > MAX_GGUF_STRING_BYTES) throw new Error('oversized');
      return load(pos, len);
    };

    const u32 = (): number => bytes(4).readUInt32LE(0);
    const u64 = (): number => {
      const v = bytes(8).readBigUInt64LE(0);
      if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('oversized');
      return Number(v);
    };
    const str = (): string => {
      const len = u64();
      if (len > MAX_GGUF_STRING_BYTES) throw new Error('oversized');
      return bytes(len).toString('utf8');
    };
    const skipValue = (type: number, depth = 0): void => {
      const scalar = GGUF_SCALAR_BYTES.get(type);
      if (scalar !== undefined) return skip(scalar);
      if (type === 8) return skip(u64());
      if (type !== 9 || depth > 16) throw new Error('bad type');
      const elem = u32();
      const count = u64();
      const elemSize = GGUF_SCALAR_BYTES.get(elem);
      if (elemSize !== undefined) return skip(count * elemSize);
      if (elem !== 8 && elem !== 9) throw new Error('bad array elem');
      for (let i = 0; i < count; i++) skipValue(elem, depth + 1);
    };

    if (bytes(4).toString('ascii') !== 'GGUF') return undefined;
    if (u32() < 3) return undefined;
    u64(); // tensor count
    const kvCount = u64();
    if (kvCount > MAX_GGUF_HEADER_KEYS) return undefined;

    const readPositiveInt = (type: number): boolean => {
      switch (type) {
        case 0:
          return bytes(1).readUInt8(0) > 0;
        case 1:
          return bytes(1).readInt8(0) > 0;
        case 2:
          return bytes(2).readUInt16LE(0) > 0;
        case 3:
          return bytes(2).readInt16LE(0) > 0;
        case 4:
          return bytes(4).readUInt32LE(0) > 0;
        case 5:
          return bytes(4).readInt32LE(0) > 0;
        case 10:
          return bytes(8).readBigUInt64LE(0) > 0n;
        case 11:
          return bytes(8).readBigInt64LE(0) > 0n;
        default:
          skipValue(type);
          return false;
      }
    };

    const meta: GgufDraftMetadata = {};
    for (let i = 0; i < kvCount; i++) {
      const key = str();
      const type = u32();
      if (type === 8 && key === 'general.architecture') {
        meta.architecture = str();
        if (meta.architecture !== 'dflash') break;
      } else if (key === 'dflash.dspark.markov_rank') {
        meta.hasMarkovRank = readPositiveInt(type);
      } else if (type === 8 && key === 'tokenizer.ggml.pre') {
        meta.tokenizerPre = str();
      } else if (type === 9 && key === 'dflash.target_layers') {
        const elem = u32();
        const count = u64();
        const elemSize = GGUF_SCALAR_BYTES.get(elem);
        if (elemSize !== undefined) {
          skip(count * elemSize);
        } else if (elem === 8 || elem === 9) {
          for (let j = 0; j < count; j++) skipValue(elem, 1);
        } else {
          throw new Error('bad array elem');
        }
        meta.hasTargetLayers = (elem === 4 || elem === 5 || elem === 10) && count > 0;
      } else {
        skipValue(type);
      }
      if (
        meta.architecture === 'dflash' &&
        meta.hasMarkovRank === true &&
        meta.tokenizerPre === 'inkling2' &&
        meta.hasTargetLayers === true
      ) {
        break;
      }
    }
    return meta;
  } catch {
    return undefined;
  } finally {
    try {
      closeSync(fd);
    } catch {
      // ignore close errors
    }
  }
}

/**
 * True when `path` is a DSpark GGUF companion for an Inkling2 target:
 * matches the native Rust check (`general.architecture == "dflash"`,
 * positive `dflash.dspark.markov_rank`, `tokenizer.ggml.pre == "inkling2"`,
 * and a non-empty integer `dflash.target_layers` array).
 */
export function isInkling2DSparkGguf(path: string): boolean {
  if (!regularFile(path)) return false;
  const meta = readGgufDraftMetadataSync(path);
  return (
    meta?.architecture === 'dflash' &&
    meta.hasMarkovRank === true &&
    meta.tokenizerPre === 'inkling2' &&
    meta.hasTargetLayers === true
  );
}

/**
 * True when `path` is a colocated Inkling2 `mtp.safetensors` DSpark draft paired
 * with a sibling `config.json` declaring a valid `mtp_config` (positive
 * `markov_rank` and non-empty `target_layer_ids`).
 */
export function isInkling2MtpSafetensors(path: string): boolean {
  if (!regularFile(path)) return false;
  const configPath = join(dirname(path), 'config.json');
  if (!regularFile(configPath)) return false;
  try {
    const config = JSON.parse(readFileSync(configPath, 'utf8')) as {
      mtp_config?: {
        markov_rank?: unknown;
        target_layer_ids?: unknown;
      };
    } | null;
    const mtp = config?.mtp_config;
    return (
      typeof mtp?.markov_rank === 'number' &&
      Number.isInteger(mtp.markov_rank) &&
      mtp.markov_rank > 0 &&
      Array.isArray(mtp.target_layer_ids) &&
      mtp.target_layer_ids.length > 0 &&
      mtp.target_layer_ids.every((v) => typeof v === 'number' && Number.isInteger(v) && v >= 0)
    );
  } catch {
    return false;
  }
}

/**
 * Discover a colocated DSpark companion (`mtp.safetensors` paired with
 * `config.json` `mtp_config`, or `dspark-*.gguf` carrying `dflash` + Inkling2
 * metadata) inside an Inkling2 model's directory.
 */
export function findInkling2DSparkDraft(modelPath: string): string | undefined {
  const isGguf = modelPath.toLowerCase().endsWith('.gguf');
  let resolved = modelPath;
  try {
    resolved = realpathSync(modelPath);
  } catch {
    // Unresolvable path: fall back to lexical-only lookup.
  }
  const modelDir = isGguf ? dirname(modelPath) : modelPath;
  const resolvedDir = isGguf ? dirname(resolved) : resolved;
  const dirs = resolvedDir !== modelDir ? [modelDir, resolvedDir] : [modelDir];

  for (const dir of dirs) {
    if (!isGguf) {
      const mtpPath = join(dir, 'mtp.safetensors');
      if (isInkling2MtpSafetensors(mtpPath)) return mtpPath;
    }
    try {
      const candidates = readdirSync(dir, { withFileTypes: true })
        .filter((entry) => direntKind(dir, entry) === 'file' && /^dspark-.*\.gguf$/i.test(entry.name))
        .sort((a, b) => a.name.localeCompare(b.name));
      for (const candidate of candidates) {
        const candidatePath = join(dir, candidate.name);
        if (isInkling2DSparkGguf(candidatePath)) return candidatePath;
      }
    } catch {
      // An absent/unreadable directory leaves ordinary target loading available.
    }
    if (isGguf) {
      const mtpPath = join(dir, 'mtp.safetensors');
      if (isInkling2MtpSafetensors(mtpPath)) return mtpPath;
    }
  }
  return undefined;
}

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

function hasMoeDFlash2Geometry(path: string): boolean {
  try {
    const config = JSON.parse(readFileSync(join(path, 'config.json'), 'utf8')) as {
      hidden_size?: unknown;
      num_target_layers?: unknown;
    } | null;
    return config?.hidden_size === 2048 && config.num_target_layers === 40;
  } catch {
    return false;
  }
}

function hasInkling2DFlash2Geometry(path: string): boolean {
  try {
    const config = JSON.parse(readFileSync(join(path, 'config.json'), 'utf8')) as {
      hidden_size?: unknown;
      num_target_layers?: unknown;
    } | null;
    return config?.hidden_size === 2304 && config.num_target_layers === 48;
  } catch {
    return false;
  }
}

function isEmbeddedDFlash2Companion(path: string, requiredGeometry: 'none' | 'moe' | 'inkling2'): boolean {
  if (!isDFlash2Companion(path)) return false;
  if (requiredGeometry === 'moe') return hasMoeDFlash2Geometry(path);
  if (requiredGeometry === 'inkling2') return hasInkling2DFlash2Geometry(path);
  return true;
}

/**
 * Shared DFlash2 companion metadata for a released target model, or `undefined`
 * when `modelPath`/`modelType` does not name a released target with a shared
 * companion.
 */
export function draftCompanionForModel(modelPath: string, modelType: string): DraftCompanion | undefined {
  const isGguf = modelPath.toLowerCase().endsWith('.gguf');
  const modelDir = isGguf ? dirname(modelPath) : modelPath;
  const matchesTarget = (pattern: RegExp): boolean =>
    pattern.test(basename(modelPath)) || (isGguf && pattern.test(basename(modelDir)));

  if (modelType === 'qwen3_5') {
    return matchesTarget(/(?:^|[-_.])qwen3[._-]8[-_.]27b(?:[-_.]|$)/i) ? QWEN38_DFLASH2 : undefined;
  }
  if (modelType === 'qwen3_5_moe') {
    return matchesTarget(/(?:^|[-_.])qwen3[._-]6[-_.]35b[-_.]a3b(?:[-_.]|$)/i) ? QWEN36_MOE_DFLASH2 : undefined;
  }
  if (modelType === 'inkling2') {
    return matchesTarget(/(?:^|[-_.])inkling[-_.]?2[-_.]mini(?:[-_.]|$)/i) ? INKLING2_DFLASH2 : undefined;
  }
  return undefined;
}

/**
 * Prefer a target's explicit `draft/`, then its family's shared companion, and
 * for `inkling2` finally fall back to bundled DSpark (`mtp.safetensors` /
 * `dspark-*.gguf`). Shared pairing requires the released target name. Renamed
 * or fine-tuned targets can opt in with `draft/` or an explicit draftModelPath;
 * geometry compatibility alone does not establish a trained companion pairing.
 */
export function findDFlash2Draft(modelPath: string, modelType: string, modelsDir?: string): string | undefined {
  if (modelType !== 'qwen3_5' && modelType !== 'qwen3_5_moe' && modelType !== 'inkling2') return undefined;
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
  // A standalone GGUF's directory can be a shared store, so its generic
  // `draft/` has no target ownership. MoE and Inkling2 discovery admit only
  // their released geometry there; repository and directory drafts remain explicit.
  const embeddedRequirement = (dir: string): 'none' | 'moe' | 'inkling2' => {
    if (!isGguf || regularFile(join(dir, 'config.json'))) return 'none';
    if (modelType === 'qwen3_5_moe') return 'moe';
    if (modelType === 'inkling2') return 'inkling2';
    return 'none';
  };
  const embedded = join(modelDir, 'draft');
  if (isEmbeddedDFlash2Companion(embedded, embeddedRequirement(modelDir))) return embedded;
  if (resolvedDir !== modelDir) {
    const resolvedEmbedded = join(resolvedDir, 'draft');
    if (isEmbeddedDFlash2Companion(resolvedEmbedded, embeddedRequirement(resolvedDir))) return resolvedEmbedded;
  }

  const companion = draftCompanionForModel(modelPath, modelType);
  if (companion !== undefined) {
    const slug = companion.hfRepo.split('/')[1].toLowerCase();
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
    // The resolved repo parent is searched even with an explicit store: a
    // symlinked GGUF's target legitimately lives outside `modelsDir`, matching
    // the resolved-dir branch above for directory targets. The config.json
    // sibling check still confines the inference to real repositories.
    if (isGguf && resolvedDir !== modelDir && regularFile(join(resolvedDir, 'config.json'))) {
      const resolvedParent = dirname(resolvedDir);
      if (!roots.includes(resolvedParent)) roots.push(resolvedParent);
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
  }
  if (modelType === 'inkling2') return findInkling2DSparkDraft(modelPath);
  return undefined;
}

/**
 * Returns an auto-discovered Inkling-2-Mini DFlash2 companion ONLY when it
 * resides inside an external `modelsDir` outside the model's own directory and
 * parent directory (where native `Inkling2Model::load` already auto-discovers
 * `draft/`, sibling `inkling-2-mini-dflash2/`, and bundled DSpark with soft
 * fallback).
 */
export function findExternalInkling2DFlash2Draft(modelPath: string, modelsDir?: string): string | undefined {
  if (modelsDir === undefined) return undefined;
  const discovered = findDFlash2Draft(modelPath, 'inkling2', modelsDir);
  if (discovered === undefined || !isDFlash2Companion(discovered)) return undefined;
  const isGguf = modelPath.toLowerCase().endsWith('.gguf');
  const modelDir = isGguf ? dirname(modelPath) : modelPath;
  let resolved = modelPath;
  try {
    resolved = realpathSync(modelPath);
  } catch {
    // Keep lexical comparison when realpath is unavailable.
  }
  const resolvedDir = isGguf ? dirname(resolved) : resolved;
  let discoveredParent = dirname(discovered);
  try {
    discoveredParent = realpathSync(discoveredParent);
  } catch {
    // Keep lexical parent.
  }
  const nativeRoots = new Set<string>([modelDir, dirname(modelDir), resolvedDir, dirname(resolvedDir)]);
  for (const dir of Array.from(nativeRoots)) {
    try {
      nativeRoots.add(realpathSync(dir));
    } catch {
      // Ignore unresolvable paths.
    }
  }
  return nativeRoots.has(discoveredParent) ? undefined : discovered;
}
