/** Shared filesystem-only model detection for inference and control-panel discovery. */
import { constants } from 'node:fs';
import { open, stat } from 'node:fs/promises';
import { dirname, extname, join } from 'node:path';

import { MODEL_FAMILY_DATA, matchFamily, type ModelType } from './family-data.js';
import { readGgufArchitecture } from './gguf-metadata.js';

export { readGgufArchitecture } from './gguf-metadata.js';

const GGUF_ARCHITECTURE_MODEL_TYPES = new Map<string, ModelType>(
  MODEL_FAMILY_DATA.flatMap((row) =>
    'ggufArchitectures' in row ? row.ggufArchitectures.map((architecture) => [architecture, row.id] as const) : [],
  ),
);

/** The family whose `ggufArchitectures` holds a GGUF header's `general.architecture`, or undefined. */
export function ggufArchitectureModelType(architecture: string): ModelType | undefined {
  return GGUF_ARCHITECTURE_MODEL_TYPES.get(architecture);
}

export async function readModelConfig(modelDir: string): Promise<unknown> {
  const file = await open(join(modelDir, 'config.json'), constants.O_RDONLY | constants.O_NONBLOCK);
  try {
    if (!(await file.stat()).isFile()) throw new Error('Model config must be a regular file');
    return JSON.parse(await file.readFile('utf8'));
  } finally {
    await file.close();
  }
}

async function detectQwen35Sidecar(sidecarDir: string, allowClef: boolean): Promise<'clef' | 'clairic' | undefined> {
  const checkSidecar = async (name: string, label: string): Promise<boolean> => {
    try {
      if (!(await stat(join(sidecarDir, name))).isFile()) {
        throw new Error(`Invalid ${label} checkpoint: ${name} must be a regular file`);
      }
      return true;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return false;
      throw error;
    }
  };
  const [clefAssets, clairicAssets] = await Promise.all([
    allowClef
      ? Promise.all(['joint_head_config.json', 'joint_head.safetensors'].map((name) => checkSidecar(name, 'CLEF')))
      : Promise.resolve([false, false]),
    Promise.all(['clair_head_config.json', 'clair_head.safetensors'].map((name) => checkSidecar(name, 'Clairic'))),
  ]);
  const hasAnyClef = clefAssets.some(Boolean);
  const hasAnyClairic = clairicAssets.some(Boolean);
  if (hasAnyClef && hasAnyClairic) {
    throw new Error('Ambiguous checkpoint: both CLEF and Clairic head assets found');
  }
  if (hasAnyClef) {
    if (!clefAssets.every(Boolean)) {
      throw new Error('Incomplete CLEF checkpoint: both joint head assets are required');
    }
    return 'clef';
  }
  if (hasAnyClairic) {
    if (!clairicAssets.every(Boolean)) {
      throw new Error(
        'Incomplete Clairic checkpoint: both clair_head_config.json and clair_head.safetensors are required',
      );
    }
    return 'clairic';
  }
  return undefined;
}

/** The loader can supply its native header validator without changing family selection. */
export async function detectModelType(
  modelPath: string,
  readArchitecture: (path: string) => string | Promise<string> = readGgufArchitecture,
): Promise<ModelType> {
  const isGguf = extname(modelPath).toLowerCase() === '.gguf';
  const sidecarDir = isGguf ? dirname(modelPath) : modelPath;
  // A sibling config.json stays authoritative: discovery cross-checks it
  // against the GGUF header and rejects files that disagree. The header is
  // only consulted when no config exists (bare GGUF loads, staged clef
  // decision checkpoints reconstruct their own sidecars anyway).
  let config: unknown;
  try {
    config = await readModelConfig(sidecarDir);
  } catch (error) {
    if (isGguf && (error as NodeJS.ErrnoException).code === 'ENOENT') {
      const architecture = await readArchitecture(modelPath);
      const type = ggufArchitectureModelType(architecture);
      if (type === undefined) throw new Error(`Unsupported GGUF architecture "${architecture}" in ${modelPath}`);
      if (type === 'qwen3_5') {
        const promoted = await detectQwen35Sidecar(sidecarDir, false);
        if (promoted) return promoted;
      }
      return type;
    }
    // `cause` preserves the real errno: an EACCES here is "couldn't look",
    // not "not found" — discovery's onEntryFailure distinguishes the two.
    throw new Error(`Cannot detect model type: config.json not found in ${modelPath}`, { cause: error });
  }
  const family = matchFamily(modelPath, config);
  if (family === 'qwen3_5') {
    const promoted = await detectQwen35Sidecar(sidecarDir, !isGguf);
    if (promoted) return promoted;
  }
  return family;
}
