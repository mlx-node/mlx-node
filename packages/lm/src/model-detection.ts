/** Shared filesystem-only model detection for inference and control-panel discovery. */
import { constants } from 'node:fs';
import { open } from 'node:fs/promises';
import { dirname, extname, join } from 'node:path';

import { MODEL_FAMILY_DATA, matchFamily, type ModelType } from './family-data.js';
import { readGgufArchitecture } from './gguf-metadata.js';

export { readGgufArchitecture } from './gguf-metadata.js';

const GGUF_ARCHITECTURE_MODEL_TYPES = new Map<string, ModelType>(
  MODEL_FAMILY_DATA.flatMap((row) =>
    'ggufArchitectures' in row ? row.ggufArchitectures.map((architecture) => [architecture, row.id] as const) : [],
  ),
);

export async function readModelConfig(modelDir: string): Promise<unknown> {
  const file = await open(join(modelDir, 'config.json'), constants.O_RDONLY | constants.O_NONBLOCK);
  try {
    if (!(await file.stat()).isFile()) throw new Error('Model config must be a regular file');
    return JSON.parse(await file.readFile('utf8'));
  } finally {
    await file.close();
  }
}

/** The loader can supply its native header validator without changing family selection. */
export async function detectModelType(
  modelPath: string,
  readArchitecture: (path: string) => string | Promise<string> = readGgufArchitecture,
): Promise<ModelType> {
  const isGguf = extname(modelPath).toLowerCase() === '.gguf';
  let config: unknown;
  try {
    config = await readModelConfig(isGguf ? dirname(modelPath) : modelPath);
  } catch (error) {
    if (isGguf && (error as NodeJS.ErrnoException).code === 'ENOENT') {
      const architecture = await readArchitecture(modelPath);
      const type = GGUF_ARCHITECTURE_MODEL_TYPES.get(architecture);
      if (type === undefined) throw new Error(`Unsupported GGUF architecture "${architecture}" in ${modelPath}`);
      return type;
    }
    // `cause` preserves the real errno: an EACCES here is "couldn't look",
    // not "not found" — discovery's onEntryFailure distinguishes the two.
    throw new Error(`Cannot detect model type: config.json not found in ${modelPath}`, { cause: error });
  }
  return matchFamily(modelPath, config);
}
