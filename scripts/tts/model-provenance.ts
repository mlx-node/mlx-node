import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

export interface ModelProvenance {
  repo: string;
  revision: string;
}

/** Prefer explicitly recorded source provenance; ordinary downloads use the
 * CLI/dashboard marker. A checkpoint without either has unknown provenance. */
export async function readModelProvenance(directory: string): Promise<ModelProvenance | null> {
  for (const name of ['reference-revision.json', '.mlx-download-complete.json']) {
    let raw: string;
    try {
      raw = await readFile(join(directory, name), 'utf8');
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') continue;
      throw error;
    }
    const value: unknown = JSON.parse(raw);
    if (
      !value ||
      typeof value !== 'object' ||
      !('repo' in value) ||
      typeof value.repo !== 'string' ||
      !value.repo.trim() ||
      !('revision' in value) ||
      typeof value.revision !== 'string' ||
      !value.revision.trim()
    )
      throw new Error(`Invalid model provenance in ${name}: expected repo and revision strings`);
    return { repo: value.repo, revision: value.revision };
  }
  return null;
}
