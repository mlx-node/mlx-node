import { writeFileSync } from 'node:fs';
import { writeFile } from 'node:fs/promises';
import { basename, dirname, join } from 'node:path';

import { buildBlockMap } from 'app-builder-lib/out/targets/blockmap/blockmap.js';

import { assertSemver, versionFromUpdateZipPath } from './release-version.js';

export const UPDATE_MANIFEST_NAME = 'latest-mac.yml';

/** This file is part of the signed bundle, including its stable cache identity. */
export function writeUpdaterConfig(resources: string): void {
  writeFileSync(
    join(resources, 'app-update.yml'),
    'provider: github\nowner: mlx-node\nrepo: mlx-node\nupdaterCacheDirName: ai.mlxnode.desktop-updater\n',
  );
}

export function assertStagingPercentage(value: number): number {
  if (!Number.isInteger(value) || value < 0 || value > 100) {
    throw new Error('Update staging percentage must be an integer from 0 to 100');
  }
  return value;
}

/** Run on the final ZIP, after notarization/stapling and before publication. */
export async function createUpdateArtifacts(options: {
  zipPath: string;
  version: string;
  minimumSystemVersion: string;
  stagingPercentage?: number;
  releaseDate?: Date;
}): Promise<{ manifestPath: string; blockmapPath: string }> {
  const version = assertSemver('update version', options.version);
  if (versionFromUpdateZipPath(options.zipPath) !== version) {
    throw new Error('Update ZIP filename must match the bundled app version');
  }
  if (!/^\d+(?:\.\d+){0,2}$/.test(options.minimumSystemVersion)) {
    throw new Error('The update must declare a valid minimum macOS version');
  }
  const stagingPercentage = assertStagingPercentage(options.stagingPercentage ?? 100);
  const releaseDate = (options.releaseDate ?? new Date()).toISOString();
  const blockmapPath = `${options.zipPath}.blockmap`;
  // Use the upstream generator in sidecar mode: never append bytes to our ZIP.
  // It streams the archive and returns its size and SHA-512 alongside the map.
  const { size, sha512 } = await buildBlockMap(options.zipPath, 'gzip', blockmapPath);
  if (size === 0) throw new Error('Cannot publish an empty update ZIP');
  const file = JSON.stringify(basename(options.zipPath));
  const manifestPath = join(dirname(options.zipPath), UPDATE_MANIFEST_NAME);
  await writeFile(
    manifestPath,
    [
      `version: ${JSON.stringify(version)}`,
      'files:',
      `  - url: ${file}`,
      `    sha512: ${JSON.stringify(sha512)}`,
      `    size: ${size}`,
      `path: ${file}`,
      `sha512: ${JSON.stringify(sha512)}`,
      `releaseDate: ${JSON.stringify(releaseDate)}`,
      `minimumSystemVersion: ${JSON.stringify(options.minimumSystemVersion)}`,
      `stagingPercentage: ${stagingPercentage}`,
      '',
    ].join('\n'),
  );
  return { manifestPath, blockmapPath };
}
