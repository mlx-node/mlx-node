import { createHash, randomBytes } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gunzipSync } from 'node:zlib';

import { computeOperations, OperationKind } from 'electron-updater/out/differentialDownloader/downloadPlanBuilder.js';
import { parseUpdateInfo } from 'electron-updater/out/providers/Provider.js';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { assertStagingPercentage, createUpdateArtifacts, writeUpdaterConfig } from '../scripts/update-artifacts.js';

const roots: string[] = [];
function temporary(): string {
  const root = mkdtempSync(join(tmpdir(), 'mlx-update-artifacts-'));
  roots.push(root);
  return root;
}
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe('update release artifacts', () => {
  it('produces metadata and blockmaps the installed updater can consume without changing the archive', async () => {
    const root = temporary();
    const archive = randomBytes(256 * 1024);
    const zipPath = join(root, 'mlx-node-0.0.14-darwin-arm64.zip');
    writeFileSync(zipPath, archive);
    const { manifestPath, blockmapPath } = await createUpdateArtifacts({
      zipPath,
      version: '0.0.14',
      minimumSystemVersion: '26.0',
      stagingPercentage: 25,
      releaseDate: new Date('2026-09-08T00:00:00Z'),
    });
    // Parse with electron-updater itself, not a matching parser of our own.
    const info = parseUpdateInfo(
      readFileSync(manifestPath, 'utf-8'),
      'latest-mac.yml',
      new URL('https://example.test/'),
    );
    expect(info).toMatchObject({
      version: '0.0.14',
      minimumSystemVersion: '26.0',
      stagingPercentage: 25,
      releaseDate: '2026-09-08T00:00:00.000Z',
      files: [
        {
          url: 'mlx-node-0.0.14-darwin-arm64.zip',
          size: archive.length,
          sha512: createHash('sha512').update(archive).digest('base64'),
        },
      ],
    });
    expect(readFileSync(zipPath)).toEqual(archive);
    const map = JSON.parse(gunzipSync(readFileSync(blockmapPath)).toString()) as Parameters<
      typeof computeOperations
    >[0];
    expect(map.version).toBe('2');
    expect(map.files[0].sizes.reduce((sum, size) => sum + size, 0)).toBe(archive.length);
  });

  it('reconstructs an updated archive using the real updater download planner and reusable old bytes', async () => {
    const root = temporary();
    const previous = randomBytes(512 * 1024);
    // Insert bytes so fixed offsets cannot accidentally pass the compatibility check.
    const updated = Buffer.concat([
      previous.subarray(0, 100_000),
      Buffer.from('new release'),
      previous.subarray(100_000),
    ]);
    const maps: Parameters<typeof computeOperations>[0][] = [];
    for (const [version, bytes] of [
      ['0.0.13', previous],
      ['0.0.14', updated],
    ] as const) {
      const zipPath = join(root, `mlx-node-${version}-darwin-arm64.zip`);
      writeFileSync(zipPath, bytes);
      const result = await createUpdateArtifacts({ zipPath, version, minimumSystemVersion: '26.0' });
      maps.push(JSON.parse(gunzipSync(readFileSync(result.blockmapPath)).toString()));
    }
    const plan = computeOperations(maps[0], maps[1], { info: vi.fn(), warn: vi.fn(), error: vi.fn() });
    const reconstructed = Buffer.concat(
      plan.map((operation) =>
        (operation.kind === OperationKind.COPY ? previous : updated).subarray(operation.start, operation.end),
      ),
    );
    expect(reconstructed).toEqual(updated);
    expect(plan.some((operation) => operation.kind === OperationKind.COPY)).toBe(true);
    expect(plan.some((operation) => operation.kind === OperationKind.DOWNLOAD)).toBe(true);
    const transferred = plan
      .filter((operation) => operation.kind === OperationKind.DOWNLOAD)
      .reduce((sum, operation) => sum + operation.end - operation.start, 0);
    expect(transferred).toBeLessThan(updated.length / 2);
  });

  it('rejects mismatched versions and missing OS eligibility before generating release files', async () => {
    const root = temporary();
    const options = {
      zipPath: join(root, 'mlx-node-0.0.14-darwin-arm64.zip'),
      version: '0.0.13',
      minimumSystemVersion: '26.0',
    };
    await expect(createUpdateArtifacts(options)).rejects.toThrow('filename must match');
    await expect(createUpdateArtifacts({ ...options, version: '0.0.14', minimumSystemVersion: '' })).rejects.toThrow(
      'minimum macOS',
    );
  });

  it.each([-1, 101, 1.5, Number.NaN, Infinity])('rejects invalid rollout percentage %s', (value) => {
    expect(() => assertStagingPercentage(value)).toThrow();
  });

  it('supports a paused rollout and full rollout', () => {
    expect(assertStagingPercentage(0)).toBe(0);
    expect(assertStagingPercentage(100)).toBe(100);
  });

  it('writes the production provider and a stable per-app cache identity outside the app directory', () => {
    const root = temporary();
    writeUpdaterConfig(root);
    expect(readFileSync(join(root, 'app-update.yml'), 'utf-8')).toBe(
      'provider: github\nowner: mlx-node\nrepo: mlx-node\nupdaterCacheDirName: ai.mlxnode.desktop-updater\n',
    );
  });
});
