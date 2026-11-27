import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, it, expect } from 'vite-plus/test';

import { listPublishedPlatformDirs, resolvePublishedTargets } from '../../packages/core/publish-targets';

const CORE = 'packages/core';
const NAME = '@mlx-node/core';

// Minimal fixture: napi.targets carries an extra unpublished target alongside
// darwin-arm64, while only darwin-arm64 is declared as a published package.
const REPO_SHAPE = {
  packageName: NAME,
  configuredTargets: ['aarch64-apple-darwin', 'aarch64-unknown-linux-gnu'],
  optionalDependencies: [`${NAME}-darwin-arm64`],
  platformPackageDirs: ['darwin-arm64'],
};

describe('resolvePublishedTargets', () => {
  it('keeps an unpublished target out of the publish job', () => {
    expect(resolvePublishedTargets(REPO_SHAPE)).toEqual(['aarch64-apple-darwin']);
  });

  it('matches the real packages/core manifest', () => {
    const manifest = JSON.parse(readFileSync(join(CORE, 'package.json'), 'utf-8'));
    const platformPackageDirs = listPublishedPlatformDirs(join(CORE, 'npm'));

    expect(manifest.napi.targets).toContain('aarch64-pc-windows-msvc');
    expect(
      resolvePublishedTargets({
        packageName: manifest.name,
        configuredTargets: manifest.napi.targets,
        optionalDependencies: Object.keys(manifest.optionalDependencies ?? {}),
        platformPackageDirs,
      }),
    ).toEqual(['aarch64-apple-darwin']);
  });

  it('counts only platform folders with a package.json that is not private', () => {
    const npmDir = mkdtempSync(join(tmpdir(), 'publish-targets-'));
    try {
      mkdirSync(join(npmDir, 'darwin-arm64'));
      writeFileSync(join(npmDir, 'darwin-arm64', 'package.json'), JSON.stringify({ name: `${NAME}-darwin-arm64` }));
      mkdirSync(join(npmDir, 'linux-arm64-gnu'));
      writeFileSync(
        join(npmDir, 'linux-arm64-gnu', 'package.json'),
        JSON.stringify({ name: `${NAME}-linux-arm64-gnu`, private: true }),
      );
      mkdirSync(join(npmDir, 'win32-arm64-msvc'));
      expect(listPublishedPlatformDirs(npmDir)).toEqual(['darwin-arm64']);
    } finally {
      rmSync(npmDir, { recursive: true, force: true });
    }
  });

  // The whole point of deriving rather than committing a second target list:
  // publishing a new platform must reach the publish job on its own.
  it('picks up a newly published platform, in napi.targets order', () => {
    expect(
      resolvePublishedTargets({
        ...REPO_SHAPE,
        optionalDependencies: [`${NAME}-linux-arm64-gnu`, `${NAME}-darwin-arm64`],
        platformPackageDirs: ['linux-arm64-gnu', 'darwin-arm64'],
      }),
    ).toEqual(['aarch64-apple-darwin', 'aarch64-unknown-linux-gnu']);
  });

  it('ignores optional dependencies that are not platform packages', () => {
    expect(
      resolvePublishedTargets({ ...REPO_SHAPE, optionalDependencies: [`${NAME}-darwin-arm64`, 'fsevents'] }),
    ).toEqual(['aarch64-apple-darwin']);
  });

  it('rejects an optional platform package that is not a configured napi target', () => {
    expect(() =>
      resolvePublishedTargets({
        ...REPO_SHAPE,
        configuredTargets: ['aarch64-apple-darwin'],
        optionalDependencies: [`${NAME}-darwin-arm64`, `${NAME}-linux-arm64-gnu`],
        platformPackageDirs: ['darwin-arm64', 'linux-arm64-gnu'],
      }),
    ).toThrow(/napi\.targets has no target for linux-arm64-gnu/);
  });

  it('rejects an optional platform package with no npm/<abi>/package.json', () => {
    expect(() =>
      resolvePublishedTargets({
        ...REPO_SHAPE,
        optionalDependencies: [`${NAME}-darwin-arm64`, `${NAME}-linux-arm64-gnu`],
      }),
    ).toThrow(/packages\/core\/npm\/linux-arm64-gnu\/package\.json is missing/);
  });

  it('rejects an npm/<abi> package that is not in optionalDependencies', () => {
    expect(() =>
      resolvePublishedTargets({ ...REPO_SHAPE, platformPackageDirs: ['darwin-arm64', 'linux-arm64-gnu'] }),
    ).toThrow(/is not in the optionalDependencies/);
  });

  it('refuses to publish nothing', () => {
    expect(() => resolvePublishedTargets({ ...REPO_SHAPE, optionalDependencies: [], platformPackageDirs: [] })).toThrow(
      /no platform package to publish an addon into/,
    );
  });

  it('rejects two napi.targets spellings of one artifact set', () => {
    expect(() =>
      resolvePublishedTargets({ ...REPO_SHAPE, configuredTargets: ['aarch64-apple-darwin', 'arm64-apple-darwin'] }),
    ).toThrow(/produce the same darwin-arm64 artifact/);
  });
});
