import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import { parseTriple } from '@napi-rs/cli';

// Which platform targets the publish job may collect `.node` artifacts for.
//
// `napi.targets` in packages/core/package.json means what napi documents it to
// mean — "all targets the crate will be compiled for". Today CI publishes
// `darwin-arm64`, `linux-arm64-gnu`, and `win32-arm64-msvc`, while any future
// target kept private or without a `packages/core/npm/<abi>/package.json` is
// excluded automatically.
//
// @napi-rs/cli 3.8 made `napi artifacts` throw when a configured target has no
// `.node` in the artifacts dir, so the publish step can no longer be handed the
// full target list. Committing a second, CI-only target list would fix today's
// break and silently rot the day a platform is added, so the publishable set is
// derived from the two places that already declare it:
//
//   - `optionalDependencies` of `@mlx-node/core` — the platform packages an
//     install resolves the addon from;
//   - `packages/core/npm/<platformArchABI>/package.json` — the workspaces that
//     `yarn workspaces foreach npm publish` actually publishes.
//
// Both already use napi's own `<packageName>-<platformArchABI>` /
// `npm/<platformArchABI>/` naming (see the scoped package napi's own
// `create-npm-dirs` generates). The two must agree with each other and with
// `napi.targets`; every disagreement throws rather than quietly publishing a
// platform package with no addon inside it.

export interface PublishTargetInputs {
  /** `name` from packages/core/package.json. */
  packageName: string;
  /** `napi.targets` from packages/core/package.json. */
  configuredTargets: readonly string[];
  /** Keys of `optionalDependencies` from packages/core/package.json. */
  optionalDependencies: readonly string[];
  /** Directory names under `packages/core/npm/` that hold a `package.json`. */
  platformPackageDirs: readonly string[];
}

export function resolvePublishedTargets(inputs: PublishTargetInputs): string[] {
  const { packageName, configuredTargets, optionalDependencies, platformPackageDirs } = inputs;
  const prefix = `${packageName}-`;

  const declared = new Set(
    optionalDependencies.filter((dep) => dep.startsWith(prefix)).map((dep) => dep.slice(prefix.length)),
  );
  const onDisk = new Set(platformPackageDirs);

  const tripleByAbi = new Map<string, string>();
  const problems: string[] = [];
  for (const triple of configuredTargets) {
    const { platformArchABI } = parseTriple(triple);
    const previous = tripleByAbi.get(platformArchABI);
    if (previous !== undefined) {
      problems.push(
        `napi.targets lists both ${previous} and ${triple}, which produce the same ${platformArchABI} artifact — keep one spelling`,
      );
      continue;
    }
    tripleByAbi.set(platformArchABI, triple);
  }

  for (const abi of [...declared].sort()) {
    if (!tripleByAbi.has(abi)) {
      problems.push(`optionalDependencies declares ${prefix}${abi} but napi.targets has no target for ${abi}`);
    }
    if (!onDisk.has(abi)) {
      problems.push(
        `optionalDependencies declares ${prefix}${abi} but packages/core/npm/${abi}/package.json is missing`,
      );
    }
  }
  for (const abi of [...onDisk].sort()) {
    if (!declared.has(abi)) {
      problems.push(
        `packages/core/npm/${abi}/package.json exists but ${prefix}${abi} is not in the optionalDependencies of ${packageName}`,
      );
    }
  }
  if (declared.size === 0) {
    problems.push(
      `${packageName} declares no ${prefix}<platformArchABI> optional dependency, so there is no platform package to publish an addon into`,
    );
  }

  if (problems.length > 0) {
    throw new Error(
      `[publish-targets] the published platform packages of ${packageName} are inconsistent:\n` +
        problems.map((problem) => `  - ${problem}`).join('\n'),
    );
  }

  return configuredTargets.filter((triple) => declared.has(parseTriple(triple).platformArchABI));
}

// A directory under packages/core/npm/ is a published platform package only
// when it holds a package.json that is not private: a desktop-only from-source
// build lands its addon in npm/<triple>/ with no manifest, and a private one is
// skipped by `yarn workspaces foreach --no-private npm publish`.
export function listPublishedPlatformDirs(npmDir: string): string[] {
  const dirs: string[] = [];
  for (const entry of readdirSync(npmDir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const manifestPath = join(npmDir, entry.name, 'package.json');
    if (!existsSync(manifestPath)) continue;
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf-8')) as { private?: unknown };
    if (manifest.private === true) continue;
    dirs.push(entry.name);
  }
  return dirs.sort();
}
