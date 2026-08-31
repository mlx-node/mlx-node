import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { NapiCli } from '@napi-rs/cli';

import { resolvePublishedTargets } from './publish-targets.js';

// CI's publish job runs this in place of a bare `napi artifacts`. It is the
// same `collectArtifacts` API the CLI calls — every guard it has (unexpected
// artifact, duplicate artifact, missing artifact, atomic write, stale-addon
// cleanup) still runs — but over the targets this repo actually publishes
// rather than every target the crate compiles for. See publish-targets.ts.

const packageRoot = dirname(fileURLToPath(import.meta.url));

interface CoreManifest {
  name?: unknown;
  optionalDependencies?: unknown;
  napi?: { targets?: unknown };
}

const manifest = JSON.parse(await readFile(join(packageRoot, 'package.json'), 'utf-8')) as CoreManifest;
if (typeof manifest.name !== 'string') {
  throw new Error('[artifacts.ts] packages/core/package.json has no "name"');
}
if (!Array.isArray(manifest.napi?.targets) || !manifest.napi.targets.every((t) => typeof t === 'string')) {
  throw new Error('[artifacts.ts] packages/core/package.json has no "napi.targets" string array');
}

const npmDir = join(packageRoot, 'npm');
const platformPackageDirs: string[] = [];
for (const entry of await readdir(npmDir, { withFileTypes: true })) {
  if (!entry.isDirectory()) continue;
  try {
    await readFile(join(npmDir, entry.name, 'package.json'), 'utf-8');
  } catch {
    // A from-source Linux build creates `npm/linux-arm64-gnu/` to land its
    // addon in (see copyNativeAddon in build.ts) without ever writing a
    // manifest there. Only a directory with a package.json is a publishable
    // platform package.
    continue;
  }
  platformPackageDirs.push(entry.name);
}

const targets = resolvePublishedTargets({
  packageName: manifest.name,
  configuredTargets: manifest.napi.targets as string[],
  optionalDependencies: Object.keys((manifest.optionalDependencies as Record<string, string> | undefined) ?? {}),
  platformPackageDirs,
});

console.log(`Collecting napi artifacts for the published targets: ${targets.join(', ')}`);
// readNapiConfig warns that the config file wins over the package.json `napi`
// field whenever both exist. That is the point: the override is targets-only,
// binaryName and packageName still come from package.json.
console.log('(the "NAPI-RS config file will be used" warning below is expected — it narrows targets only)');

const configDir = await mkdtemp(join(tmpdir(), 'mlx-core-napi-artifacts-'));
try {
  const configPath = join(configDir, 'napi.json');
  await writeFile(configPath, JSON.stringify({ targets }));
  await new NapiCli().artifacts({ cwd: packageRoot, configPath });
} finally {
  await rm(configDir, { recursive: true, force: true });
}
