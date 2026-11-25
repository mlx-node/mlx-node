#!/usr/bin/env node

import '@oxc-node/core/register';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

import { Configuration, Project, structUtils, Workspace } from '@yarnpkg/core';
import { npath } from '@yarnpkg/fslib';
import semver, { type ReleaseType } from 'semver';

const __dirname = npath.dirname(fileURLToPath(import.meta.url));
const projectRoot = npath.join(__dirname, '..', '..');
const PRIVATE_VERSIONED_WORKSPACES = new Set(['@mlx-node/desktop', '@mlx-node/dashboard']);

const args = parseArgs({
  options: {
    help: {
      type: 'boolean',
      default: false,
    },
  },
  allowPositionals: true,
});

const subcommand = args.positionals[0];

if (args.values.help || !subcommand?.length) {
  console.log('Usage: tools bump <major|minor|patch>');
  process.exit(0);
}

const semverKinds = new Set(semver.RELEASE_TYPES);

switch (subcommand) {
  case 'bump':
    const kind = args.positionals[1] as ReleaseType;
    if (!kind) {
      throw new Error('tools bump requires a kind, pass major, minor, or patch');
    }
    if (!semverKinds.has(kind)) {
      throw new Error(`Unknown kind: ${kind}, pass major, minor, or patch`);
    }
    const currentVersion = await import('../core/package.json', { with: { type: 'json' } }).then(
      (m) => m.default.version,
    );
    const newVersion = semver.inc(currentVersion, kind);
    if (!newVersion) {
      throw new Error(`Failed to bump ${kind} version from ${currentVersion}`);
    }
    console.log(`Bumping ${kind} version from ${currentVersion} to ${newVersion}`);
    const { project } = await Project.find(
      await Configuration.find(npath.toPortablePath(projectRoot), null, { strict: false }),
      npath.toPortablePath(projectRoot),
    );
    const workspaces = project.workspaces.filter(
      (pkg) =>
        pkg.manifest.private !== true ||
        (pkg.manifest.name != null && PRIVATE_VERSIONED_WORKSPACES.has(structUtils.stringifyIdent(pkg.manifest.name))),
    );
    await bumpVersion(newVersion, workspaces);
    await import('../core/build');
    execSync(`git add .`, {
      stdio: 'inherit',
      encoding: 'utf-8',
    });
    execSync(`git commit -m "${newVersion}"`, {
      stdio: 'inherit',
      encoding: 'utf-8',
    });
    execSync(`git tag -s v${newVersion} -m v${newVersion}`, {
      stdio: 'inherit',
      encoding: 'utf-8',
    });
    break;
  default:
    console.error(`Unknown subcommand: ${subcommand}`);
    process.exit(1);
}

async function bumpVersion(newVersion: string, workspaces: Workspace[]) {
  for (const workspace of workspaces) {
    workspace.manifest.version = newVersion;
    await workspace.persistManifest();
  }
}
