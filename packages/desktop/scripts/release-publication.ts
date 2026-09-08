import { statSync } from 'node:fs';
import { basename } from 'node:path';

import { waitForReleaseCI } from './release-ci.js';
import { dmgFileName, updateZipFileName, versionFromTag } from './release-version.js';

type GitHub = (args: string[]) => string;
interface Release {
  tagName: string;
  isDraft: boolean;
  isPrerelease: boolean;
  assets: { name: string; state: string; size: number }[];
}

function readRelease(tag: string, github: GitHub): Release {
  return JSON.parse(github(['release', 'view', tag, '--json', 'tagName,isDraft,isPrerelease,assets'])) as Release;
}

function assertDraft(tag: string, release: Release): void {
  if (release.tagName !== tag || release.isDraft !== true) {
    throw new Error(
      `${tag} must remain a draft until all desktop artifacts are uploaded; refusing to modify a public release`,
    );
  }
  if (release.isPrerelease !== versionFromTag(tag).includes('-')) {
    throw new Error(`${tag} has an incorrect prerelease setting`);
  }
}

/** A failed build leaves this draft hidden from the public GitHub update feed. */
export function prepareDraftRelease(tag: string, github: GitHub): void {
  const version = versionFromTag(tag);
  let release: Release;
  try {
    release = readRelease(tag, github);
  } catch {
    // Creation is draft-only and verifies the remote tag. A duplicate tag,
    // authentication failure, or network failure still aborts publication.
    github([
      'release',
      'create',
      tag,
      '--draft',
      '--verify-tag',
      '--generate-notes',
      `--prerelease=${version.includes('-')}`,
    ]);
    release = readRelease(tag, github);
  }
  assertDraft(tag, release);
}

/** Upload and verify the complete asset set while private, then publish once. */
export async function publishDesktopRelease(
  tag: string,
  commit: string,
  files: string[],
  github: GitHub,
): Promise<void> {
  const version = versionFromTag(tag);
  const zip = updateZipFileName(version);
  const expected = [dmgFileName(version), zip, `${zip}.blockmap`, 'latest-mac.yml'];
  if (
    files.length !== expected.length ||
    expected.some((name) => files.filter((file) => basename(file) === name).length !== 1)
  ) {
    throw new Error('Publication requires the matching DMG, update ZIP, blockmap, and latest-mac.yml');
  }
  const assets = files.map((file) => {
    const stat = statSync(file);
    if (!stat.isFile() || stat.size === 0) throw new Error(`Missing or empty release artifact: ${file}`);
    return { name: basename(file), size: stat.size };
  });
  assertDraft(tag, readRelease(tag, github));
  github(['release', 'upload', tag, ...files, '--clobber']);
  // The tag build can overlap main CI, but the public update feed cannot.
  // Check after uploads, then re-read the draft/assets after any CI wait.
  await waitForReleaseCI(commit, github);
  // Re-read after upload. A failed/partial upload must never expose a release
  // lacking latest-mac.yml, and reruns may only replace files on a draft.
  const release = readRelease(tag, github);
  assertDraft(tag, release);
  for (const expected of assets) {
    const matches = release.assets.filter((asset) => asset.name === expected.name);
    if (matches.length !== 1 || matches[0].state !== 'uploaded' || matches[0].size !== expected.size) {
      throw new Error(`Release artifact is not fully uploaded: ${expected.name}`);
    }
  }
  github(['release', 'edit', tag, '--draft=false', '--verify-tag']);
}
