import { mkdtempSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { prepareDraftRelease, publishDesktopRelease } from '../scripts/release-publication.js';

const tag = 'v0.0.14';
const commit = 'a'.repeat(40);
const names = [
  'mlx-node-0.0.14-arm64.dmg',
  'mlx-node-0.0.14-darwin-arm64.zip',
  'mlx-node-0.0.14-darwin-arm64.zip.blockmap',
  'latest-mac.yml',
];
let root: string;
let files: string[];

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'mlx-release-publication-'));
  files = names.map((name) => {
    const file = join(root, name);
    writeFileSync(file, name);
    return file;
  });
});
afterEach(() => rmSync(root, { recursive: true, force: true }));

function harness() {
  const release = {
    tagName: tag,
    isDraft: true,
    isPrerelease: false,
    assets: [] as { name: string; state: string; size: number }[],
  };
  const ci = {
    id: 10,
    head_sha: commit,
    head_branch: 'main',
    event: 'push',
    path: '.github/workflows/ci.yml',
    status: 'completed',
    conclusion: 'success',
    html_url: 'https://github.com/mlx-node/mlx-node/actions/runs/10',
  };
  let exists = false;
  const onUpload = vi.fn();
  const github = vi.fn((args: string[]): string => {
    if (args[0] === 'api') return JSON.stringify({ workflow_runs: [ci] });
    if (args[1] === 'view') {
      if (!exists) throw new Error('release not found');
      return JSON.stringify(release);
    }
    if (args[1] === 'create') {
      exists = true;
      release.isDraft = args.includes('--draft');
      return '';
    }
    if (args[1] === 'upload') {
      for (const file of args.slice(3).filter((value) => value !== '--clobber')) {
        release.assets = release.assets.filter((asset) => asset.name !== basename(file));
        release.assets.push({ name: basename(file), state: 'uploaded', size: statSync(file).size });
      }
      onUpload();
      return '';
    }
    if (args[1] === 'edit') {
      release.isDraft = !args.includes('--draft=false');
      return '';
    }
    throw new Error(`Unexpected GitHub operation: ${args.join(' ')}`);
  });
  return { github, release, onUpload, ci };
}

describe('desktop release publication', () => {
  it('leaves a newly prepared release hidden if the build never reaches publication', () => {
    const { github, release } = harness();
    prepareDraftRelease(tag, github);
    expect(release.isDraft).toBe(true);
    expect(release.assets).toEqual([]);
    expect(github).toHaveBeenCalledWith([
      'release',
      'create',
      tag,
      '--draft',
      '--verify-tag',
      '--generate-notes',
      '--prerelease=false',
    ]);
  });

  it('publishes only after all four assets are uploaded and CI succeeds', async () => {
    const { github, release, onUpload } = harness();
    prepareDraftRelease(tag, github);
    onUpload.mockImplementation(() => expect(release.isDraft).toBe(true));
    await publishDesktopRelease(tag, commit, files, github);
    expect(release.assets.map((asset) => asset.name)).toEqual(names);
    expect(release.isDraft).toBe(false);
    expect(github.mock.calls.at(-1)?.[0]).toEqual(['release', 'edit', tag, '--draft=false', '--verify-tag']);
  });

  it.each(['upload throws', 'missing manifest', 'wrong size', 'unfinished upload'])(
    'keeps the draft hidden when %s, then allows a successful retry',
    async (failure) => {
      const { github, release, onUpload } = harness();
      prepareDraftRelease(tag, github);
      onUpload.mockImplementationOnce(() => {
        const manifest = release.assets.find((asset) => asset.name === 'latest-mac.yml')!;
        if (failure === 'upload throws') throw new Error('connection lost');
        if (failure === 'missing manifest') release.assets.pop();
        if (failure === 'wrong size') manifest.size++;
        if (failure === 'unfinished upload') manifest.state = 'starter';
      });
      await expect(publishDesktopRelease(tag, commit, files, github)).rejects.toThrow();
      expect(release.isDraft).toBe(true);
      expect(github.mock.calls.some(([args]) => args[1] === 'edit')).toBe(false);
      const creates = github.mock.calls.filter(([args]) => args[1] === 'create').length;
      prepareDraftRelease(tag, github);
      expect(github.mock.calls.filter(([args]) => args[1] === 'create')).toHaveLength(creates);
      await publishDesktopRelease(tag, commit, files, github);
      expect(release.isDraft).toBe(false);
    },
  );

  it('refuses to overwrite a published release', async () => {
    const { github, release } = harness();
    prepareDraftRelease(tag, github);
    release.isDraft = false;
    github.mockClear();
    expect(() => prepareDraftRelease(tag, github)).toThrow('must remain a draft');
    await expect(publishDesktopRelease(tag, commit, files, github)).rejects.toThrow('must remain a draft');
    expect(github.mock.calls.every(([args]) => args[1] === 'view')).toBe(true);
  });

  it.each(['missing file', 'empty file', 'wrong version'])(
    'rejects %s before any release mutation',
    async (failure) => {
      const { github } = harness();
      if (failure === 'missing file') files.pop();
      if (failure === 'empty file') writeFileSync(files[0], '');
      if (failure === 'wrong version') files[0] = files[0].replace('0.0.14', '0.0.13');
      await expect(publishDesktopRelease(tag, commit, files, github)).rejects.toThrow();
      expect(github).not.toHaveBeenCalled();
    },
  );

  it('keeps a fully uploaded release in draft when main CI fails', async () => {
    const { github, release, ci } = harness();
    prepareDraftRelease(tag, github);
    ci.conclusion = 'failure';
    await expect(publishDesktopRelease(tag, commit, files, github)).rejects.toThrow('Release blocked');
    expect(release.assets.map((asset) => asset.name)).toEqual(names);
    expect(release.isDraft).toBe(true);
    expect(github.mock.calls.some(([args]) => args[1] === 'edit')).toBe(false);
    ci.conclusion = 'success';
    await publishDesktopRelease(tag, commit, files, github);
    expect(release.isDraft).toBe(false);
  });

  it('requires prerelease tags to remain marked as prereleases', () => {
    const { github, release } = harness();
    prepareDraftRelease(tag, github);
    release.isPrerelease = true;
    expect(() => prepareDraftRelease(tag, github)).toThrow('incorrect prerelease setting');
  });
});
