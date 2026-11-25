/** `findOnPath`: an executable regular file in an absolute `PATH` entry; nothing is run and no shell is used. */

import { chmodSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { findOnPath, readPathEnv } from '../src/chat/skills/cli-check.js';

let root: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'mlx-cli-check-'));
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

function dir(name: string): string {
  const path = join(root, name);
  mkdirSync(path, { recursive: true });
  return path;
}

function file(path: string, mode: number): void {
  writeFileSync(path, '#!/bin/sh\nexit 0\n');
  chmodSync(path, mode);
}

describe('findOnPath', () => {
  it('finds the first executable regular file, skipping a plain file, a folder and relative entries', () => {
    const plain = dir('plain');
    file(join(plain, 'firecrawl'), 0o644);
    const folder = dir('folder');
    mkdirSync(join(folder, 'firecrawl'));
    const good = dir('good');
    file(join(good, 'firecrawl'), 0o755);
    const later = dir('later');
    file(join(later, 'firecrawl'), 0o755);
    const path = ['', 'relative/bin', plain, folder, good, later].join(':');
    return expect(findOnPath('firecrawl', path)).resolves.toBe(join(good, 'firecrawl'));
  });

  it('follows a link to an executable, as npm -g installs it', async () => {
    const lib = dir('lib');
    file(join(lib, 'cli.js'), 0o755);
    const bin = dir('bin');
    symlinkSync(join(lib, 'cli.js'), join(bin, 'firecrawl'));
    expect(await findOnPath('firecrawl', bin)).toBe(join(bin, 'firecrawl'));
  });

  it('finds nothing without a PATH, on an empty one, or for a name with a slash', async () => {
    const good = dir('good');
    file(join(good, 'firecrawl'), 0o755);
    expect(await findOnPath('firecrawl', undefined)).toBeNull();
    expect(await findOnPath('firecrawl', '')).toBeNull();
    expect(await findOnPath('firecrawl', join(root, 'missing'))).toBeNull();
    expect(await findOnPath('good/firecrawl', root)).toBeNull();
    expect(await findOnPath('', good)).toBeNull();
  });

  it('readPathEnv looks up PATH case-insensitively', () => {
    expect(readPathEnv({ PATH: '/usr/bin' })).toBe('/usr/bin');
    expect(readPathEnv({ Path: 'C:\\Windows\\System32' })).toBe('C:\\Windows\\System32');
    expect(readPathEnv({ path: 'D:\\bin' })).toBe('D:\\bin');
    expect(readPathEnv({})).toBeUndefined();
  });

  it('on win32 splits on ;, strips quotes, honors PATHEXT, skips POSIX shims and directories, and rejects \\ and :', async () => {
    const shimOnly = dir('shim-only');
    // Extensionless POSIX shim (as npm -g creates alongside .cmd) must be skipped when PATHEXT is non-empty.
    file(join(shimOnly, 'firecrawl'), 0o755);
    const dirNamed = dir('dir-named');
    mkdirSync(join(dirNamed, 'firecrawl.exe'));
    const good = dir('good space');
    writeFileSync(join(good, 'firecrawl.cmd'), '@echo off\r\n');
    const later = dir('later');
    writeFileSync(join(later, 'firecrawl.exe'), 'mz');

    const path = ['', 'relative\\bin', shimOnly, dirNamed, `"${good}"`, later].join(';');
    expect(await findOnPath('firecrawl', path, { platform: 'win32' })).toBe(join(good, 'firecrawl.cmd'));
    expect(await findOnPath('firecrawl', path, { platform: 'win32', pathExt: '.EXE' })).toBe(
      join(later, 'firecrawl.exe'),
    );
    expect(await findOnPath('firecrawl', shimOnly, { platform: 'win32', pathExt: '' })).toBe(
      join(shimOnly, 'firecrawl'),
    );
    expect(await findOnPath('sub\\firecrawl', good, { platform: 'win32' })).toBeNull();
    expect(await findOnPath('C:firecrawl', good, { platform: 'win32' })).toBeNull();
  });
});
