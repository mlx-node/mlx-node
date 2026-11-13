/** `findOnPath`: an executable regular file in an absolute `PATH` entry; nothing is run and no shell is used. */

import { chmodSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { findOnPath } from '../src/chat/skills/cli-check.js';

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
});
