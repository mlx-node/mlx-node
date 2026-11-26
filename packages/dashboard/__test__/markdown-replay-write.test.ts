/** The Markdown replay's cache writer: the corpus copies answers out of 0700 homes, so the copy is owner-only too. */

import { mkdirSync, mkdtempSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vite-plus/test';

import { writeJson } from '../scripts/dev/markdown-replay.js';

const roots: string[] = [];
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

const modeOf = (path: string): number => statSync(path).mode & 0o777;

function withUmask(mask: number, run: () => void): void {
  const before = process.umask(mask);
  try {
    run();
  } finally {
    process.umask(before);
  }
}

describe('markdown replay: writeJson', () => {
  it('makes its folder 0700 and its file 0600 under umask 022', () => {
    const root = mkdtempSync(join(tmpdir(), 'mlx-md-replay-'));
    roots.push(root);
    const file = join(root, 'cache', 'markstream-corpus.json');
    withUmask(0o022, () => writeJson(file, [{ id: 'real-0', text: 'an answer' }]));
    expect(modeOf(join(root, 'cache'))).toBe(0o700);
    expect(modeOf(file)).toBe(0o600);
  });

  it('closes a cache that already exists readable to others', () => {
    const root = mkdtempSync(join(tmpdir(), 'mlx-md-replay-'));
    roots.push(root);
    const dir = join(root, 'cache');
    const file = join(dir, 'markstream-corpus.json');
    withUmask(0o022, () => {
      mkdirSync(dir, { mode: 0o755 });
      writeFileSync(file, '[]\n', { mode: 0o644 });
      writeJson(file, []);
    });
    expect(modeOf(dir)).toBe(0o700);
    expect(modeOf(file)).toBe(0o600);
  });
});
