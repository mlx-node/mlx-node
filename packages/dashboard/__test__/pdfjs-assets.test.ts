import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { afterEach, describe, expect, it } from 'vite-plus/test';

import { PDFJS_WORKER, pdfjsAssets, pdfjsBase } from '../ui/pdfjs-assets.js';

describe('pdfjsAssets', () => {
  const made: string[] = [];
  afterEach(() => {
    for (const dir of made.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  it('copies after the build into the out folder named relative to the root, not to the shell', () => {
    const root = mkdtempSync(join(tmpdir(), 'pdfjs-root-'));
    const out = `pdfjs-out-${process.pid}-${Date.now()}`;
    made.push(root, resolve(out));
    const plugin = pdfjsAssets();
    const configResolved = plugin.configResolved as unknown as (config: {
      root: string;
      build: { outDir: string };
    }) => void;
    const writeBundle = plugin.writeBundle as unknown as () => void;
    configResolved({ root, build: { outDir: out } });
    writeBundle();
    expect(existsSync(join(root, out, pdfjsBase(), PDFJS_WORKER))).toBe(true);
    expect(existsSync(resolve(out))).toBe(false);
  });
});
