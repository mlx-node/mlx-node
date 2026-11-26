import { readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

const SRC = join(import.meta.dirname, '..', 'src');
const HANDLERS = ['setPermissionCheckHandler', 'setPermissionRequestHandler'];

function sources(dir: string): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) files.push(...sources(path));
    else if (entry.name.endsWith('.ts') || entry.name.endsWith('.cts') || entry.name.endsWith('.mts')) files.push(path);
  }
  return files;
}

describe('permission handlers', () => {
  it('scans the main process and the preload', () => {
    const scanned = sources(SRC).map((file) => relative(SRC, file));
    expect(scanned).toContain(join('main', 'index.ts'));
    expect(scanned).toContain(join('preload', 'index.cts'));
  });

  // With no handler Electron grants `local-fonts`; a handler that does not name it makes queryLocalFonts() return
  // an empty list with no error, and the Fonts page then offers only the built-in fonts.
  it('a file that sets a permission handler names local-fonts', () => {
    const missing: string[] = [];
    for (const file of sources(SRC)) {
      const text = readFileSync(file, 'utf8');
      if (!HANDLERS.some((name) => text.includes(name))) continue;
      if (!text.includes("'local-fonts'") && !text.includes('"local-fonts"')) missing.push(relative(SRC, file));
    }
    expect(missing).toEqual([]);
  });
});
