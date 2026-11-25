/**
 * The Control Panel asks the user's login shell for the `PATH` a Work command gets. Without it the Firecrawl check
 * reads the app's own `PATH`, which for an app started from Finder is `/usr/bin:/bin:/usr/sbin:/sbin`.
 */

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { expect, it } from 'vite-plus/test';

const ENTRY = resolve(dirname(fileURLToPath(import.meta.url)), '../src/control-panel/index.ts');

it('the Control Panel entry passes workCommandPath as the Work PATH', () => {
  expect(readFileSync(ENTRY, 'utf8').includes('workPath: workCommandPath,')).toBe(true);
});
