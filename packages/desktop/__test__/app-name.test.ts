import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { APP_NAME, DATA_DIR_NAME } from '../src/app-name.js';

const DESKTOP = join(import.meta.dirname, '..');

describe('app name and userData folder', () => {
  it('shows Portato and keeps the mlx-node userData folder', () => {
    expect(APP_NAME).toBe('Portato');
    expect(DATA_DIR_NAME).toBe('mlx-node');
  });

  it('gives dev runs the same productName as packaged ones', () => {
    const manifest = JSON.parse(readFileSync(join(DESKTOP, 'package.json'), 'utf-8')) as { productName: string };
    expect(manifest.productName).toBe(APP_NAME);
  });

  it('pins userData before the single-instance lock takes its file there', () => {
    const source = readFileSync(join(DESKTOP, 'src', 'main', 'index.ts'), 'utf-8');
    const pin = source.indexOf("app.setPath('userData'");
    const lock = source.indexOf('app.requestSingleInstanceLock()');
    expect(pin).toBeGreaterThanOrEqual(0);
    expect(lock).toBeGreaterThan(pin);
  });

  it('creates the pinned userData folder before setPath, which throws on a missing one', () => {
    const source = readFileSync(join(DESKTOP, 'src', 'main', 'index.ts'), 'utf-8');
    const make = source.indexOf('mkdirSync(userDataDir');
    const pin = source.indexOf("app.setPath('userData'");
    expect(make).toBeGreaterThanOrEqual(0);
    expect(pin).toBeGreaterThan(make);
  });
});
