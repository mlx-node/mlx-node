import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { APP_ID, APP_NAME, APP_PUBLISHER, DATA_DIR_NAME } from '../src/app-name.js';

const DESKTOP = join(import.meta.dirname, '..');

describe('app name, ID, and userData folder', () => {
  it('shows Portato, keeps the mlx-node userData folder, and exports stable ID/publisher', () => {
    expect(APP_NAME).toBe('Portato');
    expect(APP_ID).toBe('ai.mlxnode.desktop');
    expect(APP_PUBLISHER).toBe('mlx-node');
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

  it('sets the Windows AppUserModelId outside Microsoft Store runtime before startup', () => {
    const source = readFileSync(join(DESKTOP, 'src', 'main', 'index.ts'), 'utf-8');
    const aumid = source.indexOf('app.setAppUserModelId(APP_ID)');
    const guard = source.indexOf("process.platform === 'win32' && !isWindowsStoreRuntime()");
    const lock = source.indexOf('app.requestSingleInstanceLock()');
    expect(guard).toBeGreaterThanOrEqual(0);
    expect(aumid).toBeGreaterThan(guard);
    expect(lock).toBeGreaterThan(aumid);
  });
});
