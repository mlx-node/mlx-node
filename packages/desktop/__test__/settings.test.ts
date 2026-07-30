/**
 * The settings file is the one piece of app state a user can edit by hand, a
 * backup can restore half of, and a crash can truncate. None of those may stop a
 * menubar app from starting — the app that will not start is also the app that
 * cannot be used to fix its own settings.
 */

import { chmodSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  clampBoundsToDisplays,
  DEFAULT_SETTINGS,
  loadSettings,
  normalizeSettings,
  saveSettings,
  type DesktopSettings,
} from '../src/main/settings.js';

let dir: string;
let file: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'mlx-settings-'));
  file = join(dir, 'settings.json');
});

afterEach(() => {
  // A test may have made the directory unwritable to force a rename failure.
  chmodSync(dir, 0o700);
  rmSync(dir, { recursive: true, force: true });
});

describe('normalizeSettings', () => {
  it('fills in every default for an empty object', () => {
    expect(normalizeSettings({})).toEqual({ settings: DEFAULT_SETTINGS, repaired: [] });
  });

  // The plain missing-key case, stated per key rather than as a whole-object
  // compare: an equality check against DEFAULT_SETTINGS would still pass if the
  // merge returned the defaults object itself and ignored the file entirely.
  it('applies the default for a key the file does not have', () => {
    const { settings } = normalizeSettings({ showInDock: true });
    expect(settings.showInDock).toBe(true);
    expect(settings.autoStartInference).toBe(DEFAULT_SETTINGS.autoStartInference);
    expect(settings.modelsDir).toBeNull();
    expect(settings.controlPanelWindow).toEqual(DEFAULT_SETTINGS.controlPanelWindow);
  });

  it('keeps values the file does have', () => {
    const { settings, repaired } = normalizeSettings({
      autoStartInference: false,
      showInDock: true,
      modelsDir: '/Volumes/models',
      controlPanelWindow: { width: 900, height: 700, x: 12, y: 34 },
    });
    expect(settings).toEqual({
      autoStartInference: false,
      showInDock: true,
      modelsDir: '/Volumes/models',
      controlPanelWindow: { width: 900, height: 700, x: 12, y: 34 },
    });
    expect(repaired).toEqual([]);
  });

  // One bad value must not cost the user every other setting.
  it('repairs a wrong-typed value per key and keeps the rest', () => {
    const { settings, repaired } = normalizeSettings({
      autoStartInference: 'yes',
      showInDock: true,
      modelsDir: 42,
    });
    expect(settings.autoStartInference).toBe(DEFAULT_SETTINGS.autoStartInference);
    expect(settings.showInDock).toBe(true);
    expect(settings.modelsDir).toBeNull();
    expect(repaired).toEqual(['autoStartInference', 'modelsDir']);
  });

  it('treats an empty or blank models directory as unset', () => {
    expect(normalizeSettings({ modelsDir: '   ' }).settings.modelsDir).toBeNull();
    // An explicit null is the documented "use the default", not a repair.
    expect(normalizeSettings({ modelsDir: null }).repaired).toEqual([]);
  });

  it('drops a key it does not know', () => {
    const { settings } = normalizeSettings({ theme: 'dark' });
    expect(settings).not.toHaveProperty('theme');
  });

  describe('controlPanelWindow', () => {
    it('rejects a size below the minimum the dashboard can render', () => {
      const { settings, repaired } = normalizeSettings({ controlPanelWindow: { width: 200, height: 100 } });
      expect(settings.controlPanelWindow.width).toBe(DEFAULT_SETTINGS.controlPanelWindow.width);
      expect(settings.controlPanelWindow.height).toBe(DEFAULT_SETTINGS.controlPanelWindow.height);
      expect(repaired).toEqual(['controlPanelWindow.width', 'controlPanelWindow.height']);
    });

    // Geometry is integral pixels. Every one of these round-trips through JSON
    // as a number and none of them is a window size.
    it('rejects a non-integral size', () => {
      for (const width of [Number.NaN, Number.POSITIVE_INFINITY, 1180.5, '1180']) {
        expect(
          normalizeSettings({ controlPanelWindow: { width, height: 800 } }).settings.controlPanelWindow.width,
        ).toBe(DEFAULT_SETTINGS.controlPanelWindow.width);
      }
    });

    // Half a position places the window somewhere the user never put it, which
    // reads as the app losing the setting.
    it('drops a half-written position', () => {
      const { settings, repaired } = normalizeSettings({ controlPanelWindow: { width: 900, height: 700, x: 12 } });
      expect(settings.controlPanelWindow).toMatchObject({ width: 900, height: 700, x: null, y: null });
      expect(repaired).toEqual(['controlPanelWindow.position']);
    });

    it('accepts a negative position, which is a real place on a second display', () => {
      expect(
        normalizeSettings({ controlPanelWindow: { width: 900, height: 700, x: -1440, y: 200 } }).settings
          .controlPanelWindow,
      ).toEqual({ width: 900, height: 700, x: -1440, y: 200 });
    });

    it('replaces a non-object controlPanelWindow wholesale', () => {
      const { settings, repaired } = normalizeSettings({ controlPanelWindow: 'big' });
      expect(settings.controlPanelWindow).toEqual(DEFAULT_SETTINGS.controlPanelWindow);
      expect(repaired).toContain('controlPanelWindow');
    });
  });
});

describe('loadSettings', () => {
  it('returns the defaults when there is no file at all', async () => {
    const loaded = await loadSettings(file);
    expect(loaded).toMatchObject({ source: 'missing', quarantined: null, problem: null });
    expect(loaded.settings).toEqual(DEFAULT_SETTINGS);
  });

  it('reads a good file', async () => {
    writeFileSync(file, JSON.stringify({ showInDock: true }));
    const loaded = await loadSettings(file);
    expect(loaded.source).toBe('file');
    expect(loaded.settings.showInDock).toBe(true);
  });

  // THE requirement: a truncated write or a hand edit must not brick start-up.
  // Asserted as "resolves with the defaults", not as "does not throw" — a
  // `rejects.toThrow()` inversion would be the same test written to pass for the
  // wrong reason.
  it('recovers from a truncated file instead of failing to start', async () => {
    writeFileSync(file, '{"showInDock": tr');
    const loaded = await loadSettings(file);
    expect(loaded.source).toBe('corrupt');
    expect(loaded.settings).toEqual(DEFAULT_SETTINGS);
    expect(loaded.problem).toContain('not valid JSON');
  });

  it('moves the unparseable file aside instead of overwriting the evidence', async () => {
    writeFileSync(file, '{ nope');
    const loaded = await loadSettings(file);
    expect(loaded.quarantined).toContain('settings.json.corrupt-');
    expect(readFileSync(String(loaded.quarantined), 'utf8')).toBe('{ nope');
    // The bad file must be out of the way, or the next launch recovers again and
    // the user's real settings are never written.
    expect(readdirSync(dir)).not.toContain('settings.json');
  });

  // Well-formed JSON that is not a settings object. Merging these as if they
  // were an empty object would discard a file the user may still want back.
  it('treats valid JSON that is not an object as corrupt', async () => {
    for (const text of ['[1,2]', '"hello"', 'null', '42']) {
      writeFileSync(file, text);
      const loaded = await loadSettings(file);
      expect(loaded.source, text).toBe('corrupt');
      expect(loaded.settings).toEqual(DEFAULT_SETTINGS);
    }
  });

  // A permissions error is not proof the contents are bad, so the file is left
  // strictly alone: renaming it would destroy a recoverable file to work around
  // a problem we have not diagnosed.
  it('does not quarantine a file it could not read', async () => {
    mkdirSync(file); // EISDIR on read, and nothing here is corrupt
    const loaded = await loadSettings(file);
    expect(loaded).toMatchObject({ source: 'unreadable', quarantined: null });
    expect(loaded.settings).toEqual(DEFAULT_SETTINGS);
    expect(readdirSync(dir)).toContain('settings.json');
  });

  it('still starts when the bad file cannot even be moved aside', async () => {
    writeFileSync(file, '{ nope');
    chmodSync(dir, 0o500); // readable, not writable: the rename must fail
    const loaded = await loadSettings(file);
    expect(loaded).toMatchObject({ source: 'corrupt', quarantined: null });
    expect(loaded.settings).toEqual(DEFAULT_SETTINGS);
  });
});

describe('saveSettings', () => {
  it('round-trips through loadSettings', async () => {
    const settings: DesktopSettings = {
      autoStartInference: false,
      showInDock: true,
      modelsDir: '/Volumes/models',
      controlPanelWindow: { width: 1000, height: 620, x: 40, y: 60 },
    };
    await saveSettings(file, settings);
    expect((await loadSettings(file)).settings).toEqual(settings);
  });

  it('creates the directory it is asked to write into', async () => {
    const nested = join(dir, 'deep', 'settings.json');
    await saveSettings(nested, DEFAULT_SETTINGS);
    expect((await loadSettings(nested)).source).toBe('file');
  });

  // Written to a temp name and renamed, so a crash mid-write cannot leave the
  // truncated file the corruption path above then has to recover from.
  it('leaves no temp file behind', async () => {
    await saveSettings(file, DEFAULT_SETTINGS);
    expect(readdirSync(dir)).toEqual(['settings.json']);
  });

  // Crash-atomicity itself cannot be asserted without crashing mid-write, so
  // this pins the mechanism that provides it instead. `rename(2)` needs write
  // permission on the DIRECTORY and none on the target file, while an in-place
  // `writeFile` opens the target for writing and fails with EACCES. A save that
  // is rewritten to write in place therefore fails here — which is the only
  // externally visible difference the two implementations have.
  it('replaces the file by rename, not by writing into it', async () => {
    writeFileSync(file, '{}', { mode: 0o400 });
    await saveSettings(file, { ...DEFAULT_SETTINGS, showInDock: true });
    expect((await loadSettings(file)).settings.showInDock).toBe(true);
  });
});

describe('clampBoundsToDisplays', () => {
  const primary = { x: 0, y: 0, width: 1920, height: 1080 };

  it('keeps a position on an attached display', () => {
    const bounds = { width: 900, height: 700, x: 100, y: 100 };
    expect(clampBoundsToDisplays(bounds, [primary])).toEqual(bounds);
  });

  // The case this exists for: a window saved on an external monitor that is no
  // longer plugged in. Electron will happily place it there, and the app looks
  // launched-but-broken — tray responsive, "Open Control Panel" doing nothing visible.
  it('drops a position that is off every display', () => {
    const bounds = { width: 900, height: 700, x: 2600, y: 200 };
    expect(clampBoundsToDisplays(bounds, [primary])).toMatchObject({ width: 900, height: 700, x: null, y: null });
  });

  it('keeps a window that is only partly on screen but still grabbable', () => {
    const bounds = { width: 900, height: 700, x: 1700, y: 100 };
    expect(clampBoundsToDisplays(bounds, [primary])).toEqual(bounds);
  });

  it('drops a window with only a sliver on screen', () => {
    // 40 px of width left on the display: not enough of a title bar to drag.
    const bounds = { width: 900, height: 700, x: 1880, y: 100 };
    expect(clampBoundsToDisplays(bounds, [primary]).x).toBeNull();
  });

  it('accepts a secondary display to the left of the primary', () => {
    const secondary = { x: -1440, y: 0, width: 1440, height: 900 };
    const bounds = { width: 900, height: 700, x: -1200, y: 100 };
    expect(clampBoundsToDisplays(bounds, [primary, secondary])).toEqual(bounds);
  });

  it('leaves an unpositioned window alone', () => {
    const bounds = { width: 900, height: 700, x: null, y: null };
    expect(clampBoundsToDisplays(bounds, [])).toEqual(bounds);
  });
});
