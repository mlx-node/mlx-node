/**
 * Persisted desktop settings: what they are, what a bad file means, and how the
 * app survives one.
 *
 * No Electron in here — `app.getPath('userData')` is the caller's problem — for
 * the same reason `www.ts` is split from `protocol.ts`: `import { app } from
 * 'electron'` cannot resolve outside an Electron process, and the parts that can
 * be quietly wrong (a hand-edited file, a key from a newer version, a window
 * position on a display that is no longer attached) are exactly the parts that
 * have to be provable from a plain Node test.
 *
 * Everything here is async. A synchronous read in a tray handler blocks MAIN,
 * and a blocked MAIN freezes the tray, the window chrome and all IPC — measured
 * at 2960 ms of lag for a 3 s block, against 3 ms for the same block in a
 * utilityProcess.
 */

import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

/**
 * Where the Control Panel window was last left.
 *
 * `x`/`y` are nullable together: a window with a size but no position is
 * centred, which is the right first-run behaviour and the right fallback when a
 * saved position turns out to be on a display that has since been unplugged.
 */
export interface WindowBounds {
  width: number;
  height: number;
  x: number | null;
  y: number | null;
}

export interface DesktopSettings {
  /**
   * Fork the INFERENCE sidecar at launch. Off means the app costs a menubar
   * icon and nothing else until the user asks for a model — which is the point
   * of a menubar app that supervises a 233 MB addon in another process.
   */
  autoStartInference: boolean;
  /**
   * Show a Dock icon as well as the menubar item. Default off: this is a
   * menubar app, and `app.dock.show()` is reversible at runtime while a Dock
   * icon the user did not ask for is not.
   */
  showInDock: boolean;
  /** Overrides the shared `$HOME/.mlx-node/models` discovery root. `null` = use the default. */
  modelsDir: string | null;
  controlPanelWindow: WindowBounds;
}

/**
 * A window smaller than this cannot show the dashboard's own layout, and a
 * saved size below it is far more likely to be a corrupted number than an
 * intent.
 */
const MIN_WIDTH = 720;
const MIN_HEIGHT = 480;

export const DEFAULT_SETTINGS: DesktopSettings = Object.freeze({
  autoStartInference: true,
  showInDock: false,
  modelsDir: null,
  controlPanelWindow: Object.freeze({ width: 1180, height: 800, x: null, y: null }),
});

export interface NormalizedSettings {
  settings: DesktopSettings;
  /**
   * Keys whose stored value was unusable and was replaced by the default.
   * Surfaced rather than swallowed: a setting that silently reverts every
   * launch is the kind of fault a user cannot report usefully.
   */
  repaired: readonly string[];
}

function bool(value: unknown, fallback: boolean, key: string, repaired: string[]): boolean {
  if (typeof value === 'boolean') return value;
  if (value !== undefined) repaired.push(key);
  return fallback;
}

/** A saved path is only useful as an absolute-ish non-empty string; anything else is noise. */
function optionalString(value: unknown, key: string, repaired: string[]): string | null {
  if (value === null || value === undefined) return null;
  if (typeof value === 'string' && value.trim().length > 0) return value;
  repaired.push(key);
  return null;
}

/** Window geometry is integral pixels. `NaN`, `Infinity` and `1.5` are all corruption. */
function pixel(value: unknown): number | null {
  return typeof value === 'number' && Number.isInteger(value) ? value : null;
}

function normalizeBounds(raw: unknown, repaired: string[]): WindowBounds {
  const source = typeof raw === 'object' && raw !== null ? (raw as Record<string, unknown>) : {};
  if (raw !== undefined && (typeof raw !== 'object' || raw === null)) repaired.push('controlPanelWindow');

  const width = pixel(source.width);
  const height = pixel(source.height);
  if (source.width !== undefined && (width === null || width < MIN_WIDTH)) repaired.push('controlPanelWindow.width');
  if (source.height !== undefined && (height === null || height < MIN_HEIGHT))
    repaired.push('controlPanelWindow.height');

  const x = pixel(source.x);
  const y = pixel(source.y);
  // Both or neither. Half a position places the window somewhere the user never
  // put it, which looks exactly like the app losing the setting.
  const positioned = x !== null && y !== null;
  if (!positioned && (source.x !== undefined || source.y !== undefined)) {
    if (source.x !== null || source.y !== null) repaired.push('controlPanelWindow.position');
  }

  return {
    width: width !== null && width >= MIN_WIDTH ? width : DEFAULT_SETTINGS.controlPanelWindow.width,
    height: height !== null && height >= MIN_HEIGHT ? height : DEFAULT_SETTINGS.controlPanelWindow.height,
    x: positioned ? x : null,
    y: positioned ? y : null,
  };
}

/**
 * Merge whatever was on disk over the defaults, key by key.
 *
 * Per-key, deliberately: one bad value must not cost the user every other
 * setting. A file that is not an object at all is a different failure and is
 * handled by {@link loadSettings}, which quarantines it.
 *
 * Unknown keys are dropped rather than preserved. The alternative — round-trip
 * anything we do not recognise — sounds friendlier but means a typo'd key lives
 * in the file forever looking like it does something.
 */
export function normalizeSettings(raw: unknown): NormalizedSettings {
  const repaired: string[] = [];
  const source = typeof raw === 'object' && raw !== null ? (raw as Record<string, unknown>) : {};
  return {
    settings: {
      autoStartInference: bool(
        source.autoStartInference,
        DEFAULT_SETTINGS.autoStartInference,
        'autoStartInference',
        repaired,
      ),
      showInDock: bool(source.showInDock, DEFAULT_SETTINGS.showInDock, 'showInDock', repaired),
      modelsDir: optionalString(source.modelsDir, 'modelsDir', repaired),
      controlPanelWindow: normalizeBounds(source.controlPanelWindow, repaired),
    },
    repaired,
  };
}

/**
 * How the settings we are running with were obtained.
 *
 * `corrupt` and `unreadable` are kept apart because the recovery differs: a file
 * we could read and could not parse is renamed out of the way (it is garbage,
 * and leaving it there means writing the recovered file over the evidence),
 * while a file we could not read at all is left strictly alone — a permissions
 * error is not proof the contents are bad, and renaming it would destroy a
 * recoverable file to fix a problem we have not diagnosed.
 */
export type SettingsSource = 'file' | 'missing' | 'corrupt' | 'unreadable';

export interface LoadedSettings extends NormalizedSettings {
  source: SettingsSource;
  /** Where the unparseable file was moved, when it was. */
  quarantined: string | null;
  /** One line for the log; `null` when nothing went wrong. */
  problem: string | null;
}

/**
 * Read settings, and **never** throw.
 *
 * A truncated write, a hand edit with a trailing comma, a half-synced file from
 * a crashed machine: none of them may stop a menubar app from starting, because
 * the app that cannot start is also the app that cannot be used to fix its own
 * settings. Every failure path here ends at {@link DEFAULT_SETTINGS}.
 */
export async function loadSettings(file: string): Promise<LoadedSettings> {
  let text: string;
  try {
    text = await readFile(file, 'utf8');
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') {
      return { ...normalizeSettings(undefined), source: 'missing', quarantined: null, problem: null };
    }
    return {
      ...normalizeSettings(undefined),
      source: 'unreadable',
      quarantined: null,
      problem: `could not read ${file}: ${describe(error)}`,
    };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    return corrupt(file, `not valid JSON: ${describe(error)}`);
  }
  // `[1,2]`, `"hello"` and `null` are all valid JSON and none of them is a
  // settings object. Treating them as an empty object would silently discard a
  // file the user may still want back.
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    return corrupt(file, `expected a JSON object, found ${Array.isArray(parsed) ? 'an array' : typeof parsed}`);
  }

  return { ...normalizeSettings(parsed), source: 'file', quarantined: null, problem: null };
}

async function corrupt(file: string, why: string): Promise<LoadedSettings> {
  const target = `${file}.corrupt-${new Date().toISOString().replaceAll(':', '-')}`;
  let quarantined: string | null = null;
  try {
    await rename(file, target);
    quarantined = target;
  } catch {
    // A read-only directory, a name that already exists, a file that vanished
    // between the read and the rename. None of it is worth failing a launch
    // over: the defaults are already in hand and the next save will overwrite
    // the bad file anyway.
  }
  return { ...normalizeSettings(undefined), source: 'corrupt', quarantined, problem: why };
}

/**
 * Write settings atomically.
 *
 * Temp file plus `rename`, because `rename` within a filesystem is atomic and a
 * direct `writeFile` is not: a crash (or a full disk) partway through leaves a
 * truncated file, which is precisely the corruption {@link loadSettings} then
 * has to recover from. Cheaper to not create it.
 */
export async function saveSettings(file: string, settings: DesktopSettings): Promise<void> {
  await mkdir(dirname(file), { recursive: true });
  const temp = `${file}.${process.pid}.tmp`;
  await writeFile(temp, `${JSON.stringify(settings, null, 2)}\n`, 'utf8');
  await rename(temp, file);
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** A display's usable area, as `screen.getAllDisplays()[n].workArea` reports it. */
export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Enough of the window must land on a real display for the user to grab it.
 * Roughly a title-bar-sized target.
 */
const MIN_VISIBLE_WIDTH = 120;
const MIN_VISIBLE_HEIGHT = 40;

/**
 * Drop a saved position that no longer lands on any display.
 *
 * This is not theoretical tidiness: a window saved on an external monitor at
 * `x: 2560` reopens completely off-screen once that monitor is unplugged, and
 * Electron will happily place it there. The app then looks launched-but-broken —
 * tray responsive, "Open Control Panel" doing nothing visible. Dropping the position
 * falls back to centring on the primary display.
 */
export function clampBoundsToDisplays(bounds: WindowBounds, workAreas: readonly Rect[]): WindowBounds {
  const { x, y, width, height } = bounds;
  if (x === null || y === null) return bounds;
  const visible = workAreas.some((area) => {
    const overlapX = Math.min(x + width, area.x + area.width) - Math.max(x, area.x);
    const overlapY = Math.min(y + height, area.y + area.height) - Math.max(y, area.y);
    return overlapX >= MIN_VISIBLE_WIDTH && overlapY >= MIN_VISIBLE_HEIGHT;
  });
  return visible ? bounds : { ...bounds, x: null, y: null };
}
