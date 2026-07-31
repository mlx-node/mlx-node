/**
 * The Control Panel `BrowserWindow`: create it, hide it, remember where it was, and
 * refuse to let the renderer take the shell anywhere.
 *
 * Every decision this file makes lives in `window-policy.ts` and `settings.ts`
 * so it can be tested without Electron; what is left is `BrowserWindow`
 * plumbing, held to the same "no decisions here" bar as `protocol.ts` and
 * `child-utility.ts`.
 */

import { app, BrowserWindow, ipcMain, shell, type IpcMainEvent, type WebContents } from 'electron';

import { clampBoundsToDisplays, type Rect, type WindowBounds } from './settings.js';
import {
  CONTROL_PANEL_READY_CHANNEL,
  CONTROL_PANEL_UNRESPONSIVE_CHANNEL,
  CONTROL_PANEL_URL,
  classifyNavigation,
} from './window-policy.js';

/** Below this the dashboard's own layout collapses; the user cannot recover it by dragging. */
const MIN_WIDTH = 720;
const MIN_HEIGHT = 480;

/**
 * Coalescing window for bounds writes. `resize` fires per frame while dragging,
 * and each save is a write plus a rename — nothing MAIN should do sixty times a
 * second.
 */
const BOUNDS_SAVE_DEBOUNCE_MS = 500;

export interface ControlPanelWindowOptions {
  /** Absolute path to `dist/preload/index.cjs`. */
  preload: string;
  /**
   * Where to put the window. Read at each create, not captured: the window is
   * destroyed and rebuilt if its renderer dies, and it must come back where the
   * user last left it rather than where it started.
   */
  bounds(): WindowBounds;
  /**
   * Work areas of the attached displays — `screen.getAllDisplays()`. Also a
   * function, because a monitor unplugged while the app runs is the case the
   * off-screen check exists for.
   */
  workAreas(): readonly Rect[];
  /**
   * True once the app has decided to exit.
   *
   * Read on every `close`, never captured: the window is created long before the
   * user quits, and a value copied at construction would be permanently `false`
   * — which is exactly the bug that traps a user in an app that will not exit.
   */
  isQuitting(): boolean;
  /** Called with the window's normal bounds, already debounced. */
  onBounds(bounds: WindowBounds): void;
  /**
   * **The broker seam.** Called when the renderer says it is listening, with the
   * `WebContents` that asked. The implementation creates a `MessageChannelMain`,
   * serves the CONTROL PANEL runtime on one end, and hands the other over with
   * `contents.postMessage(CONTROL_PANEL_PORT_CHANNEL, generation, [port2])`.
   *
   * Optional, and undefined until the broker exists: a window with no port
   * renders and stays empty, which is a visible, honest failure rather than a
   * crash at startup.
   */
  onControlPanelReady?(contents: WebContents): void;
  /**
   * Called only for this window when its RPC transport misses the
   * post-cancellation bound. The broker kills and replaces the runtime.
   */
  onControlPanelUnresponsive?(contents: WebContents, expectedGeneration: number): void;
}

export interface ControlPanelWindowManager {
  /** Create the window if it is gone, otherwise reveal the hidden one. */
  show(): void;
  /** The live window, or `null`. */
  current(): BrowserWindow | null;
  /** Release the IPC listener and destroy the window. Idempotent. */
  dispose(): void;
}

export function createControlPanelWindowManager(options: ControlPanelWindowOptions): ControlPanelWindowManager {
  let win: BrowserWindow | null = null;
  let boundsTimer: NodeJS.Timeout | null = null;
  let disposed = false;

  function reveal(target: BrowserWindow): void {
    if (target.isMinimized()) target.restore();
    target.show();
    target.focus();
    // An accessory app has no Dock presence to activate it. Every caller of
    // manager.show() represents an explicit launch, activation, or tray action,
    // so bringing the requested window forward is the expected behaviour.
    app.focus({ steal: true });
  }

  const onReady = (event: IpcMainEvent): void => {
    // Any renderer can send on a channel; only ours may be answered. There is
    // one window today, so this is cheap insurance rather than a live threat —
    // but the thing being handed out is a port to the runtime.
    if (win === null || win.isDestroyed() || event.sender !== win.webContents) return;
    options.onControlPanelReady?.(event.sender);
  };
  const onUnresponsive = (event: IpcMainEvent, expectedGeneration: unknown): void => {
    // Restarting the utility process is a larger capability than asking for a
    // fresh port, so keep the same exact-sender boundary.
    if (win === null || win.isDestroyed() || event.sender !== win.webContents) return;
    // The generation is also an authority boundary: malformed input and a
    // delayed report for a retired channel must never be resolved against the
    // process that happens to be current now.
    if (typeof expectedGeneration !== 'number' || !Number.isSafeInteger(expectedGeneration) || expectedGeneration < 1) {
      return;
    }
    options.onControlPanelUnresponsive?.(event.sender, expectedGeneration);
  };
  ipcMain.on(CONTROL_PANEL_READY_CHANNEL, onReady);
  ipcMain.on(CONTROL_PANEL_UNRESPONSIVE_CHANNEL, onUnresponsive);

  function persistBounds(): void {
    if (boundsTimer !== null) clearTimeout(boundsTimer);
    boundsTimer = setTimeout(() => {
      boundsTimer = null;
      if (win === null || win.isDestroyed()) return;
      // `getNormalBounds`, not `getBounds`: while maximised or full-screen the
      // latter reports the screen, and restoring that on next launch produces a
      // window the user cannot un-maximise back to a size they chose.
      const { width, height, x, y } = win.getNormalBounds();
      options.onBounds({ width, height, x, y });
    }, BOUNDS_SAVE_DEBOUNCE_MS);
    // Must not keep the process alive during quit.
    boundsTimer.unref();
  }

  function create(): BrowserWindow {
    const bounds = clampBoundsToDisplays(options.bounds(), options.workAreas());
    const created = new BrowserWindow({
      width: bounds.width,
      height: bounds.height,
      // Omitted when unknown so Electron centres the window on the primary
      // display, which is also where a position that no longer lands on any
      // display ends up — see `clampBoundsToDisplays`.
      ...(bounds.x !== null && bounds.y !== null ? { x: bounds.x, y: bounds.y } : {}),
      minWidth: MIN_WIDTH,
      minHeight: MIN_HEIGHT,
      title: 'mlx-node',
      // Painted before the SPA's first frame; without it the window opens as a
      // white flash even in dark mode.
      backgroundColor: '#101014',
      titleBarStyle: 'hiddenInset',
      // Shown from `ready-to-show` instead, so the first thing the user sees is
      // rendered content rather than an empty frame.
      show: false,
      webPreferences: {
        preload: options.preload,
        sandbox: true,
        contextIsolation: true,
        nodeIntegration: false,
        spellcheck: false,
      },
    });

    created.once('ready-to-show', () => {
      reveal(created);
    });

    created.on('resize', persistBounds);
    created.on('move', persistBounds);

    created.on('close', (event) => {
      // A menubar app's window closes to the tray. Actually destroying it would
      // throw away a loaded SPA and its MessagePort for something the user
      // meant as "get this off my screen".
      if (options.isQuitting() || disposed) return;
      event.preventDefault();
      created.hide();
    });

    created.on('closed', () => {
      if (win === created) win = null;
    });

    created.webContents.setWindowOpenHandler(({ url }) => {
      if (classifyNavigation(url) === 'external') void shell.openExternal(url);
      // Never `allow`: a second BrowserWindow would inherit this preload and
      // have no chrome to leave with.
      return { action: 'deny' };
    });

    created.webContents.on('will-navigate', (event, url) => {
      const verdict = classifyNavigation(url);
      if (verdict === 'allow') return;
      event.preventDefault();
      if (verdict === 'external') void shell.openExternal(url);
    });

    created.webContents.on('render-process-gone', (_event, details) => {
      console.error(`[mlx] control panel renderer gone: ${details.reason}`);
      // Destroy rather than leave a blank frame: `show()` then rebuilds it, and
      // the reload also re-runs the port handshake.
      if (!created.isDestroyed()) created.destroy();
    });

    // Neither awaited nor voided. A rejection here is THE packaged-app failure —
    // a blank window with nothing in any log — and under `app://` it is what a
    // missing hashed asset or an uninstalled protocol handler looks like.
    created.loadURL(CONTROL_PANEL_URL).catch((error: unknown) => {
      console.error(`[mlx] control panel window failed to load ${CONTROL_PANEL_URL}:`, error);
    });

    return created;
  }

  return {
    show(): void {
      if (disposed) return;
      if (win === null || win.isDestroyed()) {
        win = create();
        return;
      }
      reveal(win);
    },
    current(): BrowserWindow | null {
      return win !== null && !win.isDestroyed() ? win : null;
    },
    dispose(): void {
      if (disposed) return;
      disposed = true;
      ipcMain.removeListener(CONTROL_PANEL_READY_CHANNEL, onReady);
      ipcMain.removeListener(CONTROL_PANEL_UNRESPONSIVE_CHANNEL, onUnresponsive);
      if (boundsTimer !== null) {
        clearTimeout(boundsTimer);
        boundsTimer = null;
      }
      if (win !== null && !win.isDestroyed()) win.destroy();
      win = null;
    },
  };
}
