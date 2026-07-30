/**
 * MAIN: app lifecycle, and the wiring between the tray, the Control Panel window, the
 * settings file and the INFERENCE supervisor.
 *
 * Nothing is decided here. Every rule that could be wrong lives in a module with
 * no Electron import — `settings.ts`, `tray-view.ts`, `window-policy.ts`,
 * `paths.ts`, `www.ts`, `supervisor/state.ts` — because `import { app } from
 * 'electron'` cannot resolve outside an Electron process and would take this
 * file's contents out of reach of every test.
 *
 * The other rule this file obeys: **MAIN never blocks.** A 3 s synchronous block
 * here was measured at 2960 ms of event-loop lag and zero IPC round-trips, with
 * the tray, the native menus and the window chrome frozen for the duration; the
 * same block inside a utilityProcess cost MAIN 3 ms. So there is no synchronous
 * filesystem work in any handler, and everything that touches disk is awaited or
 * debounced.
 */

import { engineEnvFor, LAUNCHER_ENGINE_POLICY } from '@mlx-node/server/host/env-policy';
import { app, clipboard, Menu, screen, type MenuItemConstructorOptions, type WebContents } from 'electron';

import { DESKTOP_QUIT_DEADLINE_MS } from '../control-panel/shutdown-timings.js';
import { createControlPanelBroker, type ControlPanelBroker } from './broker.js';
import { controlPanelEnvOverrides, sidecarEnvOverrides } from './child-env.js';
import { electronBrokerDeps } from './control-panel-child.js';
import { resolveAppPaths, type AppPaths } from './paths.js';
import { installAppProtocol, registerAppScheme } from './protocol.js';
import {
  DEFAULT_SETTINGS,
  loadSettings,
  saveSettings,
  type DesktopSettings,
  type Rect,
  type WindowBounds,
} from './settings.js';
import { utilityChildTransport } from './supervisor/child-utility.js';
import { createSupervisor, type Supervisor } from './supervisor/index.js';
import { claudeConnectCommand, codexConnectCommand } from './tray-view.js';
import { createTray, type TrayController } from './tray.js';
import { createControlPanelWindowManager, type ControlPanelWindowManager } from './window.js';

/** Coalescing window for settings writes. Long enough that a drag is one write. */
const SETTINGS_SAVE_DEBOUNCE_MS = 750;

let paths: AppPaths | null = null;
let settings: DesktopSettings = DEFAULT_SETTINGS;
let saveTimer: NodeJS.Timeout | null = null;
let saveDirty = false;
/** Tail of the settings-write queue. One writer at a time — see `flushSettings`. */
let saveInFlight: Promise<void> = Promise.resolve();

let supervisor: Supervisor | null = null;
let tray: TrayController | null = null;
let controlPanel: ControlPanelWindowManager | null = null;
let broker: ControlPanelBroker<WebContents> | null = null;

/**
 * "The app is exiting", as opposed to "a window is closing". Without the
 * distinction the Control Panel window's close handler hides the window during quit and
 * the app never exits — the user is trapped in something with no Dock icon to
 * force-quit from.
 */
let quitting = false;
let shuttingDown = false;

/**
 * MUST run before `app.whenReady()`. Chromium reads the privileged-scheme table
 * once during startup and a later registration is accepted silently and ignored,
 * so getting this order wrong produces an `app://` that resolves but has no
 * origin semantics — absolute asset paths break and the SPA renders blank.
 */
registerAppScheme();

/**
 * A second launch must not start a second supervisor. Two of them fight over the
 * same models directory and the same GPU, and the first thing the user sees is
 * an inference server that mysteriously restarts.
 *
 * The loser surfaces the winner's window (via `second-instance`) and quits.
 */
if (app.requestSingleInstanceLock()) {
  wire();
} else {
  app.quit();
}

function wire(): void {
  // DO NOT rewrite as a top-level `await app.whenReady()`. It DEADLOCKS: Electron
  // waits for the ESM main module to finish evaluating before it emits `ready`,
  // so awaiting `ready` at module scope waits on an event that cannot fire until
  // the await returns. Verified on Electron 43.2.0 — the process hangs with no
  // output and has to be SIGTERMed. The `void` is deliberate.
  //
  // `.then(bootstrap).catch(…)`, NOT `.then(bootstrap, onRejected)`. The
  // two-argument form only observes a rejection from `whenReady()` itself —
  // which is resolve-only, so that handler could never run — and lets every
  // fault inside `bootstrap` escape as an unhandled rejection. Electron's
  // default handler shows a dialog and does NOT exit, so a bootstrap that
  // threw before the tray existed left an invisible process only Activity
  // Monitor could kill. `.then(bootstrap)` adopts bootstrap's promise, so the
  // `.catch` sees faults from both.
  void app
    .whenReady()
    .then(bootstrap)
    .catch((error: unknown) => {
      console.error('[mlx] failed to start:', error);
      app.exit(1);
    });

  app.on('second-instance', () => {
    controlPanel?.show();
  });

  // This handler must exist and must NOT quit. Electron's default, when nothing
  // is subscribed, is to quit once the last window closes — which for a menubar
  // app means closing the Control Panel window kills the tray and the sidecar with it.
  // The spike's `app.on('window-all-closed', () => app.quit())` was spike-only.
  app.on('window-all-closed', () => {
    // Deliberately empty: the tray is the app.
  });

  app.on('activate', () => {
    // With no Dock icon there is no gesture that produces this event, so the one
    // activation that still arrives is the app's own launch — and popping the
    // window open on every launch is not what a menubar app does. With a Dock
    // icon, clicking it must reopen the window, which is exactly this.
    if (!settings.showInDock) return;
    controlPanel?.show();
  });

  app.on('before-quit', (event) => {
    // The second pass, after `shutdown()` calls `app.quit()` again. Letting it
    // through is what actually exits.
    if (shuttingDown) return;
    shuttingDown = true;
    // Set BEFORE any window is asked to close, so the Control Panel window's close
    // handler knows this one is real.
    quitting = true;
    event.preventDefault();
    void withDeadline(shutdown(), DESKTOP_QUIT_DEADLINE_MS)
      .catch((error: unknown) => {
        console.error('[mlx] shutdown failed:', error);
      })
      .finally(() => {
        app.quit();
      });
  });
}

async function bootstrap(): Promise<void> {
  paths = resolveAppPaths({
    appPath: app.getAppPath(),
    resourcesPath: process.resourcesPath,
    packaged: app.isPackaged,
    userData: app.getPath('userData'),
  });

  // Synchronously, before the first await: a window created by anything that
  // races ahead of us (an `activate` at launch) would otherwise load `app://`
  // with no handler installed and fail as a blank frame.
  installAppProtocol(paths.wwwRoot);

  // An accessory app has no menu bar of its own, but the standard roles still
  // carry the Edit key equivalents the Control Panel window needs — a text field with no
  // Cmd+C is the first thing a user notices. It also gives the Dock-visible mode
  // a real application menu instead of Electron's default.
  Menu.setApplicationMenu(
    Menu.buildFromTemplate([
      { role: 'appMenu' },
      { role: 'editMenu' },
      { role: 'windowMenu' },
    ] as MenuItemConstructorOptions[]),
  );

  const loaded = await loadSettings(paths.settingsFile);
  settings = loaded.settings;
  if (loaded.problem !== null) {
    console.warn(
      `[mlx] settings ${loaded.source}: ${loaded.problem}` +
        (loaded.quarantined === null ? '' : ` (moved to ${loaded.quarantined})`),
    );
  }
  if (loaded.repaired.length > 0) {
    console.warn(`[mlx] settings reset to defaults for: ${loaded.repaired.join(', ')}`);
  }

  applyDockPolicy(settings.showInDock);

  supervisor = createSupervisor({
    entry: paths.sidecarEntry,
    traceDir: paths.traceDir,
    // The `child_process` transport exists for the tests and as the escape hatch
    // if the GPU watchdog ever bites; production is a utilityProcess.
    transport: utilityChildTransport,
    // Passed in rather than applied here. `@mlx-node/server/host/env-policy` is
    // the addon-free leaf of that package; its `/host` entry statically imports
    // `@mlx-node/lm` and maps the 233 MB native addon into whatever process
    // evaluates it, which is the one thing MAIN must never do.
    enginePolicyEnv: engineEnvFor(LAUNCHER_ENGINE_POLICY),
    // `NAPI_RS_NATIVE_LIBRARY_PATH` is in here, and fork time is the only moment
    // it can be set: `packages/core/index.cjs` reads it inside `requireNative()`,
    // which runs while `@mlx-node/lm` is being imported at the sidecar entry's
    // module scope. Writing it from MAIN afterwards is a no-op that looks like it
    // worked.
    env: sidecarEnvOverrides({ nativeAddon: paths.nativeAddon, modelsDir: settings.modelsDir }),
  });

  supervisor.on((event) => {
    if (event.type === 'state') {
      tray?.update(event.snapshot);
      return;
    }
    if (event.type === 'crashed') {
      // For a class-(c) failure — a C++ throw unwinding across an `extern "C"`
      // shim, which makes Rust `abort()` — the stderr line is the only
      // diagnostic that exists; the exit code says nothing.
      console.error(`[mlx] inference crashed: ${event.exit.reason}`);
      for (const line of event.exit.stderrTail.slice(-10)) console.error(`[mlx]   ${line}`);
      return;
    }
    if (event.type === 'gave-up') {
      console.error(`[mlx] inference gave up after ${event.consecutiveCrashes} crashes`);
      return;
    }
    if (event.type === 'lying') {
      // The one failure with no other symptom: `/health` says ok, no JS error,
      // no HTTP error, and the output is wrong.
      console.error(`[mlx] native error swallowed in ${event.error.context}: ${event.error.detail}`);
    }
  });

  broker = createControlPanelBroker<WebContents>({
    ...electronBrokerDeps({
      entry: paths.controlPanelEntry,
      env: { ...process.env, ...controlPanelEnvOverrides({ modelsDir: settings.modelsDir }) } as Record<string, string>,
      onLog: (stream, line) => {
        if (stream === 'stderr') console.error(`[mlx] control panel: ${line}`);
      },
    }),
    report: (event) => {
      if (event.type === 'brokered') return;
      console.error(`[mlx] control panel ${event.type}:`, event);
    },
  });

  controlPanel = createControlPanelWindowManager({
    preload: paths.controlPanelPreload,
    bounds: () => settings.controlPanelWindow,
    workAreas: () => screen.getAllDisplays().map((display): Rect => display.workArea),
    isQuitting: () => quitting,
    onBounds: (bounds: WindowBounds) => {
      settings = { ...settings, controlPanelWindow: bounds };
      scheduleSave();
    },
    // The renderer asked, so mint it a channel. Pull, not push: a transferred
    // port is consumed once, and a reload gets a fresh one because the old one
    // died with the old page.
    onControlPanelReady: (contents) => {
      broker?.attach(contents);
    },
    onControlPanelUnresponsive: (contents, expectedGeneration) => {
      broker?.recover(contents, expectedGeneration);
    },
  });

  const copyClientCommand = (build: (endpoint: string, token: string) => string): void => {
    const endpoint = supervisor?.snapshot().url;
    const token = supervisor?.connectionToken();
    // The menu items are disabled unless running, but a crash between the
    // render and the click is entirely possible — and half a command on the
    // clipboard is worse than none.
    if (endpoint === undefined || endpoint === null || token === undefined || token === null) return;
    clipboard.writeText(build(endpoint, token));
  };

  tray = createTray({
    iconPath: paths.trayIcon,
    showInDock: () => settings.showInDock,
    actions: {
      openControlPanel: () => controlPanel?.show(),
      startInference: () => {
        void supervisor?.start().catch(reportInferenceFailure);
      },
      stopInference: () => {
        void supervisor?.stop().catch(reportInferenceFailure);
      },
      restartInference: () => {
        void supervisor?.restart().catch(reportInferenceFailure);
      },
      copyClaudeConnectCommand: () => copyClientCommand(claudeConnectCommand),
      copyCodexConnectCommand: () => copyClientCommand(codexConnectCommand),
      setShowInDock: (next: boolean) => {
        settings = { ...settings, showInDock: next };
        applyDockPolicy(next);
        scheduleSave();
        refreshTray();
      },
      quit: () => {
        app.quit();
      },
    },
  });
  refreshTray();

  if (settings.autoStartInference) {
    void supervisor.start().catch(reportInferenceFailure);
  }
  // Only on the very first launch. Otherwise a menubar app that reopens its
  // window on every start is indistinguishable from one that ignores the close
  // button — and without it, a first-time user gets a menubar icon they have no
  // reason to look for.
  if (loaded.source === 'missing') controlPanel.show();

  // Materialise the file whenever we did not read one. Without this "first run"
  // never ends — nothing else writes settings until the user moves the window or
  // toggles something, so the window would reopen on every single launch.
  // `unreadable` is excluded on purpose: we could not read that file, so we do
  // not get to replace it.
  if (loaded.source === 'missing' || loaded.source === 'corrupt') scheduleSave();
}

function refreshTray(): void {
  if (supervisor === null) return;
  tray?.update(supervisor.snapshot());
}

/**
 * The supervisor already reports every failure through the tray; a rejected
 * `start()` only adds the reason. Swallowing it would leave an unhandled
 * rejection, which in a packaged app is a crash dialog with no context.
 */
function reportInferenceFailure(error: unknown): void {
  console.error('[mlx] inference:', error);
}

function applyDockPolicy(show: boolean): void {
  // Menubar-only by default. The Dock icon is opt-in because the app's whole
  // resting state is a supervisor and an icon — there is no document to return
  // to, and `app.dock.show()` makes the choice reversible without a relaunch,
  // which `LSUIElement` alone does not.
  if (show) void app.dock?.show();
  else app.dock?.hide();
}

function scheduleSave(): void {
  saveDirty = true;
  if (saveTimer !== null) return;
  saveTimer = setTimeout(() => {
    saveTimer = null;
    void flushSettings();
  }, SETTINGS_SAVE_DEBOUNCE_MS);
  // Must never be the reason the process is still alive during quit.
  saveTimer.unref();
}

/**
 * Flush pending settings, serialized against any write already in flight.
 *
 * Two writes must never overlap: `saveSettings` names its temp file after the
 * pid alone, so a second concurrent write reuses the same path and the loser's
 * `rename` fails ENOENT. Measured over 300 rounds of two overlapping saves,
 * that produced a spurious "could not save settings" every time and silently
 * kept the OLDER value 2.5% of the time.
 *
 * Returning `saveInFlight` when there is nothing new is what lets `shutdown()`
 * await a write the debounce timer started microseconds earlier — the timer
 * clears `saveDirty` before its I/O, so a plain early return told shutdown the
 * work was done while it was still in the syscall.
 */
function flushSettings(): Promise<void> {
  if (!saveDirty || paths === null) return saveInFlight;
  saveDirty = false;
  const file = paths.settingsFile;
  // Snapshot: the queued write must persist what made it dirty, not whatever
  // `settings` happens to hold once the queue reaches it.
  const snapshot = settings;
  saveInFlight = saveInFlight.then(async () => {
    try {
      await saveSettings(file, snapshot);
    } catch (error) {
      // A settings write that fails is a lost preference, not a reason to
      // refuse to quit.
      console.error('[mlx] could not save settings:', error);
    }
  });
  return saveInFlight;
}

async function shutdown(): Promise<void> {
  if (saveTimer !== null) {
    clearTimeout(saveTimer);
    saveTimer = null;
  }
  await flushSettings();
  controlPanel?.dispose();
  tray?.dispose();
  // Both awaited, and in parallel: CONTROL PANEL's shutdown drains in-flight downloads
  // so a partial multi-GB `.staging` tree is not orphaned, and INFERENCE's frees
  // its temp root. Serialising them would spend two kill-grace periods against
  // one whole-app quit deadline.
  await Promise.all([broker?.dispose(), supervisor?.dispose()]);
}

async function withDeadline(work: Promise<void>, ms: number): Promise<void> {
  let timer: NodeJS.Timeout | undefined;
  const capped = new Promise<void>((resolve) => {
    timer = setTimeout(() => {
      console.error(`[mlx] shutdown did not finish within ${ms}ms; exiting anyway`);
      resolve();
    }, ms);
    timer.unref();
  });
  try {
    await Promise.race([work, capped]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}
