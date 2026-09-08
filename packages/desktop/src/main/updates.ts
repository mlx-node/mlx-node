import type { AppUpdater } from 'electron-updater';

export const UPDATE_INTERVAL_MS = 6 * 60 * 60 * 1_000;
const STABLE_VERSION = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/;

export type UpdateStatus =
  | 'disabled'
  | 'idle'
  | 'checking'
  | 'downloading'
  | 'current'
  | 'ready'
  | 'installing'
  | 'error';

export interface UpdatePresentation {
  label: string;
  enabled: boolean;
}

export function presentUpdate(status: UpdateStatus, percent: number | null = null): UpdatePresentation {
  switch (status) {
    case 'disabled':
      return { label: 'Updates Unavailable in This Build', enabled: false };
    case 'idle':
      return { label: 'Check for Updates…', enabled: true };
    case 'checking':
      return { label: 'Checking for Updates…', enabled: false };
    case 'downloading':
      return { label: percent === null ? 'Downloading Update…' : `Downloading Update… ${percent}%`, enabled: false };
    case 'current':
      return { label: 'Up to Date — Check Again…', enabled: true };
    case 'ready':
      return { label: 'Restart to Update…', enabled: true };
    case 'installing':
      return { label: 'Restarting to Update…', enabled: false };
    case 'error':
      return { label: 'Update Failed — Try Again…', enabled: true };
  }
}

/** Only signed, packaged stable builds participate in the public release feed. */
export function canAutoUpdate(options: {
  packaged: boolean;
  enabled: boolean;
  platform: string;
  arch: string;
  version: string;
}): boolean {
  return (
    options.packaged &&
    options.enabled &&
    options.platform === 'darwin' &&
    options.arch === 'arm64' &&
    STABLE_VERSION.test(options.version)
  );
}

/** The library defaults to Darwin kernel versions; our manifest uses macOS versions. */
export function isMacUpdateSupported(systemVersion: string, minimumSystemVersion: string | undefined): boolean {
  const valid = /^\d+(?:\.\d+){0,2}$/;
  if (minimumSystemVersion === undefined || !valid.test(minimumSystemVersion) || !valid.test(systemVersion))
    return false;
  const current = systemVersion.split('.').map(Number);
  const minimum = minimumSystemVersion.split('.').map(Number);
  for (let i = 0; i < 3; i++) {
    const difference = (current[i] ?? 0) - (minimum[i] ?? 0);
    if (difference !== 0) return difference > 0;
  }
  return true;
}

export interface DesktopUpdater {
  status(): UpdateStatus;
  progress(): number | null;
  start(): void;
  check(): void;
  restartAndInstall(): void;
  /** Stop checks and UI notifications as soon as graceful shutdown starts. */
  stop(): void;
  /** Called only AFTER settings and child processes have finished shutting down. */
  installOnQuit(relaunchRequested?: boolean): boolean;
}

interface UpdateClient extends Pick<
  AppUpdater,
  | 'quitAndInstall'
  | 'autoDownload'
  | 'autoInstallOnAppQuit'
  | 'autoRunAppAfterInstall'
  | 'allowDowngrade'
  | 'allowPrerelease'
  | 'isUpdateSupported'
> {
  checkForUpdates(): Promise<{ downloadPromise?: Promise<unknown> | null } | null>;
  on(event: 'error', listener: (error: Error) => void): unknown;
  on(event: 'update-available', listener: () => void): unknown;
  on(event: 'update-not-available', listener: () => void): unknown;
  on(event: 'update-downloaded', listener: () => void): unknown;
  on(event: 'download-progress', listener: (progress: { percent: number }) => void): unknown;
}

/** Electron is injected so download and shutdown races can be tested without a GUI. */
export function createDesktopUpdater(options: {
  enabled: boolean;
  systemVersion: string;
  native: UpdateClient;
  onChange(): void;
  requestQuit(): void;
  report(error: unknown): void;
}): DesktopUpdater {
  let status: UpdateStatus = options.enabled ? 'idle' : 'disabled';
  let percent: number | null = null;
  let started = false;
  let stopped = false;
  let inFlight = false;
  let lastError: unknown;
  let installRequested = false;
  let installStarted = false;
  let timer: NodeJS.Timeout | null = null;

  function change(next: UpdateStatus): void {
    status = next;
    // Remember a download that finishes during the drain: a queued Finder
    // relaunch must still go through Squirrel, even after the tray is gone.
    if (!stopped) options.onChange();
  }

  function failed(error: unknown): void {
    // The library emits an error AND rejects its check/download promise.
    if (error === lastError) return;
    lastError = error;
    options.report(error);
    // If Squirrel fails after our shutdown, finish quitting. Leaving the app
    // alive here would leave an invisible process with no tray or children.
    if (installStarted) {
      options.requestQuit();
      return;
    }
    change('error');
  }

  if (options.enabled) {
    // Provider/cache configuration comes from Resources/app-update.yml, staged
    // before signing. Stable clients never opt into prereleases or downgrades.
    options.native.autoDownload = true;
    options.native.autoInstallOnAppQuit = true;
    options.native.allowPrerelease = false;
    options.native.allowDowngrade = false;
    options.native.isUpdateSupported = (info) =>
      STABLE_VERSION.test(info.version) && isMacUpdateSupported(options.systemVersion, info.minimumSystemVersion);
    // Keep the error listener through shutdown: an in-flight native download
    // can still fail after stop(), and an unhandled 'error' terminates MAIN.
    options.native.on('error', failed);
    options.native.on('update-available', () => {
      if (status === 'checking') change('downloading');
    });
    options.native.on('update-not-available', () => {
      if (status === 'checking') change('current');
    });
    options.native.on('update-downloaded', () => {
      if (status === 'checking' || status === 'downloading') change('ready');
    });
    options.native.on('download-progress', (progress) => {
      if (status !== 'downloading' || !Number.isFinite(progress.percent)) return;
      const next = Math.max(0, Math.min(100, Math.floor(progress.percent)));
      if (next === percent) return;
      percent = next;
      if (!stopped) options.onChange();
    });
  }

  function check(): void {
    if (stopped || !options.enabled || inFlight || !['idle', 'current', 'error'].includes(status)) return;
    // Set this before calling Electron. Overlapping checks download the same
    // archive twice, including when a manual check races the interval timer.
    change('checking');
    percent = null;
    lastError = undefined;
    inFlight = true;
    void (async () => {
      try {
        const result = await options.native.checkForUpdates();
        // Downloads have a separate promise. Observing only the check would
        // leave network/disk failures as unhandled rejections in MAIN.
        await result?.downloadPromise;
      } catch (error) {
        failed(error);
      } finally {
        inFlight = false;
      }
    })();
  }

  return {
    status: () => status,
    progress: () => percent,
    start(): void {
      if (started || stopped || !options.enabled) return;
      started = true;
      timer = setInterval(check, UPDATE_INTERVAL_MS);
      timer.unref();
      check();
    },
    check,
    restartAndInstall(): void {
      if (stopped || status !== 'ready') return;
      installRequested = true;
      change('installing');
      // Go through app.quit() first. quitAndInstall() closes windows BEFORE
      // before-quit, which otherwise hits our window's hide-on-close policy.
      options.requestQuit();
    },
    stop(): void {
      stopped = true;
      if (timer !== null) clearInterval(timer);
      timer = null;
    },
    installOnQuit(relaunchRequested = false): boolean {
      if (!installRequested && status !== 'ready') return false;
      if (!installStarted) {
        installStarted = true;
        try {
          // Wait for Squirrel to finish staging even on an ordinary quit. Only
          // an explicit restart (including Finder activation) should reopen us.
          // MacUpdater honors this flag: false calls app.quit(), while true
          // calls Electron's native quitAndInstall() to request a relaunch.
          options.native.autoRunAppAfterInstall = installRequested || relaunchRequested;
          options.native.quitAndInstall();
        } catch (error) {
          failed(error);
        }
      }
      return true;
    },
  };
}
