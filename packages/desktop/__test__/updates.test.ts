import { EventEmitter } from 'node:events';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { createQuitHandler } from '../src/main/quit.js';
import {
  canAutoUpdate,
  createDesktopUpdater,
  isMacUpdateSupported,
  presentUpdate,
  UPDATE_INTERVAL_MS,
} from '../src/main/updates.js';

const release = { packaged: true, enabled: true, platform: 'darwin', arch: 'arm64', version: '0.0.13' };

class NativeUpdater extends EventEmitter {
  autoDownload = false;
  autoInstallOnAppQuit = false;
  autoRunAppAfterInstall = false;
  allowPrerelease = true;
  allowDowngrade = true;
  isUpdateSupported = (_info: { version: string; minimumSystemVersion?: string }) => true;
  checkForUpdates = vi.fn<() => Promise<{ downloadPromise?: Promise<unknown> } | null>>().mockResolvedValue(null);
  quitAndInstall = vi.fn();
}

function harness(enabled = canAutoUpdate(release)) {
  const native = new NativeUpdater();
  const onChange = vi.fn();
  const requestQuit = vi.fn();
  const report = vi.fn();
  const updater = createDesktopUpdater({
    enabled,
    systemVersion: '26.0',
    native,
    squirrel: native,
    onChange,
    requestQuit,
    report,
  });
  return { native, updater, onChange, requestQuit, report };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe('release update feed', () => {
  it('enables signed stable Apple Silicon releases', () => {
    expect(canAutoUpdate(release)).toBe(true);
  });

  it.each([
    { packaged: false },
    { enabled: false },
    { platform: 'linux' },
    { arch: 'x64' },
    { version: '0.0.14-rc.1' },
    { version: '0.0.14+local' },
    { version: 'nightly' },
    { version: '0.01.0' },
  ])('disables updates for an ineligible build: %j', (override) => {
    expect(canAutoUpdate({ ...release, ...override })).toBe(false);
  });
});

describe('desktop updates', () => {
  it('enables background download and staging with stable-only version policy', () => {
    const { native } = harness();
    expect(native.autoDownload).toBe(true);
    expect(native.autoInstallOnAppQuit).toBe(true);
    expect(native.allowPrerelease).toBe(false);
    expect(native.allowDowngrade).toBe(false);
    expect(native.isUpdateSupported({ version: '0.0.14', minimumSystemVersion: '26.1' })).toBe(false);
    expect(native.isUpdateSupported({ version: '0.0.14', minimumSystemVersion: '26.0' })).toBe(true);
    expect(native.isUpdateSupported({ version: '0.0.14-rc.1', minimumSystemVersion: '26.0' })).toBe(false);
  });

  it('catches check and separate download promise failures once, then permits retry', async () => {
    const { updater, native, report } = harness();
    const error = new Error('check failed');
    native.checkForUpdates.mockImplementationOnce(async () => {
      native.emit('error', error);
      throw error;
    });
    updater.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(updater.status()).toBe('error');
    expect(report).toHaveBeenCalledTimes(1);

    const download = Promise.withResolvers<void>();
    native.checkForUpdates.mockImplementationOnce(async () => {
      native.emit('update-available');
      return { downloadPromise: download.promise };
    });
    updater.check();
    await vi.advanceTimersByTimeAsync(0);
    expect(updater.status()).toBe('downloading');
    const failure = new Error('download failed');
    native.emit('error', failure);
    updater.check(); // The failed download has not settled yet.
    expect(native.checkForUpdates).toHaveBeenCalledTimes(2);
    download.reject(failure);
    await vi.advanceTimersByTimeAsync(0);
    expect(report).toHaveBeenCalledTimes(2);
    expect(updater.status()).toBe('error');
    updater.check();
    expect(native.checkForUpdates).toHaveBeenCalledTimes(3);
  });

  it('shows bounded whole-number download progress without redundant menu rebuilds', () => {
    const { updater, native, onChange } = harness();
    updater.start();
    native.emit('update-available');
    onChange.mockClear();
    for (const percent of [42.1, 42.9, Number.NaN, Infinity]) native.emit('download-progress', { percent });
    expect(updater.progress()).toBe(42);
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(presentUpdate(updater.status(), updater.progress()).label).toContain('42%');
    native.emit('download-progress', { percent: 102 });
    expect(updater.progress()).toBe(100);
    updater.stop();
    onChange.mockClear();
    native.emit('download-progress', { percent: 50 });
    expect(onChange).not.toHaveBeenCalled();
  });

  it('finishes staging on an ordinary quit without requesting a relaunch', () => {
    const { updater, native } = harness();
    updater.start();
    native.emit('update-downloaded');
    updater.stop();
    expect(updater.installOnQuit(false, vi.fn())).toBe(true);
    expect(native.autoRunAppAfterInstall).toBe(false);
    expect(native.quitAndInstall).toHaveBeenCalledTimes(1);
  });

  it('never configures native updates or starts timers for disabled builds', () => {
    const { updater, native } = harness(false);
    updater.start();
    updater.check();
    updater.restartAndInstall();
    expect(updater.status()).toBe('disabled');
    expect(native.checkForUpdates).not.toHaveBeenCalled();
    expect(native.autoDownload).toBe(false);
    expect(native.quitAndInstall).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('checks on startup and periodically, without overlapping checks or downloads', async () => {
    const { updater, native } = harness();
    updater.start();
    updater.start();
    expect(updater.status()).toBe('checking');
    updater.check();
    await vi.advanceTimersByTimeAsync(UPDATE_INTERVAL_MS);
    expect(native.checkForUpdates).toHaveBeenCalledTimes(1);
    native.emit('update-not-available');
    expect(updater.status()).toBe('current');
    await vi.advanceTimersByTimeAsync(UPDATE_INTERVAL_MS);
    expect(native.checkForUpdates).toHaveBeenCalledTimes(2);
    native.emit('update-available');
    expect(updater.status()).toBe('downloading');
    updater.check();
    await vi.advanceTimersByTimeAsync(UPDATE_INTERVAL_MS * 2);
    expect(native.checkForUpdates).toHaveBeenCalledTimes(2);
  });

  it('keeps a downloaded update ready without interrupting inference or downloading again', async () => {
    const { updater, native, requestQuit } = harness();
    updater.start();
    updater.restartAndInstall();
    expect(requestQuit).not.toHaveBeenCalled();
    native.emit('update-available');
    native.emit('update-downloaded', {}, 'Notes', 'v0.0.14', new Date(), 'https://github.com/');
    expect(updater.status()).toBe('ready');
    expect(requestQuit).not.toHaveBeenCalled();
    updater.check();
    await vi.advanceTimersByTimeAsync(UPDATE_INTERVAL_MS);
    expect(native.checkForUpdates).toHaveBeenCalledTimes(1);
    expect(native.quitAndInstall).not.toHaveBeenCalled();
  });

  it('recovers from a download error on the next scheduled or manual check', async () => {
    const { updater, native, report } = harness();
    updater.start();
    native.emit('update-available');
    const error = new Error('network disconnected');
    native.emit('error', error);
    expect(report).toHaveBeenCalledWith(error);
    expect(updater.status()).toBe('error');
    await vi.advanceTimersByTimeAsync(UPDATE_INTERVAL_MS);
    expect(native.checkForUpdates).toHaveBeenCalledTimes(2);
    native.emit('update-not-available');
    await vi.advanceTimersByTimeAsync(0);
    updater.check();
    expect(native.checkForUpdates).toHaveBeenCalledTimes(3);
  });

  it('handles a synchronous check failure without throwing out of a menu action', async () => {
    const { updater, native, report } = harness();
    native.checkForUpdates.mockImplementationOnce(() => {
      throw new Error('check failed');
    });
    expect(() => updater.start()).not.toThrow();
    expect(updater.status()).toBe('error');
    expect(report).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(0);
    updater.check();
    expect(updater.status()).toBe('checking');
  });

  it('stops timers and ignores late results while retaining an error handler', async () => {
    const { updater, native, onChange, requestQuit, report } = harness();
    updater.start();
    updater.stop();
    onChange.mockClear();
    native.emit('update-available');
    native.emit('update-downloaded');
    native.emit('update-not-available');
    native.emit('error', new Error('late failure'));
    updater.check();
    updater.start();
    updater.restartAndInstall();
    await vi.advanceTimersByTimeAsync(UPDATE_INTERVAL_MS * 2);
    expect(vi.getTimerCount()).toBe(0);
    expect(native.checkForUpdates).toHaveBeenCalledTimes(1);
    expect(onChange).not.toHaveBeenCalled();
    expect(requestQuit).not.toHaveBeenCalled();
    expect(report).toHaveBeenCalledTimes(1);
  });

  it('drains once before installing and leaves relaunch to Squirrel even during activation', async () => {
    const { updater, native, requestQuit } = harness();
    const drain = Promise.withResolvers<void>();
    const shutdown = vi.fn(() => drain.promise);
    const relaunch = vi.fn();
    const finalQuit = vi.fn();
    const quit = createQuitHandler({
      beginShutdown: () => updater.stop(),
      shutdown,
      deadlineMs: 14_000,
      installUpdate: (relaunchRequested, allowQuit) => updater.installOnQuit(relaunchRequested, allowQuit),
      shouldRelaunch: () => true,
      relaunch,
      quit: finalQuit,
      report: vi.fn(),
    });
    const prevented = vi.fn();
    requestQuit.mockImplementation(() => quit({ preventDefault: prevented }));
    updater.start();
    native.emit('update-available');
    native.emit('update-downloaded');
    updater.restartAndInstall();
    updater.restartAndInstall();
    quit({ preventDefault: prevented }); // Cmd+Q during the drain must also wait.
    expect(requestQuit).toHaveBeenCalledTimes(1);
    expect(shutdown).toHaveBeenCalledTimes(1);
    expect(prevented).toHaveBeenCalledTimes(2);
    expect(native.quitAndInstall).not.toHaveBeenCalled();
    drain.resolve();
    await vi.advanceTimersByTimeAsync(0);
    expect(native.quitAndInstall).toHaveBeenCalledTimes(1);
    expect(native.autoRunAppAfterInstall).toBe(true);
    expect(relaunch).not.toHaveBeenCalled();
    expect(finalQuit).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
    updater.installOnQuit(false, vi.fn());
    expect(native.quitAndInstall).toHaveBeenCalledTimes(1);
    quit({ preventDefault: prevented }); // Squirrel's final app.quit().
    expect(prevented).toHaveBeenCalledTimes(2);
  });

  it.each(['throw', 'event'])('finishes quitting if installation fails through %s', (failure) => {
    const { updater, native, requestQuit, report } = harness();
    updater.start();
    native.emit('update-downloaded');
    updater.restartAndInstall();
    updater.stop();
    if (failure === 'throw')
      native.quitAndInstall.mockImplementationOnce(() => {
        throw new Error('install failed');
      });
    expect(updater.installOnQuit(false, vi.fn())).toBe(true);
    if (failure === 'event') native.emit('error', new Error('install failed'));
    expect(report).toHaveBeenCalledTimes(1);
    expect(requestQuit).toHaveBeenCalledTimes(2);
  });

  it.each(['before quit', 'during quit'])(
    'uses an update downloaded %s for a queued Finder relaunch',
    async (downloaded) => {
      const { updater, native, requestQuit } = harness();
      updater.start();
      native.emit('update-available');
      if (downloaded === 'before quit') native.emit('update-downloaded');
      const drain = Promise.withResolvers<void>();
      let activated = false;
      const relaunch = vi.fn();
      const finalQuit = vi.fn();
      const quit = createQuitHandler({
        beginShutdown: () => updater.stop(),
        shutdown: () => drain.promise,
        deadlineMs: 14_000,
        installUpdate: (relaunchRequested, allowQuit) => updater.installOnQuit(relaunchRequested, allowQuit),
        shouldRelaunch: () => activated,
        relaunch,
        quit: finalQuit,
        report: vi.fn(),
      });
      quit({ preventDefault: vi.fn() });
      activated = true;
      if (downloaded === 'during quit') native.emit('update-downloaded');
      drain.resolve();
      await vi.advanceTimersByTimeAsync(0);
      expect(native.quitAndInstall).toHaveBeenCalledTimes(1);
      expect(requestQuit).not.toHaveBeenCalled();
      expect(relaunch).not.toHaveBeenCalled();
      expect(finalQuit).not.toHaveBeenCalled();
    },
  );
});

describe('minimum macOS version', () => {
  it.each([
    ['26.0', '26.0.0', true],
    ['26.1', '26.0', true],
    ['26.0', '26.1', false],
    ['15.7', '26.0', false],
    ['13.7.1', '13.7.2', false],
    ['26.0', undefined, false],
    ['26.0', 'invalid', false],
  ] as const)('compares macOS %s against %s', (current, minimum, supported) => {
    expect(isMacUpdateSupported(current, minimum)).toBe(supported);
  });
});

describe('update menu', () => {
  it('makes busy and disabled states non-actionable and offers retry and restart', () => {
    for (const status of ['disabled', 'checking', 'downloading', 'installing'] as const) {
      expect(presentUpdate(status).enabled).toBe(false);
    }
    for (const status of ['idle', 'current', 'ready', 'error'] as const) {
      expect(presentUpdate(status).enabled).toBe(true);
    }
    expect(presentUpdate('ready').label).toBe('Restart to Update…');
    expect(presentUpdate('error').label).toContain('Try Again');
  });
});
