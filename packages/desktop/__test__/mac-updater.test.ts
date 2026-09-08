import { EventEmitter } from 'node:events';
import { createRequire, Module } from 'node:module';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { createQuitHandler } from '../src/main/quit.js';
import { createDesktopUpdater } from '../src/main/updates.js';

const require = createRequire(import.meta.url);
const { MacUpdater } =
  require('electron-updater/out/MacUpdater.js') as typeof import('electron-updater/out/MacUpdater.js');

function harness() {
  const squirrel = Object.assign(new EventEmitter(), {
    quitAndInstall: vi.fn(),
    checkForUpdates: vi.fn(),
  });
  const app = {
    version: '0.0.13',
    name: 'mlx-node',
    isPackaged: true,
    appUpdateConfigPath: '',
    userDataPath: '',
    baseCachePath: '',
    whenReady: async () => {},
    relaunch: vi.fn(),
    quit: vi.fn(),
    onQuit: vi.fn(),
  };

  // MacUpdater loads Electron through CommonJS in its constructor. Replace
  // only that native boundary, then immediately restore the module cache.
  // Its staging wait and quit/relaunch methods remain the shipped library code.
  const electronPath = require.resolve('electron');
  const previous = require.cache[electronPath];
  const electronModule = new Module(electronPath);
  electronModule.exports = { autoUpdater: squirrel };
  require.cache[electronPath] = electronModule;
  let client: InstanceType<typeof MacUpdater>;
  try {
    client = new MacUpdater(undefined, app);
  } finally {
    if (previous) require.cache[electronPath] = previous;
    else delete require.cache[electronPath];
  }
  client.logger = null;
  vi.spyOn(client, 'checkForUpdates').mockResolvedValue(null);
  const requestQuit = vi.fn();
  const report = vi.fn();
  const updater = createDesktopUpdater({
    enabled: true,
    systemVersion: '26.0',
    native: client,
    onChange: vi.fn(),
    requestQuit,
    report,
  });
  return { squirrel, app, client, updater, requestQuit, report };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe.each(['already staged', 'still staging'] as const)('MacUpdater with Squirrel %s', (staging) => {
  it.each(['ordinary quit', 'Restart to Update', 'Finder relaunch'] as const)(
    'honors the relaunch choice for %s after shutdown',
    async (action) => {
      const { squirrel, app, client, updater, requestQuit, report } = harness();
      const drain = Promise.withResolvers<void>();
      const quit = createQuitHandler({
        beginShutdown: () => updater.stop(),
        shutdown: () => drain.promise,
        deadlineMs: 14_000,
        installUpdate: (relaunchRequested) => updater.installOnQuit(relaunchRequested),
        shouldRelaunch: () => action === 'Finder relaunch',
        relaunch: app.relaunch,
        quit: app.quit,
        report,
      });
      const prevented = vi.fn();
      requestQuit.mockImplementation(() => quit({ preventDefault: prevented }));

      updater.start();
      client.emit('update-downloaded', {
        version: '0.0.14',
        files: [],
        path: 'update.zip',
        sha512: '',
        releaseDate: '2026-09-08T00:00:00Z',
        downloadedFile: 'update.zip',
      });
      if (staging === 'already staged') squirrel.emit('update-downloaded');
      if (action === 'Restart to Update') updater.restartAndInstall();
      else quit({ preventDefault: prevented });
      expect(prevented).toHaveBeenCalledTimes(1);
      expect(app.quit).not.toHaveBeenCalled();
      expect(squirrel.quitAndInstall).not.toHaveBeenCalled();

      drain.resolve();
      await vi.advanceTimersByTimeAsync(0);
      if (staging === 'still staging') {
        expect(app.quit).not.toHaveBeenCalled();
        expect(squirrel.quitAndInstall).not.toHaveBeenCalled();
        squirrel.emit('update-downloaded');
      }

      // Assert the actual native endpoint, not just autoRunAppAfterInstall:
      // ordinary quit must stay closed; an explicit restart belongs to Squirrel.
      expect(app.quit).toHaveBeenCalledTimes(action === 'ordinary quit' ? 1 : 0);
      expect(squirrel.quitAndInstall).toHaveBeenCalledTimes(action === 'ordinary quit' ? 0 : 1);
      expect(app.relaunch).not.toHaveBeenCalled();
      expect(squirrel.checkForUpdates).not.toHaveBeenCalled();
      expect(report).not.toHaveBeenCalled();
      expect(vi.getTimerCount()).toBe(0);
      quit({ preventDefault: prevented });
      expect(prevented).toHaveBeenCalledTimes(1);
    },
  );
});
