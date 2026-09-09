import { EventEmitter } from 'node:events';
import { createRequire, Module } from 'node:module';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { createLaunchVisibility } from '../src/main/launch-visibility.js';
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
    squirrel,
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
        installUpdate: (completeQuit) => updater.installOnQuit(completeQuit),
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
        quit({ preventDefault: prevented });
        quit({ preventDefault: prevented });
        expect(prevented).toHaveBeenCalledTimes(3);
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
      const blockedQuits = prevented.mock.calls.length;
      quit({ preventDefault: prevented });
      expect(prevented).toHaveBeenCalledTimes(blockedQuits);
    },
  );
});

it('releases a blocked quit on native staging failure and ignores late completion', async () => {
  const { squirrel, app, client, updater, requestQuit, report } = harness();
  const quit = createQuitHandler({
    beginShutdown: () => updater.stop(),
    shutdown: async () => {},
    deadlineMs: 14_000,
    installUpdate: (completeQuit) => updater.installOnQuit(completeQuit),
    shouldRelaunch: () => false,
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
  quit({ preventDefault: prevented });
  await vi.advanceTimersByTimeAsync(0);
  quit({ preventDefault: prevented });
  expect(prevented).toHaveBeenCalledTimes(2);
  const error = new Error('native staging failed');
  squirrel.emit('error', error);
  expect(report).toHaveBeenCalledWith(error);
  expect(app.quit).toHaveBeenCalledTimes(1);
  quit({ preventDefault: prevented });
  expect(prevented).toHaveBeenCalledTimes(2);
  squirrel.emit('update-downloaded');
  expect(squirrel.quitAndInstall).not.toHaveBeenCalled();
  expect(app.relaunch).not.toHaveBeenCalled();
  expect(vi.getTimerCount()).toBe(0);
});

it('finishes an explicit update quit when staging fails during the resource drain', async () => {
  const { squirrel, app, client, updater, requestQuit, report } = harness();
  const drain = Promise.withResolvers<void>();
  const quit = createQuitHandler({
    beginShutdown: () => updater.stop(),
    shutdown: () => drain.promise,
    deadlineMs: 14_000,
    installUpdate: (completeQuit) => updater.installOnQuit(completeQuit),
    shouldRelaunch: () => false,
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
  updater.restartAndInstall();
  squirrel.emit('error', new Error('staging failed during shutdown'));
  expect(app.quit).not.toHaveBeenCalled();
  drain.resolve();
  await vi.advanceTimersByTimeAsync(0);
  expect(app.quit).toHaveBeenCalledTimes(1);
  expect(squirrel.quitAndInstall).not.toHaveBeenCalled();
  quit({ preventDefault: prevented });
  expect(prevented).toHaveBeenCalledTimes(1);
  expect(report).toHaveBeenCalledTimes(1);
});

it.each(['success', 'staging error', 'install throws', 'install error'])(
  'honors a Finder/Dock open during native staging through %s',
  async (outcome) => {
    const { squirrel, app, client, updater, report } = harness();
    const visibility = createLaunchVisibility();
    const shouldRelaunch = vi.fn(() => visibility.takeRelaunchRequest());
    const quit = createQuitHandler({
      beginShutdown: () => {
        updater.stop();
        visibility.beginShutdown();
      },
      shutdown: async () => {},
      deadlineMs: 14_000,
      installUpdate: (completeQuit) => updater.installOnQuit(completeQuit),
      shouldRelaunch,
      relaunch: app.relaunch,
      quit: app.quit,
      report,
    });
    const prevented = vi.fn();
    updater.start();
    client.emit('update-downloaded', {
      version: '0.0.14',
      files: [],
      path: 'update.zip',
      sha512: '',
      releaseDate: '2026-09-08T00:00:00Z',
      downloadedFile: 'update.zip',
    });
    quit({ preventDefault: prevented });
    await vi.advanceTimersByTimeAsync(0);
    expect(shouldRelaunch).not.toHaveBeenCalled();
    visibility.activate(); // Finder/Dock reopens after the resource drain.
    quit({ preventDefault: prevented });
    expect(prevented).toHaveBeenCalledTimes(2);
    const error = new Error('native update failed');
    if (outcome === 'staging error') {
      squirrel.emit('error', error);
    } else {
      if (outcome === 'install throws')
        squirrel.quitAndInstall.mockImplementationOnce(() => {
          throw error;
        });
      squirrel.emit('update-downloaded');
      if (outcome === 'install error') squirrel.emit('error', error);
      expect(client.autoRunAppAfterInstall).toBe(true);
    }
    expect(squirrel.quitAndInstall).toHaveBeenCalledTimes(outcome === 'staging error' ? 0 : 1);
    expect(app.relaunch).toHaveBeenCalledTimes(outcome === 'success' ? 0 : 1);
    expect(app.quit).toHaveBeenCalledTimes(outcome === 'success' ? 0 : 1);
    expect(report).toHaveBeenCalledTimes(outcome === 'success' ? 0 : 1);
    quit({ preventDefault: prevented });
    expect(prevented).toHaveBeenCalledTimes(2);
  },
);
