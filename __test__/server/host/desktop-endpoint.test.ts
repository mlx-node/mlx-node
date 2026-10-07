import { chmod, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, expect, it, vi } from 'vite-plus/test';

import {
  publishDesktopEndpoint,
  readDesktopEndpoint,
  type DesktopEndpoint,
} from '../../../packages/server/src/host/desktop-endpoint.js';

const dirs: string[] = [];
afterEach(async () => {
  vi.restoreAllMocks();
  await Promise.all(dirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});
async function location() {
  const dir = await mkdtemp(join(tmpdir(), 'mlx-desktop-endpoint-'));
  dirs.push(dir);
  return join(dir, 'desktop', 'inference.json');
}
const endpoint: DesktopEndpoint = {
  models: [{ name: 'local', path: '/models/local' }],
  version: 1,
  pid: process.pid,
  url: 'http://127.0.0.1:12345',
  token: 'private',
};

it('publishes privately and removes only its own generation', async () => {
  const path = await location();
  const releaseOld = await publishDesktopEndpoint(endpoint, path);
  expect((await stat(path)).mode & 0o777).toBe(0o600);
  expect((await stat(join(path, '..'))).mode & 0o777).toBe(0o700);
  expect(await readDesktopEndpoint(path)).toEqual(endpoint);
  const releaseNew = await publishDesktopEndpoint({ ...endpoint, token: 'new' }, path);
  await releaseOld();
  expect(JSON.parse(await readFile(path, 'utf8')).token).toBe('new');
  await releaseNew();
  expect(await readDesktopEndpoint(path)).toBeUndefined();
});

it('ignores a dead process but never treats permission errors as a dead engine', async () => {
  const path = await location();
  await publishDesktopEndpoint(endpoint, path);
  const kill = vi.spyOn(process, 'kill').mockImplementation(() => {
    throw Object.assign(new Error('dead'), { code: 'ESRCH' });
  });
  expect(await readDesktopEndpoint(path)).toBeUndefined();
  kill.mockImplementation(() => {
    throw Object.assign(new Error('denied'), { code: 'EPERM' });
  });
  await expect(readDesktopEndpoint(path)).rejects.toThrow('denied');
});

it('rejects exposed credentials and non-loopback destinations', async () => {
  const path = await location();
  await publishDesktopEndpoint(endpoint, path);
  await chmod(path, 0o644);
  await expect(readDesktopEndpoint(path)).rejects.toThrow('private');
  await chmod(path, 0o600);
  await writeFile(path, JSON.stringify({ ...endpoint, url: 'https://example.com' }));
  await expect(readDesktopEndpoint(path)).rejects.toThrow('Invalid desktop');
});
