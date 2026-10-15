import { mkdtemp, readFile, rm, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, expect, it } from 'vite-plus/test';

import { publishDesktopEndpoint, type DesktopEndpoint } from '../../../packages/server/src/host/desktop-endpoint.js';

const dirs: string[] = [];
afterEach(async () => {
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
  expect(JSON.parse(await readFile(path, 'utf8'))).toEqual(endpoint);
  const releaseNew = await publishDesktopEndpoint({ ...endpoint, token: 'new' }, path);
  await releaseOld();
  expect(JSON.parse(await readFile(path, 'utf8')).token).toBe('new');
  await releaseNew();
  await expect(stat(path)).rejects.toMatchObject({ code: 'ENOENT' });
});
