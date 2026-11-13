import { spawnSync } from 'node:child_process';
import { chmod, mkdir, mkdtemp, readFile, rm, stat, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { afterEach, expect, it } from 'vite-plus/test';

import {
  publishDesktopEndpoint,
  readDesktopEndpoint,
  type DesktopEndpoint,
} from '../../../packages/server/src/host/desktop-endpoint.js';

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
  models: [{ name: 'local', path: '/models/local', base: 'local', variant: 'BF16', format: 'mlx' }],
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

async function writeRaw(path: string, text: string, mode = 0o600): Promise<void> {
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  await writeFile(path, text, { mode });
  await chmod(path, mode);
}

it('reads back a published endpoint', async () => {
  const path = await location();
  await publishDesktopEndpoint(endpoint, path);
  expect(await readDesktopEndpoint(path)).toEqual(endpoint);
});

it('reads null for a missing file or text that is not an endpoint', async () => {
  const path = await location();
  expect(await readDesktopEndpoint(path)).toBeNull();
  await writeRaw(path, '{not json');
  expect(await readDesktopEndpoint(path)).toBeNull();
  await writeRaw(path, 'null');
  expect(await readDesktopEndpoint(path)).toBeNull();
  await writeRaw(path, JSON.stringify({ ...endpoint, models: [null] }));
  expect(await readDesktopEndpoint(path)).toBeNull();
});

it('reads null for a model without a base, a build or a known format', async () => {
  const path = await location();
  const model = endpoint.models[0]!;
  const bad: unknown[] = [
    { name: model.name, path: model.path, variant: 'BF16', format: 'mlx' },
    { ...model, base: '' },
    { ...model, variant: '' },
    { ...model, variant: 4 },
    { ...model, format: 'safetensors' },
    { name: model.name, path: model.path, base: 'local', variant: 'BF16' },
  ];
  for (const one of bad) {
    await writeRaw(path, JSON.stringify({ ...endpoint, models: [one] }));
    expect(await readDesktopEndpoint(path)).toBeNull();
  }
  await expect(publishDesktopEndpoint({ ...endpoint, models: [{ ...model, base: '' }] }, path)).rejects.toThrow();
});

it('reads null when group or other may read the file', async () => {
  const path = await location();
  await writeRaw(path, JSON.stringify(endpoint), 0o640);
  expect(await readDesktopEndpoint(path)).toBeNull();
  await chmod(path, 0o604);
  expect(await readDesktopEndpoint(path)).toBeNull();
  await chmod(path, 0o600);
  expect(await readDesktopEndpoint(path)).toEqual(endpoint);
});

it('reads null through a symlink', async () => {
  const path = await location();
  const real = join(dirname(path), 'real.json');
  await writeRaw(real, JSON.stringify(endpoint));
  await symlink(real, path);
  expect(await readDesktopEndpoint(path)).toBeNull();
});

it('reads null for an origin that is not loopback http, or an empty token', async () => {
  const path = await location();
  for (const url of [
    'http://localhost:12345',
    'https://127.0.0.1:12345',
    'http://127.0.0.1:12345/v1',
    'http://10.0.0.1:12345',
  ]) {
    await writeRaw(path, JSON.stringify({ ...endpoint, url }));
    expect(await readDesktopEndpoint(path)).toBeNull();
  }
  await writeRaw(path, JSON.stringify({ ...endpoint, token: '' }));
  expect(await readDesktopEndpoint(path)).toBeNull();
});

it('reads null when the publishing process is gone', async () => {
  const path = await location();
  const child = spawnSync(process.execPath, ['-e', '']);
  expect(child.pid).toBeGreaterThan(0);
  await writeRaw(path, JSON.stringify({ ...endpoint, pid: child.pid }));
  expect(await readDesktopEndpoint(path)).toBeNull();
});
