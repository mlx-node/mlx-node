/** Native-free discovery of the desktop's existing inference host. */
import { randomUUID } from 'node:crypto';
import { chmod, mkdir, readFile, rename, unlink, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, join } from 'node:path';

import { resolveMlxNodeHome } from './paths.js';

export interface DesktopEndpoint {
  version: 1;
  pid: number;
  url: string;
  token: string;
  models: { name: string; path: string }[];
}

export function desktopEndpointPath(): string {
  return join(resolveMlxNodeHome(), 'desktop', 'inference.json');
}

function validate(value: DesktopEndpoint): void {
  const url = new URL(value.url);
  if (
    value.version !== 1 ||
    !Number.isSafeInteger(value.pid) ||
    value.pid < 1 ||
    !Array.isArray(value.models) ||
    !value.models.every(
      (model) => typeof model.name === 'string' && typeof model.path === 'string' && isAbsolute(model.path),
    ) ||
    value.url !== url.origin ||
    typeof value.token !== 'string' ||
    !value.token ||
    url.protocol !== 'http:' ||
    url.hostname !== '127.0.0.1' ||
    !url.port ||
    url.username ||
    url.password ||
    url.pathname !== '/' ||
    url.search ||
    url.hash
  )
    throw new Error('Invalid desktop inference endpoint. Restart the mlx-node app.');
}

/** Atomic publication; an older generation must never remove a newer one's credentials. */
export async function publishDesktopEndpoint(
  endpoint: DesktopEndpoint,
  path = desktopEndpointPath(),
): Promise<() => Promise<void>> {
  validate(endpoint);
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  await chmod(dirname(path), 0o700);
  const temp = `${path}.${randomUUID()}.tmp`;
  try {
    await writeFile(temp, JSON.stringify(endpoint), { mode: 0o600, flag: 'wx' });
    await rename(temp, path);
  } finally {
    await unlink(temp).catch(() => {});
  }
  return async () => {
    try {
      const current = JSON.parse(await readFile(path, 'utf8')) as DesktopEndpoint;
      if (current.token === endpoint.token && current.pid === endpoint.pid) await unlink(path);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
    }
  };
}
