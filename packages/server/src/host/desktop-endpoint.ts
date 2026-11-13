/** Native-free discovery of the desktop's existing inference host. */
import { randomUUID } from 'node:crypto';
import { chmod, lstat, mkdir, readFile, rename, unlink, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, join } from 'node:path';

import { resolveMlxNodeHome } from './paths.js';

export interface DesktopEndpoint {
  version: 1;
  pid: number;
  url: string;
  token: string;
  models: DesktopEndpointModel[];
}

/** A served chat model: its id, its path, and the base and build the app's model picker groups by. */
export interface DesktopEndpointModel {
  name: string;
  path: string;
  base: string;
  variant: string;
  format: 'gguf' | 'mlx';
}

function validModel(model: unknown): boolean {
  if (typeof model !== 'object' || model === null) return false;
  const { name, path, base, variant, format } = model as Record<string, unknown>;
  return (
    typeof name === 'string' &&
    typeof path === 'string' &&
    isAbsolute(path) &&
    typeof base === 'string' &&
    base !== '' &&
    typeof variant === 'string' &&
    variant !== '' &&
    (format === 'gguf' || format === 'mlx')
  );
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
    !value.models.every(validModel) ||
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

function pidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === 'EPERM';
  }
}

/**
 * The published endpoint, or null unless the file is a regular file owned by this user with no group or other bits,
 * holds a valid endpoint, and names a live process. Never throws.
 */
export async function readDesktopEndpoint(path = desktopEndpointPath()): Promise<DesktopEndpoint | null> {
  try {
    const info = await lstat(path);
    if (!info.isFile() || (info.mode & 0o077) !== 0) return null;
    if (typeof process.getuid === 'function' && info.uid !== process.getuid()) return null;
    const value = JSON.parse(await readFile(path, 'utf8')) as DesktopEndpoint;
    if (typeof value !== 'object' || value === null) return null;
    validate(value);
    return pidAlive(value.pid) ? value : null;
  } catch {
    return null;
  }
}
