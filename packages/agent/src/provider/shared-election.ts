import { randomUUID } from 'node:crypto';
import { link, mkdir, readFile, readdir, unlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import type { SharedEndpoint } from './shared-protocol.js';

async function isLive(endpoint: SharedEndpoint): Promise<boolean> {
  try {
    process.kill(endpoint.pid, 0);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ESRCH') return false;
    if ((error as NodeJS.ErrnoException).code !== 'EPERM') throw error;
  }
  // Check identity as well as PID, which the OS may have reused after a crash.
  // A timeout is conservatively live: loading/inference can stall the JS loop.
  try {
    const response = await fetch(`http://127.0.0.1:${endpoint.port}/health`, {
      headers: { authorization: `Bearer ${endpoint.token}` },
      signal: AbortSignal.timeout(1000),
      redirect: 'error',
    });
    if (!response.ok && response.status !== 503) {
      await response.body?.cancel();
      return false;
    }
    const health = (await response.json()) as { pid?: number; protocol?: string };
    return health.pid === endpoint.pid && health.protocol === endpoint.protocol;
  } catch (error) {
    return !(error instanceof Error && (error.cause as NodeJS.ErrnoException)?.code === 'ECONNREFUSED');
  }
}

/**
 * Elect one service in the user's private directory, independently of TCP ports.
 * Publish complete claims with an exclusive hard link. Never reuse a generation:
 * a delayed contender must not replace a newer owner after reading a dead one.
 * Small retired claim files are retained to prevent that stale-lock deletion race.
 */
export async function claimSharedService(directory: string, endpoint: SharedEndpoint): Promise<() => Promise<void>> {
  const claims = join(directory, 'claims');
  await mkdir(claims, { recursive: true, mode: 0o700 });
  const generations = (await readdir(claims)).flatMap((name) => {
    const match = /^(\d+)\.json$/.exec(name);
    return match ? [Number(match[1])] : [];
  });
  let generation = generations.reduce((latest, value) => Math.max(latest, value), 0);
  const prepared = join(claims, `candidate-${randomUUID()}.json`);
  await writeFile(prepared, JSON.stringify(endpoint), { mode: 0o600 });
  try {
    for (;;) {
      const claim = join(claims, `${generation}.json`);
      try {
        await link(prepared, claim);
        return () => writeFile(`${claim}.released`, '', { mode: 0o600 });
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
      }
      let released = false;
      try {
        await readFile(`${claim}.released`);
        released = true;
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
      }
      if (!released && (await isLive(JSON.parse(await readFile(claim, 'utf8')) as SharedEndpoint))) {
        throw Object.assign(new Error('Another shared delegate service owns this user directory.'), {
          code: 'EADDRINUSE',
        });
      }
      generation++;
    }
  } finally {
    await unlink(prepared);
  }
}
