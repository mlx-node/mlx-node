import { spawnSync } from 'node:child_process';
import { mkdir, mkdtemp, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  hostTempDirPrefix,
  isProcessAlive,
  sweepOrphanHostTempRoots,
} from '../../../packages/server/src/host/temp-root.js';

/**
 * An inference host's headline deployment is an Electron `utilityProcess`,
 * which can be SIGKILLed. SIGKILL runs no handler, so `close()` — and with it
 * `PagedConfigOverrideManager.cleanup()` — never happens and the temp root
 * survives. The only way to reclaim it later is to be able to tell whose it
 * was, which is what the pid in the directory name buys.
 */

/**
 * A pid that is definitely dead. `spawnSync` waits for the child AND reaps it,
 * so by the time it returns the pid is no longer addressable. (Recycling it
 * within the same millisecond would require the kernel to wrap its whole pid
 * space; on macOS/Linux pids increment.)
 */
function reapedPid(): number {
  const result = spawnSync(process.execPath, ['-e', '0'], { stdio: 'ignore' });
  if (typeof result.pid !== 'number' || result.pid <= 0) {
    throw new Error('spawnSync did not report a pid');
  }
  return result.pid;
}

describe('isProcessAlive', () => {
  it('reports our own process as alive', () => {
    expect(isProcessAlive(process.pid)).toBe(true);
  });

  it('reports a reaped child as dead', () => {
    expect(isProcessAlive(reapedPid())).toBe(false);
  });

  it('refuses to probe pid 0 and negative pids, reporting them alive', () => {
    // `process.kill(0, sig)` addresses the caller's entire process group and a
    // negative pid a group by id. Neither is ever a temp root's owner, and a
    // probe that answered "dead" would arm the sweeper against them.
    expect(isProcessAlive(0)).toBe(true);
    expect(isProcessAlive(-1)).toBe(true);
    expect(isProcessAlive(-process.pid)).toBe(true);
  });

  it('reports pid 1 as alive (init always exists; also covers the EPERM path)', () => {
    // pid 1 exists on every POSIX box and is root-owned, so an unprivileged
    // probe gets EPERM rather than success. EPERM must read as ALIVE — a
    // sweeper that deleted other users' live roots would be a real bug.
    expect(isProcessAlive(1)).toBe(true);
  });
});

describe('hostTempDirPrefix', () => {
  it('embeds the pid so the owner is recoverable from the directory name', () => {
    expect(hostTempDirPrefix(4242)).toBe('mlx-inference-host-4242-');
  });

  it('defaults to this process', () => {
    expect(hostTempDirPrefix()).toBe(`mlx-inference-host-${process.pid}-`);
  });
});

describe('sweepOrphanHostTempRoots', () => {
  let root: string;

  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), 'mlx-sweep-test-'));
  });

  afterEach(async () => {
    await rm(root, { recursive: true, force: true });
  });

  async function seed(name: string): Promise<string> {
    const dir = join(root, name);
    await mkdir(dir, { recursive: true });
    // Non-empty, so a `rmdir`-shaped implementation would fail where `rm -r`
    // succeeds. A real leaked root always holds at least one clone.
    await mkdir(join(dir, 'model-abc123'), { recursive: true });
    await writeFile(join(dir, 'model-abc123', 'config.json'), '{}', 'utf-8');
    return dir;
  }

  it('removes a dead-pid root and spares a live one', async () => {
    const deadPid = reapedPid();
    await seed(`mlx-inference-host-${deadPid}-aaaaaa`);
    await seed(`mlx-inference-host-${process.pid}-bbbbbb`);

    const removed = await sweepOrphanHostTempRoots({ root });

    expect(removed).toEqual([join(root, `mlx-inference-host-${deadPid}-aaaaaa`)]);
    expect(await readdir(root)).toEqual([`mlx-inference-host-${process.pid}-bbbbbb`]);
  });

  it('spares our own root even when the liveness probe claims we are dead', async () => {
    // Defence in depth: `selfPid` is checked before `isAlive` is consulted, so
    // a probe that mis-answers can never delete the root we are actively
    // resolving clones into.
    await seed(`mlx-inference-host-${process.pid}-cccccc`);
    const removed = await sweepOrphanHostTempRoots({ root, isAlive: () => false });
    expect(removed).toEqual([]);
    expect(await readdir(root)).toEqual([`mlx-inference-host-${process.pid}-cccccc`]);
  });

  it('leaves directories that are not host temp roots alone', async () => {
    // The sweeper runs against the shared OS temp dir, so "matches our exact
    // stem" has to be the whole test for ownership. Every decoy below carries
    // a `<digits>-` run somewhere in its name — the shape a loosened pattern
    // would latch onto — including a bare `PagedConfigOverrideManager` root
    // from a build that predates the pid-scoped naming, and an unrelated
    // tool's pid-stamped scratch dir.
    await seed('mlx-paged-overrides-Ab3xYz');
    await seed('mlx-paged-overrides-1234-old');
    await seed('some-other-tool-1234-scratch');
    await seed('not-mlx-inference-host-1234-abc');
    await seed('mlx-inference-host-notapid-dddddd');
    await seed('some-unrelated-dir');
    await seed('mlx-inference-host-');

    const removed = await sweepOrphanHostTempRoots({ root, isAlive: () => false });

    expect(removed).toEqual([]);
    expect((await readdir(root)).sort()).toEqual(
      [
        'mlx-inference-host-',
        'mlx-inference-host-notapid-dddddd',
        'mlx-paged-overrides-1234-old',
        'mlx-paged-overrides-Ab3xYz',
        'not-mlx-inference-host-1234-abc',
        'some-other-tool-1234-scratch',
        'some-unrelated-dir',
      ].sort(),
    );
  });

  it('classifies purely by the pid in the name, not by mtime or contents', async () => {
    await seed('mlx-inference-host-101-eeeeee');
    await seed('mlx-inference-host-202-ffffff');

    const removed = await sweepOrphanHostTempRoots({
      root,
      selfPid: 999,
      isAlive: (pid) => pid === 202,
    });

    expect(removed).toEqual([join(root, 'mlx-inference-host-101-eeeeee')]);
    expect(await readdir(root)).toEqual(['mlx-inference-host-202-ffffff']);
  });

  it('returns [] instead of throwing when the scan root does not exist', async () => {
    // Housekeeping must never be the reason a host refuses to start.
    await expect(sweepOrphanHostTempRoots({ root: join(root, 'nope') })).resolves.toEqual([]);
  });
});
