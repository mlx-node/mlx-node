import { spawnSync, type ChildProcess, type SpawnOptions } from 'node:child_process';
import { createHash } from 'node:crypto';
import { EventEmitter } from 'node:events';
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  symlinkSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { createServer as createHttpServer } from 'node:http';
import { connect, createServer, type AddressInfo, type Server } from 'node:net';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { PassThrough } from 'node:stream';

import { describe, expect, it, vi, afterEach, beforeEach } from 'vitest';

import {
  acquireInstallLock,
  lockDirPath,
  agentTagFromBinPath,
  agentUpdateStatus,
  launchGrokAgent,
  leasePinnedManaged,
  leaseProbePath,
  writeLeaseMarker,
  applyAgentUpdate,
  downloadAgentBinary,
  managedAgentBinPath,
  markerProbePath,
  parseLockMarker,
  probeAgentInstall,
  releaseInstallLock,
  removeIfUnchanged,
  resolveAgentBinary,
  scanMembers,
  resolveReleaseTag,
  MLX_AGENT_BIN_ENV,
  MLX_AGENT_TAG_ENV,
  MLX_AGENT_MIN_RELEASE,
  type AgentInstallProbe,
} from '../../packages/cli/src/commands/agent/grok-build.js';
import { run } from '../../packages/cli/src/commands/agent/index.js';

const MANAGED = { source: 'managed', path: '/m/mlx-agent' } satisfies Partial<AgentInstallProbe>;

function status(overrides: Partial<AgentInstallProbe> = {}): () => AgentInstallProbe {
  return () => ({ managedTag: 'mlx-agent-v0.2.0', effectiveTag: 'mlx-agent-v0.2.0', ...MANAGED, ...overrides });
}

const EXE = process.platform === 'win32' ? '.exe' : '';
const versioned = (home: string, tag: string) => join(home, '.mlx-node', 'bin', `${tag}${EXE}`);
const mkdirBin = (home: string) => mkdirSync(join(home, '.mlx-node', 'bin'), { recursive: true });

describe('probeAgentInstall', () => {
  it('honors the MLX_AGENT_BIN pin', () => {
    const root = mkdtempSync(join(tmpdir(), 'probe-'));
    try {
      const bin = join(root, 'custom-agent');
      writeFileSync(bin, 'x');
      const probe = probeAgentInstall(undefined, { [MLX_AGENT_BIN_ENV]: bin }, root);
      expect(probe).toMatchObject({ source: 'env', path: bin });
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('rejects a missing MLX_AGENT_BIN path', () => {
    expect(() => probeAgentInstall(undefined, { [MLX_AGENT_BIN_ENV]: '/no/such/bin' })).toThrow(MLX_AGENT_BIN_ENV);
  });

  it('resolves the versioned binary the .tag pointer names', () => {
    const root = mkdtempSync(join(tmpdir(), 'probe-'));
    try {
      mkdirBin(root);
      writeFileSync(versioned(root, 'mlx-agent-v0.2.0'), 'v020');
      writeFileSync(`${managedAgentBinPath(root)}.tag`, 'mlx-agent-v0.2.0');
      const probe = probeAgentInstall(root, {}, root);
      expect(probe).toMatchObject({
        source: 'managed',
        path: versioned(root, 'mlx-agent-v0.2.0'),
        managedTag: 'mlx-agent-v0.2.0',
      });
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('prefers the versioned binary the pointer names while an older versioned file remains', () => {
    const root = mkdtempSync(join(tmpdir(), 'probe-'));
    try {
      mkdirBin(root);
      writeFileSync(versioned(root, 'mlx-agent-v0.2.0'), 'v020');
      writeFileSync(versioned(root, 'mlx-agent-v0.3.0'), 'v030');
      writeFileSync(`${managedAgentBinPath(root)}.tag`, 'mlx-agent-v0.3.0');
      const probe = probeAgentInstall(root, {}, root);
      expect(probe).toMatchObject({ source: 'managed', path: versioned(root, 'mlx-agent-v0.3.0') });
      expect(readFileSync(probe.path!, 'utf-8')).toBe('v030');
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('treats an orphaned .tag pointer as a broken cache, not a legacy binary', () => {
    const root = mkdtempSync(join(tmpdir(), 'probe-'));
    try {
      mkdirBin(root);
      const legacy = managedAgentBinPath(root);
      // The versioned file the pointer names is gone; the leftover
      // unversioned binary predates tagging, so it must NOT inherit the
      // orphaned tag — reporting v0.2.0 would suppress the update that
      // repairs this cache.
      writeFileSync(legacy, 'legacy');
      writeFileSync(`${legacy}.tag`, 'mlx-agent-v0.2.0');
      const probe = probeAgentInstall(root, {}, root);
      expect(probe).toMatchObject({ source: 'none', managedTag: undefined });
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('treats an orphaned .tag pointer as no install', () => {
    const root = mkdtempSync(join(tmpdir(), 'probe-'));
    try {
      mkdirBin(root);
      writeFileSync(`${managedAgentBinPath(root)}.tag`, 'mlx-agent-v0.2.0');
      const probe = probeAgentInstall(root, {}, root);
      expect(probe).toMatchObject({ source: 'none', managedTag: undefined });
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('honors an MLX_AGENT_TAG pin over a different cached tag', () => {
    const root = mkdtempSync(join(tmpdir(), 'probe-'));
    try {
      mkdirBin(root);
      writeFileSync(versioned(root, 'mlx-agent-v0.2.0'), 'v020');
      writeFileSync(versioned(root, 'mlx-agent-v0.3.0'), 'v030');
      writeFileSync(`${managedAgentBinPath(root)}.tag`, 'mlx-agent-v0.3.0');
      const probe = probeAgentInstall(root, { [MLX_AGENT_TAG_ENV]: 'mlx-agent-v0.2.0' }, root);
      // The pin selects its exact release even though the pointer names a newer one.
      expect(probe).toMatchObject({
        source: 'managed',
        path: versioned(root, 'mlx-agent-v0.2.0'),
        managedTag: 'mlx-agent-v0.3.0',
        effectiveTag: 'mlx-agent-v0.2.0',
      });
      // A cache-hit pin must be leased too: another process's sweep can
      // delete the binary between this probe and the actual spawn.
      expect(existsSync(`${probe.path}.lease-${process.pid}`)).toBe(true);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('resolves the bundled binary when its stamped tag matches the pin', () => {
    const root = mkdtempSync(join(tmpdir(), 'probe-'));
    try {
      mkdirBin(root);
      writeFileSync(versioned(root, 'mlx-agent-v0.3.0'), 'v030');
      writeFileSync(`${managedAgentBinPath(root)}.tag`, 'mlx-agent-v0.3.0');
      // The packaged app bundles exactly the pinned release, but the
      // managed cache lacks it — a managed miss must not trigger a
      // download when the bundle already carries that tag (offline
      // launches would fail for nothing).
      const bundled = join(root, 'app', 'native', `mlx-agent${EXE}`);
      mkdirSync(dirname(bundled), { recursive: true });
      writeFileSync(bundled, 'bundled-v020');
      writeFileSync(`${bundled}.tag`, 'mlx-agent-v0.2.0');
      const probe = probeAgentInstall(root, { [MLX_AGENT_TAG_ENV]: 'mlx-agent-v0.2.0' }, root, bundled);
      expect(probe).toMatchObject({
        source: 'bundled',
        path: bundled,
        bundledTag: 'mlx-agent-v0.2.0',
        effectiveTag: 'mlx-agent-v0.2.0',
      });
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('resolves none when the pinned release is not cached', () => {
    const root = mkdtempSync(join(tmpdir(), 'probe-'));
    try {
      mkdirBin(root);
      writeFileSync(versioned(root, 'mlx-agent-v0.3.0'), 'v030');
      writeFileSync(`${managedAgentBinPath(root)}.tag`, 'mlx-agent-v0.3.0');
      // Cached v0.3.0 must not satisfy a v0.4.0 pin — the launcher downloads it.
      const probe = probeAgentInstall(root, { [MLX_AGENT_TAG_ENV]: 'mlx-agent-v0.4.0' }, root);
      expect(probe).toMatchObject({ source: 'none', managedTag: 'mlx-agent-v0.3.0' });
      expect(probe.path).toBeUndefined();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('rejects a malformed MLX_AGENT_TAG pin at probe time', () => {
    const root = mkdtempSync(join(tmpdir(), 'probe-'));
    try {
      mkdirBin(root);
      writeFileSync(versioned(root, 'mlx-agent-v0.3.0'), 'v030');
      writeFileSync(`${managedAgentBinPath(root)}.tag`, 'mlx-agent-v0.3.0');
      expect(() => probeAgentInstall(root, { [MLX_AGENT_TAG_ENV]: 'mlx-agent-latest' }, root)).toThrow(
        MLX_AGENT_TAG_ENV,
      );
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});

describe('resolveAgentBinary', () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;
  let savedTag: string | undefined;

  beforeEach(() => {
    savedTag = process.env[MLX_AGENT_TAG_ENV];
    delete process.env[MLX_AGENT_TAG_ENV];
    fetchSpy = vi.spyOn(globalThis, 'fetch');
  });
  afterEach(() => {
    if (savedTag === undefined) delete process.env[MLX_AGENT_TAG_ENV];
    else process.env[MLX_AGENT_TAG_ENV] = savedTag;
    fetchSpy.mockRestore();
  });

  it('leases a cached pinned binary under the install lock', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-resolve-'));
    try {
      mkdirBin(home);
      writeFileSync(versioned(home, 'mlx-agent-v0.2.0'), 'v020');
      writeFileSync(`${managedAgentBinPath(home)}.tag`, 'mlx-agent-v0.3.0');
      const env = { [MLX_AGENT_TAG_ENV]: 'mlx-agent-v0.2.0' };
      // `home` doubles as a non-repo root so the vendored probe stays cold.
      const bin = await resolveAgentBinary(home, env, home);
      expect(bin).toBe(versioned(home, 'mlx-agent-v0.2.0'));
      // The serialized lease survives — it is what a concurrent unpinned
      // sweep consults before deleting this binary.
      expect(existsSync(`${bin}.lease-${process.pid}`)).toBe(true);
      // The lock is released once the lease is in place.
      expect(existsSync(`${managedAgentBinPath(home)}.lock`)).toBe(false);
      expect(fetchSpy).not.toHaveBeenCalled();
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('leases an unpinned managed hit the same way — the sweep protects it until spawn', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-resolve-'));
    try {
      mkdirBin(home);
      writeFileSync(versioned(home, 'mlx-agent-v0.3.0'), 'v030');
      writeFileSync(`${managedAgentBinPath(home)}.tag`, 'mlx-agent-v0.3.0');
      // No pin: the pointer resolves the managed cache — the same
      // probe→spawn gap a concurrent updater's sweep can hit.
      const bin = await resolveAgentBinary(home, {}, home);
      expect(bin).toBe(versioned(home, 'mlx-agent-v0.3.0'));
      expect(existsSync(`${bin}.lease-${process.pid}`)).toBe(true);
      expect(existsSync(`${managedAgentBinPath(home)}.lock`)).toBe(false);
      expect(fetchSpy).not.toHaveBeenCalled();
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('does not lease a non-managed path — no sweep can touch it', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-resolve-'));
    try {
      const bin = join(home, 'custom-agent');
      writeFileSync(bin, 'x');
      const resolved = await resolveAgentBinary(home, { [MLX_AGENT_BIN_ENV]: bin }, home);
      expect(resolved).toBe(bin);
      expect(existsSync(`${bin}.lease-${process.pid}`)).toBe(false);
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('leases a cache miss too — the download is not safe from the next sweep', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-resolve-'));
    try {
      mkdirBin(home);
      const file = join(home, 'agent-bin');
      writeFileSync(file, 'v030');
      const packed = spawnSync('tar', ['-czf', '-', '-C', home, 'agent-bin']);
      rmSync(file, { force: true });
      const sha = createHash('sha256').update(packed.stdout).digest('hex');
      fetchSpy.mockImplementation(async (url: RequestInfo | URL) => {
        const u = url instanceof URL ? url.href : typeof url === 'string' ? url : url.url;
        if (u.endsWith('.sha256')) return new Response(sha);
        if (u.endsWith('.tar.gz')) return new Response(new Uint8Array(packed.stdout));
        return new Response('nope', { status: 404 });
      });
      // Nothing cached and metadata 404s → the download path must return
      // the fallback (minimum) binary already leased under the lock, or a
      // second install's sweep can remove it before the launcher finishes
      // serve startup.
      const bin = await resolveAgentBinary(home, {}, home);
      expect(bin).toBe(versioned(home, MLX_AGENT_MIN_RELEASE));
      expect(existsSync(`${bin}.lease-${process.pid}`)).toBe(true);
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('downloads the pinned release when the cache cannot satisfy it', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-resolve-'));
    try {
      mkdirBin(home);
      writeFileSync(versioned(home, 'mlx-agent-v0.3.0'), 'v030');
      writeFileSync(`${managedAgentBinPath(home)}.tag`, 'mlx-agent-v0.3.0');
      process.env[MLX_AGENT_TAG_ENV] = 'mlx-agent-v0.2.0';

      const file = join(home, 'agent-bin');
      writeFileSync(file, 'v020');
      const packed = spawnSync('tar', ['-czf', '-', '-C', home, 'agent-bin']);
      rmSync(file, { force: true });
      expect(packed.status).toBe(0);
      const sha = createHash('sha256').update(packed.stdout).digest('hex');
      fetchSpy.mockImplementation(async (url: RequestInfo | URL) => {
        const u = url instanceof URL ? url.href : typeof url === 'string' ? url : url.url;
        if (u.endsWith('.sha256')) return new Response(sha);
        if (u.endsWith('.tar.gz')) return new Response(new Uint8Array(packed.stdout));
        return new Response('nope', { status: 404 });
      });

      const bin = await resolveAgentBinary(home, process.env, home);
      expect(bin).toBe(versioned(home, 'mlx-agent-v0.2.0'));
      // A pinned download is a side-install: the unpinned pointer is untouched.
      expect(readFileSync(`${managedAgentBinPath(home)}.tag`, 'utf-8')).toBe('mlx-agent-v0.3.0');
      // Freshly downloaded managed binaries carry the same locked lease as
      // cache hits — a concurrent updater's sweep must see it before spawn.
      expect(existsSync(`${bin}.lease-${process.pid}`)).toBe(true);
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('leasePinnedManaged re-verifies under the lock and reports a swept binary', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-resolve-'));
    try {
      mkdirBin(home);
      const bin = versioned(home, 'mlx-agent-v0.2.0');
      writeFileSync(bin, 'v020');
      const managed = managedAgentBinPath(home);
      // Present → locked lease written, path returned.
      expect(await leasePinnedManaged(managed, bin)).toBe(bin);
      expect(existsSync(`${bin}.lease-${process.pid}`)).toBe(true);
      // Swept before the locked recheck → undefined, so the caller
      // downloads instead of spawning ENOENT.
      rmSync(bin);
      expect(await leasePinnedManaged(managed, bin)).toBeUndefined();
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });
});

describe('downloadAgentBinary', () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, 'fetch');
  });
  afterEach(() => {
    fetchSpy.mockRestore();
  });

  /** Real tar.gz on stdout containing a single file holding `content`. */
  function makeTarball(dir: string, content: string): Buffer {
    const file = join(dir, 'agent-bin');
    writeFileSync(file, content);
    const out = spawnSync('tar', ['-czf', '-', '-C', dir, 'agent-bin']);
    rmSync(file, { force: true });
    expect(out.status, `tar pack failed: ${out.stderr?.toString()}`).toBe(0);
    return out.stdout;
  }

  function mockMirrors(tar: Buffer): string {
    const sha = createHash('sha256').update(tar).digest('hex');
    fetchSpy.mockImplementation(async (url: RequestInfo | URL) => {
      const u = url instanceof URL ? url.href : typeof url === 'string' ? url : url.url;
      if (u.endsWith('.sha256')) return new Response(sha);
      if (u.endsWith('.tar.gz')) return new Response(new Uint8Array(tar));
      return new Response('nope', { status: 404 });
    });
    return sha;
  }

  it('installs the versioned binary + pointer and never writes the unversioned base', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    try {
      mockMirrors(makeTarball(home, 'fake-agent-binary'));
      const dest = managedAgentBinPath(home);
      const path = await downloadAgentBinary(dest, 'mlx-agent-v0.3.0');
      expect(path).toBe(join(dirname(dest), `mlx-agent-v0.3.0${process.platform === 'win32' ? '.exe' : ''}`));
      expect(readFileSync(path, 'utf-8')).toBe('fake-agent-binary');
      expect(readFileSync(`${dest}.tag`, 'utf-8')).toBe('mlx-agent-v0.3.0');
      expect(existsSync(dest)).toBe(false);

      const probe = probeAgentInstall(home, {}, home);
      expect(probe).toMatchObject({ source: 'managed', path, managedTag: 'mlx-agent-v0.3.0' });
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('flips the pointer to the new versioned binary and keeps the prior release', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    try {
      const dest = managedAgentBinPath(home);
      const versionedName = (tag: string) => join(dirname(dest), `${tag}${process.platform === 'win32' ? '.exe' : ''}`);

      mockMirrors(makeTarball(home, 'v020'));
      const old = await downloadAgentBinary(dest, 'mlx-agent-v0.2.0');
      expect(old).toBe(versionedName('mlx-agent-v0.2.0'));

      mockMirrors(makeTarball(home, 'v030'));
      const current = await downloadAgentBinary(dest, 'mlx-agent-v0.3.0');
      expect(current).toBe(versionedName('mlx-agent-v0.3.0'));
      expect(readFileSync(current, 'utf-8')).toBe('v030');
      expect(readFileSync(`${dest}.tag`, 'utf-8')).toBe('mlx-agent-v0.3.0');
      // The prior release survives the sweep — it is the rollback target a
      // side-install's pointer dance can no longer destroy.
      expect(existsSync(old)).toBe(true);
      // The install lock is released once the install completes.
      expect(existsSync(`${dest}.lock`)).toBe(false);

      const probe = probeAgentInstall(home, {}, home);
      expect(probe).toMatchObject({ source: 'managed', path: current, managedTag: 'mlx-agent-v0.3.0' });
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('sweeps the legacy unversioned binary once a versioned pointer is live', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    try {
      const dest = managedAgentBinPath(home);
      // A pre-pointer install: unversioned binary, no .tag.
      mkdirSync(dirname(dest), { recursive: true });
      writeFileSync(dest, 'legacy-binary');

      mockMirrors(makeTarball(home, 'v030'));
      await downloadAgentBinary(dest, 'mlx-agent-v0.3.0');
      // The pointer now resolves every launch to the versioned name — the
      // migrated unversioned binary is dead weight and must not linger.
      expect(readFileSync(`${dest}.tag`, 'utf-8')).toBe('mlx-agent-v0.3.0');
      expect(existsSync(dest)).toBe(false);
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('keeps the unversioned base while there is no pointer, and when MLX_AGENT_BIN selects it', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    const savedTag = process.env[MLX_AGENT_TAG_ENV];
    const savedBin = process.env[MLX_AGENT_BIN_ENV];
    try {
      const dest = managedAgentBinPath(home);
      const versionedName = (tag: string) => join(dirname(dest), `${tag}${process.platform === 'win32' ? '.exe' : ''}`);
      mkdirSync(dirname(dest), { recursive: true });
      writeFileSync(dest, 'legacy-binary');

      // A pinned side-install on a pointer-less legacy install leaves no
      // .tag — the unversioned base is still the resolvable default and
      // must survive the sweep.
      process.env[MLX_AGENT_TAG_ENV] = 'mlx-agent-v0.2.0';
      mockMirrors(makeTarball(home, 'v020'));
      await downloadAgentBinary(dest, 'mlx-agent-v0.2.0');
      expect(existsSync(dest)).toBe(true);
      expect(existsSync(versionedName('mlx-agent-v0.2.0'))).toBe(true);

      // With the pointer now live, an $MLX_AGENT_BIN that selects exactly
      // the base path protects it — sweeping would break the user's pin.
      delete process.env[MLX_AGENT_TAG_ENV];
      process.env[MLX_AGENT_BIN_ENV] = dest;
      mockMirrors(makeTarball(home, 'v030'));
      await downloadAgentBinary(dest, 'mlx-agent-v0.3.0');
      expect(existsSync(dest)).toBe(true);
      expect(readFileSync(dest, 'utf-8')).toBe('legacy-binary');

      // A symlinked override must protect it the same way: resolve()
      // normalizes text but never dereferences, so the comparison has to
      // match the override's real path against dest.
      const link = join(home, 'agent-link');
      symlinkSync(dest, link);
      process.env[MLX_AGENT_BIN_ENV] = link;
      mockMirrors(makeTarball(home, 'v040'));
      await downloadAgentBinary(dest, 'mlx-agent-v0.4.0');
      expect(existsSync(dest)).toBe(true);
      expect(readFileSync(dest, 'utf-8')).toBe('legacy-binary');
    } finally {
      if (savedTag === undefined) delete process.env[MLX_AGENT_TAG_ENV];
      else process.env[MLX_AGENT_TAG_ENV] = savedTag;
      if (savedBin === undefined) delete process.env[MLX_AGENT_BIN_ENV];
      else process.env[MLX_AGENT_BIN_ENV] = savedBin;
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('reuses an already-installed same-tag binary instead of renaming over it', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    try {
      const dest = managedAgentBinPath(home);
      const versionedName = join(dirname(dest), `mlx-agent-v0.3.0${process.platform === 'win32' ? '.exe' : ''}`);

      mockMirrors(makeTarball(home, 'first-install'));
      await downloadAgentBinary(dest, 'mlx-agent-v0.3.0');
      expect(readFileSync(versionedName, 'utf-8')).toBe('first-install');

      // A second install of the SAME tag (the concurrent-first-launch case:
      // both processes downloaded before either held the lock) adopts the
      // existing binary — renaming over a running `<tag>.exe` fails on
      // Windows. The pointer is repaired, the bytes are untouched.
      rmSync(`${dest}.tag`, { force: true });
      mockMirrors(makeTarball(home, 'second-install'));
      const path = await downloadAgentBinary(dest, 'mlx-agent-v0.3.0');
      expect(path).toBe(versionedName);
      expect(readFileSync(versionedName, 'utf-8')).toBe('first-install');
      expect(readFileSync(`${dest}.tag`, 'utf-8')).toBe('mlx-agent-v0.3.0');
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('keeps a pinned download as a side-install: pointer and prior release untouched', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    const saved = process.env[MLX_AGENT_TAG_ENV];
    try {
      const dest = managedAgentBinPath(home);
      const versionedName = (tag: string) => join(dirname(dest), `${tag}${process.platform === 'win32' ? '.exe' : ''}`);

      // Active release: v0.3.0.
      delete process.env[MLX_AGENT_TAG_ENV];
      mockMirrors(makeTarball(home, 'v030'));
      await downloadAgentBinary(dest, 'mlx-agent-v0.3.0');
      expect(readFileSync(`${dest}.tag`, 'utf-8')).toBe('mlx-agent-v0.3.0');

      // A pinned launch fetches v0.2.0 — it must not repoint the default or
      // sweep the active release.
      process.env[MLX_AGENT_TAG_ENV] = 'mlx-agent-v0.2.0';
      mockMirrors(makeTarball(home, 'v020'));
      const path = await downloadAgentBinary(dest, 'mlx-agent-v0.2.0');
      expect(path).toBe(versionedName('mlx-agent-v0.2.0'));
      expect(readFileSync(path, 'utf-8')).toBe('v020');
      expect(readFileSync(`${dest}.tag`, 'utf-8')).toBe('mlx-agent-v0.3.0');
      expect(existsSync(versionedName('mlx-agent-v0.3.0'))).toBe(true);
    } finally {
      if (saved === undefined) delete process.env[MLX_AGENT_TAG_ENV];
      else process.env[MLX_AGENT_TAG_ENV] = saved;
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('a later unpinned update keeps the prior release; side-installed binaries are reclaimed', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    const saved = process.env[MLX_AGENT_TAG_ENV];
    try {
      const dest = managedAgentBinPath(home);
      const versionedName = (tag: string) => join(dirname(dest), `${tag}${process.platform === 'win32' ? '.exe' : ''}`);

      delete process.env[MLX_AGENT_TAG_ENV];
      mockMirrors(makeTarball(home, 'v030'));
      await downloadAgentBinary(dest, 'mlx-agent-v0.3.0');

      // Side-install v0.2.0 for a pin.
      process.env[MLX_AGENT_TAG_ENV] = 'mlx-agent-v0.2.0';
      mockMirrors(makeTarball(home, 'v020'));
      await downloadAgentBinary(dest, 'mlx-agent-v0.2.0');

      // Later unpinned update to v0.4.0: the pointer's previous target
      // (v0.3.0) survives as the rollback release. The side-installed
      // v0.2.0 survives too — its lease names this (live) process, so a
      // sweep must not delete a binary a pinned launch can still resolve.
      delete process.env[MLX_AGENT_TAG_ENV];
      mockMirrors(makeTarball(home, 'v040'));
      await downloadAgentBinary(dest, 'mlx-agent-v0.4.0');
      expect(readFileSync(`${dest}.tag`, 'utf-8')).toBe('mlx-agent-v0.4.0');
      expect(existsSync(versionedName('mlx-agent-v0.4.0'))).toBe(true);
      expect(existsSync(versionedName('mlx-agent-v0.3.0'))).toBe(true);
      expect(existsSync(versionedName('mlx-agent-v0.2.0'))).toBe(true);

      // A lease whose pinner is dead no longer protects the binary: the
      // next sweep removes both it and the stale lease.
      const staleBin = versionedName('mlx-agent-v0.1.0');
      writeFileSync(staleBin, 'v010');
      writeFileSync(`${staleBin}.lease-999999`, '');
      mockMirrors(makeTarball(home, 'v050'));
      await downloadAgentBinary(dest, 'mlx-agent-v0.5.0');
      expect(existsSync(staleBin)).toBe(false);
      expect(existsSync(`${staleBin}.lease-999999`)).toBe(false);
    } finally {
      if (saved === undefined) delete process.env[MLX_AGENT_TAG_ENV];
      else process.env[MLX_AGENT_TAG_ENV] = saved;
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('sweeps a fresh-mtime lease whose pinner is dead — liveness, not age, decides', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    const saved = process.env[MLX_AGENT_TAG_ENV];
    try {
      delete process.env[MLX_AGENT_TAG_ENV];
      const dest = managedAgentBinPath(home);
      mkdirBin(home);
      const staleBin = versioned(home, 'mlx-agent-v0.1.0');
      writeFileSync(staleBin, 'v010');
      // A marker a dead pinner JUST wrote (or another sweeper refreshed):
      // mtime is no alibi — the recorded pid is dead, so both it and the
      // binary it names go. 999999 cannot be live on any real host.
      const lease = `${staleBin}.lease-999999`;
      writeFileSync(lease, '');
      mockMirrors(makeTarball(home, 'v030'));
      await downloadAgentBinary(dest, 'mlx-agent-v0.3.0');
      expect(existsSync(staleBin)).toBe(false);
      expect(existsSync(lease)).toBe(false);
    } finally {
      if (saved === undefined) delete process.env[MLX_AGENT_TAG_ENV];
      else process.env[MLX_AGENT_TAG_ENV] = saved;
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('sweeps an aged lease whose live pid cannot prove ownership (recycled pid)', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    const saved = process.env[MLX_AGENT_TAG_ENV];
    try {
      delete process.env[MLX_AGENT_TAG_ENV];
      const dest = managedAgentBinPath(home);
      mkdirBin(home);
      const staleBin = versioned(home, 'mlx-agent-v0.1.0');
      writeFileSync(staleBin, 'v010');
      // A crashed pinner's marker whose pid the OS handed to us: live pid,
      // no endpoint. Aged past the mid-bind grace so only a REAL owner —
      // not merely a reused pid — protects the binary.
      const lease = `${staleBin}.lease-${process.pid}`;
      writeFileSync(lease, '');
      const aged = new Date(Date.now() - 60_000);
      utimesSync(lease, aged, aged);
      mockMirrors(makeTarball(home, 'v030'));
      await downloadAgentBinary(dest, 'mlx-agent-v0.3.0');
      expect(existsSync(staleBin)).toBe(false);
      expect(existsSync(lease)).toBe(false);
    } finally {
      if (saved === undefined) delete process.env[MLX_AGENT_TAG_ENV];
      else process.env[MLX_AGENT_TAG_ENV] = saved;
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('keeps an aged lease whose probe endpoint still answers', { timeout: 20_000 }, async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    const saved = process.env[MLX_AGENT_TAG_ENV];
    try {
      delete process.env[MLX_AGENT_TAG_ENV];
      const dest = managedAgentBinPath(home);
      mkdirBin(home);
      const bin = versioned(home, 'mlx-agent-v0.2.0');
      writeFileSync(bin, 'v020');
      writeLeaseMarker(bin);
      const lease = `${bin}.lease-${process.pid}`;
      expect(existsSync(lease)).toBe(true);
      // The marker file lands synchronously; the endpoint binds async —
      // poll for it so the sweep exercises the live-endpoint path rather
      // than the fresh-marker grace.
      const probe = leaseProbePath(lease);
      for (let i = 0; i < 100; i++) {
        const up = await new Promise<boolean>((res) => {
          const sock = connect(probe);
          sock.once('connect', () => {
            sock.destroy();
            res(true);
          });
          sock.once('error', () => res(false));
        });
        if (up) break;
        await new Promise((r) => setTimeout(r, 25));
      }
      // Past the mid-bind grace: only a live endpoint may keep the lease.
      const aged = new Date(Date.now() - 60_000);
      utimesSync(lease, aged, aged);
      mockMirrors(makeTarball(home, 'v030'));
      await downloadAgentBinary(dest, 'mlx-agent-v0.3.0');
      expect(existsSync(bin)).toBe(true);
      expect(existsSync(lease)).toBe(true);
    } finally {
      if (saved === undefined) delete process.env[MLX_AGENT_TAG_ENV];
      else process.env[MLX_AGENT_TAG_ENV] = saved;
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('refreshes a stale marker on re-lease so the mid-bind grace covers pid reuse', () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    try {
      mkdirBin(home);
      const bin = versioned(home, 'mlx-agent-v0.2.0');
      writeFileSync(bin, 'v020');
      // Simulate the recycled-pid case: a marker OUR dead predecessor left,
      // aged past the sweep's mid-bind grace.
      const lease = `${bin}.lease-${process.pid}`;
      writeFileSync(lease, '');
      const aged = new Date(Date.now() - 60_000);
      utimesSync(lease, aged, aged);
      writeLeaseMarker(bin);
      // Re-advertising must renew the mtime — without it a sweep between
      // now and the async bind completing treats us as an aged refused
      // marker and deletes the binary we are about to spawn.
      expect(Date.now() - statSync(lease).mtimeMs).toBeLessThan(5_000);
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('proceeds through a stale lock left by a dead installer', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    try {
      const dest = managedAgentBinPath(home);
      mkdirSync(dirname(dest), { recursive: true });
      const lock = `${dest}.lock`;
      // A leftover node whose holder is gone refuses connections — the
      // installer unlinks it and binds its own. (The legacy plain-file lock
      // is reclaimed by the same probe.)
      writeFileSync(lock, '999999');

      mockMirrors(makeTarball(home, 'fake-agent-binary'));
      const path = await downloadAgentBinary(dest, 'mlx-agent-v0.3.0');
      expect(readFileSync(path, 'utf-8')).toBe('fake-agent-binary');
      expect(existsSync(lock)).toBe(false);
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('reclaims a fresh lock node whose holder is dead', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    try {
      const dest = managedAgentBinPath(home);
      mkdirSync(dirname(dest), { recursive: true });
      // Fresh mtime, no listener — connection refusal, not age, proves
      // staleness.
      writeFileSync(`${dest}.lock`, '999999');

      mockMirrors(makeTarball(home, 'fake-agent-binary'));
      const path = await downloadAgentBinary(dest, 'mlx-agent-v0.3.0');
      expect(readFileSync(path, 'utf-8')).toBe('fake-agent-binary');
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('serializes concurrent acquirers: held while owned, reclaimable after release', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    try {
      const lock = join(home, 'mlx-agent.lock');
      await acquireInstallLock(lock);
      try {
        // A live listener owns the name — a second acquirer can never unlink
        // or bind over it, so it waits out its deadline and refuses.
        await expect(acquireInstallLock(lock, { deadlineMs: 400 })).rejects.toThrow('install lock');
      } finally {
        releaseInstallLock(lock);
      }
      // The name frees on release — the next acquirer binds immediately.
      await acquireInstallLock(lock, { deadlineMs: 2_000 });
      releaseInstallLock(lock);
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('reclaims membership markers left behind by crashed holders', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    try {
      const lock = join(home, 'mlx-agent.lock');
      const dir = lockDirPath(lock).dir;
      mkdirSync(dir, { recursive: true });
      // A dead member's marker and the legacy single-name lock are both
      // reclaimed unconditionally: the owning pid cannot exist.
      writeFileSync(join(dir, 'b-1-999999-0'), '');
      writeFileSync(join(dir, 'm-999999-0'), '');
      writeFileSync(join(dir, 'c-999999-0'), '');
      writeFileSync(lock, '999999');
      await acquireInstallLock(lock, { deadlineMs: 2_000 });
      expect(existsSync(lock)).toBe(false);
      const members = readdirSync(dir);
      expect(members).toHaveLength(1);
      // The surviving marker is our b-<ticket>-<pid>-<seq>-<nonce> socket.
      const parts = members[0]!.split('-');
      expect(parts[0]).toBe('b');
      expect(parts).toHaveLength(5);
      expect(parts[1]!.length > 0 && Number.isInteger(Number(parts[1]))).toBe(true);
      expect(parts[2]).toBe(String(process.pid));
      expect(parts[3]!.length > 0 && Number.isInteger(Number(parts[3]))).toBe(true);
      expect(parts[4]!).toMatch(/^[0-9a-f]{10}$/);
      releaseInstallLock(lock);
      expect(readdirSync(dir)).toEqual([]);
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it(
    'orders by bakery ticket, not pid order: an earlier advertiser always blocks us',
    { timeout: 20_000 },
    async () => {
      const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
      const lock = join(home, 'mlx-agent.lock');
      const dir = lockDirPath(lock).dir;
      const servers: Server[] = [];
      // A live member marker must actually hold its endpoint: on unix the
      // marker is a listening socket, on Windows a named pipe plus the
      // marker file — a live pid with a dead endpoint is a recycled pid
      // and gets swept on both.
      const advertise = async (name: string): Promise<void> => {
        const file = join(dir, name);
        const server = createServer((sock) => sock.destroy());
        servers.push(server);
        await new Promise<void>((res, rej) => {
          server.once('error', rej);
          server.listen(markerProbePath(file), res);
        });
        if (process.platform === 'win32') writeFileSync(file, '');
      };
      try {
        mkdirSync(dir, { recursive: true });
        // A member from the previous `m-` protocol — live (our pid), high
        // seq. Under the superseded (pid, seq) ticket order this member
        // sorted "behind" a new acquire with a lower seq, letting a later
        // contender in while it held. Under bakery it maps to ticket 0 —
        // it advertised before any `b-` member — so we MUST queue.
        await advertise(`m-${process.pid}-999`);
        await expect(acquireInstallLock(lock, { deadlineMs: 400 })).rejects.toThrow('install lock');

        // Same for a low bakery ticket under our pid: ticket 2 < whatever we
        // choose next, so the earlier advertiser wins regardless of seq.
        for (const server of servers.splice(0)) {
          server.close();
        }
        rmSync(join(dir, `m-${process.pid}-999`), { force: true });
        await advertise(`b-2-${process.pid}-999`);
        await expect(acquireInstallLock(lock, { deadlineMs: 400 })).rejects.toThrow('install lock');
      } finally {
        for (const server of servers) {
          server.close();
        }
        rmSync(home, { recursive: true, force: true });
      }
    },
  );

  it('sweeps a dead-endpoint member whose pid was recycled by an unrelated process', { timeout: 20_000 }, async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    try {
      const lock = join(home, 'mlx-agent.lock');
      const dir = lockDirPath(lock).dir;
      mkdirSync(dir, { recursive: true });
      // A bakery marker left by a crashed holder whose pid the OS handed
      // to us: live pid, dead endpoint — and OLD (a fresh refusal is a
      // member mid-bind, so only aged markers get swept). pidAlive alone
      // would pin every acquire behind a lock nobody owns; the refusal
      // plus age is what identifies the recycle.
      const marker = join(dir, `b-1-${process.pid}-0`);
      writeFileSync(marker, '');
      const aged = new Date(Date.now() - 60_000);
      const { utimesSync } = await import('node:fs');
      utimesSync(marker, aged, aged);
      await acquireInstallLock(lock, { deadlineMs: 2_000 });
      releaseInstallLock(lock);
      expect(existsSync(marker)).toBe(false);
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it(
    'refuses the grant when our own marker was swept and only higher tickets remain',
    { timeout: 20_000 },
    async () => {
      // A member suspended past the stale bound with an unanswerable
      // endpoint gets its live marker removed; on resume, every() alone
      // can pass over the winner's higher ticket. Runs on Windows too:
      // the recycled-pid probe there is the derived named pipe.
      const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
      const lock = join(home, 'mlx-agent.lock');
      const dir = lockDirPath(lock).dir;
      const servers: Server[] = [];
      const listenAt = async (name: string): Promise<void> => {
        const file = join(dir, name);
        const server = createServer((sock) => sock.destroy());
        servers.push(server);
        await new Promise<void>((res, rej) => {
          server.once('error', rej);
          server.listen(markerProbePath(file), res);
        });
        if (process.platform === 'win32') writeFileSync(file, '');
      };
      try {
        mkdirSync(dir, { recursive: true });
        // An ahead member keeps our acquire polling until the watcher
        // fires; it is swept alongside our marker below.
        const ahead = `b-1-${process.pid}-0`;
        await listenAt(ahead);
        let swapped = false;
        // Simulate the racing sweeper: as soon as our marker exists,
        // delete it plus the ahead member and advertise the winner's
        // higher ticket — the state a suspended-then-resumed contender
        // would find on its next scan.
        const watcher = setInterval(() => {
          if (swapped) return;
          try {
            const ours = readdirSync(dir).find((e) => {
              const parsed = parseLockMarker(e);
              return (
                parsed !== undefined &&
                !parsed.choosing &&
                parsed.member.ticket > 0 &&
                e !== ahead &&
                e !== `b-99-${process.pid}-0`
              );
            });
            if (ours === undefined) return;
            swapped = true;
            rmSync(join(dir, ours), { force: true });
            rmSync(join(dir, ahead), { force: true });
            void listenAt(`b-99-${process.pid}-0`);
          } catch {
            // retry on the next tick
          }
        }, 5);
        try {
          await expect(acquireInstallLock(lock, { deadlineMs: 1_500 })).rejects.toThrow('install lock');
          expect(swapped).toBe(true);
        } finally {
          clearInterval(watcher);
        }
      } finally {
        for (const server of servers) {
          server.close();
        }
        rmSync(home, { recursive: true, force: true });
      }
    },
  );

  it('keeps a live choose marker in the choosing set even when it is old', { timeout: 20_000 }, async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    const lock = join(home, 'mlx-agent.lock');
    const dir = lockDirPath(lock).dir;
    const marker = join(dir, `c-${process.pid}-0`);
    let server: Server | undefined;
    try {
      mkdirSync(dir, { recursive: true });
      // A suspended contender is indistinguishable from a recycled pid
      // by age alone — sweeping its live marker lets a later contender
      // choose the same ticket and enter concurrently, so an OLD but
      // owned marker must keep blocking until it advertises or dies.
      server = createServer((sock) => sock.destroy());
      await new Promise<void>((res, rej) => {
        server!.once('error', rej);
        server!.listen(markerProbePath(marker), res);
      });
      if (process.platform === 'win32') writeFileSync(marker, '');
      try {
        const past = new Date(Date.now() - 60_000);
        utimesSync(marker, past, past);
      } catch {
        // aging is best-effort — some filesystems refuse socket utimes
      }
      await expect(acquireInstallLock(lock, { deadlineMs: 1_000 })).rejects.toThrow('install lock');
      expect(existsSync(marker)).toBe(true);
    } finally {
      server?.close();
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('sweeps a choose marker whose endpoint refuses — its pid was recycled', { timeout: 20_000 }, async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    try {
      const lock = join(home, 'mlx-agent.lock');
      const dir = lockDirPath(lock).dir;
      mkdirSync(dir, { recursive: true });
      // A `c-` marker whose endpoint is not listening: a real contender
      // binds one, so a live pid with a dead endpoint is debris fronting
      // a recycled pid. Aged past the mid-bind grace so the sweep applies
      // immediately.
      const marker = join(dir, `c-${process.pid}-0`);
      writeFileSync(marker, '');
      const past = new Date(Date.now() - 60_000);
      utimesSync(marker, past, past);
      await acquireInstallLock(lock, { deadlineMs: 2_000 });
      releaseInstallLock(lock);
      expect(existsSync(marker)).toBe(false);
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('reports a vanished mid-probe marker as still choosing so the caller rescans', { timeout: 20_000 }, async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    try {
      const dir = join(home, 'lock.d');
      mkdirSync(dir, { recursive: true });
      // A live-pid c- marker that is NOT a listening socket. The race:
      // readdirSync snapshots it, the probe refuses, and the owner swaps
      // it for a b- member before statSync runs. Dropping it as "absent"
      // would let an acquirer grant on a member list that missed the new
      // b- marker; the scan must instead say "still choosing".
      const marker = join(dir, `c-${process.pid}-0`);
      writeFileSync(marker, '');
      // scanMembers runs its readdirSync synchronously at call time, so
      // deleting right after the call guarantees the marker existed in the
      // snapshot but is gone before the async probes complete — the exact
      // readdir-then-unlink window a real contender's choose→bind swap hits.
      const scanPromise = scanMembers(dir);
      rmSync(marker, { force: true });
      const scan = await scanPromise;
      expect(scan.choosing).toBe(true);
      expect(scan.members).toEqual([]);
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('nonced marker names still sweep on a dead pid and parse as contenders', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    try {
      const dir = join(home, 'lock.d');
      mkdirSync(dir, { recursive: true });
      const deadPid = spawnSync(process.execPath, ['-e', '']).pid!;
      const marker = join(dir, `b-3-${deadPid}-7-deadbeef99`);
      writeFileSync(marker, '');
      // A nonce belongs to its dead owner — this name can never be rebound
      // by a live process, so the sweep's unlink-by-name cannot hit a
      // replacement mid-flight. Parsing must still see through the suffix.
      expect(parseLockMarker(`b-3-${deadPid}-7-deadbeef99`)).toMatchObject({
        member: { ticket: 3, pid: deadPid, seq: 7 },
        choosing: false,
      });
      expect(parseLockMarker(`c-${deadPid}-7-deadbeef99`)).toMatchObject({ choosing: true });
      expect(parseLockMarker(`b-3-${deadPid}-7`)).toMatchObject({ member: { ticket: 3 } });
      const scan = await scanMembers(dir);
      expect(existsSync(marker)).toBe(false);
      expect(scan).toEqual({ members: [], choosing: false });
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('sweeps a dead-pid marker without flagging a rescan', { timeout: 20_000 }, async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    try {
      const dir = join(home, 'lock.d');
      mkdirSync(dir, { recursive: true });
      // spawnSync's child is already reaped — its pid is dead by the time
      // the scan reads liveness, so the marker must be gone and absent.
      const deadPid = spawnSync(process.execPath, ['-e', '']).pid!;
      const marker = join(dir, `c-${deadPid}-0`);
      writeFileSync(marker, '');
      const scan = await scanMembers(dir);
      expect(existsSync(marker)).toBe(false);
      expect(scan.choosing).toBe(false);
      expect(scan.members).toEqual([]);
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('removeIfUnchanged unlinks only the observed node, never a replacement', () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    try {
      const file = join(home, 'marker');
      writeFileSync(file, 'stale');
      const baseline = statSync(file);
      expect(removeIfUnchanged(file, baseline)).toBe(true);
      expect(existsSync(file)).toBe(false);

      // The pid-reuse race the sweep guards: a new owner rebinds the same
      // name between observation and unlink. The replacement node is a
      // different inode/mtime — it must survive and force a rescan.
      writeFileSync(file, 'live');
      expect(removeIfUnchanged(file, baseline)).toBe(false);
      expect(readFileSync(file, 'utf8')).toBe('live');

      // An owner refresh (utimes) is also a changed node — do not unlink.
      const liveBaseline = statSync(file);
      utimesSync(file, new Date(), new Date());
      expect(removeIfUnchanged(file, liveBaseline)).toBe(false);
      expect(existsSync(file)).toBe(true);

      // A vanished marker: nothing left to unlink, still stale.
      rmSync(file, { force: true });
      expect(removeIfUnchanged(file, liveBaseline)).toBe(false);
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('refuses to install while a live process holds the lock', { timeout: 20_000 }, async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    try {
      const dest = managedAgentBinPath(home);
      mkdirSync(dirname(dest), { recursive: true });
      const lock = `${dest}.lock`;
      // The lock is a bound socket: a live holder accepts connections, so
      // even an AGED file must read as genuinely held — the install waits
      // out its deadline and refuses rather than proceeding unlocked.
      await acquireInstallLock(lock);
      try {
        mockMirrors(makeTarball(home, 'fake-agent-binary'));
        await expect(downloadAgentBinary(dest, 'mlx-agent-v0.3.0')).rejects.toThrow('install lock');
        // A refused install must not have touched the managed cache.
        expect(existsSync(`${dest}.tag`)).toBe(false);
      } finally {
        releaseInstallLock(lock);
      }
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('rejects a mirror that declares an oversized Content-Length', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    try {
      const dest = managedAgentBinPath(home);
      const tar = makeTarball(home, 'fake-agent-binary');
      const sha = createHash('sha256').update(tar).digest('hex');
      fetchSpy.mockImplementation(async (url: RequestInfo | URL) => {
        const u = url instanceof URL ? url.href : typeof url === 'string' ? url : url.url;
        if (u.endsWith('.sha256')) return new Response(sha);
        if (u.endsWith('.tar.gz')) {
          if (u.includes('github.com')) return new Response('nope', { status: 404 });
          // R2 lies: correct bytes, absurd declared size. The cap rejects it
          // before a byte is buffered; GitHub's 404 then fails the install.
          return new Response(new Uint8Array(tar), { headers: { 'content-length': '999999999' } });
        }
        return new Response('nope', { status: 404 });
      });
      await expect(downloadAgentBinary(dest, 'mlx-agent-v0.3.0')).rejects.toThrow('verified download failed');
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('rejects a mirror that streams past the archive byte cap', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    try {
      const dest = managedAgentBinPath(home);
      const sha = createHash('sha256').update(makeTarball(home, 'x')).digest('hex');
      fetchSpy.mockImplementation(async (url: RequestInfo | URL) => {
        const u = url instanceof URL ? url.href : typeof url === 'string' ? url : url.url;
        if (u.endsWith('.sha256')) return new Response(sha);
        if (u.endsWith('.tar.gz')) {
          if (u.includes('github.com')) return new Response('nope', { status: 404 });
          // No Content-Length: only the streaming cap can stop this body.
          const body = new ReadableStream<Uint8Array>({
            start(controller) {
              controller.enqueue(new Uint8Array(300 * 1024 * 1024));
              controller.close();
            },
          });
          return new Response(body);
        }
        return new Response('nope', { status: 404 });
      });
      await expect(downloadAgentBinary(dest, 'mlx-agent-v0.3.0')).rejects.toThrow('verified download failed');
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('an unpinned download falls through to the next candidate when the resolved tag fails to fetch', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    const saved = process.env[MLX_AGENT_TAG_ENV];
    try {
      delete process.env[MLX_AGENT_TAG_ENV];
      const dest = managedAgentBinPath(home);
      const tar = makeTarball(home, 'v020');
      const sha = createHash('sha256').update(tar).digest('hex');
      // The R2 pointer names v0.3.0 and its presence probe answers — but the
      // archive 404s when the real download GETs it (assets yanked between
      // resolve and fetch, the race presence probes cannot close). GitHub
      // latest offers v0.2.0, which must carry the launch.
      let badArchiveGets = 0;
      fetchSpy.mockImplementation(async (url: RequestInfo | URL) => {
        const u = url instanceof URL ? url.href : typeof url === 'string' ? url : url.url;
        if (u.endsWith('.sha256')) return new Response(sha);
        if (u.includes('releases/latest')) return new Response(JSON.stringify({ tag_name: 'mlx-agent-v0.2.0' }));
        if (u.endsWith('.tar.gz')) {
          if (u.includes('mlx-agent-v0.3.0')) {
            badArchiveGets += 1;
            return badArchiveGets === 1 ? new Response('probe-ok') : new Response('nope', { status: 404 });
          }
          return new Response(new Uint8Array(tar));
        }
        if (u.includes('dl.portato.si')) return new Response('mlx-agent-v0.3.0');
        return new Response('nope', { status: 404 });
      });
      const path = await downloadAgentBinary(dest, undefined, {});
      expect(path).toBe(versioned(home, 'mlx-agent-v0.2.0'));
      expect(readFileSync(path, 'utf-8')).toBe('v020');
      expect(readFileSync(`${dest}.tag`, 'utf-8')).toBe('mlx-agent-v0.2.0');
    } finally {
      if (saved === undefined) delete process.env[MLX_AGENT_TAG_ENV];
      else process.env[MLX_AGENT_TAG_ENV] = saved;
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('a floored walk skips candidates at or below the installed release', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    const saved = process.env[MLX_AGENT_TAG_ENV];
    try {
      delete process.env[MLX_AGENT_TAG_ENV];
      const dest = managedAgentBinPath(home);
      const tar = makeTarball(home, 'v030');
      const sha = createHash('sha256').update(tar).digest('hex');
      fetchSpy.mockImplementation(async (url: RequestInfo | URL) => {
        const u = url instanceof URL ? url.href : typeof url === 'string' ? url : url.url;
        if (u.endsWith('.sha256')) return new Response(sha);
        if (u.includes('releases/latest')) return new Response(JSON.stringify({ tag_name: 'mlx-agent-v0.2.9' }));
        if (u.endsWith('.tar.gz')) return new Response(new Uint8Array(tar));
        if (u.includes('dl.portato.si')) return new Response('mlx-agent-v0.3.0');
        return new Response('nope', { status: 404 });
      });
      // Candidates v0.3.0, v0.2.9, v0.2.0 — only v0.3.0 beats the v0.2.9
      // floor; the walk must land there, never on an older complete release.
      const path = await downloadAgentBinary(dest, undefined, {}, 'mlx-agent-v0.2.9');
      expect(path).toBe(versioned(home, 'mlx-agent-v0.3.0'));
      expect(readFileSync(`${dest}.tag`, 'utf-8')).toBe('mlx-agent-v0.3.0');
    } finally {
      if (saved === undefined) delete process.env[MLX_AGENT_TAG_ENV];
      else process.env[MLX_AGENT_TAG_ENV] = saved;
      rmSync(home, { recursive: true, force: true });
    }
  });

  it('a floored walk fails instead of installing an older candidate', async () => {
    const home = mkdtempSync(join(tmpdir(), 'agent-dl-'));
    const saved = process.env[MLX_AGENT_TAG_ENV];
    try {
      delete process.env[MLX_AGENT_TAG_ENV];
      const dest = managedAgentBinPath(home);
      fetchSpy.mockImplementation(async (url: RequestInfo | URL) => {
        const u = url instanceof URL ? url.href : typeof url === 'string' ? url : url.url;
        if (u.endsWith('.sha256')) return new Response('a'.repeat(64));
        if (u.includes('releases/latest')) return new Response(JSON.stringify({ tag_name: 'mlx-agent-v0.3.0' }));
        if (u.endsWith('.tar.gz')) return new Response('archive-bytes');
        if (u.includes('dl.portato.si')) return new Response('mlx-agent-v0.3.0');
        return new Response('nope', { status: 404 });
      });
      // Every candidate is at or below the v0.4.0 floor: the walk refuses
      // to move the pointer backward rather than "updating" to an older tag.
      await expect(downloadAgentBinary(dest, undefined, {}, 'mlx-agent-v0.4.0')).rejects.toThrow(
        'no release newer than mlx-agent-v0.4.0',
      );
      expect(existsSync(`${dest}.tag`)).toBe(false);
    } finally {
      if (saved === undefined) delete process.env[MLX_AGENT_TAG_ENV];
      else process.env[MLX_AGENT_TAG_ENV] = saved;
      rmSync(home, { recursive: true, force: true });
    }
  });
});

describe('resolveReleaseTag', () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;
  let savedTag: string | undefined;

  /** Any parseable digest — resolution only needs the shape, not the value. */
  const DIGEST = 'a'.repeat(64);

  function githubLatest(tag: string): Response {
    return new Response(JSON.stringify({ tag_name: tag }), { status: 200 });
  }

  beforeEach(() => {
    savedTag = process.env[MLX_AGENT_TAG_ENV];
    delete process.env[MLX_AGENT_TAG_ENV];
    fetchSpy = vi.spyOn(globalThis, 'fetch');
  });
  afterEach(() => {
    if (savedTag === undefined) delete process.env[MLX_AGENT_TAG_ENV];
    else process.env[MLX_AGENT_TAG_ENV] = savedTag;
    fetchSpy.mockRestore();
  });

  it('accepts a valid R2 latest pointer whose release verifies installable', async () => {
    fetchSpy.mockImplementation(async (url: RequestInfo | URL) => {
      const u = url instanceof URL ? url.href : typeof url === 'string' ? url : url.url;
      if (u.endsWith('.sha256')) return new Response(DIGEST);
      if (u.endsWith('.tar.gz')) return new Response('archive-bytes');
      if (u.includes('dl.portato.si')) return new Response('mlx-agent-v0.3.0');
      return new Response('nope', { status: 404 });
    });
    await expect(resolveReleaseTag()).resolves.toBe('mlx-agent-v0.3.0');
  });

  it('falls through to GitHub latest when the R2 tag has no release assets', async () => {
    fetchSpy.mockImplementation(async (url: RequestInfo | URL) => {
      const u = url instanceof URL ? url.href : typeof url === 'string' ? url : url.url;
      // R2 names a well-formed, above-minimum tag that never shipped — its
      // checksum 404s, so the mirror's word is not trusted.
      if (u.includes('releases/download/mlx-agent-v9.9.9')) return new Response('nope', { status: 404 });
      if (u.endsWith('.sha256')) return new Response(DIGEST);
      if (u.endsWith('.tar.gz')) return new Response('archive-bytes');
      if (u.includes('dl.portato.si')) return new Response('mlx-agent-v9.9.9');
      return githubLatest('mlx-agent-v0.3.0');
    });
    await expect(resolveReleaseTag()).resolves.toBe('mlx-agent-v0.3.0');
  });

  it('falls through to GitHub when the R2 tag checksum is empty or malformed', async () => {
    for (const bad of ['', 'not-a-digest', '<html>404</html>']) {
      fetchSpy.mockImplementation(async (url: RequestInfo | URL) => {
        const u = url instanceof URL ? url.href : typeof url === 'string' ? url : url.url;
        // The mirrored tag's checksum sidecar exists but carries no usable
        // digest — a 200 alone must not qualify the release.
        if (u.includes('releases/download/mlx-agent-v9.9.9')) return new Response(bad);
        if (u.endsWith('.sha256')) return new Response(DIGEST);
        if (u.endsWith('.tar.gz')) return new Response('archive-bytes');
        if (u.includes('dl.portato.si')) return new Response('mlx-agent-v9.9.9');
        return githubLatest('mlx-agent-v0.3.0');
      });
      await expect(resolveReleaseTag()).resolves.toBe('mlx-agent-v0.3.0');
    }
  });

  it('falls through to GitHub when the R2 tag archive is missing on both mirrors', async () => {
    fetchSpy.mockImplementation(async (url: RequestInfo | URL) => {
      const u = url instanceof URL ? url.href : typeof url === 'string' ? url : url.url;
      if (u.endsWith('.sha256')) return new Response(DIGEST);
      // The checksum shipped but no archive did — present-but-undownloadable.
      if (u.includes('mlx-agent-v9.9.9') && u.endsWith('.tar.gz')) return new Response('nope', { status: 404 });
      if (u.endsWith('.tar.gz')) return new Response('archive-bytes');
      if (u.includes('dl.portato.si')) return new Response('mlx-agent-v9.9.9');
      return githubLatest('mlx-agent-v0.3.0');
    });
    await expect(resolveReleaseTag()).resolves.toBe('mlx-agent-v0.3.0');
  });

  it('falls back to the minimum when no discovered tag is installable', async () => {
    fetchSpy.mockImplementation(async (url: RequestInfo | URL) => {
      const u = url instanceof URL ? url.href : typeof url === 'string' ? url : url.url;
      // Neither mirror can produce a checksum for the tags they name.
      if (u.endsWith('.sha256')) return new Response('nope', { status: 404 });
      if (u.endsWith('.tar.gz')) return new Response('archive-bytes');
      if (u.includes('dl.portato.si')) return new Response('mlx-agent-v0.3.0');
      return githubLatest('mlx-agent-v0.3.0');
    });
    await expect(resolveReleaseTag()).resolves.toBe(MLX_AGENT_MIN_RELEASE);
  });

  it('falls through to GitHub when the R2 latest is below the minimum', async () => {
    fetchSpy.mockImplementation(async (url: RequestInfo | URL) => {
      const u = url instanceof URL ? url.href : typeof url === 'string' ? url : url.url;
      if (u.endsWith('.sha256')) return new Response(DIGEST);
      if (u.endsWith('.tar.gz')) return new Response('archive-bytes');
      if (u.includes('dl.portato.si')) return new Response('mlx-agent-v0.1.0');
      return githubLatest('mlx-agent-v0.3.0');
    });
    await expect(resolveReleaseTag()).resolves.toBe('mlx-agent-v0.3.0');
  });

  it('falls through to GitHub when the R2 latest pointer streams past the byte cap', async () => {
    fetchSpy.mockImplementation(async (url: RequestInfo | URL) => {
      const u = url instanceof URL ? url.href : typeof url === 'string' ? url : url.url;
      if (u.endsWith('.sha256')) return new Response(DIGEST);
      if (u.endsWith('.tar.gz')) return new Response('archive-bytes');
      if (u.includes('github.com')) return githubLatest('mlx-agent-v0.3.0');
      // A hostile R2 streams an unbounded body before the tag can be read;
      // the cap rejects it instead of buffering it all.
      return new Response(
        new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(new Uint8Array(8 * 1024 * 1024));
            controller.close();
          },
        }),
      );
    });
    expect(await resolveReleaseTag()).toBe('mlx-agent-v0.3.0');
  });

  it('prefers the newer GitHub latest over a lagging but installable R2 pointer', async () => {
    fetchSpy.mockImplementation(async (url: RequestInfo | URL) => {
      const u = url instanceof URL ? url.href : typeof url === 'string' ? url : url.url;
      if (u.endsWith('.sha256')) return new Response(DIGEST);
      if (u.endsWith('.tar.gz')) return new Response('archive-bytes');
      // Mirror order is not version order: R2 still names the older v0.2.0
      // while GitHub's latest is v0.3.0 — the newest verified tag must win.
      if (u.includes('dl.portato.si')) return new Response('mlx-agent-v0.2.0');
      return githubLatest('mlx-agent-v0.3.0');
    });
    await expect(resolveReleaseTag()).resolves.toBe('mlx-agent-v0.3.0');
  });

  it('falls back to the minimum when every discovered tag is below it', async () => {
    fetchSpy.mockImplementation(async (url: RequestInfo | URL) => {
      const u = url instanceof URL ? url.href : typeof url === 'string' ? url : url.url;
      if (u.includes('dl.portato.si')) return new Response('mlx-agent-v0.1.0');
      return githubLatest('mlx-agent-v0.1.0');
    });
    await expect(resolveReleaseTag()).resolves.toBe(MLX_AGENT_MIN_RELEASE);
  });

  it('rejects a pinned tag below the minimum instead of downloading a stale cache', async () => {
    process.env[MLX_AGENT_TAG_ENV] = 'mlx-agent-v0.1.0';
    await expect(resolveReleaseTag()).rejects.toThrow(MLX_AGENT_TAG_ENV);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('rejects a malformed pinned tag instead of looping the download', async () => {
    for (const bad of [
      'v0.3.0',
      'mlx-agent-latest',
      'mlx-agent-v0.3',
      // semver.parse normalizes 'v0.3.0' and ' 0.3.0' to 0.3.0 — the pin is
      // used literally, so only the canonical form may pass.
      'mlx-agent-vv0.3.0',
      'mlx-agent-v 0.3.0',
    ]) {
      process.env[MLX_AGENT_TAG_ENV] = bad;
      await expect(resolveReleaseTag()).rejects.toThrow(MLX_AGENT_TAG_ENV);
    }
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('honors a pinned tag at or above the minimum', async () => {
    process.env[MLX_AGENT_TAG_ENV] = 'mlx-agent-v0.9.0';
    await expect(resolveReleaseTag()).resolves.toBe('mlx-agent-v0.9.0');
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});

describe('agentUpdateStatus', () => {
  it('reports updateAvailable when latest is newer than the managed tag', async () => {
    const s = await agentUpdateStatus({
      probe: status(),
      resolveTag: async () => 'mlx-agent-v0.3.0',
      managedPath: '/m/mlx-agent',
    });
    expect(s).toMatchObject({
      current: 'mlx-agent-v0.2.0',
      latest: 'mlx-agent-v0.3.0',
      updateAvailable: true,
      managedPath: '/m/mlx-agent',
      pinned: 'managed',
    });
  });

  it('reports up-to-date when managed matches latest', async () => {
    const s = await agentUpdateStatus({
      probe: status({ managedTag: 'mlx-agent-v0.3.0', effectiveTag: 'mlx-agent-v0.3.0' }),
      resolveTag: async () => 'mlx-agent-v0.3.0',
    });
    expect(s.updateAvailable).toBe(false);
  });

  it('treats an unstamped/missing install as updatable', async () => {
    const s = await agentUpdateStatus({
      probe: status({ source: 'none', path: undefined, managedTag: undefined, effectiveTag: undefined }),
      resolveTag: async () => 'mlx-agent-v0.3.0',
    });
    expect(s.current).toBeNull();
    expect(s.updateAvailable).toBe(true);
  });

  it('requests the pinned tag when a different release is cached', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'agent-pin-'));
    const saved = process.env[MLX_AGENT_TAG_ENV];
    process.env[MLX_AGENT_TAG_ENV] = 'mlx-agent-v0.2.0';
    try {
      const managed = join(dir, 'mlx-agent');
      // Cache holds the newer v0.3.0, but the pin wants exactly v0.2.0 —
      // an exact pin makes that an update (downgrade included).
      const s = await agentUpdateStatus({
        probe: status({ managedTag: 'mlx-agent-v0.3.0', effectiveTag: 'mlx-agent-v0.3.0' }),
        resolveTag: async () => 'mlx-agent-v0.2.0',
        managedPath: managed,
      });
      expect(s.latest).toBe('mlx-agent-v0.2.0');
      expect(s.updateAvailable).toBe(true);
    } finally {
      if (saved === undefined) delete process.env[MLX_AGENT_TAG_ENV];
      else process.env[MLX_AGENT_TAG_ENV] = saved;
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('is satisfied when the pinned release is already cached', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'agent-pin-'));
    const saved = process.env[MLX_AGENT_TAG_ENV];
    process.env[MLX_AGENT_TAG_ENV] = 'mlx-agent-v0.2.0';
    try {
      const managed = join(dir, 'mlx-agent');
      const exe = process.platform === 'win32' ? '.exe' : '';
      writeFileSync(join(dir, `mlx-agent-v0.2.0${exe}`), 'v020');
      const s = await agentUpdateStatus({
        probe: status({ managedTag: 'mlx-agent-v0.3.0', effectiveTag: 'mlx-agent-v0.2.0' }),
        resolveTag: async () => 'mlx-agent-v0.2.0',
        managedPath: managed,
      });
      expect(s.updateAvailable).toBe(false);
    } finally {
      if (saved === undefined) delete process.env[MLX_AGENT_TAG_ENV];
      else process.env[MLX_AGENT_TAG_ENV] = saved;
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('applyAgentUpdate', () => {
  it('downloads into the managed path when newer and reports the new current', async () => {
    const download = vi.fn(async (dest: string, tag?: string) => {
      writeFileSync(`${dest}.tag`, tag ?? 'unpinned');
      return dest;
    });
    const dir = mkdtempSync(join(tmpdir(), 'agent-update-'));
    try {
      const managed = join(dir, 'mlx-agent');
      const { status: s, downloaded } = await applyAgentUpdate({
        probe: status(),
        resolveTag: async () => 'mlx-agent-v0.3.0',
        managedPath: managed,
        download,
      });
      expect(download).toHaveBeenCalledWith(managed, 'mlx-agent-v0.3.0');
      expect(downloaded).toBe(true);
      expect(s.current).toBe('mlx-agent-v0.3.0');
      expect(s.updateAvailable).toBe(false);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('skips the download when already current', async () => {
    const download = vi.fn(async () => '/m/mlx-agent');
    const { downloaded } = await applyAgentUpdate({
      probe: status({ managedTag: 'mlx-agent-v0.3.0', effectiveTag: 'mlx-agent-v0.3.0' }),
      resolveTag: async () => 'mlx-agent-v0.3.0',
      download,
    });
    expect(downloaded).toBe(false);
    expect(download).not.toHaveBeenCalled();
  });

  it('retries the candidate walk when the latest tag cannot be fetched', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'agent-update-'));
    try {
      const managed = join(dir, 'mlx-agent');
      const download = vi.fn(async (dest: string, tag?: string) => {
        if (tag !== undefined) throw new Error('archive vanished between resolve and fetch');
        // The fallback (undefined = candidate walk) lands the older release.
        writeFileSync(`${dest}.tag`, 'mlx-agent-v0.2.0');
        return join(dir, 'mlx-agent-v0.2.0');
      });
      const { status: s, downloaded } = await applyAgentUpdate({
        probe: status({ managedTag: undefined, effectiveTag: undefined, source: 'none', path: undefined }),
        resolveTag: async () => 'mlx-agent-v0.3.0',
        managedPath: managed,
        download,
      });
      expect(downloaded).toBe(true);
      expect(download).toHaveBeenNthCalledWith(1, managed, 'mlx-agent-v0.3.0');
      expect(download).toHaveBeenNthCalledWith(2, managed, undefined, undefined);
      // current reports the ACTUALLY installed tag, not the requested one —
      // and the update must still read as available: the installed release
      // trails latest, so callers must not claim "updated to latest".
      expect(s.current).toBe('mlx-agent-v0.2.0');
      expect(s.latest).toBe('mlx-agent-v0.3.0');
      expect(s.updateAvailable).toBe(true);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('floors the fallback walk at the installed release so it cannot downgrade', async () => {
    const download = vi.fn(async (dest: string, tag?: string) => {
      if (tag !== undefined) throw new Error('latest archive corrupt');
      return dest;
    });
    const dir = mkdtempSync(join(tmpdir(), 'agent-update-'));
    try {
      const managed = join(dir, 'mlx-agent');
      // Managed v0.4.0 with v0.5.0 latest whose fetch fails: the candidate
      // walk may offer v0.3.0/v0.2.0, but anything at or below the install
      // is a downgrade — the floor must reach the downloader.
      const { status: s } = await applyAgentUpdate({
        probe: status({ managedTag: 'mlx-agent-v0.4.0', effectiveTag: 'mlx-agent-v0.4.0' }),
        resolveTag: async () => 'mlx-agent-v0.5.0',
        managedPath: managed,
        download,
      });
      expect(download).toHaveBeenNthCalledWith(2, managed, undefined, 'mlx-agent-v0.4.0');
      // A pinned install whose managed cache is NEWER than the pin still
      // floors at the cache tag — unpinning must not expose a downgrade.
      const pinnedDownload = vi.fn(async (dest: string, tag?: string) => {
        if (tag !== undefined) throw new Error('latest archive corrupt');
        return dest;
      });
      const saved = process.env[MLX_AGENT_TAG_ENV];
      try {
        // A pin naming an uninstalled tag keeps the update eligible; the
        // floor is still the managed cache's v0.4.0, not the pin.
        process.env[MLX_AGENT_TAG_ENV] = 'mlx-agent-v0.6.0';
        await applyAgentUpdate({
          probe: status({ managedTag: 'mlx-agent-v0.4.0', effectiveTag: 'mlx-agent-v0.3.0' }),
          resolveTag: async () => 'mlx-agent-v0.5.0',
          managedPath: managed,
          download: pinnedDownload,
        });
        expect(pinnedDownload).toHaveBeenNthCalledWith(2, managed, undefined, 'mlx-agent-v0.4.0');
      } finally {
        if (saved === undefined) delete process.env[MLX_AGENT_TAG_ENV];
        else process.env[MLX_AGENT_TAG_ENV] = saved;
      }
      expect(s.newestInstalled).toBe('mlx-agent-v0.4.0');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('agentTagFromBinPath', () => {
  it('reads the release tag back from a versioned install path', () => {
    expect(agentTagFromBinPath('/cache/mlx-agent-v1.2.3')).toBe('mlx-agent-v1.2.3');
    expect(agentTagFromBinPath('/cache/mlx-agent-v1.2.3.exe')).toBe('mlx-agent-v1.2.3');
    expect(agentTagFromBinPath('/cache/mlx-agent')).toBeUndefined();
    expect(agentTagFromBinPath('/cache/not-a-tag.bin')).toBeUndefined();
  });
});

describe('mlx agent update (run)', () => {
  let savedExitCode: typeof process.exitCode;
  let log: ReturnType<typeof vi.spyOn>;
  let warn: ReturnType<typeof vi.spyOn>;
  let err: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    savedExitCode = process.exitCode;
    process.exitCode = 0;
    log = vi.spyOn(console, 'log').mockImplementation(() => {});
    warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    err = vi.spyOn(console, 'error').mockImplementation(() => {});
  });
  afterEach(() => {
    process.exitCode = savedExitCode;
    vi.restoreAllMocks();
  });

  it('update --check prints the status JSON and does not download', async () => {
    const updateStatus = vi.fn(async () => ({
      current: 'mlx-agent-v0.2.0',
      latest: 'mlx-agent-v0.3.0',
      updateAvailable: true,
      managedPath: '/m/mlx-agent',
      pinned: 'managed' as const,
      newestInstalled: 'mlx-agent-v0.2.0',
    }));
    const applyUpdate = vi.fn();
    await run(['update', '--check'], { updateStatus, applyUpdate });
    expect(updateStatus).toHaveBeenCalledOnce();
    expect(applyUpdate).not.toHaveBeenCalled();
    expect(JSON.parse(log.mock.calls[0]![0] as string)).toMatchObject({ latest: 'mlx-agent-v0.3.0' });
    expect(process.exitCode).toBe(0);
  });

  it('update downloads and reports the new tag', async () => {
    const applyUpdate = vi.fn(async () => ({
      downloaded: true,
      status: {
        current: 'mlx-agent-v0.3.0',
        latest: 'mlx-agent-v0.3.0',
        updateAvailable: false,
        managedPath: '/m/mlx-agent',
        pinned: 'managed' as const,
        newestInstalled: null,
      },
    }));
    await run(['update'], { applyUpdate });
    expect(log).toHaveBeenCalledWith(expect.stringContaining('updated to mlx-agent-v0.3.0'));
    expect(process.exitCode).toBe(0);
  });

  it('update reports the installed tag and warns when a fallback lands below latest', async () => {
    const applyUpdate = vi.fn(async () => ({
      downloaded: true,
      status: {
        current: 'mlx-agent-v0.2.0',
        latest: 'mlx-agent-v0.3.0',
        updateAvailable: true,
        managedPath: '/m/mlx-agent',
        pinned: 'managed' as const,
        newestInstalled: null,
      },
    }));
    await run(['update'], { applyUpdate });
    expect(log).toHaveBeenCalledWith(expect.stringContaining('updated to mlx-agent-v0.2.0'));
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('mlx-agent-v0.3.0'));
    expect(process.exitCode).toBe(0);
  });

  it('update ends a piped run with the status JSON the desktop tray parses', async () => {
    // The desktop's parse takes the LAST JSON stdout line — a fallback
    // install below latest must surface updateAvailable there, not just
    // in the human warning.
    const applyUpdate = vi.fn(async () => ({
      downloaded: true,
      status: {
        current: 'mlx-agent-v0.2.0',
        latest: 'mlx-agent-v0.3.0',
        updateAvailable: true,
        managedPath: '/m/mlx-agent',
        pinned: 'managed' as const,
        newestInstalled: null,
      },
    }));
    const isTTY = Object.getOwnPropertyDescriptor(process.stdout, 'isTTY');
    Object.defineProperty(process.stdout, 'isTTY', { value: false, configurable: true });
    try {
      await run(['update'], { applyUpdate });
    } finally {
      if (isTTY !== undefined) Object.defineProperty(process.stdout, 'isTTY', isTTY);
      else delete (process.stdout as { isTTY?: boolean }).isTTY;
    }
    const last = log.mock.calls.at(-1)![0] as string;
    expect(JSON.parse(last)).toMatchObject({
      current: 'mlx-agent-v0.2.0',
      latest: 'mlx-agent-v0.3.0',
      updateAvailable: true,
    });
  });

  it('update on a pinned env install warns that the pin shadows the download', async () => {
    const applyUpdate = vi.fn(async () => ({
      downloaded: true,
      status: {
        current: 'mlx-agent-v0.3.0',
        latest: 'mlx-agent-v0.3.0',
        updateAvailable: false,
        managedPath: '/m/mlx-agent',
        pinned: 'env' as const,
        newestInstalled: null,
      },
    }));
    await run(['update'], { applyUpdate });
    expect(warn).toHaveBeenCalledWith(expect.stringContaining(MLX_AGENT_BIN_ENV));
  });

  it('update failures exit non-zero with the reason', async () => {
    const applyUpdate = vi.fn(async () => {
      throw new Error('checksum mismatch');
    });
    await run(['update'], { applyUpdate });
    expect(err).toHaveBeenCalledWith(expect.stringContaining('checksum mismatch'));
    expect(process.exitCode).toBe(1);
  });

  it('update --help prints usage without touching the network', async () => {
    const applyUpdate = vi.fn();
    await run(['update', '--help'], { applyUpdate });
    expect(log).toHaveBeenCalledWith(expect.stringContaining('Usage: mlx agent update'));
    expect(applyUpdate).not.toHaveBeenCalled();
  });
});

/**
 * `needsServer` is the CALLER's routing decision: `scanAgentArgs` walks argv
 * value-aware, so a `--help`/`--version` token sitting in another option's
 * value slot stays data. The launcher must honor `opts.needsServer` verbatim
 * — a raw `argv.includes('--help')` rescan here would starve turns like
 * `--system-prompt-override --help` of the inference host.
 */
describe('launchGrokAgent server gating', () => {
  let root: string;
  let health: ReturnType<typeof createHttpServer>;
  let serveBase: string;
  let savedAgentBin: string | undefined;
  let savedAgentTag: string | undefined;

  const serveKills: ReturnType<typeof vi.fn>[] = [];
  const spawnServe = vi.fn((_argv: string[], _so: SpawnOptions) => {
    const kill = vi.fn();
    serveKills.push(kill);
    const child = Object.assign(new EventEmitter(), {
      stdout: new PassThrough(),
      kill,
    });
    child.stdout.end(`listening on ${serveBase}\n`);
    return child as unknown as ChildProcess;
  });

  function spawnAgent(_bin: string, _argv: string[], so: SpawnOptions): ChildProcess {
    spawnAgent.envs.push(so.env ?? {});
    const agent = Object.assign(new EventEmitter(), { kill: vi.fn() }) as unknown as ChildProcess;
    queueMicrotask(() => agent.emit('exit', 0));
    return agent;
  }
  spawnAgent.envs = [] as NodeJS.ProcessEnv[];

  beforeEach(async () => {
    root = mkdtempSync(join(tmpdir(), 'launch-grok-'));
    const bin = join(root, 'mlx-agent');
    writeFileSync(bin, 'x');
    savedAgentBin = process.env[MLX_AGENT_BIN_ENV];
    savedAgentTag = process.env[MLX_AGENT_TAG_ENV];
    process.env[MLX_AGENT_BIN_ENV] = bin;
    delete process.env[MLX_AGENT_TAG_ENV];
    health = createHttpServer((_, res) => res.end('ok'));
    await new Promise<void>((resolvePromise) => health.listen(0, '127.0.0.1', resolvePromise));
    serveBase = `http://127.0.0.1:${(health.address() as AddressInfo).port}`;
    spawnServe.mockClear();
    serveKills.length = 0;
    spawnAgent.envs.length = 0;
  });

  afterEach(async () => {
    if (savedAgentBin === undefined) delete process.env[MLX_AGENT_BIN_ENV];
    else process.env[MLX_AGENT_BIN_ENV] = savedAgentBin;
    if (savedAgentTag === undefined) delete process.env[MLX_AGENT_TAG_ENV];
    else process.env[MLX_AGENT_TAG_ENV] = savedAgentTag;
    await new Promise<void>((resolvePromise) => health.close(() => resolvePromise()));
    rmSync(root, { recursive: true, force: true });
  });

  it('serves on an unset needsServer even when argv contains a --help-looking token', async () => {
    const code = await launchGrokAgent({
      argv: ['--system-prompt-override', '--help', '-p', 'hi'],
      spawnServe,
      spawnAgent,
    });
    expect(code).toBe(0);
    expect(spawnServe).toHaveBeenCalledOnce();
    expect(spawnAgent.envs[0]!.GROK_MODELS_BASE_URL).toBe(`${serveBase}/v1`);
  });

  it('skips the server only when the caller sets needsServer: false', async () => {
    const code = await launchGrokAgent({
      argv: ['--version'],
      needsServer: false,
      spawnServe,
      spawnAgent,
    });
    expect(code).toBe(0);
    expect(spawnServe).not.toHaveBeenCalled();
    expect(spawnAgent.envs[0]!.GROK_MODELS_BASE_URL).toBeUndefined();
  });

  it('kills the serve child when readiness fails', async () => {
    // No banner line — the child lives but never reports a port. The
    // rejection must still reap it, or `mlx agent` leaves an orphan serve
    // consuming model memory.
    spawnServe.mockImplementationOnce((_argv: string[], _so: SpawnOptions) => {
      const kill = vi.fn();
      serveKills.push(kill);
      return Object.assign(new EventEmitter(), {
        stdout: new PassThrough(),
        kill,
      }) as unknown as ChildProcess;
    });
    await expect(
      launchGrokAgent({ argv: ['-p', 'hi'], spawnServe, spawnAgent, serveReadyTimeoutMs: 50 }),
    ).rejects.toThrow('timed out waiting for `mlx serve`');
    expect(serveKills[0]!).toHaveBeenCalledWith('SIGTERM');
  });

  it('kills the serve child when the agent fails to spawn', async () => {
    // spawn() emits 'error' without 'exit' for a non-executable binary.
    // Emitting inside the factory keeps the microtask ordered AFTER the
    // launcher attaches its own 'error' listener.
    const agentKill = vi.fn();
    const agent = Object.assign(new EventEmitter(), { kill: agentKill }) as unknown as ChildProcess;
    await expect(
      launchGrokAgent({
        argv: ['-p', 'hi'],
        spawnServe,
        spawnAgent: () => {
          queueMicrotask(() => agent.emit('error', new Error('spawn EACCES')));
          return agent;
        },
      }),
    ).rejects.toThrow('spawn EACCES');
    expect(serveKills[0]!).toHaveBeenCalledWith('SIGTERM');
    expect(agentKill).not.toHaveBeenCalled();
  });

  it('forwards SIGTERM to both children and re-raises it', async () => {
    const agentKill = vi.fn();
    const agent = Object.assign(new EventEmitter(), { kill: agentKill }) as unknown as ChildProcess;
    const baseline = new Set(process.listeners('SIGTERM'));
    const killSpy = vi.spyOn(process, 'kill').mockImplementation(() => true);
    try {
      const launch = launchGrokAgent({ argv: ['-p', 'hi'], spawnServe, spawnAgent: () => agent });
      // Poll: the listeners land only after serve readiness resolves.
      let teardown: ((signal: 'SIGTERM') => void) | undefined;
      for (let i = 0; i < 100 && teardown === undefined; i++) {
        await new Promise((resolvePromise) => setTimeout(resolvePromise, 10));
        teardown = process.listeners('SIGTERM').find((listener) => !baseline.has(listener));
      }
      expect(teardown).toBeDefined();
      teardown!('SIGTERM');
      expect(serveKills[0]!).toHaveBeenCalledWith('SIGTERM');
      expect(agentKill).toHaveBeenCalledWith('SIGTERM');
      expect(killSpy).toHaveBeenCalledWith(process.pid, 'SIGTERM');
      agent.emit('exit', 143, 'SIGTERM');
      await expect(launch).resolves.toBe(143);
    } finally {
      killSpy.mockRestore();
    }
  });
});
