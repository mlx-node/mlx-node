import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { describe, expect, it, vi, afterEach, beforeEach } from 'vitest';

import {
  acquireInstallLock,
  agentUpdateStatus,
  applyAgentUpdate,
  downloadAgentBinary,
  managedAgentBinPath,
  probeAgentInstall,
  releaseInstallLock,
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

  const EXE = process.platform === 'win32' ? '.exe' : '';
  const versioned = (home: string, tag: string) => join(home, '.mlx-node', 'bin', `${tag}${EXE}`);
  const mkdirBin = (home: string) => mkdirSync(join(home, '.mlx-node', 'bin'), { recursive: true });

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
      const dir = `${lock}.d`;
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
      expect(members[0]).toMatch(/^b-\d+-\d+-\d+$/);
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
      try {
        const lock = join(home, 'mlx-agent.lock');
        const dir = `${lock}.d`;
        mkdirSync(dir, { recursive: true });
        // A member from the previous `m-` protocol — live (our pid), high
        // seq. Under the superseded (pid, seq) ticket order this member
        // sorted "behind" a new acquire with a lower seq, letting a later
        // contender in while it held. Under bakery it maps to ticket 0 —
        // it advertised before any `b-` member — so we MUST queue.
        writeFileSync(join(dir, `m-${process.pid}-999`), '');
        await expect(acquireInstallLock(lock, { deadlineMs: 400 })).rejects.toThrow('install lock');

        // Same for a low bakery ticket under our pid: ticket 2 < whatever we
        // choose next, so the earlier advertiser wins regardless of seq.
        rmSync(join(dir, `m-${process.pid}-999`));
        writeFileSync(join(dir, `b-2-${process.pid}-999`), '');
        await expect(acquireInstallLock(lock, { deadlineMs: 400 })).rejects.toThrow('install lock');
      } finally {
        rmSync(home, { recursive: true, force: true });
      }
    },
  );

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
});

describe('resolveReleaseTag', () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;
  let savedTag: string | undefined;

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

  it('accepts a valid R2 latest pointer at or above the minimum', async () => {
    fetchSpy.mockResolvedValue(new Response('mlx-agent-v0.3.0'));
    await expect(resolveReleaseTag()).resolves.toBe('mlx-agent-v0.3.0');
  });

  it('falls through to GitHub when the R2 latest is below the minimum', async () => {
    fetchSpy.mockImplementation(async (url: RequestInfo | URL) => {
      const u = url instanceof URL ? url.href : typeof url === 'string' ? url : url.url;
      if (u.includes('dl.portato.si')) return new Response('mlx-agent-v0.1.0');
      return githubLatest('mlx-agent-v0.3.0');
    });
    await expect(resolveReleaseTag()).resolves.toBe('mlx-agent-v0.3.0');
  });

  it('falls through to GitHub when the R2 latest pointer streams past the byte cap', async () => {
    fetchSpy.mockImplementation(async (url: RequestInfo | URL) => {
      const u = url instanceof URL ? url.href : typeof url === 'string' ? url : url.url;
      if (u.includes('github.com')) return new Response(JSON.stringify({ tag_name: 'mlx-agent-v0.3.0' }));
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
    for (const bad of ['v0.3.0', 'mlx-agent-latest', 'mlx-agent-v0.3']) {
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
    const download = vi.fn(async (dest: string, tag: string) => {
      writeFileSync(`${dest}.tag`, tag);
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
      },
    }));
    await run(['update'], { applyUpdate });
    expect(log).toHaveBeenCalledWith(expect.stringContaining('updated to mlx-agent-v0.3.0'));
    expect(process.exitCode).toBe(0);
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
