import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it, vi, afterEach, beforeEach } from 'vitest';

import {
  agentUpdateStatus,
  applyAgentUpdate,
  probeAgentInstall,
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
