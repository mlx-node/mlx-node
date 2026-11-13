/**
 * `mlx agent` argv scan — the mlx-owned `--no-persist-cache` switch and the
 * Qwen3.5 SSD state tier flags it lifts and forwards to the `mlx serve` it
 * spawns (`--max-cache-disk`, `--cache-dir`, Splash `serve_options.py`). Like
 * the other mlx-owned flags they never reach the agent binary, and a flag in a
 * binary value-consumer's value slot stays data.
 */

import { homedir } from 'node:os';
import { join } from 'node:path';

import { coldTierRestoreFamilyList } from '@mlx-node/agent/catalog';
import { describe, expect, it } from 'vite-plus/test';

import { agentPreambleText, scanAgentArgs } from '../src/commands/agent/index.js';

describe('scanAgentArgs --no-persist-cache', () => {
  // Mutation TPA1: the agent's default is `byDefault: false` → red.
  it('keeps the paged KV disk cache on when the flag is absent', () => {
    const scan = scanAgentArgs(['-c'], {});
    expect(scan.persistPagedCache).toEqual({ byDefault: true });
    expect(scan.passthrough).toEqual(['-c']);
  });

  it('lifts --no-persist-cache out of the argv as the explicit flag', () => {
    const scan = scanAgentArgs(['--no-persist-cache', '-c'], {});
    expect(scan.persistPagedCache).toEqual({ flag: false, byDefault: true });
    expect(scan.passthrough).toEqual(['-c']);
  });

  it('keeps the flag when it sits in a binary value slot', () => {
    const scan = scanAgentArgs(['--system-prompt', '--no-persist-cache'], {});
    // "--no-persist-cache" is the system-prompt VALUE here, so it stays on
    // and is forwarded verbatim.
    expect(scan.persistPagedCache).toEqual({ byDefault: true });
    expect(scan.passthrough).toEqual(['--system-prompt', '--no-persist-cache']);
  });

  it('no longer knows --persistent-cache: it goes to the binary like any unknown flag', () => {
    for (const flag of ['--persistent-cache', '--no-persistent-cache']) {
      const scan = scanAgentArgs([flag, '-c'], {});
      expect(scan.persistPagedCache).toEqual({ byDefault: true });
      expect(scan.passthrough).toEqual([flag, '-c']);
    }
  });
});

describe('scanAgentArgs SSD state tier flags', () => {
  it('leaves both unset when none is given', () => {
    const scan = scanAgentArgs(['-c'], {});
    expect(scan.cacheDisk).toBeUndefined();
    expect(scan.cacheDir).toBeUndefined();
    expect(scan.cacheDiskError).toBeUndefined();
  });

  it('lifts the flags out of the argv in both spellings', () => {
    const spaced = scanAgentArgs(['--max-cache-disk', '16G', '--cache-dir', '/c', '-c'], {});
    expect(spaced).toMatchObject({ cacheDisk: String(16 * 1024 ** 3), cacheDir: '/c' });
    expect(spaced.passthrough).toEqual(['-c']);

    const inline = scanAgentArgs(['--max-cache-disk=0', '-c'], {});
    expect(inline.cacheDisk).toBe('0');
    expect(inline.passthrough).toEqual(['-c']);

    const home = scanAgentArgs(['--cache-dir=~/x'], {});
    expect(home.cacheDir).toBe(join(homedir(), 'x'));
    expect(home.passthrough).toEqual([]);
  });

  it('reports a bad value and the Splash conflicts as usage errors', () => {
    expect(scanAgentArgs(['--max-cache-disk', 'lots'], {}).cacheDiskError).toBe(
      '--max-cache-disk must be auto, 0 to disable, or a size such as 16G',
    );
    expect(scanAgentArgs(['--max-cache-disk'], {}).cacheDiskError).toBe('Missing value for --max-cache-disk');
    expect(scanAgentArgs(['--cache-dir', '/c'], {}).cacheDiskError).toBeUndefined();
    expect(scanAgentArgs(['--max-cache-disk', '0'], {}).cacheDiskError).toBeUndefined();
    expect(scanAgentArgs(['--no-persist-cache', '--cache-dir', '/c'], {}).cacheDiskError).toBe(
      '--cache-dir needs the persistent cache --no-persist-cache turns off',
    );
    expect(scanAgentArgs(['--cache-dir', '/c'], { MLX_PERSIST_PAGED_CACHE: '0' }).cacheDiskError).toBe(
      '--cache-dir needs the persistent cache MLX_PERSIST_PAGED_CACHE turns off',
    );
    expect(scanAgentArgs(['--max-cache-disk', '0', '--cache-dir', '/c'], {}).cacheDiskError).toBe(
      '--cache-dir needs the SSD cache --max-cache-disk 0 turns off',
    );
  });
});

describe('mlx agent help — --no-persist-cache', () => {
  it('names every restore-eligible family and the Qwen3.5 state tier', () => {
    const help = agentPreambleText();
    expect(help).toContain('--no-persist-cache');
    expect(help).toContain(coldTierRestoreFamilyList().join(', '));
    expect(help).toContain('Qwen3.5 dense and MoE state tier');
    expect(help).toContain('MLX_PERSIST_PAGED_CACHE=0');
    expect(help).toContain('--max-cache-disk <size>');
    expect(help).toContain('--cache-dir <dir>');
    expect(help).not.toContain('--persistent-cache');
  });
});
