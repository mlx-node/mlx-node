/**
 * `mlx agent` argv scan — the SSD cache flags it lifts and forwards to the
 * `mlx serve` it spawns (`--max-cache-disk`, `--persistent-cache`,
 * `--cache-dir`, Splash `serve_options.py`). Like the other mlx-owned flags
 * they never reach the agent binary, and a flag in a binary value-consumer's
 * value slot stays data.
 */

import { homedir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { agentPreambleText, scanAgentArgs } from '../src/commands/agent/index.js';

describe('scanAgentArgs SSD cache flags', () => {
  it('leaves every cache flag unset when none is given (a temporary cache, auto quota)', () => {
    const scan = scanAgentArgs(['-c']);
    expect(scan.cacheDisk).toBeUndefined();
    expect(scan.persistentCache).toBe(false);
    expect(scan.cacheDir).toBeUndefined();
    expect(scan.cacheDiskError).toBeUndefined();
    expect(scan.passthrough).toEqual(['-c']);
  });

  // Mutation TSF1: stop lifting `--persistent-cache` (forward it to grok) → red.
  it('lifts the three flags out of the argv in both spellings', () => {
    const spaced = scanAgentArgs(['--max-cache-disk', '16G', '--persistent-cache', '--cache-dir', '/c', '-c']);
    expect(spaced).toMatchObject({ cacheDisk: String(16 * 1024 ** 3), persistentCache: true, cacheDir: '/c' });
    expect(spaced.passthrough).toEqual(['-c']);

    const inline = scanAgentArgs(['--max-cache-disk=0', '-c']);
    expect(inline.cacheDisk).toBe('0');
    expect(inline.passthrough).toEqual(['-c']);

    const home = scanAgentArgs(['--persistent-cache', '--cache-dir=~/x']);
    expect(home.cacheDir).toBe(join(homedir(), 'x'));
    expect(home.passthrough).toEqual([]);
  });

  it('keeps a cache flag sitting in a binary value slot', () => {
    const scan = scanAgentArgs(['--system-prompt', '--persistent-cache']);
    expect(scan.persistentCache).toBe(false);
    expect(scan.passthrough).toEqual(['--system-prompt', '--persistent-cache']);
  });

  it('reports a bad value and the Splash conflicts as usage errors', () => {
    expect(scanAgentArgs(['--max-cache-disk', 'lots']).cacheDiskError).toBe(
      '--max-cache-disk must be auto, 0 to disable, or a size such as 16G',
    );
    expect(scanAgentArgs(['--max-cache-disk']).cacheDiskError).toBe('Missing value for --max-cache-disk');
    expect(scanAgentArgs(['--cache-dir', '/c']).cacheDiskError).toBe('--cache-dir needs --persistent-cache');
    expect(scanAgentArgs(['--persistent-cache', '--max-cache-disk', '0']).cacheDiskError).toBe(
      '--persistent-cache needs the SSD cache --max-cache-disk 0 turns off',
    );
  });

  it('no longer knows --no-persist-cache: it goes to the binary like any unknown flag', () => {
    const scan = scanAgentArgs(['--no-persist-cache', '-c']);
    expect(scan.persistentCache).toBe(false);
    expect(scan.passthrough).toEqual(['--no-persist-cache', '-c']);
  });
});

describe('mlx agent help — SSD cache flags', () => {
  it('documents the three flags', () => {
    const help = agentPreambleText();
    expect(help).toContain('--max-cache-disk <size>');
    expect(help).toContain('--persistent-cache');
    expect(help).toContain('--cache-dir <dir>');
    expect(help).not.toContain('--no-persist-cache');
  });
});
