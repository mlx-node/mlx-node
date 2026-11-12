/**
 * `mlx serve` SSD cache flags (Splash `serve_options.py`): `--max-cache-disk`
 * (default auto), `--persistent-cache` (default off), `--cache-dir` (needs
 * `--persistent-cache`), and the two usage errors of `check_serve_arguments`.
 */

import { homedir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { parseServeArgs } from '../src/commands/serve.js';

describe('mlx serve cache flags', () => {
  it('defaults to an automatic temporary cache', () => {
    const args = parseServeArgs([]);
    expect(args).toMatchObject({ cacheDisk: 'auto', persistentCache: false });
    expect(args.cacheDir).toBeUndefined();
  });

  it('parses --max-cache-disk as Splash: auto, 0 or a size', () => {
    expect(parseServeArgs(['--max-cache-disk', '0']).cacheDisk).toBe('0');
    expect(parseServeArgs(['--max-cache-disk', ' 0 ']).cacheDisk).toBe('0');
    expect(parseServeArgs(['--max-cache-disk', '64G']).cacheDisk).toBe(String(64 * 1024 ** 3));
    expect(parseServeArgs(['--max-cache-disk=2.5G']).cacheDisk).toBe(String(2.5 * 1024 ** 3));
    expect(parseServeArgs(['--max-cache-disk', 'AUTO']).cacheDisk).toBe('auto');
    for (const bad of ['', 'lots', '-1', '1.5', '0G']) {
      expect(() => parseServeArgs([`--max-cache-disk=${bad}`])).toThrow(
        '--max-cache-disk must be auto, 0 to disable, or a size such as 16G',
      );
    }
  });

  it('keeps a persistent cache under --cache-dir, ~ expanded', () => {
    expect(parseServeArgs(['--persistent-cache'])).toMatchObject({ cacheDisk: 'auto', persistentCache: true });
    expect(parseServeArgs(['--persistent-cache', '--cache-dir', '/var/c']).cacheDir).toBe('/var/c');
    expect(parseServeArgs(['--persistent-cache', '--cache-dir', '~/c']).cacheDir).toBe(join(homedir(), 'c'));
    expect(() => parseServeArgs(['--persistent-cache', '--cache-dir', ' '])).toThrow(
      '--cache-dir must name a directory',
    );
  });

  it('refuses the flag pairs Splash refuses', () => {
    expect(() => parseServeArgs(['--cache-dir', '/var/c'])).toThrow('--cache-dir needs --persistent-cache');
    expect(() => parseServeArgs(['--persistent-cache', '--max-cache-disk', '0'])).toThrow(
      '--persistent-cache needs the SSD cache --max-cache-disk 0 turns off',
    );
  });
});
