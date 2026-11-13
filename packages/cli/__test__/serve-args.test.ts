/**
 * `mlx serve` paged KV disk cache flags: `--persist-cache` /
 * `--no-persist-cache` (the switch every family shares; flag >
 * `MLX_PERSIST_PAGED_CACHE` > on), and the Qwen3.5 SSD state tier's
 * `--max-cache-disk` (default auto) and `--cache-dir` with the usage errors
 * of Splash `check_serve_arguments`.
 */

import { homedir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { parseServeArgs } from '../src/commands/serve.js';

describe('mlx serve cache flags', () => {
  it('defaults to an automatic tier and leaves the switch to the host default (on)', () => {
    const args = parseServeArgs([], {});
    expect(args.cacheDisk).toBe('auto');
    expect(args.persistPagedCache).toBeUndefined();
    expect(args.cacheDir).toBeUndefined();
  });

  it('reads --persist-cache and --no-persist-cache as the explicit flag', () => {
    expect(parseServeArgs(['--persist-cache'], {}).persistPagedCache).toBe(true);
    expect(parseServeArgs(['--no-persist-cache'], {}).persistPagedCache).toBe(false);
    expect(parseServeArgs(['--persist-cache'], { MLX_PERSIST_PAGED_CACHE: '0' }).persistPagedCache).toBe(true);
    expect(() => parseServeArgs(['--persist-cache', '--no-persist-cache'], {})).toThrow(
      '--persist-cache and --no-persist-cache cannot be used together',
    );
  });

  it('no longer knows --persistent-cache', () => {
    expect(() => parseServeArgs(['--persistent-cache'], {})).toThrow(/persistent-cache/);
    expect(() => parseServeArgs(['--no-persistent-cache'], {})).toThrow(/persistent-cache/);
  });

  it('parses --max-cache-disk as Splash: auto, 0 or a size', () => {
    expect(parseServeArgs(['--max-cache-disk', '0'], {}).cacheDisk).toBe('0');
    expect(parseServeArgs(['--max-cache-disk', ' 0 '], {}).cacheDisk).toBe('0');
    expect(parseServeArgs(['--max-cache-disk', '64G'], {}).cacheDisk).toBe(String(64 * 1024 ** 3));
    expect(parseServeArgs(['--max-cache-disk=2.5G'], {}).cacheDisk).toBe(String(2.5 * 1024 ** 3));
    expect(parseServeArgs(['--max-cache-disk', 'AUTO'], {}).cacheDisk).toBe('auto');
    for (const bad of ['', 'lots', '-1', '1.5', '0G']) {
      expect(() => parseServeArgs([`--max-cache-disk=${bad}`], {})).toThrow(
        '--max-cache-disk must be auto, 0 to disable, or a size such as 16G',
      );
    }
  });

  it('keeps the switch on beside --max-cache-disk 0: it also serves the other families', () => {
    expect(parseServeArgs(['--persist-cache', '--max-cache-disk', '0'], {})).toMatchObject({
      cacheDisk: '0',
      persistPagedCache: true,
    });
  });

  it('takes --cache-dir while the switch is on, ~ expanded', () => {
    expect(parseServeArgs(['--cache-dir', '/var/c'], {}).cacheDir).toBe('/var/c');
    expect(parseServeArgs(['--persist-cache', '--cache-dir', '/var/c'], {}).cacheDir).toBe('/var/c');
    expect(parseServeArgs(['--cache-dir', '~/c'], {}).cacheDir).toBe(join(homedir(), 'c'));
    expect(() => parseServeArgs(['--cache-dir', ' '], {})).toThrow('--cache-dir must name a directory');
  });

  it('refuses a --cache-dir nothing keeps', () => {
    expect(() => parseServeArgs(['--no-persist-cache', '--cache-dir', '/var/c'], {})).toThrow(
      '--cache-dir needs the persistent cache --no-persist-cache turns off',
    );
    expect(() => parseServeArgs(['--cache-dir', '/var/c'], { MLX_PERSIST_PAGED_CACHE: 'off' })).toThrow(
      '--cache-dir needs the persistent cache MLX_PERSIST_PAGED_CACHE turns off',
    );
    expect(() => parseServeArgs(['--max-cache-disk', '0', '--cache-dir', '/var/c'], {})).toThrow(
      '--cache-dir needs the SSD cache --max-cache-disk 0 turns off',
    );
  });
});
