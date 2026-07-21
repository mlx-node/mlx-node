/**
 * `mlx agent` argv scan — the mlx-owned `--no-persist-cache` flag.
 *
 * The flag disables the qwen3 cold tier the agent otherwise enables by
 * default. Like the other mlx-owned flags it is lifted out of the argv and
 * never forwarded to pi, and it must not be hijacked when it lands in a pi
 * value-consumer's value slot.
 */

import { describe, expect, it } from 'vite-plus/test';

import { scanAgentArgs } from '../src/commands/agent/index.js';

describe('scanAgentArgs --no-persist-cache', () => {
  it('defaults persistPagedCache to true when the flag is absent', () => {
    const scan = scanAgentArgs(['-c']);
    expect(scan.persistPagedCache).toBe(true);
    expect(scan.passthrough).toEqual(['-c']);
  });

  it('is mlx-owned: flips the flag false and is not forwarded to pi', () => {
    const scan = scanAgentArgs(['--no-persist-cache', '-c']);
    expect(scan.persistPagedCache).toBe(false);
    expect(scan.passthrough).toEqual(['-c']);
  });

  it('does not hijack the flag when it sits in a pi value-consumer slot', () => {
    const scan = scanAgentArgs(['--system-prompt', '--no-persist-cache']);
    // "--no-persist-cache" is the system-prompt VALUE here, so it stays enabled
    // and passes through verbatim.
    expect(scan.persistPagedCache).toBe(true);
    expect(scan.passthrough).toEqual(['--system-prompt', '--no-persist-cache']);
  });
});
