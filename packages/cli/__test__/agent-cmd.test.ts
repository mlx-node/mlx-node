/**
 * `mlx agent` argv scan — the mlx-owned `--no-persist-cache` flag.
 *
 * The flag disables the SSD cold tier the agent otherwise enables by default
 * for EVERY allowlisted family. Like the other mlx-owned flags it is lifted out
 * of the argv and never forwarded to pi, and it must not be hijacked when it
 * lands in a pi value-consumer's value slot.
 */

import { coldTierRestoreFamilyList } from '@mlx-node/agent/catalog';
import { describe, expect, it } from 'vite-plus/test';

import { agentPreambleText, scanAgentArgs } from '../src/commands/agent/index.js';

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

/**
 * The shipped help said the flag disabled the tier "for qwen3 dense paged
 * prefix blocks (other families unaffected)". Running it against qwen3_5_moe
 * disproved that: the flag is ONE process-wide boolean that
 * `MlxModelHost.runWithResident` hands to every load whose family is on the
 * allowlist. "Other families unaffected" was doubly wrong — non-allowlisted
 * families are unaffected because they can NEVER persist, not because the flag
 * spares them.
 */
describe('mlx agent help — --no-persist-cache', () => {
  it('names every family the flag actually disables, from the allowlist itself', () => {
    const help = agentPreambleText();
    const families = coldTierRestoreFamilyList();
    expect(families.length).toBeGreaterThan(1);
    for (const family of families) {
      expect(help, `help text must name ${family}`).toContain(family);
    }
  });

  it('no longer promises that other cold-tier families keep persisting', () => {
    expect(agentPreambleText()).not.toContain('other families unaffected');
  });

  it('does not single out qwen3 dense as the only affected family', () => {
    expect(agentPreambleText()).not.toContain('qwen3\n                            dense');
  });
});
