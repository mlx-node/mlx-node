import { afterEach, describe, expect, it } from 'vite-plus/test';

import {
  applyEnginePolicy,
  engineEnvFor,
  ENGINE_POLICY_ENV_VARS,
  LAUNCHER_ENGINE_POLICY,
} from '../../../packages/server/src/host/env-policy.js';

/**
 * `MLX_PAGED_PREFILL_CHUNK_SIZE` is launcher POLICY, not an engine default:
 * unset, the Rust side reads it as `0` (no chunking), and 2048 is the value
 * `mlx launch claude` has historically applied to bound the cold-prefill
 * memory peak. The distinction only stays real if nothing writes the var
 * unless a caller explicitly asks — hence the assertions below on the
 * NEGATIVE case as much as the positive one.
 *
 * The var latches native-side through a `OnceLock` on first read, so these
 * tests restore whatever the surrounding process had; a leaked value would
 * silently retune every later test in the worker.
 */

const CHUNK = 'MLX_PAGED_PREFILL_CHUNK_SIZE';

const original = process.env[CHUNK];
afterEach(() => {
  if (original === undefined) delete process.env[CHUNK];
  else process.env[CHUNK] = original;
});

describe('engineEnvFor', () => {
  it('renders the launcher policy as the child env map', () => {
    expect(engineEnvFor(LAUNCHER_ENGINE_POLICY)).toEqual({ [CHUNK]: '2048' });
  });

  it('omits knobs the policy leaves unset', () => {
    expect(engineEnvFor({})).toEqual({});
  });

  it('renders an explicit 0 rather than dropping it', () => {
    // 0 is the engine's "no chunking" value and a legitimate policy choice;
    // a falsy-check bug would silently turn it into "unset", which happens
    // to mean the same thing today but would stop meaning it the moment the
    // Rust default changes.
    expect(engineEnvFor({ pagedPrefillChunkSize: 0 })).toEqual({ [CHUNK]: '0' });
  });

  it('only ever names vars from the declared inventory', () => {
    expect(Object.keys(engineEnvFor(LAUNCHER_ENGINE_POLICY))).toEqual(
      expect.arrayContaining([...ENGINE_POLICY_ENV_VARS]),
    );
    for (const name of Object.keys(engineEnvFor(LAUNCHER_ENGINE_POLICY))) {
      expect(ENGINE_POLICY_ENV_VARS).toContain(name);
    }
  });

  it('returns a fresh object each call so a caller cannot mutate the policy', () => {
    const first = engineEnvFor(LAUNCHER_ENGINE_POLICY);
    first[CHUNK] = 'tampered';
    expect(engineEnvFor(LAUNCHER_ENGINE_POLICY)[CHUNK]).toBe('2048');
  });
});

describe('applyEnginePolicy', () => {
  it('writes the policy value into an env that does not have it', () => {
    const env: NodeJS.ProcessEnv = {};
    expect(applyEnginePolicy(LAUNCHER_ENGINE_POLICY, env)).toEqual([CHUNK]);
    expect(env[CHUNK]).toBe('2048');
  });

  it('leaves a value the user already set alone', () => {
    const env: NodeJS.ProcessEnv = { [CHUNK]: '512' };
    expect(applyEnginePolicy(LAUNCHER_ENGINE_POLICY, env)).toEqual([]);
    expect(env[CHUNK]).toBe('512');
  });

  it('treats an explicit empty string as "the user has an opinion"', () => {
    // Matches the historical `== null` guard. An empty value is how a shell
    // script says "unset this for the child"; overwriting it would resurrect
    // a default the operator deliberately cleared.
    const env: NodeJS.ProcessEnv = { [CHUNK]: '' };
    expect(applyEnginePolicy(LAUNCHER_ENGINE_POLICY, env)).toEqual([]);
    expect(env[CHUNK]).toBe('');
  });

  it('writes nothing at all for an empty policy', () => {
    const env: NodeJS.ProcessEnv = {};
    expect(applyEnginePolicy({}, env)).toEqual([]);
    expect(Object.keys(env)).toEqual([]);
  });

  it('defaults to process.env', () => {
    delete process.env[CHUNK];
    expect(applyEnginePolicy(LAUNCHER_ENGINE_POLICY)).toEqual([CHUNK]);
    expect(process.env[CHUNK]).toBe('2048');
  });
});
