import { __parseEnvPositiveInt, __parseEnvSeconds, createServer } from '@mlx-node/server';
import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

/**
 * `createServer` config validation.
 *
 * These tests cover caller-supplied positive-integer knobs that can take
 * the server offline if passed zero / negative / non-integer — specifically
 * `maxQueueDepthPerModel` (a `0` makes `queuedCount >= limit` true for
 * every request, instantly returning HTTP 429) and `responseRetentionSec`
 * (a `0` stamps `expires_at = now` on every row, which the next cleanup
 * sweep promptly deletes). Validation happens synchronously in
 * `createServer` so a bad config fails fast with a descriptive error
 * instead of silently falling through to the env / default path.
 */
describe('createServer config validation', () => {
  // Each `createServer` call we are NOT awaiting for validation errors
  // would otherwise try to bind a port. The validation we are testing
  // throws synchronously before the `httpCreateServer` / `listen` call,
  // so the returned promise rejects before any socket is opened. Kept
  // here as a safety net in case a test falls through.
  let openedServers: Array<{ close: () => Promise<void> }> = [];

  beforeEach(() => {
    openedServers = [];
    // Make sure env overrides from other test files in the same suite
    // cannot mask validation errors (e.g. if `MLX_MAX_QUEUE_DEPTH_PER_MODEL`
    // were left set, a `??` fallback would hide a silent-coerce regression).
    delete process.env.MLX_MAX_QUEUE_DEPTH_PER_MODEL;
    delete process.env.MLX_RESPONSE_RETENTION_SECONDS;
  });

  afterEach(async () => {
    for (const srv of openedServers) {
      await srv.close().catch(() => {});
    }
  });

  describe('maxQueueDepthPerModel', () => {
    it('rejects 0 with a descriptive error', async () => {
      await expect(createServer({ maxQueueDepthPerModel: 0, disableStore: true, port: 0 })).rejects.toThrow(
        /maxQueueDepthPerModel must be a positive integer/,
      );
    });

    it('rejects negative values', async () => {
      await expect(createServer({ maxQueueDepthPerModel: -5, disableStore: true, port: 0 })).rejects.toThrow(
        /maxQueueDepthPerModel must be a positive integer/,
      );
    });

    it('rejects non-integer values', async () => {
      await expect(createServer({ maxQueueDepthPerModel: 1.5, disableStore: true, port: 0 })).rejects.toThrow(
        /maxQueueDepthPerModel must be a positive integer/,
      );
    });

    it('rejects NaN', async () => {
      await expect(createServer({ maxQueueDepthPerModel: Number.NaN, disableStore: true, port: 0 })).rejects.toThrow(
        /maxQueueDepthPerModel must be a positive integer/,
      );
    });

    it('rejects Infinity', async () => {
      await expect(
        createServer({ maxQueueDepthPerModel: Number.POSITIVE_INFINITY, disableStore: true, port: 0 }),
      ).rejects.toThrow(/maxQueueDepthPerModel must be a positive integer/);
    });

    it('accepts the minimum valid value of 1', async () => {
      const srv = await createServer({ maxQueueDepthPerModel: 1, disableStore: true, port: 0 });
      openedServers.push(srv);
      expect(srv.registry).toBeDefined();
    });

    it('accepts undefined (no opt-in)', async () => {
      const srv = await createServer({ maxQueueDepthPerModel: undefined, disableStore: true, port: 0 });
      openedServers.push(srv);
      expect(srv.registry).toBeDefined();
    });

    it('falls through to env when config value is absent', async () => {
      process.env.MLX_MAX_QUEUE_DEPTH_PER_MODEL = '4';
      try {
        const srv = await createServer({ disableStore: true, port: 0 });
        openedServers.push(srv);
        expect(srv.registry).toBeDefined();
      } finally {
        delete process.env.MLX_MAX_QUEUE_DEPTH_PER_MODEL;
      }
    });

    it('rejects bogus config even if env has a valid value', async () => {
      // Fail-fast: a caller who explicitly passes a bad value should see
      // the error instead of silently using the env fallback.
      process.env.MLX_MAX_QUEUE_DEPTH_PER_MODEL = '4';
      try {
        await expect(createServer({ maxQueueDepthPerModel: 0, disableStore: true, port: 0 })).rejects.toThrow(
          /maxQueueDepthPerModel must be a positive integer/,
        );
      } finally {
        delete process.env.MLX_MAX_QUEUE_DEPTH_PER_MODEL;
      }
    });
  });

  describe('responseRetentionSec', () => {
    it('rejects 0 with a descriptive error', async () => {
      await expect(createServer({ responseRetentionSec: 0, disableStore: true, port: 0 })).rejects.toThrow(
        /responseRetentionSec must be a positive integer/,
      );
    });

    it('rejects negative values', async () => {
      await expect(createServer({ responseRetentionSec: -10, disableStore: true, port: 0 })).rejects.toThrow(
        /responseRetentionSec must be a positive integer/,
      );
    });

    it('rejects non-integer values', async () => {
      await expect(createServer({ responseRetentionSec: 3.14, disableStore: true, port: 0 })).rejects.toThrow(
        /responseRetentionSec must be a positive integer/,
      );
    });

    it('rejects NaN', async () => {
      await expect(createServer({ responseRetentionSec: Number.NaN, disableStore: true, port: 0 })).rejects.toThrow(
        /responseRetentionSec must be a positive integer/,
      );
    });

    it('accepts the minimum valid value of 1', async () => {
      const srv = await createServer({ responseRetentionSec: 1, disableStore: true, port: 0 });
      openedServers.push(srv);
      expect(srv.registry).toBeDefined();
    });

    it('accepts undefined (uses default)', async () => {
      const srv = await createServer({ responseRetentionSec: undefined, disableStore: true, port: 0 });
      openedServers.push(srv);
      expect(srv.registry).toBeDefined();
    });
  });
});

/**
 * Env-var parser validation.
 *
 * `parseEnvSeconds` and `parseEnvPositiveInt` back `MLX_RESPONSE_RETENTION_SECONDS`
 * and `MLX_MAX_QUEUE_DEPTH_PER_MODEL` respectively. A prior implementation
 * applied `Math.floor` to the parsed number, so a typo like
 * `MLX_RESPONSE_RETENTION_SECONDS="1.5"` (intended as `"15"`) was silently
 * truncated to 1 — persisted response rows then expired almost immediately
 * and `previous_response_id` continuity broke across turns. Same class of
 * bug on queue depth: `"1.5"` clamped to 1 instead of being rejected.
 *
 * Fix: reject non-integer positive values by returning `undefined` so the
 * caller falls through to its documented default (7 days retention /
 * unbounded queue). We deliberately do NOT throw on bad env — env vars
 * are routinely set by orchestrators and CI templates, and a
 * startup-crash-on-typo is harsher than falling through to a safe
 * default.
 */
describe('env var parsing', () => {
  const RETENTION_VAR = 'MLX_RESPONSE_RETENTION_SECONDS';
  const QUEUE_VAR = 'MLX_MAX_QUEUE_DEPTH_PER_MODEL';

  // Snapshot and restore originals so these tests do not leak env state
  // to sibling test files (especially important for queue-depth, which
  // affects registry construction elsewhere).
  let origRetention: string | undefined;
  let origQueue: string | undefined;

  beforeEach(() => {
    origRetention = process.env[RETENTION_VAR];
    origQueue = process.env[QUEUE_VAR];
    delete process.env[RETENTION_VAR];
    delete process.env[QUEUE_VAR];
  });

  afterEach(() => {
    if (origRetention === undefined) delete process.env[RETENTION_VAR];
    else process.env[RETENTION_VAR] = origRetention;
    if (origQueue === undefined) delete process.env[QUEUE_VAR];
    else process.env[QUEUE_VAR] = origQueue;
  });

  describe('parseEnvSeconds (MLX_RESPONSE_RETENTION_SECONDS)', () => {
    it('rejects fractional MLX_RESPONSE_RETENTION_SECONDS and falls back to default', () => {
      // A `"1.5"` typo intended as `"15"` must NOT silently truncate to 1s.
      // Returning undefined lets the caller apply its 7-day default.
      process.env[RETENTION_VAR] = '1.5';
      expect(__parseEnvSeconds(RETENTION_VAR)).toBeUndefined();
    });

    it('accepts integer MLX_RESPONSE_RETENTION_SECONDS', () => {
      process.env[RETENTION_VAR] = '15';
      expect(__parseEnvSeconds(RETENTION_VAR)).toBe(15);
    });

    it('accepts a large valid integer without coercion drift', () => {
      // 7 days in seconds — the documented default. Sanity check that
      // the common production value round-trips cleanly.
      process.env[RETENTION_VAR] = '604800';
      expect(__parseEnvSeconds(RETENTION_VAR)).toBe(604800);
    });

    it('rejects MLX_RESPONSE_RETENTION_SECONDS=0', () => {
      process.env[RETENTION_VAR] = '0';
      expect(__parseEnvSeconds(RETENTION_VAR)).toBeUndefined();
    });

    it('rejects negative MLX_RESPONSE_RETENTION_SECONDS', () => {
      process.env[RETENTION_VAR] = '-5';
      expect(__parseEnvSeconds(RETENTION_VAR)).toBeUndefined();
    });

    it('rejects MLX_RESPONSE_RETENTION_SECONDS=NaN', () => {
      process.env[RETENTION_VAR] = 'NaN';
      expect(__parseEnvSeconds(RETENTION_VAR)).toBeUndefined();
    });

    it('rejects MLX_RESPONSE_RETENTION_SECONDS=Infinity', () => {
      process.env[RETENTION_VAR] = 'Infinity';
      expect(__parseEnvSeconds(RETENTION_VAR)).toBeUndefined();
    });

    it('rejects non-numeric MLX_RESPONSE_RETENTION_SECONDS', () => {
      process.env[RETENTION_VAR] = 'abc';
      expect(__parseEnvSeconds(RETENTION_VAR)).toBeUndefined();
    });

    it('returns undefined when MLX_RESPONSE_RETENTION_SECONDS is unset', () => {
      // No env assignment — parseEnvSeconds should report absent.
      expect(__parseEnvSeconds(RETENTION_VAR)).toBeUndefined();
    });

    it('returns undefined for empty MLX_RESPONSE_RETENTION_SECONDS', () => {
      process.env[RETENTION_VAR] = '';
      expect(__parseEnvSeconds(RETENTION_VAR)).toBeUndefined();
    });
  });

  describe('parseEnvPositiveInt (MLX_MAX_QUEUE_DEPTH_PER_MODEL)', () => {
    it('rejects fractional MLX_MAX_QUEUE_DEPTH_PER_MODEL and leaves queue unbounded', () => {
      // `"1.5"` must NOT silently coerce to 1 — that would clamp the
      // queue to depth=1 and return 429 for every second request.
      process.env[QUEUE_VAR] = '1.5';
      expect(__parseEnvPositiveInt(QUEUE_VAR)).toBeUndefined();
    });

    it('accepts integer MLX_MAX_QUEUE_DEPTH_PER_MODEL', () => {
      process.env[QUEUE_VAR] = '8';
      expect(__parseEnvPositiveInt(QUEUE_VAR)).toBe(8);
    });

    it('rejects MLX_MAX_QUEUE_DEPTH_PER_MODEL=0', () => {
      process.env[QUEUE_VAR] = '0';
      expect(__parseEnvPositiveInt(QUEUE_VAR)).toBeUndefined();
    });

    it('rejects negative MLX_MAX_QUEUE_DEPTH_PER_MODEL', () => {
      process.env[QUEUE_VAR] = '-2';
      expect(__parseEnvPositiveInt(QUEUE_VAR)).toBeUndefined();
    });

    it('rejects MLX_MAX_QUEUE_DEPTH_PER_MODEL=NaN', () => {
      process.env[QUEUE_VAR] = 'NaN';
      expect(__parseEnvPositiveInt(QUEUE_VAR)).toBeUndefined();
    });

    it('rejects MLX_MAX_QUEUE_DEPTH_PER_MODEL=Infinity', () => {
      process.env[QUEUE_VAR] = 'Infinity';
      expect(__parseEnvPositiveInt(QUEUE_VAR)).toBeUndefined();
    });

    it('rejects non-numeric MLX_MAX_QUEUE_DEPTH_PER_MODEL', () => {
      process.env[QUEUE_VAR] = 'lots';
      expect(__parseEnvPositiveInt(QUEUE_VAR)).toBeUndefined();
    });

    it('returns undefined when MLX_MAX_QUEUE_DEPTH_PER_MODEL is unset', () => {
      expect(__parseEnvPositiveInt(QUEUE_VAR)).toBeUndefined();
    });

    it('returns undefined for empty MLX_MAX_QUEUE_DEPTH_PER_MODEL', () => {
      process.env[QUEUE_VAR] = '';
      expect(__parseEnvPositiveInt(QUEUE_VAR)).toBeUndefined();
    });
  });
});
