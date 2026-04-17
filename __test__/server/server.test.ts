import { createServer } from '@mlx-node/server';
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
