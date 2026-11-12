import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { LoadableModel } from '@mlx-node/lm';
import { createInferenceHost } from '@mlx-node/server/host';
import { describe, expect, it, vi } from 'vite-plus/test';

import {
  claudeChildEnv,
  launchClaudeHostOptions,
  loadServedWindow,
} from '../../../packages/cli/src/commands/launch-claude/index.js';

/**
 * The one launcher behaviour the shared-host extraction could silently drop.
 *
 * `mlx launch claude` has always written `MLX_PAGED_PREFILL_CHUNK_SIZE=2048`
 * before loading anything, to bound the cold-prefill memory peak. That is
 * LAUNCHER policy, not an engine default — unset, the Rust side reads the var
 * as `0` (no chunking) — and the var latches through a `OnceLock` on first
 * read, so a launcher that forgot to declare it would degrade silently rather
 * than fail. Pinning it here needs no `claude` binary and no model.
 */
describe('launchClaudeHostOptions', () => {
  it('declares the 2048-token paged-prefill chunk policy', () => {
    expect(launchClaudeHostOptions({}).enginePolicy).toEqual({ pagedPrefillChunkSize: 2048 });
  });

  it('declares the policy regardless of which flags were passed', () => {
    expect(launchClaudeHostOptions({ port: 8080, model: 'qwen' }).enginePolicy).toEqual({
      pagedPrefillChunkSize: 2048,
    });
  });

  it('forwards each flag onto the matching host option', () => {
    expect(
      launchClaudeHostOptions({ port: 8080, host: '0.0.0.0', modelsDir: '/m', model: 'q', logDir: '/l' }),
    ).toMatchObject({
      port: 8080,
      host: '0.0.0.0',
      modelsDir: '/m',
      model: 'q',
      logDir: '/l',
    });
  });

  it('leaves unset flags undefined so the host applies its own defaults', () => {
    const opts = launchClaudeHostOptions({});
    expect(opts.port).toBeUndefined();
    expect(opts.host).toBeUndefined();
    expect(opts.modelsDir).toBeUndefined();
    expect(opts.model).toBeUndefined();
    expect(opts.logDir).toBeUndefined();
  });

  it('forwards the per-launch token onto the host', () => {
    // Dropping it here is invisible: `claude` still gets its token and still
    // works, because a host with no `authToken` accepts everyone. The server is
    // simply open to every other local process and to any web page the user
    // visits.
    expect(launchClaudeHostOptions({ authToken: 'per-launch-secret' }).authToken).toBe('per-launch-secret');
  });
});

/**
 * The environment the spawned `claude` runs under.
 *
 * MEASURED against Claude Code 2.1.220 pointed at a local base URL, not
 * assumed — `ANTHROPIC_AUTH_TOKEN` alone produces
 * `authorization: Bearer <token>`, and adding `ANTHROPIC_API_KEY` to the
 * environment makes it send BOTH that bearer AND `x-api-key: <the key>`. The
 * gate reads `x-api-key` first, so an inherited key beats our bearer and 401s
 * the entire session.
 */
describe('claudeChildEnv', () => {
  const opts = { baseUrl: 'http://127.0.0.1:51234', model: 'qwen3.5-9b', authToken: 'per-launch-secret' };

  it('hands `claude` the token the host is actually enforcing', () => {
    // Was the constant `mlx-node-local`, against a host with no token at all —
    // decorative, since nothing checked it.
    expect(claudeChildEnv({}, opts).ANTHROPIC_AUTH_TOKEN).toBe('per-launch-secret');
  });

  it('drops an inherited ANTHROPIC_API_KEY, which would otherwise 401 every turn', () => {
    const env = claudeChildEnv({ ANTHROPIC_API_KEY: 'sk-ant-the-users-own-key' }, opts);
    expect('ANTHROPIC_API_KEY' in env).toBe(false);
    // Not merely emptied: `x-api-key: ''` still loses to the bearer under a
    // gate that checks `x-api-key` first.
    expect(env.ANTHROPIC_API_KEY).toBeUndefined();
  });

  it("does not leak the user's Anthropic key into a request log", () => {
    // `--verbose` writes every request header to `requests.ndjson`. A key that
    // never reaches the child is a key that cannot be logged in the first
    // place; the logger's own redaction is the second line, not the first.
    expect(JSON.stringify(claudeChildEnv({ ANTHROPIC_API_KEY: 'sk-ant-secret' }, opts))).not.toContain('sk-ant-secret');
  });

  // Splash `_reach_loopback_directly`. Mutation: skip `withLoopbackNoProxy` → red.
  it('exempts loopback from an inherited proxy so the bearer token never reaches it', () => {
    const env = claudeChildEnv({ HTTP_PROXY: 'http://proxy.example:3128', NO_PROXY: 'corp.example' }, opts);
    expect(env.NO_PROXY).toBe('corp.example,127.0.0.1,localhost,::1');
    expect(env.no_proxy).toBe('corp.example,127.0.0.1,localhost,::1');
    expect('NO_PROXY' in claudeChildEnv({}, opts)).toBe(false);
  });

  // Mutation: drop the two window vars → red.
  it('hands Claude Code the served window when one is known', () => {
    const env = claudeChildEnv({ CLAUDE_CODE_MAX_CONTEXT_TOKENS: '1000000' }, { ...opts, contextTokens: 139257 });
    expect(env.CLAUDE_CODE_MAX_CONTEXT_TOKENS).toBe('139257');
    expect(env.CLAUDE_CODE_AUTO_COMPACT_WINDOW).toBe('139257');
    const unknown = claudeChildEnv({}, opts);
    expect('CLAUDE_CODE_MAX_CONTEXT_TOKENS' in unknown).toBe(false);
    expect('CLAUDE_CODE_AUTO_COMPACT_WINDOW' in unknown).toBe(false);
  });

  it('points the child at this host and passes everything else through', () => {
    const env = claudeChildEnv({ PATH: '/usr/bin', TERM: 'xterm' }, opts);
    expect(env.ANTHROPIC_BASE_URL).toBe(opts.baseUrl);
    expect(env.ANTHROPIC_MODEL).toBe(opts.model);
    expect(env.PATH).toBe('/usr/bin');
    expect(env.TERM).toBe('xterm');
  });

  it('leaves the haiku overrides unset so the swap controller aliases them', () => {
    const env = claudeChildEnv({}, opts);
    expect(env.ANTHROPIC_SMALL_FAST_MODEL).toBeUndefined();
    expect(env.ANTHROPIC_DEFAULT_HAIKU_MODEL).toBeUndefined();
  });

  it('does not mutate the parent environment', () => {
    const parent = { ANTHROPIC_API_KEY: 'sk-ant-the-users-own-key' };
    claudeChildEnv(parent, opts);
    expect(parent.ANTHROPIC_API_KEY).toBe('sk-ant-the-users-own-key');
  });
});

/**
 * The window handed to Claude Code is the one the host serves after it has
 * loaded the bound model, through the real host and registry.
 */
describe('loadServedWindow', () => {
  // Mutation: skip the load → red (undefined: nothing is resident yet).
  // Mutation: read the window before the load settles → red (undefined).
  it('loads the bound model and returns its served window, or undefined when the load fails', async () => {
    const modelsDir = await mkdtemp(join(tmpdir(), 'mlx-launch-claude-'));
    try {
      await mkdir(join(modelsDir, 'qwen'), { recursive: true });
      await writeFile(join(modelsDir, 'qwen', 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }));
      const model = {
        contextLimits: () => ({
          trainedWindowTokens: 262144,
          effectiveWindowTokens: 73696,
          pagedBlockCapacity: 0,
          pagedBlockSize: 16,
          speculativeWindowTokens: 139257,
        }),
        hasMtpWeights: () => true,
      };
      let fail = false;
      const log = vi.spyOn(console, 'log').mockImplementation(() => undefined);
      const error = vi.spyOn(console, 'error').mockImplementation(() => undefined);
      const host = await createInferenceHost({
        port: 0,
        modelsDir,
        disableStore: true,
        sweepOrphanTempRoots: false,
        pagedModelTypes: [],
        loadModel: async () => {
          await new Promise((resolve) => setTimeout(resolve, 10));
          if (fail) throw new Error('weights unreadable');
          return model as unknown as LoadableModel;
        },
      });
      try {
        expect(host.server.registry.servedContext(host.boundModel)).toBeUndefined();
        await expect(loadServedWindow(host)).resolves.toBe(139257);

        host.server.registry.unregister(host.boundModel);
        fail = true;
        await expect(loadServedWindow(host)).resolves.toBeUndefined();
        expect(error.mock.calls.flat().join('\n')).toContain('load of "qwen" failed: weights unreadable');
      } finally {
        await host.close();
        log.mockRestore();
        error.mockRestore();
      }
    } finally {
      await rm(modelsDir, { recursive: true, force: true });
    }
  });
});
