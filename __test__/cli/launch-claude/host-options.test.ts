import { describe, expect, it } from 'vite-plus/test';

import { launchClaudeHostOptions } from '../../../packages/cli/src/commands/launch-claude/index.js';

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
});
