import {
  copyFileSync,
  existsSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { Qwen3Model, type ChatMessage } from '@mlx-node/core';
import { describe, it, expect, beforeAll, afterAll } from 'vite-plus/test';

/**
 * Regression: the block-paged Qwen3 forward used to fail with
 *
 *   chat_stream_sync_core_paged: zero-delta prompt (every token cached) is
 *   not yet supported on the block-paged path; flat path required for
 *   this corner case
 *
 * whenever the live cache (or a previous turn's residue) covered every
 * token of the new prompt — the production trigger was a client retrying
 * a prompt verbatim after an earlier 600 s HTTP timeout. The fix is the
 * vLLM-style `max_cache_hit_tokens = prompt.len() - 1` cap, applied at
 * every paged dispatch site, so the cache lookup always leaves at least
 * one suffix token to prefill.
 *
 * This test reproduces the production scenario through the public NAPI
 * surface: send the same prompt twice on the same paged model and assert
 * the second call does not surface the zero-delta error. Gated on
 * `QWEN3_PAGED_PARITY_MODEL_PATH` (same gate as
 * `qwen3-paged-parity.test.ts`) so it never blocks default CI.
 */
function findModelPath(): string | null {
  const env = process.env.QWEN3_PAGED_PARITY_MODEL_PATH;
  if (env && existsSync(env)) return env;
  return null;
}

function clonePagedModelDir(src: string): string {
  const dst = mkdtempSync(join(tmpdir(), `mlx-paged-zero-delta-`));
  for (const entry of readdirSync(src)) {
    const from = join(src, entry);
    const to = join(dst, entry);
    if (statSync(from).isFile()) copyFileSync(from, to);
  }
  const cfgPath = join(dst, 'config.json');
  const raw = readFileSync(cfgPath, 'utf-8');
  const cfg = JSON.parse(raw) as Record<string, unknown>;
  cfg.use_block_paged_cache = true;
  cfg.paged_cache_memory_mb = 256;
  cfg.paged_block_size = 16;
  writeFileSync(cfgPath, JSON.stringify(cfg, null, 2));
  return dst;
}

describe('Qwen3Model — block-paged zero-delta regression', () => {
  const modelPath = findModelPath();
  const modelExists = modelPath !== null;
  let pagedDir: string | null = null;
  let pagedModel: Qwen3Model | null = null;

  beforeAll(async () => {
    if (!modelExists || !modelPath) return;
    pagedDir = clonePagedModelDir(modelPath);
    pagedModel = await Qwen3Model.load(pagedDir);
  }, 120_000);

  afterAll(() => {
    if (pagedDir) rmSync(pagedDir, { recursive: true, force: true });
  });

  it.runIf(modelExists)(
    'accepts a byte-identical retry of a prompt the live cache already covers',
    async () => {
      if (!pagedModel) throw new Error('paged model failed to load');

      const messages: ChatMessage[] = [{ role: 'user', content: 'Say hi.' }];
      const opts = {
        maxNewTokens: 8,
        temperature: 0,
        repetitionPenalty: 1.0,
        thinkingTokenBudget: 8,
        reuseCache: true,
      };

      // Turn 1 — populates the paged adapter's live cache with the full
      // (prompt + decoded tokens) trace.
      const first = await pagedModel.chatSessionStart(messages, opts);
      expect(first.text.length).toBeGreaterThan(0);

      // Turn 2 — same prompt verbatim. Before the fix, the cache lookup
      // returned `cached_prefix_len == prompt.len()` and the inner forward
      // bailed with a "zero-delta prompt … not yet supported" error. With
      // the vLLM cap in place, the lookup is bounded at `prompt.len() - 1`
      // so at least one suffix token survives for the prefill chunk.
      const second = await pagedModel.chatSessionStart(messages, opts);
      expect(second.text.length).toBeGreaterThan(0);
    },
    300_000,
  );
});
