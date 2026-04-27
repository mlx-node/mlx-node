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
 * TypeScript smoke parity test for the block-paged KV cache adapter.
 *
 * Pairs with the Rust integration tests in
 * `crates/mlx-core/tests/qwen3_paged_vs_flat_parity.rs` to gate the
 * `use_block_paged_cache` default flip from `None` to `Some(true)`. The
 * Rust tests cover real-weights Qwen3 + LFM2 across single-turn and
 * two-turn dialogs; this TS smoke test pins the same parity property
 * end-to-end through the public NAPI surface that production traffic
 * actually hits, catching any TS-only divergence (config plumbing,
 * tokenizer wiring) the Rust tests would miss.
 *
 * Uses `QWEN3_MODEL_PATH` env var with a `.cache/models/qwen3-0.6b-mlx-bf16`
 * default fallback. Skipped via `it.runIf` when no model directory
 * exists, so the test never blocks CI on machines without weights.
 */

/**
 * Resolve the model path used for the parity gate.
 *
 * The test only runs when the user explicitly opts in by setting
 * `QWEN3_PAGED_PARITY_MODEL_PATH` (analogous to the Rust integration
 * tests' `MLX_TEST_MODEL_PATH` `#[ignore]` env var) — without the var the
 * test is skipped via `it.runIf` so it never blocks default CI on
 * machines that happen to have a `.cache/models/qwen3-0.6b-mlx-bf16`
 * directory left over from other tests. The default-flip workflow in
 * CLAUDE.md > "Parity gate" describes how to enable the env var when
 * evaluating the gate.
 */
function findModelPath(): string | null {
  const env = process.env.QWEN3_PAGED_PARITY_MODEL_PATH;
  if (env && existsSync(env)) return env;
  return null;
}

/**
 * Copy the source model directory into a fresh tempdir. If
 * `useBlockPagedCache` is true, patch the cloned `config.json` to enable
 * the block-paged adapter (and pin a small adapter pool so the test
 * doesn't take excessive memory).
 */
function cloneModelDir(src: string, suffix: string, useBlockPagedCache: boolean): string {
  const dst = mkdtempSync(join(tmpdir(), `mlx-paged-parity-${suffix}-`));
  const entries = readdirSync(src);
  for (const entry of entries) {
    const from = join(src, entry);
    const to = join(dst, entry);
    if (statSync(from).isFile()) {
      copyFileSync(from, to);
    }
    // Top-level only — Qwen3 model dirs have no nested subdirs that
    // matter for inference. (`.git`, `tokenizer/` etc would be stripped
    // here too if present.)
  }

  if (useBlockPagedCache) {
    const cfgPath = join(dst, 'config.json');
    const raw = readFileSync(cfgPath, 'utf-8');
    const cfg = JSON.parse(raw) as Record<string, unknown>;
    cfg.use_block_paged_cache = true;
    cfg.paged_cache_memory_mb = 256;
    cfg.paged_block_size = 16;
    writeFileSync(cfgPath, JSON.stringify(cfg, null, 2));
  }
  return dst;
}

describe('Qwen3Model — paged-vs-flat parity', () => {
  const modelPath = findModelPath();
  const modelExists = modelPath !== null;
  let flatDir: string | null = null;
  let pagedDir: string | null = null;
  let flatModel: Qwen3Model | null = null;
  let pagedModel: Qwen3Model | null = null;

  beforeAll(async () => {
    if (!modelExists || !modelPath) {
      console.log('  Skipping (no Qwen3 model found at QWEN3_MODEL_PATH or .cache fallback)');
      return;
    }
    flatDir = cloneModelDir(modelPath, 'flat', false);
    pagedDir = cloneModelDir(modelPath, 'paged', true);
    flatModel = await Qwen3Model.load(flatDir);
    pagedModel = await Qwen3Model.load(pagedDir);
  }, 120_000);

  afterAll(() => {
    if (flatDir) rmSync(flatDir, { recursive: true, force: true });
    if (pagedDir) rmSync(pagedDir, { recursive: true, force: true });
  });

  it.runIf(modelExists)(
    'produces byte-equal greedy output between flat and paged',
    async () => {
      if (!flatModel || !pagedModel) {
        throw new Error('models failed to load — beforeAll did not complete');
      }

      const prompts: string[] = [
        'Say hi in one short word.',
        'What is 2 + 3? Answer with just the number.',
        'Name a primary color.',
        'Complete: the sky is',
      ];

      for (const [idx, prompt] of prompts.entries()) {
        // Each prompt is a fresh user message that does not share a
        // prefix with any previous turn, so `verify_cache_prefix`
        // returns 0 inside `chat_sync_core` and the implicit cache
        // reset fires before each prefill. No explicit `resetCaches`
        // is needed (and calling it from a JS-async test risks the same
        // "blocking_recv inside runtime" trap the Rust integration
        // tests hit when invoked from a tokio task).

        const messages: ChatMessage[] = [{ role: 'user', content: prompt }];

        // `chatSessionStart` dispatches through `chat_sync_core`, which
        // routes to `chat_sync_core_paged` (and thus
        // `forward_paged_adapter`) when `use_block_paged_cache` is on.
        // We deliberately use this entry point (and NOT `generate()`):
        // `generate_sync` always uses fresh flat caches and never
        // consults `paged_adapter`, so it would mask any paged-vs-flat
        // divergence and falsely report parity.
        const flatResult = await flatModel.chatSessionStart(messages, {
          maxNewTokens: 32,
          temperature: 0,
          repetitionPenalty: 1.0,
          thinkingTokenBudget: 32,
          reuseCache: true,
        });
        const pagedResult = await pagedModel.chatSessionStart(messages, {
          maxNewTokens: 32,
          temperature: 0,
          repetitionPenalty: 1.0,
          thinkingTokenBudget: 32,
          reuseCache: true,
        });

        // ChatResult does not surface raw token IDs — `text` and
        // `numTokens` are the byte-level comparison surface. At
        // temperature=0 with greedy sampling, byte-identical text +
        // matching numTokens is the strongest equivalence we can pin
        // through the public NAPI surface.
        if (flatResult.text !== pagedResult.text) {
          // Find first byte-level divergence for a compact repro hint.
          const flatBytes = Buffer.from(flatResult.text, 'utf-8');
          const pagedBytes = Buffer.from(pagedResult.text, 'utf-8');
          const minLen = Math.min(flatBytes.length, pagedBytes.length);
          let firstDiffByte = -1;
          for (let i = 0; i < minLen; i++) {
            if (flatBytes[i] !== pagedBytes[i]) {
              firstDiffByte = i;
              break;
            }
          }
          if (firstDiffByte === -1) firstDiffByte = minLen;

          throw new Error(
            `Text mismatch on prompt #${idx} (${JSON.stringify(prompt)}). first_diff_byte=${firstDiffByte}\n` +
              `  flat  (${flatResult.numTokens} tokens) text=${JSON.stringify(flatResult.text)}\n` +
              `  paged (${pagedResult.numTokens} tokens) text=${JSON.stringify(pagedResult.text)}`,
          );
        }

        expect(flatResult.text).toBe(pagedResult.text);
        expect(flatResult.numTokens).toBe(pagedResult.numTokens);

        console.log(`  prompt #${idx}: ${flatResult.numTokens} tokens matched`);
      }
    },
    600_000,
  );
});
