/// <reference types="node" />
/**
 * Runs one shard of the compiled `mlx-core --lib` test binary, so CI builds it once
 * (`rust-lib-build`) and runs it on N runners. Shards are disjoint and cover every
 * non-ignored test; the weights below only balance the shards, they never filter.
 *
 *   node .github/scripts/run-rust-lib-shard.ts <test-binary> <shard-1-based> <shard-count>
 *   node .github/scripts/run-rust-lib-shard.ts <test-binary> --verify <shard-count>
 */

import { execFileSync, spawn } from 'node:child_process';
import { accessSync, constants, statSync } from 'node:fs';
import { createInterface } from 'node:readline';

// Keep each argv well below macOS ARG_MAX (1 MiB, shared with the environment).
const MAX_CHUNK_ARGV_BYTES = 256 * 1024;

// Seconds on hosted macos-26 for the slowest tests.
const TEST_WEIGHTS = new Map<string, number>([
  ['models::qwen4_exp::tests::draft_attach::a_sampled_drafted_turn_keeps_the_target_distribution', 39.9],
  ['models::qwen4_exp::tests::draft_cycle::a_sampled_cycle_keeps_the_target_distribution', 26.7],
  ['models::qwen4_exp::tests::draft_cycle::a_sampled_cycle_of_four_rows_keeps_the_target_pair_distribution', 22.6],
  ['models::qwen4_exp::tests::draft_switch::a_store_that_moves_up_drafts_the_turn_after_its_fill', 21.6],
  ['models::qwen4_exp::tests::route_step::early_cut_is_exact', 20.0],
  ['engine::dspark_device_sampled_tests::device_sampled_acceptance_keeps_the_target_law', 19.2],
  ['models::qwen4_exp::tests::route_step::early_cut_with_groups_in_flight_is_exact', 18.5],
  ['engine::dspark_device_sampled_tests::host_acceptance_of_device_drafts_keeps_the_target_law', 16.2],
  ['models::deepseek_v41::store::tests::tiny_cache_bitwise_any_size', 15.2],
  [
    'models::qwen4_exp::tests::batched_prefill_matches_reference_across_chunks_sparse_boundaries_and_continuation',
    14.6,
  ],
  ['models::qwen4_exp::tests::promote::a_host_gate_store_moves_up_at_its_last_close_and_decodes_as_a_whole_load', 12.7],
  ['models::qwen4_exp::tests::prefill_store::a_fill_never_writes_under_a_pending_reader', 12.0],
  ['models::qwen4_exp::tests::draft_rows::every_row_choice_gives_the_plain_text', 11.2],
  ['models::deepseek_v41::store::tests::tiny_prefetch_bitwise', 10.8],
  ['models::qwen4_exp::tests::route_step::device_replay_fills_protect_the_layers_ids', 9.8],
  ['models::qwen4_exp::tests::lanes::moved::the_residency_is_read_at_every_step', 9.7],
  ['models::qwen4_exp::tests::prefill_store::residents_are_submitted_before_the_fill_finishes', 9.0],
  ['models::qwen4_exp::tests::batched_qsa_matches_per_token_flat_and_paged', 8.6],
  ['models::deepseek_v41::tests::tiny_logits_bf16_within_floor', 8.6],
  ['models::qwen4_exp::tests::kquant_nax_gathers_match_dequantized_experts_with_tail_tiles', 8.3],
  ['models::deepseek_v41::store::tests::tiny_store_matches_resident', 8.1],
  ['models::qwen4_exp::tests::prefill_store::an_int8_group_matches_windows_bit_for_bit', 7.9],
  ['models::qwen4_exp::tests::contiguous_prompt_windows_preserve_rows_frontiers_and_continuation', 7.9],
  ['models::qwen4_exp::tests::route_step::device_replay_is_exact', 7.6],
  ['models::qwen4_exp::tests::promote::a_refused_allocation_half_way_leaves_a_partial_store_for_the_next_turn', 6.9],
  ['models::qwen4_exp::tests::promote::a_state_written_on_halved_passes_does_not_serve_the_moved_store', 6.6],
  ['models::qwen4_exp::tests::row_batched_decode_kernels_match_single_row_calls_bit_for_bit', 6.4],
  ['models::qwen3_5::gated_delta::gdn_bf16_gates_tests::bf16_scalars_keep_the_chain_bits', 6.1],
  ['models::deepseek_v41::store::tests::tiny_cache_bank_switch_mid_session', 5.9],
  ['models::deepseek_v41::store::tests::tiny_cache_split_bitwise', 5.5],
  ['models::qwen4_exp::tests::route_step::tail_replay_is_exact_at_every_layer_kind', 5.5],
  ['models::qwen4_exp::tests::draft_rows::the_rows_repeat_and_each_drafted_turn_starts_its_own_record', 5.2],
  ['models::qwen4_exp::tests::draft_cycle::every_accept_length_gives_the_plain_text_and_state', 5.0],
  ['models::inkling2::scheduler::tests::ws4_multi_phase_recon_verify_synthesize_and_parent_tip_ssd_durability', 4.9],
  ['models::deepseek_v41::tests::invariants_chunk_sizes', 4.8],
  ['models::deepseek_v41::tests::invariants_cancel_restores', 4.8],
  ['models::deepseek_v41::store::tests::tiny_layer_skip_bitwise', 4.7],
  ['models::qwen4_exp::tests::draft_switch::a_peer_ends_the_drafted_cycles_at_a_cycle_boundary', 4.7],
  ['models::qwen4_exp::tests::draft_attach::a_proposal_error_fails_the_turn', 4.3],
  ['models::hy_v4_flash::ops::tests::hc_fused_suite_and_prefill_row2_parity_and_invariance', 4.2],
]);

// For tests not in TEST_WEIGHTS; the most specific prefix comes first.
const PREFIX_WEIGHTS: [string, number][] = [
  ['models::qwen4_exp::tests::draft_cycle::', 3.5],
  ['models::qwen4_exp::tests::draft_turns::', 2.4],
  ['models::qwen4_exp::tests::lanes::', 1.8],
  ['models::qwen4_exp::tests::promote::', 1.7],
  ['models::qwen4_exp::tests::prefill_store::', 1.2],
  ['models::qwen4_exp::tests::draft_attach::', 1.15],
  ['models::qwen4_exp::tests::route_step::', 0.9],
  ['models::qwen4_exp::', 0.45],
  ['models::deepseek_v41::store::', 1.1],
  ['models::deepseek_v41::', 0.45],
  ['engine::dspark_device_sampled_tests::', 0.95],
  ['models::hy_v4_flash::', 0.22],
  ['models::inkling2::', 0.13],
  ['engine::hybrid_scheduler::', 0.09],
  ['models::qwen3_5::', 0.07],
  ['models::gemma4::', 0.03],
];

const DEFAULT_TEST_WEIGHT = 0.02;

const byCodePoint = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);
const argvBytes = (name: string): number => Buffer.byteLength(name) + 1;

function weight(name: string): number {
  const exact = TEST_WEIGHTS.get(name);
  if (exact !== undefined) return exact;
  return PREFIX_WEIGHTS.find(([prefix]) => name.startsWith(prefix))?.[1] ?? DEFAULT_TEST_WEIGHT;
}

function listTests(binary: string, ignored: boolean): string[] {
  const args = ['--list', '--format', 'terse', ...(ignored ? ['--ignored'] : [])];
  const out = execFileSync(binary, args, { encoding: 'utf-8', maxBuffer: 256 * 1024 * 1024 });
  const names: string[] = [];
  for (const raw of out.split('\n')) {
    const line = raw.trim();
    if (line === '') continue;
    if (!line.endsWith(': test')) throw new Error(`Unexpected line from ${binary} ${args.join(' ')}: ${raw}`);
    names.push(line.slice(0, -': test'.length));
  }
  if (new Set(names).size !== names.length) throw new Error(`Duplicate test names from ${binary} ${args.join(' ')}`);
  return names;
}

function activeTests(binary: string): { active: string[]; ignoredCount: number } {
  const all = listTests(binary, false);
  const ignored = new Set(listTests(binary, true));
  const allSet = new Set(all);
  const unknown = [...ignored].filter((name) => !allSet.has(name));
  if (unknown.length > 0)
    throw new Error(`--list --ignored returned tests absent from --list: ${unknown.slice(0, 5).join(', ')}`);
  const active = all.filter((name) => !ignored.has(name)).sort(byCodePoint);
  if (active.length === 0) throw new Error(`No active tests in ${binary}`);
  return { active, ignoredCount: ignored.size };
}

/** Longest-first greedy packing: each test goes to the lightest shard (ties: fewer tests, lower index). */
function partition(active: string[], shardCount: number): string[][] {
  const ordered = [...active].sort((a, b) => weight(b) - weight(a) || byCodePoint(a, b));
  const loads = Array.from({ length: shardCount }, () => 0);
  const buckets: string[][] = Array.from({ length: shardCount }, () => []);
  for (const name of ordered) {
    let target = 0;
    for (let i = 1; i < shardCount; i++) {
      if (loads[i] < loads[target] || (loads[i] === loads[target] && buckets[i].length < buckets[target].length)) {
        target = i;
      }
    }
    buckets[target].push(name);
    loads[target] += weight(name);
  }
  for (const bucket of buckets) bucket.sort(byCodePoint);
  return buckets;
}

function chunks(names: string[]): string[][] {
  const out: string[][] = [];
  let current: string[] = [];
  let bytes = 0;
  for (const name of names) {
    const size = argvBytes(name);
    if (current.length > 0 && bytes + size > MAX_CHUNK_ARGV_BYTES) {
      out.push(current);
      current = [];
      bytes = 0;
    }
    current.push(name);
    bytes += size;
  }
  if (current.length > 0) out.push(current);
  return out;
}

function stripAnsi(line: string): string {
  let out = '';
  for (let i = 0; i < line.length; i++) {
    if (line[i] === '\x1b' && line[i + 1] === '[') {
      let j = i + 2;
      while (j < line.length && ((line[j] >= '0' && line[j] <= '9') || line[j] === ';')) j++;
      const end = line[j] ?? '';
      if ((end >= 'A' && end <= 'Z') || (end >= 'a' && end <= 'z')) {
        i = j;
        continue;
      }
    }
    out += line[i];
  }
  return out;
}

/** Parses `test result: ok. 3 passed; 0 failed; 1 ignored; …`. */
function parseResult(line: string): { status: string; passed: number; failed: number; ignored: number } | null {
  const prefix = 'test result: ';
  if (!line.startsWith(prefix)) return null;
  const [head, ...counts] = line.slice(prefix.length).split('; ');
  const status = head.split('. ')[0];
  const count = (field: string, text: string | undefined): number | null => {
    const [n, label] = (text ?? '').split(' ');
    const named = label === field || label === `${field};`;
    return named && n !== '' && Number.isInteger(Number(n)) ? Number(n) : null;
  };
  const passed = count('passed', head.split('. ')[1]);
  const failed = count('failed', counts[0]);
  const ignored = count('ignored', counts[1]);
  if ((status !== 'ok' && status !== 'FAILED') || passed === null || failed === null || ignored === null) return null;
  return { status, passed, failed, ignored };
}

async function runChunk(binary: string, chunk: string[], index: number, total: number): Promise<number> {
  console.log(`==> Running chunk ${index}/${total} (${chunk.length} tests)`);
  const child = spawn(binary, ['--test-threads=1', '--exact', ...chunk], {
    env: { ...process.env, RUST_TEST_THREADS: '1' },
    stdio: ['ignore', 'pipe', 'inherit'],
  });
  const exited = new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolve, reject) => {
    child.on('error', reject);
    child.on('close', (code: number | null, signal: NodeJS.Signals | null) => resolve({ code, signal }));
  });
  let last: ReturnType<typeof parseResult> = null;
  for await (const line of createInterface({ input: child.stdout })) {
    process.stdout.write(`${line}\n`);
    last = parseResult(stripAnsi(line).trim()) ?? last;
  }
  const { code, signal } = await exited;
  if (signal !== null) {
    console.error(`Chunk ${index}/${total} terminated by signal ${signal}`);
    return 1;
  }
  if (code !== 0) return code ?? 1;
  if (last === null) throw new Error(`Chunk ${index}/${total} exited 0 without a libtest summary line`);
  if (last.status !== 'ok' || last.passed !== chunk.length || last.failed !== 0 || last.ignored !== 0) {
    throw new Error(
      `Chunk ${index}/${total} summary mismatch: status=${last.status} passed=${last.passed} failed=${last.failed} ` +
        `ignored=${last.ignored}, expected passed=${chunk.length} failed=0 ignored=0`,
    );
  }
  return 0;
}

function verify(binary: string, shardCount: number): void {
  const { active, ignoredCount } = activeTests(binary);
  const seen: string[] = [];
  partition(active, shardCount).forEach((mine, i) => {
    if (mine.length === 0) throw new Error(`Shard ${i + 1}/${shardCount} has 0 tests`);
    const bytes = mine.reduce((sum, name) => sum + argvBytes(name), 0);
    const load = mine.reduce((sum, name) => sum + weight(name), 0);
    console.log(
      `shard ${i + 1}/${shardCount}: ${mine.length} tests (est=${load.toFixed(1)}s, ` +
        `${bytes.toLocaleString('en-US')} argv bytes, ${chunks(mine).length} chunk(s))`,
    );
    seen.push(...mine);
  });
  if (seen.length !== active.length || new Set(seen).size !== active.length) {
    throw new Error(
      `Shard union mismatch: got ${seen.length} (${new Set(seen).size} unique), expected ${active.length}`,
    );
  }
  console.log(
    `Verified ${shardCount} shards: ${active.length} active tests covered ` +
      `(0 dropped, 0 duplicated; ${ignoredCount} ignored skipped).`,
  );
}

async function main(argv: string[]): Promise<number> {
  if (argv.length !== 3) {
    console.error(
      'Usage:\n  run-rust-lib-shard.ts <test-binary> <shard-1-based> <shard-count>\n  run-rust-lib-shard.ts <test-binary> --verify <shard-count>',
    );
    return 2;
  }
  const [binary, shardArg, countArg] = argv;
  try {
    if (!statSync(binary).isFile()) throw new Error('not a file');
    accessSync(binary, constants.X_OK);
  } catch {
    console.error(`Test binary is missing or not executable: ${binary}`);
    return 2;
  }
  const shardCount = Number(countArg);
  if (shardArg === '--verify') {
    if (!Number.isInteger(shardCount) || shardCount < 1) {
      console.error(`Invalid shard count: ${countArg}`);
      return 2;
    }
    verify(binary, shardCount);
    return 0;
  }
  const shard = Number(shardArg);
  if (!Number.isInteger(shard) || !Number.isInteger(shardCount) || shardCount < 1 || shard < 1 || shard > shardCount) {
    console.error(`Invalid shard ${shardArg}/${countArg} (expected 1 <= shard <= count)`);
    return 2;
  }
  const { active, ignoredCount } = activeTests(binary);
  const mine = partition(active, shardCount)[shard - 1];
  if (mine.length === 0) throw new Error(`Shard ${shard}/${shardCount} selected 0 of ${active.length} active tests`);
  const parts = chunks(mine);
  const load = mine.reduce((sum, name) => sum + weight(name), 0);
  console.log(
    `Shard ${shard}/${shardCount}: running ${mine.length} of ${active.length} active tests ` +
      `(est=${load.toFixed(1)}s, ${ignoredCount} ignored skipped) in ${parts.length} chunk(s)`,
  );
  for (const [i, chunk] of parts.entries()) {
    const exitCode = await runChunk(binary, chunk, i + 1, parts.length);
    if (exitCode !== 0) return exitCode;
  }
  console.log(`Shard ${shard}/${shardCount}: all ${mine.length} tests passed.`);
  return 0;
}

process.exitCode = await main(process.argv.slice(2));
