#!/usr/bin/env python3
"""Run one deterministic LPT-balanced shard of a compiled Rust libtest binary.

Used by `.github/workflows/ci.yml` (`rust-lib-test-1..N`) so `mlx-core --lib`
is compiled once in `rust-lib-build` and executed in parallel across N runners
with zero duplicate compilation and mathematically disjoint, complete coverage.

Timing hints below are ONLY scheduling weights (never filters): if a test is
renamed, added, or removed, it still gets assigned cleanly via its module prefix
or default weight, and `--verify` guarantees 0 dropped and 0 duplicated tests.

Usage:
  python3 scripts/run-rust-lib-shard.py <test-binary> <shard-1-based> <shard-count>
  python3 scripts/run-rust-lib-shard.py <test-binary> --verify <shard-count>
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

# Keep each argv invocation safely below macOS ARG_MAX (1 MiB, shared with env).
MAX_CHUNK_ARGV_BYTES = 256 * 1024

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
RESULT_RE = re.compile(
    r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;"
)

# Relative duration hints (seconds on hosted macos-26) for the heaviest unit
# tests so LPT bin packing spreads them evenly across shards.
TEST_WEIGHTS: dict[str, float] = {
    "models::qwen4_exp::tests::draft_attach::a_sampled_drafted_turn_keeps_the_target_distribution": 39.9,
    "models::qwen4_exp::tests::draft_cycle::a_sampled_cycle_keeps_the_target_distribution": 26.7,
    "models::qwen4_exp::tests::draft_cycle::a_sampled_cycle_of_four_rows_keeps_the_target_pair_distribution": 22.6,
    "models::qwen4_exp::tests::draft_switch::a_store_that_moves_up_drafts_the_turn_after_its_fill": 21.6,
    "models::qwen4_exp::tests::route_step::early_cut_is_exact": 20.0,
    "engine::dspark_device_sampled_tests::device_sampled_acceptance_keeps_the_target_law": 19.2,
    "models::qwen4_exp::tests::route_step::early_cut_with_groups_in_flight_is_exact": 18.5,
    "engine::dspark_device_sampled_tests::host_acceptance_of_device_drafts_keeps_the_target_law": 16.2,
    "models::deepseek_v41::store::tests::tiny_cache_bitwise_any_size": 15.2,
    "models::qwen4_exp::tests::batched_prefill_matches_reference_across_chunks_sparse_boundaries_and_continuation": 14.6,
    "models::qwen4_exp::tests::promote::a_host_gate_store_moves_up_at_its_last_close_and_decodes_as_a_whole_load": 12.7,
    "models::qwen4_exp::tests::prefill_store::a_fill_never_writes_under_a_pending_reader": 12.0,
    "models::qwen4_exp::tests::draft_rows::every_row_choice_gives_the_plain_text": 11.2,
    "models::deepseek_v41::store::tests::tiny_prefetch_bitwise": 10.8,
    "models::qwen4_exp::tests::route_step::device_replay_fills_protect_the_layers_ids": 9.8,
    "models::qwen4_exp::tests::lanes::moved::the_residency_is_read_at_every_step": 9.7,
    "models::qwen4_exp::tests::prefill_store::residents_are_submitted_before_the_fill_finishes": 9.0,
    "models::qwen4_exp::tests::batched_qsa_matches_per_token_flat_and_paged": 8.6,
    "models::deepseek_v41::tests::tiny_logits_bf16_within_floor": 8.6,
    "models::qwen4_exp::tests::kquant_nax_gathers_match_dequantized_experts_with_tail_tiles": 8.3,
    "models::deepseek_v41::store::tests::tiny_store_matches_resident": 8.1,
    "models::qwen4_exp::tests::prefill_store::an_int8_group_matches_windows_bit_for_bit": 7.9,
    "models::qwen4_exp::tests::contiguous_prompt_windows_preserve_rows_frontiers_and_continuation": 7.9,
    "models::qwen4_exp::tests::route_step::device_replay_is_exact": 7.6,
    "models::qwen4_exp::tests::promote::a_refused_allocation_half_way_leaves_a_partial_store_for_the_next_turn": 6.9,
    "models::qwen4_exp::tests::promote::a_state_written_on_halved_passes_does_not_serve_the_moved_store": 6.6,
    "models::qwen4_exp::tests::row_batched_decode_kernels_match_single_row_calls_bit_for_bit": 6.4,
    "models::qwen3_5::gated_delta::gdn_bf16_gates_tests::bf16_scalars_keep_the_chain_bits": 6.1,
    "models::deepseek_v41::store::tests::tiny_cache_bank_switch_mid_session": 5.9,
    "models::deepseek_v41::store::tests::tiny_cache_split_bitwise": 5.5,
    "models::qwen4_exp::tests::route_step::tail_replay_is_exact_at_every_layer_kind": 5.5,
    "models::qwen4_exp::tests::draft_rows::the_rows_repeat_and_each_drafted_turn_starts_its_own_record": 5.2,
    "models::qwen4_exp::tests::draft_cycle::every_accept_length_gives_the_plain_text_and_state": 5.0,
    "models::inkling2::scheduler::tests::ws4_multi_phase_recon_verify_synthesize_and_parent_tip_ssd_durability": 4.9,
    "models::deepseek_v41::tests::invariants_chunk_sizes": 4.8,
    "models::deepseek_v41::tests::invariants_cancel_restores": 4.8,
    "models::deepseek_v41::store::tests::tiny_layer_skip_bitwise": 4.7,
    "models::qwen4_exp::tests::draft_switch::a_peer_ends_the_drafted_cycles_at_a_cycle_boundary": 4.7,
    "models::qwen4_exp::tests::draft_attach::a_proposal_error_fails_the_turn": 4.3,
    "models::hy_v4_flash::ops::tests::hc_fused_suite_and_prefill_row2_parity_and_invariance": 4.2,
}

# Module-level fallback weights (most specific prefix first) for tests not in
# TEST_WEIGHTS.
PREFIX_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("models::qwen4_exp::tests::draft_cycle::", 3.5),
    ("models::qwen4_exp::tests::draft_turns::", 2.4),
    ("models::qwen4_exp::tests::lanes::", 1.8),
    ("models::qwen4_exp::tests::promote::", 1.7),
    ("models::qwen4_exp::tests::prefill_store::", 1.2),
    ("models::qwen4_exp::tests::draft_attach::", 1.15),
    ("models::qwen4_exp::tests::route_step::", 0.9),
    ("models::qwen4_exp::", 0.45),
    ("models::deepseek_v41::store::", 1.1),
    ("models::deepseek_v41::", 0.45),
    ("engine::dspark_device_sampled_tests::", 0.95),
    ("models::hy_v4_flash::", 0.22),
    ("models::inkling2::", 0.13),
    ("engine::hybrid_scheduler::", 0.09),
    ("models::qwen3_5::", 0.07),
    ("models::gemma4::", 0.03),
)

DEFAULT_TEST_WEIGHT = 0.02


def estimate_weight(name: str) -> float:
    exact = TEST_WEIGHTS.get(name)
    if exact is not None:
        return exact
    for prefix, weight in PREFIX_WEIGHTS:
        if name.startswith(prefix):
            return weight
    return DEFAULT_TEST_WEIGHT


def list_tests(binary: Path, *, ignored: bool = False) -> list[str]:
    cmd = [str(binary), "--list", "--format", "terse"]
    if ignored:
        cmd.append("--ignored")
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    names: list[str] = []
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        if not line.endswith(": test"):
            raise RuntimeError(
                f"Unexpected line from {' '.join(cmd)}: {raw!r}"
            )
        names.append(line[: -len(": test")])
    if len(names) != len(set(names)):
        raise RuntimeError(f"Duplicate test names returned by {' '.join(cmd)}")
    return names


def compute_active_tests(binary: Path) -> tuple[list[str], int]:
    all_tests = list_tests(binary, ignored=False)
    ignored_tests = set(list_tests(binary, ignored=True))
    unknown_ignored = ignored_tests - set(all_tests)
    if unknown_ignored:
        raise RuntimeError(
            f"--list --ignored returned tests absent from --list: {sorted(unknown_ignored)[:5]}"
        )
    # Deterministic ASCII code-point sort (independent of locale).
    active = sorted(name for name in all_tests if name not in ignored_tests)
    if not active:
        raise RuntimeError(f"No active tests discovered in {binary}")
    return active, len(ignored_tests)


def partition_shards(active: list[str], shard_count: int) -> list[list[str]]:
    # Longest-Processing-Time-First (LPT) greedy packing: place heaviest tests
    # first into the lightest shard, breaking ties by test count then index.
    ordered = sorted(active, key=lambda name: (-estimate_weight(name), name))
    loads = [0.0] * shard_count
    counts = [0] * shard_count
    buckets: list[list[str]] = [[] for _ in range(shard_count)]
    for name in ordered:
        weight = estimate_weight(name)
        target = min(range(shard_count), key=lambda i: (loads[i], counts[i], i))
        buckets[target].append(name)
        loads[target] += weight
        counts[target] += 1
    for bucket in buckets:
        bucket.sort()
    return buckets


def chunk_names(names: list[str], max_bytes: int = MAX_CHUNK_ARGV_BYTES) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        size = len(name.encode("utf-8")) + 1
        if current and current_bytes + size > max_bytes:
            chunks.append(current)
            current = [name]
            current_bytes = size
        else:
            current.append(name)
            current_bytes += size
    if current:
        chunks.append(current)
    return chunks


def verify_shards(binary: Path, shard_count: int) -> int:
    active, ignored_count = compute_active_tests(binary)
    buckets = partition_shards(active, shard_count)
    seen: list[str] = []
    for shard, mine in enumerate(buckets, start=1):
        if not mine:
            raise RuntimeError(f"Shard {shard}/{shard_count} has 0 tests")
        argv_bytes = sum(len(n.encode("utf-8")) + 1 for n in mine)
        est_load = sum(estimate_weight(n) for n in mine)
        chunks = chunk_names(mine)
        print(
            f"shard {shard}/{shard_count}: {len(mine)} tests "
            f"(est={est_load:.1f}s, {argv_bytes:,} argv bytes, {len(chunks)} chunk(s))"
        )
        seen.extend(mine)
    if len(seen) != len(active) or set(seen) != set(active):
        raise RuntimeError(
            f"Shard union mismatch: got {len(seen)} ({len(set(seen))} unique), "
            f"expected {len(active)}"
        )
    print(
        f"Verified {shard_count} shards: {len(active)} active tests covered "
        f"(0 dropped, 0 duplicated; {ignored_count} ignored skipped)."
    )
    return 0


def run_chunk(binary: Path, chunk: list[str], chunk_idx: int, total_chunks: int) -> None:
    cmd = [str(binary), "--test-threads=1", "--exact", *chunk]
    env = os.environ.copy()
    env["RUST_TEST_THREADS"] = "1"

    print(
        f"==> Running chunk {chunk_idx}/{total_chunks} ({len(chunk)} tests)",
        flush=True,
    )
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    last_result: tuple[str, int, int, int] | None = None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        cleaned = ANSI_RE.sub("", line).strip()
        match = RESULT_RE.match(cleaned)
        if match:
            last_result = (
                match.group(1),
                int(match.group(2)),
                int(match.group(3)),
                int(match.group(4)),
            )

    exit_code = proc.wait()
    if exit_code != 0:
        raise SystemExit(exit_code)

    if last_result is None:
        raise RuntimeError(
            f"Chunk {chunk_idx}/{total_chunks} exited 0 without printing a libtest summary line"
        )
    status, passed, failed, ignored = last_result
    if status != "ok" or passed != len(chunk) or failed != 0 or ignored != 0:
        raise RuntimeError(
            f"Chunk {chunk_idx}/{total_chunks} summary mismatch: "
            f"status={status} passed={passed} failed={failed} ignored={ignored}, "
            f"expected passed={len(chunk)} failed=0 ignored=0"
        )


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(
            "Usage:\n"
            f"  {argv[0]} <test-binary> <shard-1-based> <shard-count>\n"
            f"  {argv[0]} <test-binary> --verify <shard-count>",
            file=sys.stderr,
        )
        return 2

    binary = Path(argv[1])
    if not binary.is_file() or not os.access(binary, os.X_OK):
        print(f"Test binary is missing or not executable: {binary}", file=sys.stderr)
        return 2

    if argv[2] == "--verify":
        shard_count = int(argv[3])
        if shard_count < 1:
            print(f"Invalid shard count: {shard_count}", file=sys.stderr)
            return 2
        return verify_shards(binary, shard_count)

    shard_index = int(argv[2])
    shard_count = int(argv[3])
    if shard_count < 1 or not (1 <= shard_index <= shard_count):
        print(
            f"Invalid shard {shard_index}/{shard_count} (expected 1 <= shard <= count)",
            file=sys.stderr,
        )
        return 2

    active, ignored_count = compute_active_tests(binary)
    mine = partition_shards(active, shard_count)[shard_index - 1]
    if not mine:
        raise RuntimeError(
            f"Shard {shard_index}/{shard_count} selected 0 of {len(active)} active tests"
        )

    chunks = chunk_names(mine)
    est_load = sum(estimate_weight(n) for n in mine)
    print(
        f"Shard {shard_index}/{shard_count}: running {len(mine)} of {len(active)} "
        f"active tests (est={est_load:.1f}s, {ignored_count} ignored skipped) in {len(chunks)} chunk(s)",
        flush=True,
    )
    for idx, chunk in enumerate(chunks, start=1):
        run_chunk(binary, chunk, idx, len(chunks))

    print(
        f"Shard {shard_index}/{shard_count}: all {len(mine)} tests passed.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
