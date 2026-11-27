#!/usr/bin/env python3
"""Run one deterministic round-robin shard of a compiled Rust libtest binary.

Used by `.github/workflows/ci.yml` (`rust-lib-test`) so `mlx-core --lib` is
compiled once in `rust-lib-build` and executed in parallel across N runners
with zero duplicate compilation and mathematically disjoint, complete coverage.

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


def slice_shard(active: list[str], shard_index: int, shard_count: int) -> list[str]:
    # Round-robin across sorted test names deals consecutive tests from every
    # heavy module (e.g. qwen4_exp, deepseek_v41, hy_v4_flash) evenly across
    # shards without maintaining brittle manual module lists.
    return active[shard_index - 1 :: shard_count]


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
    seen: list[str] = []
    for shard in range(1, shard_count + 1):
        mine = slice_shard(active, shard, shard_count)
        if not mine:
            raise RuntimeError(f"Shard {shard}/{shard_count} has 0 tests")
        argv_bytes = sum(len(n.encode("utf-8")) + 1 for n in mine)
        chunks = chunk_names(mine)
        print(
            f"shard {shard}/{shard_count}: {len(mine)} tests "
            f"({argv_bytes:,} argv bytes, {len(chunks)} chunk(s))"
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
    mine = slice_shard(active, shard_index, shard_count)
    if not mine:
        raise RuntimeError(
            f"Shard {shard_index}/{shard_count} selected 0 of {len(active)} active tests"
        )

    chunks = chunk_names(mine)
    print(
        f"Shard {shard_index}/{shard_count}: running {len(mine)} of {len(active)} "
        f"active tests ({ignored_count} ignored skipped) in {len(chunks)} chunk(s)",
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
