# Remove unproven performance experiments

September 22, 2026. The policy is one default: keep optimizations with a valid,
repeatable performance benefit and remove the rest, including their switches.

The five phase-6 experiments passed correctness checks but did not establish
isolated, repeatable model-level gains. Combined 6K results were promising;
short/32K and old/new default results varied materially with background activity.
Synthetic graph-construction and isolated projection gains do not establish a
model-level improvement. We therefore removed all five instead of promoting
the entire combination or keeping permanent configuration options.

## Removed

- Verifier-only GDN recurrence/preparation APIs, shaders and output-policy plumbing.
- Per-turn compiled state-settlement graphs and their separate cache ownership.
- Replay-tape query aliasing for shorter retention.
- Indexed compiled-graph replay plans and their alternate test target.
- Aligned Q5/Q6 word extraction, AOT/JIT dispatch and SIMD owner-load prototypes.
- All five environment switches, benchmark metadata entries and experiment-only
  harnesses/tests. The pre-removal sources remain in the evidence archive.

## Retained

The pre-phase-6 runtime selections, including segmented verifier attention and
its runtime device/pipeline checks, remain intact. No device-name tuning or
replacement switches were introduced.

The growing-prefix attention correctness fix remains active: unsupported unfused
attention uses eager verification because its causal mask contains host-side
prefix lengths. Supported vector/segmented attention retains compiled verification.
Regression tests retain every kept-token count from one through eight, repeated
cycles, draft-window wraps and continuation for both paths. The diagnostic
improvement that reports the first state mismatch also remains.

## Verification

The canonical native rebuild passed. Its `mlx.metallib` is byte-for-byte equal
to candidate-9; the packaged addon contains none of the five removed switch
names, and the GPU library contains no word-load experimental symbols.

The final addon matches the retained baseline in all six deterministic lifecycle
scenarios: 321-token first/streaming/cached/cold turns plus 6K first/follow-up
requests. All non-timing result fields and every acceptance statistic match;
long continuation retains 6,251 cached tokens. Sampled generation passes a smoke
check. Monitored peak RSS was 17.15 GB; CPU compilation overlapped, so this is a
correctness check, not a timing comparison.

Three unchanged debug-assert tests passed separately in their existing debug
executable. The freshly compiled release suite passed 3,824 tests (129 ignored; the three
debug-only cases above were filtered). Fifteen fresh processes then each passed
both every-prefix continuation regressions, covering compiled and eager verifier
paths. Current-source Clippy with warnings denied, Rust/document formatting,
benchmark/continuation TypeScript type checks and whitespace checks also passed.
Independent review found no actionable cleanup issue.

The final packaged addon SHA-256 is
`5680fd5658f053207f5dd1bdb82e8493a01619756d81d0558427ab8f71628b96`.
Runtime source and all three packaged artifacts match the candidate-14 manifest.
See `validation-summary.json` and `lifecycle-parity.json` for the final checks.
No new performance gain is claimed by this cleanup. Source restoration is checked
against the frozen candidate-9 snapshot and vendored HEAD, preserving earlier
work. Evidence is under `.cache/benchmarks/splash-qwen38-phase7/`, including the
pre-removal root/vendor patches and source archives.
