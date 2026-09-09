# Qwen dense agent: early Metal submission

Submit the first four completed decode layers asynchronously while the host
builds later layers, following [mlx.fast's early graph submission](https://github.com/Layr-Labs/mlxfast-gemma4-26b-a4b-engine/blob/27c821c466c9799e87162e9436618863b7d0a0ba/Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift#L31-L100).
The [implementation](../../../crates/mlx-core/src/models/qwen3_5/paged_forward.rs)
covers Qwen3.5 dense paged AR decode, MTP Step A and MTP verification on Metal.
Prefill, model arithmetic, PagedAttention layout, GDN sidecars and cache
publication boundaries remain unchanged; submitted layers retain their cache-write dependencies.

`MLX_QWEN35_DECODE_EARLY_EVAL_LAYERS=0` disables the optimization for a
same-binary control. The default is four; the value is read once per process.

## Measured agent benefit

On an M5 Max with 128 GiB unified memory and an external P4510 SSD,
`qwen3.8-27b-mxfp4-mlx` improved **2.50% across 12 paired warm continuations**:
**2.33% after full prefill** and **2.67% after SSD restart**. Eleven pairs
improved and one regressed. Speedup is the geometric mean of each pair's
`control wall time / optimized wall time`, excluding setup prefills.

All times below are seconds for a 256-token continuation.

| Turn            | After full prefill: control |  Optimized | After SSD restart: control |  Optimized |
| --------------- | --------------------------: | ---------: | -------------------------: | ---------: |
| 1               |                      13.250 |     12.873 |                     12.403 |     12.027 |
| 2               |                      13.669 |     13.588 |                     12.901 |     12.837 |
| 3               |                      13.535 |     13.835 |                     13.289 |     12.370 |
| 4               |                      14.015 |     13.535 |                     12.747 |     12.597 |
| 5               |                      14.601 |     13.649 |                     13.905 |     13.593 |
| 6               |                      13.135 |     12.836 |                     12.563 |     12.365 |
| Arithmetic mean |                  **13.701** | **13.386** |                 **12.968** | **12.631** |

All 14 output pairs, including setup turns, matched; MTP cycle and acceptance
counts matched too. Each SSD arm restored **3,920 tokens (6.2% of the input)**,
installed one recurrent sidecar and read 259,241,115 bytes. Every writer drain
succeeded, with zero cold-write errors or corruption. SSD capture retained its
default 128-block/turn and 250 ms budget.

Setup peak active allocation was identical between arms: 34,204,061,256 bytes
after full prefill and 28,230,471,224 after SSD restart. Every paired warm peak
was within 16 KiB. These results support faster fixed-work agent continuation;
prefill speed, SSD I/O speed, full-history recovery and other hardware were
not established by this comparison.

## Measurement protocol

- Replay a recorded coding-agent context through production `MlxModelHost`
  and its streaming adapter: 133 historical messages and 62,892 input tokens,
  including source reads and tool results. The system prompt was reconstructed;
  historical tools were treated as data and new tool calls received offline results.
- Use temperature zero, high thinking, 256 output tokens, normal inline MTP,
  2,048-token prefill chunks, persistent paged caching and stock Metal
  command-buffer limits of 50 MiB / 50 operations in both arms.
- Keep two processes resident, alternating six growing-context continuations
  with only one inference workload active. Wait for native producer completion
  and the SSD writer drain before releasing the next turn. Exclude model loading
  and setup prefills; require matching inputs and outputs for each pair.
- Repeat after restarting from byte-verified copies of the same SSD snapshot,
  reversing loading and turn order. Require a recurrent-sidecar installation;
  KV hits alone do not prove a valid hybrid-model restore.
- Cap each allocator cache at 2 GiB to fit both models; this is a benchmark
  control, not an agent default. Runs used AC high-performance mode and zero
  swap. Background desktop activity was uncontrolled, so small gains have noise.

Measurements were collected on September 9, 2026 using baseline
`0413d14d7d538774a5ed2f8a396eccf3c5fa8b38` and MLX
`6d45ab90cfec5e7fe0cfe25bc635a23ac8a351bb`. Native inference and agent source
were unchanged by the subsequent rebase. Both arms loaded addon SHA-256
`653b265c4e6dab6b715c88b78cf3a66ad9dc44169aa52009280137611045924b` and used
context SHA-256 `6900ee940d7f347acb12aad71af5c533af1007b66122a5ac378ed4542af0a3a1`.

The checkpoint uses 168 MXFP4 and 233 MXFP8 modules with intact BF16 inline MTP.
The old local name `qwen3.8-27b-unsloth-mxfp4-mlx` was a directory symlink
skipped by model discovery; the canonical model existed and needed no reconversion.

## Validation and limits

- Native and TypeScript builds, focused script lint, Rust formatting and diff
  checks passed during evaluation. **42 targeted Qwen dense native tests and
  four paged dispatch stress tests passed**, including MTP owner replay and
  recurrent-cache checks. Stress tests compare against an explicitly
  synchronized reference and reject a no-write baseline.
- Broader test filters hit existing unsupported 16-dimensional KV-head fixtures
  before reaching the changed helper. The targeted release run excluded these
  and a debug-assertion-only test; the full suite was not green.
- A separate high-thinking `mlx agent` read/edit/test task hit its **600-second
  limit** after three reads and no edit; independent acceptance failed. The
  loaded addon matched the measured candidate. Without a matched task control,
  this establishes neither completed coding-task speed nor the timeout's cause.

## Candidates not retained

Compact grouped GDN heads, joined gate/up projections, multiply fusion and
folded RMSNorm scaling produced no repeatable agent gain. Full sigmoid fusion
and dequantize-then-GEMM encountered numerical differences. Larger command
buffers gained only 0.49% in paired turns while increasing peak prefill
allocation by 3.27 GB. An 8,192-token prefill experiment stalled, and the
matrix-based chunked GDN screen was slower than the existing Metal path.
None of these experiments changed production defaults.
