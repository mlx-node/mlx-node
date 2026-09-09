# Qwen dense agent: Metal submission and CPU cache transfers

Two changes are retained for `mlx agent`:

- [Early Metal submission](../../../crates/mlx-core/src/models/qwen3_5/paged_forward.rs):
  submit the first four completed decode layers while the host builds later
  layers, following [mlx.fast](https://github.com/Layr-Labs/mlxfast-gemma4-26b-a4b-engine/blob/27c821c466c9799e87162e9436618863b7d0a0ba/Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift#L31-L100).
  Covers Qwen3.5 dense paged AR decode, MTP Step A and MTP verification.
  Submitted tensors retain their cache-write dependencies; arithmetic is unchanged.
- [CPU cache transfers](../../../crates/mlx-paged-attn/src/layer_kv_pool.rs):
  allocate shared paged-pool buffers on unified-memory devices. SSD capture
  makes one owned snapshot copy; restore copies validated bytes directly into
  reserved slots. This removes GPU staging allocations, extra copies and
  command-buffer waits across the shared paged-cache implementation.
  Growth preserves storage mode. Other devices retain the private-buffer path.

Both default on. Set `MLX_QWEN35_DECODE_EARLY_EVAL_LAYERS=0` or
`MLX_PAGED_CPU_TRANSFER=0` before process startup for the respective control.
Neither changes PagedAttention layout, GDN sidecar identity, cache publication
rules or capture budgets. Shared access still requires completed GPU writes
and exclusive restore destinations; the SSD writer receives owned bytes,
not a borrowed view of reusable pool memory.

## Agent measurements

M5 Max, 128 GiB unified memory, external P4510 SSD; September 9, 2026.
The primary checkpoint is `qwen3.8-27b-mxfp4-mlx`: Qwen3.5 dense architecture,
168 MXFP4 and 233 MXFP8 modules, with BF16 inline MTP. The old local name
`qwen3.8-27b-unsloth-mxfp4-mlx` was a symlink skipped by model discovery;
the canonical model existed and needed no reconversion.

### Early submission

**2.50% paired speedup across 12 warm continuations**, with 11 wins and one
regression. These measurements predate CPU transfers; the gains are separate.

| Warm continuation | Mean control | Mean optimized | Paired speedup |
| --- | ---: | ---: | ---: |
| After full prefill | 13.701 s | 13.386 s | 2.33% |
| After SSD restart | 12.968 s | 12.631 s | 2.67% |

All 14 output pairs, including setup, and their MTP statistics matched. Each
SSD arm restored 3,920 tokens (6.2% of input), installed one recurrent sidecar
and read 259,241,115 bytes. Setup peak active allocation was identical;
paired warm peaks differed by at most 16 KiB. Setup prefills are excluded.

### CPU cache transfers

Four fresh restart pairs restored **62,880 of 62,892 input tokens**, installed
one recurrent sidecar and read **4,158,439,110 bytes per arm**. All four pairs
improved; outputs, MTP cycles and acceptance statistics matched exactly.
Early submission stayed enabled in both arms.

| Pair (first arm) | Native TTFT: GPU transfers | CPU transfers | Whole turn: GPU | CPU |
| --- | ---: | ---: | ---: | ---: |
| 1 (GPU) | 3.184 s | 2.814 s | 16.748 s | 16.558 s |
| 2 (CPU) | 3.281 s | 2.573 s | 16.815 s | 16.219 s |
| 3 (GPU) | 3.311 s | 2.658 s | 17.294 s | 16.451 s |
| 4 (CPU) | 3.205 s | 2.479 s | 17.843 s | 16.089 s |

Native time to first token averaged **3.245 → 2.631 s**: **19.0% lower paired
latency**. This includes cache preparation and the uncached suffix. The first
visible streaming event averaged **5.250 → 4.673 s** (11.0% lower paired
latency); whole turns averaged **17.175 → 16.329 s** (4.9% lower paired latency,
equivalent to 5.15% speedup). These are four-pair results on this machine.

For the actual 16-layer, 4-KV-head, 256-head-dimension, BF16 pool, one 16-token
block is 1 MiB. The production transfer API measured capture **0.256 → 0.014 ms**
and restore **0.161 → 0.012 ms**, comparing the existing batched GPU path with
CPU access. The ignored `bench_block_io_cost_by_model_family` test reproduces
this screen with `MLX_PAGED_CPU_TRANSFER=0/1`; these are transfer costs, not SSD
throughput. A separate four-pair warm-continuation screen was 1.2% slower overall,
with matching outputs and a first-turn timing outlier: no warm-turn gain is established.

**Full-restore boundary:** both capture and restart used
`MLX_PAGED_CACHE_INITIAL_MB=4096`. With the agent's default 2 GiB initial pool,
the synchronous restore exhausted its slots after 32,768 tokens, short of the
saved recurrent boundary, and reused nothing. Changing the initial size also
changes the config fingerprint, so the snapshot must be created with the same
setting. The comparison preserves this existing limitation; it does not prove
full-history restoration under default pool sizing. Only seed creation raised
capture limits to 4,096 blocks / 60 seconds to persist the whole snapshot;
timed arms retained the default 128-block / 250 ms capture budgets.

## Protocol and validation

- Replay a recorded coding-agent context through production `MlxModelHost` and
  its streaming adapter: 133 historical messages, including source reads and
  tool results. Reconstruct the system prompt; treat historical tools as data
  and supply offline results for new tool calls. Use temperature zero, high
  thinking, 256 output tokens, normal inline MTP and 2,048-token prefill chunks.
- Use the same native binary within each comparison, persistent paged caching,
  stock Metal command-buffer limits (50 MiB / 50 operations) and a 2 GiB
  allocator-cache cap per process. The cap is a measurement control.
  Alternate execution order with one inference workload active at a time.
  Verify copied SSD snapshots byte-for-byte before loading; require actual
  recurrent-sidecar installation, not just KV hit counters. Paired speedup is
  `geomean(control / optimized) - 1`; latency reduction is
  `1 - geomean(optimized / control)`.
- Every measured writer drain succeeded; cold-write errors and corruption were
  zero. Runs used AC high-performance mode. Background desktop activity was
  uncontrolled, so small timing differences have noise.
- Native build, Rust formatting, Clippy with warnings denied, **209 paged-cache
  unit tests and four GPU dispatch stress tests passed**. The new regression
  checks FP16/BF16/FP8 bytes across CPU→GPU and GPU→CPU transfers, malformed-chain
  rejection before writes, growth, absence of shared-path staging and private
  fallback preservation.
- The earlier submission change passed 42 targeted Qwen dense tests and four
  paged dispatch stress tests. Broader filters encountered existing unsupported
  16-dimensional KV-head fixtures; this is not a full-repository green claim.
  A separate high-thinking read/edit/test agent fixture timed out after 600 s
  without an edit. No matched task control exists, so completed coding-task
  speed is not established.

MLX revision: `6d45ab90cfec5e7fe0cfe25bc635a23ac8a351bb`.
Early-submission addon SHA-256:
`653b265c4e6dab6b715c88b78cf3a66ad9dc44169aa52009280137611045924b`.
CPU-transfer addon SHA-256:
`af01acb630f7367b1927d3f4084f8dc4f2830f5aa546ce8672b82626d95409bf`.
Context SHA-256:
`6900ee940d7f347acb12aad71af5c533af1007b66122a5ac378ed4542af0a3a1`.
Raw evidence and experimental harnesses remain local; this is the consolidated report.

## Candidates not retained

- CPU GDN gates, FFN token/channel splits and SIMD argmax were slower or within
  noise of the GPU path. The long CPU-gate replay took 135.6 s to first token;
  separate GPU references ranged from 115.1 to 130.5 s. This does not establish
  a precise percentage regression.
- A fused ANE FFN split moved 2,048 of 17,408 channels to ANE in 56 MXFP4 layers.
  Direct IOSurface access worked, including GPU layout packing. Isolated FFN
  repeats improved 2–8%, but the agent replay took 148.0 s to first token and
  a continuation changed text. Actual activation error was 0.4–2.4% relative
  RMS. The prototype was removed; no private ANE runtime ships in production.
- Compact GDN heads, joined projections, multiply fusion and folded RMSNorm
  scaling produced no repeatable agent gain. Sigmoid fusion and dequantized
  GEMM changed numerics. Larger command buffers gained only 0.49% while adding
  3.27 GB peak prefill allocation; an 8,192-token prefill stalled, and chunked
  matrix GDN was slower.

Zero-copy access alone is not a speed result. Handoffs and GPU-produced inputs
were included in the offload screens. Primary references:
[Apple shared storage](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus),
[MLX unified memory and streams](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html),
[the pinned ANE bridge](https://github.com/maderix/ane-prefill-bench/tree/bfa6bb4a83c17879efc09eda3b6dcbd5dc4d7c5f),
and [FusionML](https://arxiv.org/abs/2607.22785). Their results do not substitute
for measurements on this mixed MXFP4/MXFP8 checkpoint.
