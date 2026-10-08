# Qwen3.8-27B + DFlash2 vs Splash: performance reference

This file merges the `docs/research/splash-qwen38/` reports (Sep 2026) and the
PR #171 results (Oct 2026). It keeps only facts, decisions, dead ends and open
work. The gate tools `benchmark.ts` and `validate.ts` stay in
`docs/research/splash-qwen38/`. The full history and the other tools are in git
at `69ccaf9d`: `git show 69ccaf9d:docs/research/splash-qwen38/<file>`.

## 1. Workload and identities

| Item       | Value                                                                                                                                                                                                                                 |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hardware   | Apple M5 Max, 40 GPU cores, 128 GiB, macOS 27.0                                                                                                                                                                                       |
| Target     | `.cache/models/qwen3.8-27b-gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf`, 17,559,178,144 B, sha256 `3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e`                                                                             |
| Target mix | Q5_K 191 matrices (7.21 GB codes), IQ4_XS 70 (2.94 GB), Q6_K 56 (2.62 GB), Q4_K 68 (2.08 GB), other 120 (0.56 GB); Q8_0 affine for small tensors                                                                                      |
| Draft      | `.cache/models/qwen3.8-27b-dflash2` = `incoai/Qwen3.8-27B-DFlash2` rev `dedf8df68adfb1afeaf7b7480c0a0243108177b4`, 3,848,817,896 B, 81 BF16 tensors, sha256 `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`        |
| Splash     | `/Users/brooklyn/workspace/github/splash` at `7e3c67e8e3a9e9912ff6e02521457017cc4c65d0`; package `incoai/Qwen3.8-27B-Splash` snapshot `9d27070b71f7142c6b6025f03ac011d70a73cb48` (packed affine Q4/g64 target and draft, Q8 paged KV) |
| Fixtures   | `.cache/benchmarks/fixtures-public-2026-09-11/qwen38-inputs.json` (sha256 `c6c13733…bf5b`, local only): `6k` 6,219 tokens, `16k` 18,754, `32k` 32,488. `short` = 87 tokens (TypeScript LRU-cache prompt), hard-coded in the scripts   |

Geometry:

```
target  64 layers = 48 GDN (Hk 16, Hv 48, Dk=Dv 128, conv 4) + 16 full attention (4 KV heads x 256, GQA 6)
        hidden 5120, inter 17408, vocab 248320; BF16 KV = 65,536 B per context row (2 GiB at 32K)
draft   5 layers, 8 KV heads x 128, taps from target layers 5/19/33/47/61, window 2048 (tail limit 2047)
        shares target embedding + output head; proposes 7 tokens -> verify block = 8 rows
cycle   propose -> verify (8 rows) -> accept (host argmax read) -> stop clamp -> commit (GDN replay) -> emit -> eval_boundary
```

At 1,024 tokens every fixture is still in reasoning when it stops. The runs
measure fixed-length decode, not answer quality.

## 2. Current state (PR #171, base `04ce9b2b`)

| Commit                  | Change                                                                                                                                                                                                                                                                                   | Exact |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| `20635793`              | One GPU generation stream per model thread. Before: each turn made a stream, re-traced the compiled verify (~29 ms/request) and leaked +12.5 MB/turn                                                                                                                                     | yes   |
| `f370524b` + `6576948f` | Reserve the DFlash2 target KV once per turn. Cap = live memory headroom; a warm cache is charged the whole replacement buffer. Growth copies 4 -> 0 per request                                                                                                                          | yes   |
| `844cc1f9`              | Async-submit the draft graph before the verify build; submit draft-context roots with the cycle boundary. GPU idle/cycle ~2.0 -> ~0.7 ms                                                                                                                                                 | yes   |
| `53fb60b5`              | One segmented SDPA call for all 8 verify rows (then a fork kernel; now mlx-node's `segmented_sdpa_verify_2pass_1`; routes single / one_pass / unified / split). 32K verify SDPA 31.6 -> 22.0 ms/cycle                                                                                    | yes   |
| `85852d93`              | Strided `gdn_prepare`, mixed BF16/F32 affine `qmv_wide`, residual add fused into the next `add_rmsnorm`, one draft mask per propose: -310..-330 dispatches/cycle                                                                                                                         | yes   |
| `0b164eac`              | Evaluate GDN derived constants at load. A persistent compiled tape replays any lazy constant it captured on every cycle (+96 dispatches)                                                                                                                                                 | yes   |
| `0f7e1da5`              | Draft is affine Q4/g64 only (selector projection keeps checkpoint precision; target head shared). Resident draft 3.58 -> 1.18 GiB                                                                                                                                                        | no    |
| `0326f2ad`              | `sg8`: simdgroup-matrix K-quant matvec for exactly M=8 rows (q4k/q5k/q6k/iq4xs, BF16, aligned, N%32==0) on GPU gen >= 17. Splash `0x4300\|q` code-to-BF16 trick. ~1 ULP on 0.02-0.1% of outputs, same argmax                                                                             | no    |
| (stage 2, this PR)      | `qk_norm_rope`: one `fast::metal_kernel` per FA verify layer for q_norm + k_norm + rope(q) + rope(k) (plus the strided-input and partial-rotary copies those ops made). Dispatches/cycle 1197 -> 1095 (short, int8 KV); hash and 275 cycles unchanged; +2..4% tok/s in ABBA (noise +-5%) | yes   |

Result vs `04ce9b2b` (5-8 ABBA fresh-process pairs, clock-matched):

|                                                    | short                   | 6K                    | 32K                   |
| -------------------------------------------------- | ----------------------- | --------------------- | --------------------- |
| ms per committed token, same text (teacher-forced) | 23.26 -> 18.80 (-19%)   | 20.16 -> 14.85 (-26%) | 39.63 -> 28.72 (-28%) |
| ms per cycle                                       | 86.8 -> 69.1            | 95.5 -> 69.7          | 125.9 -> 91.9         |
| raw decode tok/s                                   | 43.1 -> 58.1            | 50.1 -> 41.9          | 25.3 -> 37.3          |
| cycles for 1,024 tokens                            | 274 -> 291              | 216 -> 358            | 322 -> 302            |
| dispatches / steady cycle                          | 1671 -> 1627            | 1733 -> 1653          | 1733 -> 1653          |
| AR decode                                          | same tokens, same speed |                       |                       |

Raw 6K is slower only because the non-exact commits change the text (358 vs
216 cycles). Target QMM is still 59-78% of cycle GPU time; 32K verify SDPA is
26%.

Draft precision study (teacher-forced, fixed BF16 transcripts, 13+ prompts):

|                                   | BF16                  | Q8/g64                | Q4/g64                |
| --------------------------------- | --------------------- | --------------------- | --------------------- |
| acceptance vs BF16 (short/6K/32K) | 1                     | 0.996 / 0.991 / 0.997 | 0.996 / 0.995 / 1.009 |
| ms per committed token, summed    | 80.97                 | 79.69                 | 78.23 (-3.4%)         |
| raw E2E tok/s (own transcript)    | 42.49 / 47.78 / 24.69 | 43.88 / 30.18 / 28.18 | 52.23 / 36.33 / 26.99 |

The old "Q8 drops 6K acceptance 4.736 -> 3.081" was transcript drift. On the
same text Q8/g32 keeps 4.714.

## 2b. Splash-parity session (PR #183, base `0.0.16`, target UD-Q4_K_M)

Target for this session is `Qwen3.8-27B-UD-Q4_K_M.gguf` (the `_M` Unsloth
quant, 15.33 GB; Splash selector `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M`). Same
draft, fixtures, 1,024 tokens, greedy, reasoning on. Splash 1.2.1 from
Homebrew on the same HF blob.

| Commit                              | Change                                                                                                                                                                                                                                       | Exact           |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `fc747f5d9`                         | Discover `UD-Q*_K_M` (and later the `UD-Q2_K` / `UD-IQ*`) GGUF targets                                                                                                                                                                       | yes             |
| `bbcda9770`                         | Verify SDPA on a simdgroup-matrix tile. 32K kernel 0.61 -> 0.40 ms/layer                                                                                                                                                                     | ~1 BF16 ulp     |
| `a5f510a69`                         | Tensor-op (`matmul2d`) M=8 K-quant matmul. Won only for Q3_K / IQ4_NL on row-major; the math shape was not the gap                                                                                                                           | no (rounding)   |
| `63ac73999`                         | Verify SDPA on `matmul2d` (NAX, gen >= 17). 32K kernel 0.355 ms/layer; vector / tile routes stay as fallbacks                                                                                                                                | ~1 BF16 ulp     |
| `a7c7c1c77`                         | 64-row tiled K-quant layout (`[N/64][K/32][64][unit]`, scales/biases tiled per super-block). Same kernel math on Splash-like tiles ran 1.02-1.04x Splash's kernel; row-major was 1.4-1.8x behind. The gap was page footprint, not arithmetic | yes             |
| `bf87bd9cb`                         | Release row-major sources after tiling (load peak 33 -> 17 GB); pad short rows into merged tiles                                                                                                                                             | yes             |
| `a09fce90b`                         | Calibrate the SDPA vector/block crossover per device at first use (M5 Max: 512 keys, 34 ms). No device constants                                                                                                                             | yes             |
| `ca106593b`                         | Remove the A/B switches and losing paths (-918/+283)                                                                                                                                                                                         | yes             |
| `6e5185c79` `76fc74a59`             | Int8 flat K/V (Splash-aligned default where head_dim 256 kernels serve; `kvFormat: 'bf16'` opts out). 32K KV 2.1 -> 1.07 GB; KL 0.01069 -> 0.01067                                                                                           | no              |
| `32f501bcc`                         | Bound the MLX buffer cache during decode (idle cache 1.2 -> 0.2 GiB at short; the cap grows with context, the allocator never refuses on it)                                                                                                 | yes             |
| `923c1f3d9` `6bb6d3725` `971bbe10c` | Q2_K, Q5_0, MXFP4, IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS, IQ3_S imported packed at ggml bpw with in-kernel grids; bit-exact vs vendored llama.cpp decoders; cache v7                                                                                | yes (new types) |
| `d8a33ea84` `33f85213d`             | Packed GDN state blobs, one-kernel GDN commit, one-dispatch KV row stores                                                                                                                                                                    | yes             |
| `df754198d` `1c7626ba3`             | `convert` writes `layout: "t64"`; every family's loader tiles; `.biases` tiled in super-block units (review blocker)                                                                                                                         | yes             |
| `3615896e5`                         | Fused q/k RMSNorm + RoPE for the verify block. Dispatches/cycle 1197 -> 1095                                                                                                                                                                 | yes             |

Gate history (fresh process per run, ABBA, `macmon` clock; `norm` = ms/cycle
scaled to 1550 MHz; Splash rows from the same sessions):

| Build                        | short norm / raw tok/s / cycles | 6K norm / raw / cycles | 32K norm / raw / cycles        |
| ---------------------------- | ------------------------------- | ---------------------- | ------------------------------ |
| baseline `fc747f5d9`         | 62.3 / 51.4 / 263               | 62.6 / 38.9 / 351      | 82.4 / 37.5 / 290              |
| + SDPA tile `bbcda9770`      | -                               | 61.7 / 43.3 / 338      | 70.9 / 39.3 / 292              |
| + M=8 matmul2d `a5f510a69`   | 57.9 / 53.3 / 295               | 62.7 / 41.4 / 358      | 69.9 / 34.5 / 291              |
| + tiled layout `a7c7c1c77`   | 54.0 / 74.6 / 239               | 54.6 / 82.7 / 217      | 60.8 / 51.9 / 297              |
| + int8 opt-in, cache cap     | 53.5 / 64.9 / 241               | 57.4 / 68.1 / 217      | 66.6 / 40.1 / 297 (slow clock) |
| Splash 1.2.1 (same sessions) | 52.1 / 63.1 / 231               | 49.3 / 48.9 / 363      | 55.1 / 58.2 / 289              |

Paired ABBA at `a09fce90b` (4 pairs per case, clock-matched): mlx-node /
Splash ms per cycle 1.12x short, 1.09x 6K, 1.10x 32K (was 1.20 / 1.27 / 1.49
at the baseline). Raw 6K tok/s flatters mlx-node (4.71 accepted per cycle on
its transcript vs 2.82 on Splash's); use ms per cycle.

Final run at `3615896e5` (int8 K/V default): only the first pair ran at full
clock, short 53.9 vs 47.3 ms/cycle = 1.14x. The GPU then dropped to ~900 MHz
and stayed there. In that state mlx-node lost 46% decode tok/s (69 -> 37) and
Splash 30% (93 -> 65): our cycle tracks GPU clock, theirs mostly does not, so
we are still more ALU/issue-bound (dispatch count 1095 vs ~680, and more
shift/mask work per dequantized byte). Memory at 32K: 21.5 GB (bf16 K/V) ->
20.5 GB (int8) vs Splash 20.2 GB.

Acceptance differs by transcript, not engine: short 4.43 (Splash) vs 3.72
(mlx, int8 K/V) / 4.28 (bf16); 6K 2.82 vs 4.71; 32K 3.54 vs 3.52. Only a
teacher-forced run on the same text can compare engines (see §6).

### 2c. Second round (same PR): FP32 GDN state, one GDN kernel, glue, tracing

| Commit      | Change                                                                                                                                                                                                                                   | Exact                |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| `455026426` | Teacher-forced acceptance harness restored (`MLX_DFLASH2_TF_RECORD` / `MLX_DFLASH2_TF_DIR`, `tf-acceptance.ts`). Forcing a build onto its own transcript reproduces the free run exactly (hash, cycles, by-position, flip rate 0)        | tooling              |
| `6c7c61429` | GDN recurrent state carried in f32 (Splash-aligned); `y` and conv stay bf16. T=1 chain == T=8 window bit-for-bit. State 72 -> 144 MiB; sidecar fingerprint v2                                                                            | no (state precision) |
| `f37bee5cf` | One Metal primitive per GDN verify layer: prologue (conv, q/k norm, gates) + recurrence + gated RMSNorm + z-gate (Splash `verify_gdn_fused` shape). 0 ulps vs the chain. Full accept adopts the verify state (parity swap, no GdnCommit) | yes                  |
| `4cd4f0837` | 16 KvStoreRows -> 1 (address table); draft K/V flat window with in-place stores; draft q\|k\|v, gate\|up, cross-layer k\|v merged; one RoPE; fused q/k norm + RoPE in the draft. Commit 50 -> 18 and propose 134 -> 99 dispatches        | yes                  |
| `c2650b622` | Per-cycle timeline: `[dspark-span]` host spans on the Metal trace clock + `timeline.ts` (per-phase GPU attribution, idle gaps, Perfetto export)                                                                                          | tooling              |
| `09d35f662` | Q8_K stays refused, by name, as Splash does                                                                                                                                                                                              | -                    |

Teacher-forced gate for the f32 state (1,024 tokens, short/6K/32K, real
compiled verify): each build forced onto the OTHER build's transcript loses
the same ~1.5% of cycles (bf16 build 934 vs its own 923; f32 build 899 vs 885) with a 3% flip rate either way, so the precision change is neutral on
acceptance. Paired ABBA was flat (B/A 1.013 at a slow clock).

Wave 2 (fused GDN kernel + glue) cut dispatches per cycle ~1190 -> ~845 and
bought only 1.2% (short) / 1.8% (32K) in clock-matched pairs: dispatch
overhead was not where the gap lived. The timeline shows why: GPU busy 49.6
of 50.2 ms per cycle, idle 0.6 ms at the cycle tail (`eval_boundary` -> next
`propose`), verify 42 ms (~85%), draft propose 6.2 ms (~13%), commit ~1 ms.
Device-side accept/commit (Splash's `encodeBatchAcceptance`) is therefore
capped at ~1% here and was not built.

Kernel facts measured this round (M5 Max, DRAM-cold ring, 8 matmuls per
eval, `kquant_tiled_bench` / `kquant_small_m_bench`):

| Kernel                                   | M=1          | M=8                            | Note                                                              |
| ---------------------------------------- | ------------ | ------------------------------ | ----------------------------------------------------------------- |
| K-quant `@t64` (q4k/q5k/q6k, big shapes) | 440-487 GB/s | 1.14-1.16x the M=1 time        | was 1.54-2.60x row-major: the M=8 K-quant path is bandwidth-bound |
| MLX affine Q4/g64 (the draft's format)   | 438-496 GB/s | 190-230 GB/s (per-row `qmv`)   | the draft runs at ~2.2x its floor                                 |
| lm_head q6k `@t64`, N=248320             | 1.97 ms      | 2.24 ms; M=7 2.53 (`qmv_wide`) | run the draft head on 8 rows, not 7                               |

So the "cheaper dequant per byte" idea (Splash chunk order inside the 16 B
units, ~700 LOC) can win at most ~15% of QMM time and is parked; the draft's
matmul route was the next lever:

| Commit      | Change                                                                                                                                                                                                                                                                                                                                                                                        | Exact               |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| `827c363e7` | MLX affine Q4/g64 as `a4g64@t64` on the tiled K-quant kernels; the draft is tiled at load (same bytes) and its head runs on 8 rows. M=8 445 vs 188 GB/s; propose 6.2 -> 4.8 ms; teacher-forced 928 vs 923 cycles, 2.32 draft matches both                                                                                                                                                     | no (draft rounding) |
| `52c9cb8cd` | Sorted MoE expert matmul at prefill widths (`B / E >= 4`) on a tensor-op `gather_qmm_rhs_nax` kernel, one expert per 32/64-row tile; 112-220 -> 240-410 GB/s (1.8-2.4x) on the LFM2.5 / Qwen3.6-35B-A3B shapes; decode and verify untouched (`gather_qmv`)                                                                                                                                    | <= 1 bf16 ulp       |
| `707096991` | Affine modes trimmed to the one with a producer (`a4g64`)                                                                                                                                                                                                                                                                                                                                     | -                   |
| `e051db2ea` | MLX-affine linears tiled at load in every family (`a4g64@t64`, `a8g64@t64`; N >= 6144 and K >= 1536, bf16 companions, no calibration), with 16/32-row tensor-op tiers for M 9..32 and 8/16-row `qmv_t64` sub-tiles for small N. Qwen3.5-4B affine: AR +0..6%, prefill +15-25%, peak -1.1 GB, `mlx eval` identical to the row-major route at M=1; Gemma-4-E2B 0.997x; K-quant routes untouched | affine only         |

Head-to-head at `827c363e7` (ABBA, fresh processes, 1,024 tokens; Splash
1.2.1 on the same GGUF and draft):

|                           | short                                            | 6K                                                 | 32K                                        |
| ------------------------- | ------------------------------------------------ | -------------------------------------------------- | ------------------------------------------ |
| ms per cycle mlx / Splash | 52.2 / 48.5 = 1.077x (1 pair at 1519 / 1546 MHz) | 50.1 / 48.6 = 1.03x (normalized; ~1250 / 1300 MHz) | 60.0 / 55.7 = 1.078x (4 vs 4 at ~1430 MHz) |
| raw decode tok/s          | 61.0 / 90.7                                      | 47.6 / 59.2                                        | 60.2 / 61.0                                |
| accepted per cycle        | 3.18 / 4.43                                      | 2.94 / 3.04                                        | 3.61 / 3.42                                |
| prefill tok/s             | 538 / 323                                        | 715 / 1066                                         | 730 / 875                                  |

Session start was 1.20 / 1.27 / 1.49x. Under the slow GPU clock both engines
now scale the same way (mlx 52 -> 88 ms at 896 MHz, Splash 48 -> 78 at 1012):
the ALU/issue-bound behaviour from the morning is gone. The short-prompt
tok/s gap is the transcript (Splash's text accepts 4.43 per cycle, ours
3.18); on the same text the draft matches per cycle are within 0.1%.

## 3. Splash comparison (last run 2026-09-22, pre-#171 runtime, BF16 draft)

|                               | short         | 6K            | 32K             |
| ----------------------------- | ------------- | ------------- | --------------- |
| mlx decode tok/s              | 35.99         | 41.23         | 22.46           |
| Splash decode tok/s           | 63.92         | 82.96         | 51.46           |
| full request mlx / Splash (s) | 28.59 / 16.15 | 34.73 / 21.38 | 98.50 / 76.35   |
| TTFT mlx / Splash (s)         | 0.169 / 0.246 | 9.907 / 9.096 | 51.788 / 56.557 |
| cycles mlx / Splash           | 274 / 274     | 216 / 206     | 322 / 303       |

- Same cycle count on short: the gap is per-cycle cost, not acceptance.
- Timing boundaries differ. mlx decode counts 1,023 tokens after the first and
  includes final settle. Splash excludes its first emitted batch (6-8 tokens)
  and may skip the terminal KV row. Full request time is the fairest number.
- Splash README numbers (74 tok/s short) are from an M5 Pro 16-core. Do not
  compare them.
- The current share of Splash is unmeasured. Re-run before any parity claim.

Splash design vs mlx-node:

|                            | Splash                                                                                                                                                  | mlx-node                                                                     |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| target weights             | packed affine Q4/g64 (14.44 GB)                                                                                                                         | mixed K-quant GGUF (16.84 GB); not interchangeable                           |
| cycle                      | new CPU dispatch list per cycle, one command buffer, one serial encoder (`Runtime.mm:2208-2266`, `MetalBackend.mm:1400-1497`). Not pre-recorded, no ICB | MLX graph + compiled verify, MLX evaluator                                   |
| accept + commit            | on GPU in the same command (`encodeBatchAcceptance`, `encodeBatchGdnCommit`, `encodeDraftStateCommitBatch`); host swaps state parity after completion   | host argmax read (<= 60 B), host stop clamp, lazy commit with GDN replay     |
| stop contract              | budget + 2 stop tokens on device; cancel drains the in-flight command                                                                                   | cancel, observer, extra EOS, repetition, budget on host, in fixed order      |
| GDN state                  | FP32, ping-pong buffers                                                                                                                                 | BF16, rounded per token by replay (FP32 alignment in progress, see §2b)      |
| memory                     | fixed-plan arenas (`MemoryPlan.cpp`, `RuntimeArenas.hpp`); persistent draft rings                                                                       | MLX allocator, buffer cache bounded during decode                            |
| KV                         | Q8 paged                                                                                                                                                | int8 flat by default since PR #183 (DFlash has no paged KV or batching)      |
| weights                    | 256-row planar tiles, lane-chunk order, GPU repack at every start                                                                                       | 64-row tiles (`t64`), ggml bit order inside 16 B units, written by `convert` |
| profile (512-token prompt) | 52.03 ms/cycle, matrix pipelines ~90%                                                                                                                   | see section 2                                                                |

## 4. Facts that hold

- Host sync is not the lever. Pre-#171 trace: GPU busy ~97% of a cycle, host
  idle 2.3-2.9 ms/cycle. Host acceptance math < 0.3 ms/cycle.
- Target QMM at M=8 is ALU/issue-bound, not bandwidth-bound: M=8 57.7 ms vs M=1
  31.6 ms per forward; cache-resident weights only 1.06x faster. M8/M1 by format:
  Q6_K 1.54, Q5_K 1.69, Q4_K 1.98, IQ4_XS 2.17, Q3_K 2.60.
- GDN replay must run even on full acceptance. Verify carries FP32 state across
  the block; serial commit rounds to BF16 per token. Taking the verify state
  changes outputs.
- MLX `compile()` replays a CPU graph, not encoded GPU commands. Encoding,
  allocation and hazards run on every call. Warm compiled verify call: median
  0.83 ms.
- MLX tracks hazards per whole Metal buffer. One big arena creates false
  dependencies; use separate buffers per lifetime.
- MLX keeps a multi-output primitive until all outputs are dead. Custom kernels
  allocate every output.
- `concatenate_gpu` always copies. `SliceUpdate` copies unless the buffer can be
  donated.
- GDN snapshots alias state; there is no snapshot copy to save.
- State sizes per cycle: GDN state 72 MiB (48 x 1.5 MiB BF16), conv history
  2.8 MiB, replay tape ~15.1 MiB. Discarded verify outputs: 74.8 MiB and 96
  allocations.
- M5 Max counters: timestamps at stage boundaries only. No per-dispatch,
  DRAM, occupancy or ALU counters. Instruments' default Metal template gives no
  per-kernel rows.
- Metal 4.0 cooperative matrix ops: BF16 x uint8/int8 (K 16/32/64) and BF16 x
  packed uint4 (K 32/64). No packed uint2/5/6.
- GGUF scale groups (Q4_K/Q5_K/IQ4_XS 32, Q6_K 16) cannot merge into Splash's
  K=64 single-scale dot.
- Chat template trims trailing whitespace. A fixture that ends in whitespace
  cold-prefills on the next turn. End fixtures on non-whitespace.

## 5. Dead ends (do not retry without a new idea)

Kernels (ratio > 1 = faster than baseline):

| Experiment                                                                                                     | Result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Register-blocked `qmv_wide` (R rows per lane)                                                                  | bit-exact, 1.4-2.6x slower                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Integer cooperative-matrix (MPP) Q5/Q4 at M=8, 8 variants (expanded bytes, split-K 4, packed, tiled, paired)   | 0.26-0.74x of `qmv_wide`; gate was 1.25x                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Wider K-lanes KL16 / KL32                                                                                      | lost 10/13 and 13/13 cases                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Threadgroup grouping SG4 / SG8 at KL8                                                                          | 1.008/1.010 and 0.995/0.989; Q6 head regressed. (Not the same as sg8, which landed)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Owner-load + `simd_shuffle` activation broadcast                                                               | Q5 0.04-0.18x, Q6 0.02-0.11x                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Aligned Q5/Q6 word loads (`MLX_KQUANT_QMV_WORD_LOADS`)                                                         | operator 0.94-1.17x, exact, no model win; removed                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Forced split-K at M=8                                                                                          | -16/-38/-6%, outputs changed                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| BM16 NAX for K/IQ verify                                                                                       | 16x64 tile silently gave zero accumulators (fake speedup); 16x128 slower than vector                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| K/IQ residual epilogue / SwiGLU epilogue                                                                       | +1.7/+1.1% (noise) / -4..-17% (register pressure)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| SwiGLU store epilogue on `qmm_m8_nax_t64` (Splash `_a` gate then `_g` up, fp32 silu, up rounded to BF16 first) | Oct 2026, M5 Max, M=8, K=5120, N=17408/proj, Tiled64, DRAM-cold, 21 interleaved samples x 3 runs. ms/block today (merged gate\|up + compiled swiglu) vs A (gate + up `_g`): q4k 0.222-0.239 vs 0.224-0.235, q5k 0.259-0.293 vs 0.264-0.275, q6k 0.294-0.317 vs 0.294-0.323, iq4xs 0.227-0.245 vs 0.228-0.238; A/today 0.92-1.05x with no stable sign. today ~= merged-only ~= two separate matmuls, so the standalone swiglu is <= 1% of the block: no SwiGLU fusion (store epilogue, or a down_proj prologue that would recompute it per tile) can reach +3%. Exactness: 97.5-99.7% equal, max 3 BF16 ulps vs the chain. Removed |
| Native Metal chunked GDN (BT32)                                                                                | outputs changed, +3.5/-37/-34%                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Four-column GDN E48 Qwen override                                                                              | chain -3%, synchronized +4%, model mixed; removed (Qwen4 route kept)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Persistent draft attention ring                                                                                | 10.5% slower                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| First segmented SDPA prototype "+80.6% at 32K"                                                                 | withdrawn: per-key segment select in the hot loop; quiet repeats favored concat                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |

Scheduling and memory:

| Experiment                                                                                                                            | Result                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Bigger command buffers (256 ops/512 MiB, 1024/4096)                                                                                   | -1..-6%; 1024 preset peaked at 55.4 GB vs 27.1                          |
| Skip empty finalize                                                                                                                   | buffers 20,084 -> 10,971, decode -12..-15%                              |
| Early commit submission / settle before emit                                                                                          | +0.4..+4.8%, pair signs flipped                                         |
| Direct conv replay                                                                                                                    | no gain                                                                 |
| Pressure cache trim                                                                                                                   | no robust gain, cache 1.23 -> 7.70 GB                                   |
| Phase-6 set: verify-only GDN outputs, per-keep compiled commit (8 graphs), replay-tape Q release, indexed compiled replay, word loads | each exact; combined +1/+7/+1% under -9..+2% control drift; all removed |
| Fewer 256-token sync+clear events (per-stepper cadence)                                                                               | short-prompt allocator cache 1.23 -> 2.04 GB; dropped                   |
| Eval-boundary move in the shared decode loop                                                                                          | no gain at short/6K                                                     |

Draft and policy:

| Experiment                                                                                                                                        | Result                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Depth 3 vs 7                                                                                                                                      | short +14.8%, 6K -21.9% (216 -> 391 cycles); no generic rule                                                                                                                                                                                                                                                                                                                                                            |
| Adaptive AR fallback                                                                                                                              | 6K switched to AR after 2 cycles, -46% decode; removed                                                                                                                                                                                                                                                                                                                                                                  |
| Imported Splash packed Q4 draft                                                                                                                   | -5/-27/+19% raw (drift); removed. Draft bytes are not the Splash gap: at 32K it needed fewer cycles than Splash and was still slower                                                                                                                                                                                                                                                                                    |
| Q8 head clone for the draft                                                                                                                       | +0.39 GB; never clone the head. Q4 head clone for the draft: only 0.2 ms                                                                                                                                                                                                                                                                                                                                                |
| Splash integer-dot / scale-bias-sum arithmetic, Splash Q4 target package                                                                          | change numerics or the model; not neutral optimizations                                                                                                                                                                                                                                                                                                                                                                 |
| Draft re-quantized to Q4_K at load so it takes the `@t64` M=8 kernels (Oct 2026)                                                                  | draft phase -37%, cycle -5%, but -2% draft matches on 9 teacher-forced prompts (6 of 9 worse; Q4_K's 6-bit sub-scales vs g64's bf16 scales). Rejected; the fix is an affine M=8 kernel, not a format change                                                                                                                                                                                                             |
| Device-side greedy accept + commit (Splash `encodeBatchAcceptance`)                                                                               | not built: the timeline shows GPU idle 0.6 ms per 50 ms cycle, all at the cycle tail, so the ceiling is ~1%                                                                                                                                                                                                                                                                                                             |
| `add_rmsnorm` fused into `qmm_m8_nax_t64` (residual epilogue + whole-grid sumsq tail on the producer, RMSNorm prologue on the consumer; Oct 2026) | bit-exact (0 ulps), but a loss: the tail costs +15-28 us (one threadgroup re-reads 80 KB), the prologue +3-30% because the tensor op must read A from threadgroup memory instead of device memory (a copy-only prologue costs the same), while the separate `add_rmsnorm` costs 0-4 us per site back to back. Removed. Also closes the "merge `ba` into in_proj" idea: 48 launch-bound dispatches at ~3 us are ~0.15 ms |

## 6. Method

- Non-exact changes: decide by clock-matched ms/cycle / teacher-forced
  committed tokens per cycle (fixed reference text through the real compiled
  verify, many prompts, 95% CI per bucket). Report raw E2E tok/s beside it.
  Raw single-fixture tok/s measures transcript luck.
- Exact changes: full output hash + cycles + acceptance by position must equal
  the base (DFlash and AR), plus lifecycle records. Unit tests must go red on a
  named mutation, then green.
- 1,024 tokens. 128-256 token pilots reversed sign (Q8 pilot +8-12% became
  -12/-35%).
- Fresh process per fixture, ABBA order, >= 5 pairs for +-5% calls, 180 s idle
  between processes, `macmon` telemetry. Paired-delta noise floor is about
  +-2.5% with 2 samples.
- Thermal: GPU reaches 93-98 C; tok/s tracks GPU MHz. A slow clock mode
  (570-1290 MHz) hits 20-40% of runs at random. Fit ms/cycle against MHz and
  report slow-mode runs separately.
- One GPU job at a time. The desktop alone keeps the GPU 30-80% "active", so
  gate on GPU power < 6 W, not active ratio. Reject runs with host load.
- Trace runs never count for timing. CPU and GPU stamps share one host clock
  on macOS (`timeline.ts` prints the check), but a GPU interval is still not a
  host cost. Fewer command buffers does not mean fewer kernels or bytes.
- Rebuild check: a header-only Metal edit once shipped a stale metallib after a
  "successful" build. Check metallib hashes and kernel names after kernel edits.
- Token equality is not state equality. Compare caches, frontiers and the next
  proposal, then a cached turn.
- New kernel routes are keyed on pipeline properties, shape, dtype, quant mode
  and GPU generation. Never on device names or Splash constants.

### Command trace

`MLX_METAL_COMMAND_TRACE=1` logs `[metal-command]`, `[mlx-evaluation]` and
`[mlx-compiled]` lines; `=2` adds a `[metal-op]` line per primitive, `=3` its
dtypes. Each `[metal-command]` line counts `dispatches` and `primitives`
(every `gpu::eval`, including no-dispatch ones, so `primitives ≥ dispatches`).
The deleted `analyze-command-trace.mjs` unioned GPU intervals per benchmark
window (commands, dispatches, barriers, gaps). Rules:
logging perturbs timing; the eval wait holds real deferred GPU work, not CPU
overhead; `resourceBytes` is not traffic; the `mlx-compiled` span is an upper
bound on replay cost.

### Tools

Gate tools in `docs/research/splash-qwen38/`:

```sh
oxnode docs/research/splash-qwen38/benchmark.ts <addon.node> <out.json> [ar|dflash] [short,6k,16k,32k] [runs] [tokens]
oxnode docs/research/splash-qwen38/validate.ts <addon.node> <out.json> [tokens]
```

- Run them from the repo root. They read `.cache/` relative to the working
  directory.
- `benchmark.ts`: greedy, high reasoning, cache reset per case, 16-token
  warmup. Gate on `outputHash`, `performance.mtpCycles` and
  `performance.mtpAcceptanceByPosition`; time with
  `performance.decodeTokensPerSecond`.
- `validate.ts` records:
  - `first`
  - `continued` (full prefix reused)
  - `cold-continuation`
  - `stream` (must equal `first`)
  - `sampled` (no seed, so not repeatable; do not gate on it)
  - `long-first`
  - `long-continued` (crosses the 2,048-row draft window)

Teacher-forced acceptance (`tf-acceptance.ts`, one fresh process per case):

```sh
oxnode docs/research/splash-qwen38/tf-acceptance.ts record <addon.node> <ref-dir> [short,6k,32k] [tokens=1024]
oxnode docs/research/splash-qwen38/tf-acceptance.ts force <addon.node> <ref-dir> <label> [short,6k,32k] [tokens=1024]
```

- `record` runs `benchmark.ts` with `MLX_DFLASH2_TF_RECORD=<ref-dir>` and
  writes one `ref-<prompt key>.json` per case from the base build.
- `force` runs the candidate build with `MLX_DFLASH2_TF_DIR=<ref-dir>` and
  `MLX_DFLASH2_TF_LABEL=<label>`: the reference ids go through the real
  verify path, so KV and GDN state follow the reference text. It then prints
  cycles, mean committed per cycle (`a_live + 1`), mean `a_ref` (drafts that
  match the reference), flip rate (target argmax left the reference), accept
  rate by position and ms per committed token from `cycles-<label>.jsonl`.
  Compare labels on the same `<ref-dir>`. `MLX_BENCH_TARGET` overrides the
  target GGUF path for `benchmark.ts`.

Per-cycle host/GPU timeline (`timeline.ts`, one combined stderr log):

```sh
MLX_PROFILE_DECODE=1 MLX_METAL_COMMAND_TRACE=1 oxnode docs/research/splash-qwen38/benchmark.ts <addon.node> out.json dflash short 1 256 > run.log 2>&1
oxnode docs/research/splash-qwen38/timeline.ts run.log [--turn N] [--cycles 20-60] [--gap-us 50] [--chrome trace.json]
```

- Joins the `[dspark-span]` host phases with the `[metal-command]` GPU
  intervals of the last turn (the benchmark's measured one). A command buffer
  belongs to the phase that encoded it (`encodeStartCpu` inside the span); the
  `gpu(overlap)` column shows where it ran instead. Per cycle: GPU busy
  (union), idle gaps over the threshold with the phase they fall in, host-only
  time; then medians and p90. `--chrome` writes a two-track Perfetto trace.
- Both stamps are `std::chrono::steady_clock`; Metal's `GPUStartTime` is the
  same host clock here (the tool prints `gpuStart - submitCpu`, min ~10 µs).
  Timing from a traced run still never counts (§6 rules).

Deleted tools (restore from `69ccaf9d` if needed):

| Script                      | Use                                                                                    |
| --------------------------- | -------------------------------------------------------------------------------------- |
| `compare-runs.mjs`          | exactness + speed A/B of two `benchmark.ts` outputs                                    |
| `analyze-command-trace.mjs` | command-trace summary (see above)                                                      |
| `splash-local.mjs`          | runs Splash's server per prompt (port 18938, 40 GiB, 40,960 context), checks token ids |
| `architecture-audit.mjs`    | offline byte audit bound to an old trace; historical                                   |

## 7. Open work

1. Target QMM is still the largest cost. GPUs below gen 17 have no sg8. Ideas
   for them (unmeasured): aligned word extraction, byte-preserving interleave
   with look-ahead.
2. Device-side accept + GDN commit, as Splash does. It needs a device-count
   commit contract that keeps the host stop order (cancel, observer, EOS
   exclusion, terminal token, partial-accept rollback). Gain comes from fewer
   fences, not acceptance math.
3. Drop dead verify outputs (74.8 MiB stores, 96 allocations per cycle) with a
   verify-only output policy in the recurrence and prepare kernels. An earlier
   flag version was exact but showed no clear model win.
4. Remove the 3 sync+clear events per request without short-prompt memory
   growth (make context-sized scratch reusable first).
5. Explicit current/candidate (ping-pong) GDN and draft-context ownership. It
   is the prerequisite for any native Splash-style plan.
6. DFlash on paged KV with continuous batching; Q8 KV (separate quality gate).
7. Re-check `MLX_QMM_SPLITK_MIN_M`, `MLX_SDPA_BLOCKS` and buffer heuristics
   against the sg8 route.
8. Not measured: sampled (T > 0) acceptance with the Q4 draft, older GPUs,
   CUDA, and sg8 in other model families (8-row MTP verify, 8-token prefill
   chunks).
9. Re-run the Splash comparison on the current runtime. Done in §2b for
   `0.0.16` + PR #183; repeat on a quiet machine before any parity claim.
10. Affine on the `@t64` kernel family landed for the DFlash2 draft
    (`827c363e7`) and is now default-on for every bf16-companion MLX-affine
    4/64 and 8/64 linear at load (the Splash model: one layout, one kernel
    family, every M), see [perf.md](../perf.md). What it took: the tensor-op
    kernel grew 16- and 32-row tiers (`qmm_m16/m32_nax_t64`, x staged per
    step with zero rows past M, shared by the two simdgroups) so affine
    M = 9..32 stop paying the GEMM's 64-row tiles or qmv_wide's re-reads
    (M = 12 on the Qwen3.8 MLP shapes: 0.19 vs MLX 0.78 ms), a grid rule
    that leaves the tiny shapes (Gemma4 k/v at N = 256, routers) to
    qmv_wide, and a cap on the split-K partials for the taller tiers. The
    tiers, the grid rule, the cap and the `qmv_t64` sub-tiles are the affine
    contracts' only: the K-quant modes route and sum exactly as they
    shipped, so the Qwen3.8 GGUF target is bit-for-bit unchanged (AR short
    bench hash `e5b8ed3563` before and after; forcing the affine modes onto
    the shipped rules reproduces the DFlash2 short bench's `bf80b142cb` /
    321 cycles exactly). The DFlash2 draft is affine (`a4g64@t64`), so it
    takes the new affine routes and proposes differently: the DFlash2 bench
    hash and cycle count move with the draft (`67e565ac65`, 342 cycles on
    the short prompt; the target's verify is unchanged). The M = 1 long-K `qmv_t64` gap did
    not reproduce in overlapped throughput (8..32 k-splits within 3% at
    K = 5120..25600), but a dependency-chain bench (`MLX_KQUANT_SMALL_M_CHAIN=1`,
    the latency a decode step pays) showed `qmv_t64`'s 32-row threadgroups
    5-12% behind MLX's `qmv` on narrow N and Gemma-4-E2B decode 5% slower;
    `qmv_t64` now takes 8- or 16-row sub-tiles of the 64-row tile unless the
    grid is already 16 threadgroups per core (model decode back within noise
    of MLX's route, chain latency 0.92-1.02x). Remaining: the 32-row tier on
    wide-N short-K shapes (N >= 12288, K = 2560) is 0.9x of MLX's GEMM; M = 8
    `qmv_wide_t64` at N = 2560, K = 4096 is 0.84x of MLX's `qmv_wide`, and
    `qmv_wide_t64` at K = 25600 is 0.87-0.89x in chain latency (its k-steps
    jump a tile row, 1 KB, where the row-major kernel walks contiguous
    bytes). The generic loaders therefore tile an affine linear only from
    `[6144, 1536]` up (`kquant_affine_tiled_shape`); narrower ones keep
    MLX's route, which the chain bench puts at or ahead of the tiled
    kernels there (Gemma-4-E2B: only `gate|up` tiles; its attention
    projections, N = 256..2048, do not).
11. K-quant MoE experts on `@t64`: measured, not worth it (`kquant_moe_bench`,
    `MLX_KQUANT_MOE_BENCH=1`). At the verify width (8 tokens x top-8, ~54 of
    256 experts, Qwen3.6-35B-A3B shapes) today's one-dispatch `gather_qmv`
    runs at 424-432 GB/s = 88-90% of the floor and beats one dense M=8
    tiled pass over the same rows (A/C 0.79-0.83); a per-expert tiled
    dispatch is 2.2x slower (54 launches, split-K counters serialise them);
    AR decode (8 experts) is launch-bound at ~20 us per node on every route.
    A grouped tile-descriptor kernel could recover <= 12% of expert time
    (~9% of a verify cycle) for 800-1500 LOC: parked. The real MoE gap is
    prefill: the sorted `gather_qmm_rhs` route (B/E >= 4) runs at 113-132
    GB/s, 3.3-3.7x slower than a dense M=8 tiled pass. Closed in
    `52c9cb8cd`: a tensor-op route with one expert per row tile (Splash's
    tile-descriptor shape) reaches 240-410 GB/s; the remaining gap to a
    dense pass (~400 GB/s on LFM2) is the un-overlapped MMA phase with 1-3
    resident threadgroups per core.
12. Found while gating the affine route, pre-existing: Gemma4 decode picked
    its paged-attention `grouped_stripes` by a timing sweep at context > 512
    (`engine/decode_tuning.rs`), and every partition count gives a different
    bf16 transcript (the stripes round their online-softmax weights and
    partials to bf16 over different page subsets, then merge), so a greedy
    transcript could flip on a near-tie between runs, and the first 20
    tokens past 512 mixed partitions within one run. Resolved by a
    device rule instead of the sweep (then `gemma4/attention.rs`
    `grouped_d512_rule_stripes`): the smallest power of two whose
    query-head SIMD groups reach 16 per GPU core, bounded by the work tiles
    and 256. Measured on the 40-core M5 Max with forced stripes
    (gemma-4-e2b-it-4bit, 512 tokens): throughput rises monotonically with
    the partition count up to 128 (8 heads x 128 = 1024 SIMD groups) and is
    flat at 256; generic V2 is level with the best stripes at 1K and 4-13%
    behind at 4K/8K, so the rule never picks it. gemma-4-12b-it-qat-q4_0
    (Hq16/Hkv1) at 8K: generic 42.7, 32 -> 44.3, 64 -> 46.1 (the rule's
    pick), 128 -> 46.6, 256 -> 45.0 tok/s. Default build A/B, 5 interleaved
    runs: 156.3 -> 157.6 (short), 141.5 -> 144.8 (4K), 130.4 -> 144.1 (8K)
    tok/s median, HEAD producing 2/5/2 distinct transcripts per context and
    the rule one. A canonical per-tile reduction (partition-independent
    numerics) was rejected: the kernel template is shared with the
    D128/D256 routes, and fixed-size partials cost up to 50% extra
    attention traffic at 32K or the short-context parallelism the kernel
    lives on. `MLX_GEMMA4_DECODE_TUNING` is gone; the remaining
    submission-depth sweep does not change numerics (hash `e9a263b752`
    for depths 0/4/16/34). Muse-Glimmer's D128 partition, the first
    leftover, picked by the same timing sweep (`muse_glimmer/model.rs` via
    `begin_with_limit`): resolved the same way. The rule is now one shared
    helper (`engine/decode_tuning.rs` `grouped_partition_stripes`; D512
    caps it at 256, Muse at its live temporary-storage limit, which stays a
    hard cap), and the attention stage of the sweep is deleted (as is
    `MLX_MUSE_DECODE_TUNING`). muse-glimmer-30b-q4k (Hq32/Hkv2, D128, 13
    global layers), fresh processes, 512 greedy tokens: the sweep gave 4
    transcripts in 12 runs at 1K and 7 in 12 at 4K (it settled on generic,
    4, 8, 16, 32, 64 or 128 partitions from run to run); the rule (32 on 40
    cores) gives one per context (1K 12/12, 4K 12/12, 8K 10/10), the same
    transcript as the sweep build forced to 32, for every submission depth
    tried (0 to 51). Captured-input replay (60K capture cut to 1K-60K, 13
    dependent calls) puts 32 partitions level with generic V2 at 1K and
    1.2-2.3x ahead from 4K to 60K, within 6% of the fastest count to 8K;
    from 16K, 128-512 partitions are 9-11% faster in attention (under 1.5%
    of a decode step), left unclaimed: a context term is unmeasured for
    D512. Decode tok/s A/B was within noise (machine under heavy outside
    load, 10-19 tok/s spread per build; paired new/old medians 1.10, 0.98
    and 1.00 at 1K/4K/8K over 6, 6 and 10 interleaved pairs).
    The second leftover, a transcript that depended on the internal route,
    is resolved for the paths below (512 greedy tokens unless noted, fresh
    processes, 2 runs per hash; the 147- and 54-token rows are 1 run each):
    - Raw vs graph decode route: the raw (synchronous) route passed the
      whole block table plus the window mask while the graph route trimmed
      the sliding layers' table to the live window, so the generic kernel
      cut the same keys at different 512-token partition boundaries. Both
      now read `decode_read_span` (`paged_kv_cache_adapter.rs`), which
      covers every sliding family (Gemma4, Muse-Glimmer).
      gemma-4-e2b-it-4bit, 4K prompt, `MLX_PAGED_GRAPH_DECODE_GATHER=0`
      vs default: grouped D512 on `1af569649a` → `3b8ca77f44` = graph;
      off (`MLX_PAGED_GROUPED_D512=0`) `ba0c056b64` → `f5c51b41dc` =
      graph. Graph hashes unchanged.
    - Muse-Glimmer whole-turn lane vs scheduled lane: the whole-turn lane
      decoded with generic V2 and prefilled the whole suffix in one slice;
      it now takes the scheduled single-row plan (grouped D128, the rule's
      partitions) and the scheduled prefill grid (512-token slices from the
      cached prefix). The one-slice prefill also failed above ~2.5K prompt
      tokens (`context_length_exceeded`: the sliding group reserved the
      whole prompt in its window-sized pool), which slicing fixes.
      muse-glimmer-30b-q4k, whole-turn → scheduled hash: 147-token prompt
      and 1,024 tokens `8e15c00659` → `ced7eb1218`; 1K `d891e132e4` →
      `1ba153515b`; 2.2K `27e622b7df` → `4b91469bee`; 4K error →
      `72f5808af4`. Scheduled hashes unchanged.
    - `MLX_MUSE_GROUPED_STRIPES` is now bounded by the same live cap as the
      rule; the D512 override was already limited to its reducer's set.
    - K2-Horizon's `ForceD128` took a context table (32 to 4K, 64 to 8K,
      ...) that changed the partition count mid-turn; it now takes the
      same rule (32 on 40 cores at every context beyond 512), and
      `mlx_paged_grouped_d128_default_stripes` is gone (the C++ table stays
      for the Qwen3.5 D256 route only). k2-horizon-7b-fp8, 3.4K prompt and
      768 tokens across 4096: 4/4 `d7bf25daf0`. This checkpoint did not
      flip on these prompts under any partition (4K: HEAD's table and the
      rule give `d7bf25daf0`; 7K: HEAD's 64, the rule's 32 and generic V2
      all give `9120fe91bd`); a breakpoint on
      `mlx_paged_grouped_d128_max_stripes` confirmed the rule is resolved
      once per decode token.
    - The rule's work-tile bound still moves the count inside short turns
      (40 cores: 8-head Gemma4 32 → 64 → 128 at 1,009 and 2,033 tokens;
      the 32-head D128 models never move beyond 512). It is a function of
      device, heads and context, so a turn reproduces; kept. Any count at
      or above the page count gives the same bits (empty stripes add exact
      zeros; forced 128 and 256 partitions, 54-token prompt, 1,024 tokens:
      one hash `4cc97efe2f`), so dropping the bound would give one count
      per turn; not done because the empty-threadgroup cost could not be
      separated from load noise here (gemma-4-e2b 1K, rule vs 128: 45-98
      vs 55-90 tok/s).
    - Decode tok/s vs `c86d991bc`, interleaved and ABBA runs on a loaded
      machine (medians): gemma-4-e2b 4K 129.7 → 129.4 (13 runs each, 65-148
      spread); muse-glimmer 4K scheduled lane 15.7 → 15.3 (9 each, 10-18
      spread). The graph routes these lanes take did not change; no
      difference is established.
    - Gemma4's lane split is now closed, along the lines sketched for it
      above. Its scheduled single-row decode passed no plan to the D512
      layers (generic V2 where the whole-turn lane takes the grouped kernel;
      54-token prompt, 1,024 tokens: scheduled `9631187a81` = whole-turn
      with the grouped kernel off, whole-turn default `b252314b4e`), and at
      4K the lanes differed even with both on generic V2 (scheduled
      `c0305fa44c`, whole-turn `f5c51b41dc`) because the scheduled lane
      pinned different prefill breaks than the whole-turn chunking walked.
      A one-row scheduled wave now forwards through the whole-turn
      single-token step inside the same `decode_tuning` `PlanScope`, so it
      resolves the identical D512 policy (SDPA, grouped at the rule's
      stripes, generic, and their memory guards) instead of the batched
      gather, and prefill boundaries come from one helper
      (`engine/hybrid_scheduler.rs` `prefill_slice_ends` — the family's
      slice grid from the cached prefix plus its extra breaks) shared by
      admission, SSD restore, preemption replay, and the whole-turn chunk
      loop. gemma-4-e2b-it-4bit, 512 greedy tokens, 2 runs per lane:
      54-token `c611d4f4c3`, 1K `749930e373`, 4K `3b8ca77f44` (the grouped
      hash), 4K with the SSD cold tier `5c19395c11`, 4K grouped-off
      `f5c51b41dc` — scheduled = whole-turn in every configuration.
      Interleaved decode tok/s vs the pre-fix build at 4K (sched 48-134 vs
      48-130, wt 95-124 vs 58-137; the machine was under heavy outside
      load) shows no established difference. A tiny-model unit test
      (`gemma4_scheduled_slice_walk_matches_whole_turn_prefill_and_decode`)
      crosses a slice boundary and compares whole-turn vs scheduled
      prefill and single-row decode logits bit-for-bit.
    - Still route-dependent, deliberately left: multi-row scheduled waves
      keep generic V2 (the grouped kernels index one sequence per dispatch),
      so a scheduled row's bits depend on co-scheduling; grouped D128 has no
      raw-route kernel, so Muse's and K2's global layers run generic V2 when
      the graph gather is off or fails.
13. Dequant bit order inside the 16 B units (Splash chunk order, `t64p`):
    parked, <= 15% of QMM time now that M=8 is bandwidth-bound.
