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

| Commit                  | Change                                                                                                                                                                                                       | Exact |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| `20635793`              | One GPU generation stream per model thread. Before: each turn made a stream, re-traced the compiled verify (~29 ms/request) and leaked +12.5 MB/turn                                                         | yes   |
| `f370524b` + `6576948f` | Reserve the DFlash2 target KV once per turn. Cap = live memory headroom; a warm cache is charged the whole replacement buffer. Growth copies 4 -> 0 per request                                              | yes   |
| `844cc1f9`              | Async-submit the draft graph before the verify build; submit draft-context roots with the cycle boundary. GPU idle/cycle ~2.0 -> ~0.7 ms                                                                     | yes   |
| `53fb60b5`              | One segmented SDPA call for all 8 verify rows (fork kernel `sdpa_vector_segmented_verify_2pass_1`; routes single / one_pass / unified / split). 32K verify SDPA 31.6 -> 22.0 ms/cycle                        | yes   |
| `85852d93`              | Strided `gdn_prepare`, mixed BF16/F32 affine `qmv_wide`, residual add fused into the next `add_rmsnorm`, one draft mask per propose: -310..-330 dispatches/cycle                                             | yes   |
| `0b164eac`              | Evaluate GDN derived constants at load. A persistent compiled tape replays any lazy constant it captured on every cycle (+96 dispatches)                                                                     | yes   |
| `0f7e1da5`              | Draft is affine Q4/g64 only (selector projection keeps checkpoint precision; target head shared). Resident draft 3.58 -> 1.18 GiB                                                                            | no    |
| `0326f2ad`              | `sg8`: simdgroup-matrix K-quant matvec for exactly M=8 rows (q4k/q5k/q6k/iq4xs, BF16, aligned, N%32==0) on GPU gen >= 17. Splash `0x4300\|q` code-to-BF16 trick. ~1 ULP on 0.02-0.1% of outputs, same argmax | no    |

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

|                            | Splash                                                                                                                                                  | mlx-node                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| target weights             | packed affine Q4/g64 (14.44 GB)                                                                                                                         | mixed K-quant GGUF (16.84 GB); not interchangeable                       |
| cycle                      | new CPU dispatch list per cycle, one command buffer, one serial encoder (`Runtime.mm:2208-2266`, `MetalBackend.mm:1400-1497`). Not pre-recorded, no ICB | MLX graph + compiled verify, MLX evaluator                               |
| accept + commit            | on GPU in the same command (`encodeBatchAcceptance`, `encodeBatchGdnCommit`, `encodeDraftStateCommitBatch`); host swaps state parity after completion   | host argmax read (<= 60 B), host stop clamp, lazy commit with GDN replay |
| stop contract              | budget + 2 stop tokens on device; cancel drains the in-flight command                                                                                   | cancel, observer, extra EOS, repetition, budget on host, in fixed order  |
| GDN state                  | FP32, ping-pong buffers                                                                                                                                 | BF16, rounded per token by replay                                        |
| memory                     | fixed-plan arenas (`MemoryPlan.cpp`, `RuntimeArenas.hpp`); persistent draft rings                                                                       | MLX allocator                                                            |
| KV                         | Q8 paged                                                                                                                                                | BF16 flat (DFlash has no paged KV or batching)                           |
| profile (512-token prompt) | 52.03 ms/cycle, matrix pipelines ~90%                                                                                                                   | see section 2                                                            |

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

| Experiment                                                                                                   | Result                                                                               |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| Register-blocked `qmv_wide` (R rows per lane)                                                                | bit-exact, 1.4-2.6x slower                                                           |
| Integer cooperative-matrix (MPP) Q5/Q4 at M=8, 8 variants (expanded bytes, split-K 4, packed, tiled, paired) | 0.26-0.74x of `qmv_wide`; gate was 1.25x                                             |
| Wider K-lanes KL16 / KL32                                                                                    | lost 10/13 and 13/13 cases                                                           |
| Threadgroup grouping SG4 / SG8 at KL8                                                                        | 1.008/1.010 and 0.995/0.989; Q6 head regressed. (Not the same as sg8, which landed)  |
| Owner-load + `simd_shuffle` activation broadcast                                                             | Q5 0.04-0.18x, Q6 0.02-0.11x                                                         |
| Aligned Q5/Q6 word loads (`MLX_KQUANT_QMV_WORD_LOADS`)                                                       | operator 0.94-1.17x, exact, no model win; removed                                    |
| Forced split-K at M=8                                                                                        | -16/-38/-6%, outputs changed                                                         |
| BM16 NAX for K/IQ verify                                                                                     | 16x64 tile silently gave zero accumulators (fake speedup); 16x128 slower than vector |
| K/IQ residual epilogue / SwiGLU epilogue                                                                     | +1.7/+1.1% (noise) / -4..-17% (register pressure)                                    |
| Native Metal chunked GDN (BT32)                                                                              | outputs changed, +3.5/-37/-34%                                                       |
| Four-column GDN E48 Qwen override                                                                            | chain -3%, synchronized +4%, model mixed; removed (Qwen4 route kept)                 |
| Persistent draft attention ring                                                                              | 10.5% slower                                                                         |
| First segmented SDPA prototype "+80.6% at 32K"                                                               | withdrawn: per-key segment select in the hot loop; quiet repeats favored concat      |

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

| Experiment                                                                               | Result                                                                                                                               |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Depth 3 vs 7                                                                             | short +14.8%, 6K -21.9% (216 -> 391 cycles); no generic rule                                                                         |
| Adaptive AR fallback                                                                     | 6K switched to AR after 2 cycles, -46% decode; removed                                                                               |
| Imported Splash packed Q4 draft                                                          | -5/-27/+19% raw (drift); removed. Draft bytes are not the Splash gap: at 32K it needed fewer cycles than Splash and was still slower |
| Q8 head clone for the draft                                                              | +0.39 GB; never clone the head. Q4 head clone for the draft: only 0.2 ms                                                             |
| Splash integer-dot / scale-bias-sum arithmetic, FP32 GDN state, Splash Q4 target package | change numerics or the model; not neutral optimizations                                                                              |

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
- Trace runs never count for timing. CPU and GPU clocks are separate domains.
  Fewer command buffers does not mean fewer kernels or bytes.
- Rebuild check: a header-only Metal edit once shipped a stale metallib after a
  "successful" build. Check metallib hashes and kernel names after kernel edits.
- Token equality is not state equality. Compare caches, frontiers and the next
  proposal, then a cached turn.
- New kernel routes are keyed on pipeline properties, shape, dtype, quant mode
  and GPU generation. Never on device names or Splash constants.

### Command trace

`MLX_METAL_COMMAND_TRACE=1` logs `[metal-command]`, `[mlx-evaluation]` and
`[mlx-compiled]` lines. The deleted `analyze-command-trace.mjs` unioned GPU
intervals per benchmark window (commands, dispatches, barriers, gaps). Rules:
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
9. Re-run the Splash comparison on the current runtime.
