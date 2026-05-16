# Autoresearch: FlashQLA-inspired Metal GDN kernel speedup

## Objective

Reduce **prefill latency** of `Qwen3.5-4B-mlx` on Apple Silicon Metal by
porting algorithmic ideas from
[QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA) into our existing
gated-delta Metal kernels (`crates/mlx-sys/src/metal/gated_delta_*.metal.inc`
and the C++ launch surface in `crates/mlx-sys/src/mlx_gated_delta.cpp`).
Correctness must hold — the first 8 decoded tokens at temperature=0 must
match the main-branch baseline byte-for-byte.

FlashQLA is the Qwen team's CUDA SM90+ kernel for the same chunked
gated-delta-rule math we run in `gated_delta.rs` / `gated_delta_net.rs`. It
claims 2-3× forward speedup over the FLA Triton baseline on Hopper through
three mechanisms (gate-driven CP, algebraic reformulation, warp-specialized
fusion). On Apple GPUs warp specialization has no direct analog, but the
**algebraic reformulation** and the **CP idea** translate.

## Environment

- Host: Apple M3 Max (40 GPU cores, gen 15, Metal 4)
- Important: the existing chunked kernel (`gated_delta_chunked`) is gated on
  `CHUNK_MIN_GPU_GEN >= 17` (M5+) because, on M1–M4, the per-step kernel
  beats it. So *as shipped today, M3 runs prefill through `gated_delta_kernel`
  (the per-step kernel, same code path as decode)*. Part of the research arc
  is: can the chunked kernel — once we apply FlashQLA-style improvements —
  beat the per-step kernel on M3, allowing us to drop the M5 gate?

## Metrics

- **Primary**: `prefill_ms_median` — median across `runs` measurements of
  time from `sendStream` start to first delta event (~prefill cost).
  Lower is better.
- **Correctness gate**: `max_abs_diff` — 0.0 if the 8-token greedy
  fingerprint matches the baseline, 1.0 otherwise. **A run with
  `max_abs_diff != 0` is rejected, regardless of latency.**
- **Secondary** (logged but not selected on):
  - `prefill_ms_min`, `prefill_ms_max`
  - `peak_mem_gb`
  - `build_seconds`

## How to Run

```bash
./autoresearch.sh
```

The driver:
1. Reads `autoresearch.config.json`.
2. Runs `yarn build:native` unless `skipBuild=true`. Incremental rebuilds
   triggered by kernel edits are ~70 s.
3. Runs `scripts/bench-gdn-prefill.ts` against `Qwen3.5-4B-mlx`. Warms up
   `warmup` times then takes `runs` measurements.
4. Compares the 8-token fingerprint to
   `.cache/autoresearch-flashqla-baseline.json`. Fails the run on mismatch.

Per-experiment cost ≈ 70 s (build) + 20 s (load) + ~5 × prefill_ms ≈ 90–110 s
for the default 1024-token prompt. Most of that is rebuild; if we only
tweak `autoresearch.config.json` knobs (no kernel changes) set
`skipBuild: true`.

## Files in Scope

- `autoresearch.config.json` — promptTokens, runs, warmup, plus a notes
  field the agent edits each iteration.
- `autoresearch.sh`, `scripts/bench-gdn-prefill.ts` — driver and bench.
  Edit only if the experiment surface needs to grow (e.g. multi-shape
  sweep).
- `crates/mlx-sys/src/metal/gated_delta_chunked.metal.inc` — chunked prefill
  Metal kernel (BT=32, 5-phase). **Main target for FlashQLA ports.**
- `crates/mlx-sys/src/metal/gated_delta_step{,_vec,_mask,_vec_mask}.metal.inc` —
  per-step kernels (drive prefill on M3 today). Secondary target.
- `crates/mlx-sys/src/metal/gated_delta_fused_gating.metal.inc` — beta/g
  fusion. Probably already tight.
- `crates/mlx-sys/src/mlx_gated_delta.cpp` — kernel launch parameters
  (BT, DV_PER_TG, grid/threadgroup tuples). Many tunings live here.
- `crates/mlx-core/src/models/qwen3_5/gated_delta.rs` — Rust dispatcher
  with the `CHUNK_THRESHOLD` and `CHUNK_MIN_GPU_GEN` gates.

## Off Limits

- The bench / driver / config schema — the harness must remain reproducible
  for cross-experiment comparisons. Add knobs, don't change the metric
  definition.
- The `Qwen3.5-4B-mlx` checkpoint — read-only.
- The submodule `crates/mlx-sys/mlx` — pinned, don't bump.
- Anything outside the Qwen3.5 GDN code path (other models, training,
  server, tokenizer). One model, one feature.

## Constraints

- All Rust code changes must pass `cargo clippy --all -- -D warnings` and
  `cargo fmt --check` (project memory: "Clippy before commit").
- C++ / Metal kernel changes need a `rm -rf target/release/build/mlx-sys-*`
  if the file count changed (cc-crate caching, see CLAUDE.md).
- A kernel change requires a full `yarn build:native` (~70 s incremental).
- Correctness is non-negotiable: a 2× speedup with wrong output is rejected.
- Single GPU; runs are serial. Do not start an experiment while another
  GPU job is in flight on the host.

## Search Surface

Priority-ordered. Items at the top have the best (estimated) gain/cost ratio.
See `autoresearch.ideas.md` for the full ideas catalogue with FlashQLA
citations.

1. **Cache `exp(gcum[i])` in threadgroup memory** (chunked kernel).
   Phase 3 currently calls `exp(gcum[i] - gcum[j])` inside the inner BT×BT
   loop — that's O(BT²) SFU calls per chunk. Precomputing `decay[i] =
   exp(gcum[i])` once and using `decay[i] / decay[j]` (1 FMA + 1 div) is
   much cheaper. Same idea in Phase 4. Trivial to try; ~30 LOC change.

2. **Enable chunked on M3 with optimized version**. Re-bench chunked-vs-
   per-step after the optimizations land. If chunked wins, drop the
   `CHUNK_MIN_GPU_GEN >= 17` gate (or lower the threshold).

3. **Try `BT = 64`**. Matches FlashQLA's chunk size. Doubles arithmetic
   intensity (more reuse of state registers) but quadruples `kk_dot`
   threadgroup memory (1024 → 4096 floats = 16 KB). Threadgroup memory
   budget on Apple GPUs is 32 KB, so feasible. Need to also bump
   `DV_PER_TG` for occupancy.

4. **Drop redundant `threadgroup_barrier` in Phase 4 loop**. Each
   chunk's Phase 4 has a barrier inside the i-loop — but only `q_buf` is
   reused across i, and only the first BK threads write it. Audit whether
   the barrier is actually needed every iteration.

5. **WY-precomputation kernel** (FlashQLA's `kkt_solve` analog). Split
   the dense chunked kernel into (i) a small kernel that computes
   `A = (I + strict_lower(K_beta @ K^T * decay))^{-1}` per chunk, then
   (ii) the main kernel becomes purely parallel — no sequential
   forward-substitution. Higher risk, higher payoff.

6. **Gate-driven early termination**. If `gcum[i] - gcum[chunk_end] <
   -10`, the contribution of timestep i to the chunk-end state is
   numerically zero (< 4.5e-5 in fp32). Skip those terms in Phase 5's
   state-update accumulation. Most useful for long-context prefill.

7. **Per-step kernel BK tiling**. The per-step kernel uses
   `n_per_t = Dk/32 = 4` for Dk=128. On M3 a wider SIMD (32 lanes ×
   4 floats = 128) saturates one SIMD; could try processing 2 v-columns
   per simdgroup to share key loads.

8. **Algebraic reformulation of `kk_dot * decay`**. Fold the `decay_mask`
   into `k_chunk` once at load time (so `k_chunk[t,d] *= sqrt(exp(gcum[t]))`)
   — then `kk_dot[i,j] = sum_d k'[i,d] * k'[j,d]` is the decayed dot
   directly. Removes per-(i,j) `exp(gcum[i] - gcum[j])` multiplications
   from Phases 2 and 3.

## Decision rules

- **Adopt** an experiment when: `prefill_ms_median` improves by ≥3% over
  the current best AND `max_abs_diff == 0`. Lock the change into the
  kernels and update the baseline best.
- **Discard** otherwise: revert the kernel/config change, log the result
  with the hypothesis, move to the next idea.
- **Investigate** when correctness fails or latency is unchanged but you
  suspect a measurement issue — capture extra signals (per-run latencies,
  Metal counters via `MTL_CAPTURE_ENABLED=1`) before declaring win/loss.
