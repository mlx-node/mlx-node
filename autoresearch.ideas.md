# Autoresearch ideas — FlashQLA → Metal port

Catalogue of optimization ideas distilled from
[QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA) (commit at /tmp/FlashQLA
on 2026-05-16) that may translate to our Apple Silicon Metal GDN kernels.
Each entry: the idea, its source, expected gain, and complexity.

References within FlashQLA:
- `flash_qla/ops/gated_delta_rule/chunk/__init__.py` — high-level orchestration
- `flash_qla/ops/gated_delta_rule/chunk/cp_context.py` — gate-driven CP preprocess
- `flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py` — fused forward
- `flash_qla/ops/gated_delta_rule/chunk/hopper/kkt_solve.py` — WY transform precompute
- `flash_qla/ops/gated_delta_rule/chunk/hopper/prepare_h.py` — state buildup
- `tests/ref_gdr.py` — clean PyTorch reference (the math, no tricks)

References within mlx-node:
- `crates/mlx-sys/src/metal/gated_delta_chunked.metal.inc` — our chunked kernel
- `crates/mlx-sys/src/metal/gated_delta_step.metal.inc` — our per-step kernel
- `crates/mlx-core/src/models/qwen3_5/gated_delta.rs` — Rust dispatch

---

## A. Cheap algebraic wins (try first)

### A1. Cache `exp(gcum[i])` in threadgroup memory
**Source**: FlashQLA `fused_fwd.py` precomputes `g.exp()` once per chunk.
**Where**: `gated_delta_chunked.metal.inc` Phases 3 and 4 currently call
`exp(gcum[i] - gcum[j])` inside the BT×BT inner loops — O(BT²) SFU calls
per chunk, per head, per batch.
**Idea**: After Phase 1b's prefix-sum, add a step that computes
`decay[i] = exp(gcum[i])` for i in [0..BT). Use it as `decay[i] / decay[j]`
(or `decay[i] * rdecay[j]` if we also store `rdecay = exp(-gcum)`).
**Gain estimate**: -O(BT²) × (cost_of_exp - cost_of_div) per chunk.
On M3, SFU `exp` is ~16 cycles; div is ~8. With BT=32 and 64 chunks
across 8B params worth of layers, this is meaningful.
**Cost**: ~30 LOC, low risk.

### A2. Fold decay into k_chunk at load time
**Source**: FlashQLA `fused_fwd.py` scales k by `exp(g)` before the inner
matmul: `k_beta = k * beta * exp(g)`.
**Where**: Same kernel, Phase 1 load.
**Idea**: At load time, store `k_chunk[t,d] = k[t,d] * sqrt(exp(gcum[t]))`.
Then `kk_dot[i,j] = sum_d k_chunk[i,d] * k_chunk[j,d]` already incorporates
the `exp((gcum[i] + gcum[j])/2)` factor. Combined with A1 this removes
the per-(i,j) exp from Phase 2 entirely.
**Risk**: numerical — sqrt of small numbers can underflow. Alt: scale only
one side and absorb the asymmetry. Validate with parity test.
**Gain**: removes another O(BT²) exp.

### A3. Reduce barriers in Phase 4
**Where**: Phase 4 (output computation), `gated_delta_chunked.metal.inc`
lines 142–177.
**Idea**: The `threadgroup_barrier` at the bottom of the `for i` loop
guards `q_buf` — but `q_buf` is written only by threads 0..BK-1 and read
by all. The cooperative load writes BK elements then `simd_sum` reduces.
A more careful audit might show one barrier per chunk suffices instead
of per token.
**Gain**: small, but barriers are surprisingly expensive on Apple GPUs.

### A4. Compute `1/decay` (`rdecay`) once per chunk for state update
**Where**: Phase 5 (state update). Currently:
```
s_new = s[p] * exp(g_total);
for (t) s_new += k_chunk[t,d] * delta[t] * exp(g_total - gcum[t]);
```
This is BT exp() calls per d per chunk.
**Idea**: With A1 in place, `exp(g_total - gcum[t]) = decay_total /
decay[t]`. One div per t inside the loop instead of one exp.

---

## B. Algorithmic restructurings (medium risk)

### B1. WY-transform precomputation (FlashQLA `kkt_solve`)
**Source**: FlashQLA factors out the lower-triangular solve into a
dedicated kernel — see `tests/ref_gdr.py::torch_solve`.
**Where**: Currently embedded in `gated_delta_chunked.metal.inc` Phase 3
(`for i { ... for j < i { ... } }` forward substitution). This is the
serial bottleneck.
**Idea**: Run a separate small Metal kernel before the main pass that
computes `A = (I + strict_lower(L * decay))^{-1}` per chunk, where
`L[i,j] = beta[i] * <k[i], k[j]>`. Then the main kernel uses A directly:
`delta = beta * (v - kv_mem) + A @ partial_corrections`.
**Gain**: high — eliminates O(BT) sequential dependency.
**Cost**: significant refactor, parity test critical.

### B2. Separate `prepare_h` kernel
**Source**: FlashQLA splits state buildup from output computation
(`prepare_h.py`).
**Where**: Currently Phase 5 in our chunked kernel does both intra-chunk
output and state update.
**Idea**: Pre-build `h[i]` (state at chunk start) for all i in a single
kernel, then a separate kernel computes outputs in parallel across
chunks.
**Trade-off**: more global memory traffic (writing h between kernels) vs
more parallelism. Probably a loss on bandwidth-bound Apple GPU; tested
last.

### B3. Gate-driven early termination
**Source**: FlashQLA's CP module thresholds chunks at `g < -10` because
exp(-10) ≈ 4.5e-5 is below bf16 epsilon.
**Where**: Per-step kernel state update; chunked kernel Phase 5 accumulator.
**Idea**: When `gcum[i] - gcum[chunk_end] < -10`, skip term i in the
state update (or the entire chunk's contribution to far-future steps).
**Gain**: high on long-context prefill where many tokens contribute
negligibly. Especially relevant since Qwen3.5 has long context targets.

---

## C. Tile / launch-parameter tuning (cheap to try)

### C1. `BT = 64`
**Source**: FlashQLA uses chunk_size=64. Our kernel uses BT=32.
**Cost**: `kk_dot` grows from 32×32×4=4 KB to 64×64×4=16 KB; `k_chunk`
from 32×128×4=16 KB to 64×128×4=32 KB. Apple GPU threadgroup memory
budget is 32 KB on most variants — feasible but tight. May force
DV_PER_TG=2.
**Benefit**: better arithmetic intensity, fewer chunks per sequence.

### C2. `DV_PER_TG` sweep
**Where**: `mlx_gated_delta.cpp:182` — `DV_PER_TG = min(4, Dv)`.
**Idea**: Try 2, 4, 8. Each value trades occupancy (more TGs in flight)
vs reuse of `k_chunk` and `kk_dot` (more reuse with larger DV_PER_TG).
**Cost**: 1 LOC + bench.

### C3. Threadgroup grid layout
**Where**: Grid is `(32, Dv, B*Hv)` with TG `(32, DV_PER_TG, 1)`.
**Idea**: For B=1 and large S, B*Hv may not saturate 40 SMs. Try
splitting along S into a `cp_chunks` outer dim — same insight as B3 but
expressed as launch parallelism.

### C4. Drop M5 gate after A/B wins
**Where**: `crates/mlx-core/src/models/qwen3_5/gated_delta.rs:15`,
`CHUNK_MIN_GPU_GEN: i32 = 17`.
**Idea**: After optimizations land, re-bench chunked-vs-per-step on M3
(gen 15). If chunked wins, lower the gate (or remove entirely).
**Cost**: 1 LOC + verification on M3 hardware.

---

## D. Per-step kernel (used during prefill on M3 today)

### D1. Process 2 v-columns per simdgroup
**Source**: classic register-blocking pattern.
**Where**: `gated_delta_step.metal.inc` — each simdgroup currently handles
one v-column. Each lane holds n_per_t=4 state values.
**Idea**: A lane holds n_per_t=4 state values for v_col_A AND another 4
for v_col_B. Shares the `k_*` load and the `kv_mem` simd_sum across two
columns.
**Cost**: ~50 LOC; risk: register pressure.

### D2. Pre-reduce mask
**Where**: The masked variants `gated_delta_step_mask.metal.inc`.
**Idea**: For prefill, mask is monotone (no holes — just a left-aligned
length). A single `mask_len` int beats per-timestep mask loads.

---

## E. Out-of-scope but noted

- **Warp specialization** (FlashQLA innovation 3): no analog on Apple
  GPUs. Skip.
- **TileLang DSL**: not available; we hand-write Metal. Skip.
- **Tensor Cores**: only M5+ has the equivalent (Neural Accelerators).
  Already exploited by chunked kernel via simd_sum. Anything more is M5+
  specific.

## Experimental order recommendation

1. **Baseline** (no changes) — establish numbers.
2. **A1** (`exp(gcum[i])` cache) — quick win, 30 LOC.
3. **A4** (`1/decay` for state update) — extends A1.
4. **A2** (fold decay into k_chunk) — depends on parity check.
5. **C2** (`DV_PER_TG` sweep) — 1 LOC each.
6. **C1** (`BT = 64`) — bigger change, more impactful.
7. **C4** (drop M5 gate) — only after chunked beats per-step on M3.
8. **B3** (early termination) — long-context wins.
9. **B1** (WY precompute) — high payoff, high risk; tackle last in the
   "easy wins" tranche.
