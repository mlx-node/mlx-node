# NAX research arc 2026-08 — worklog & banked levers

Host: Apple **M5 Max, gen 17** (`applegpu_g17s`, 's'-class), macOS 26.x, MLX fork `nax-macos-26-0-floor`, submodule pin `fef8890f` (perf/qmm-splitk-nax era, 15 commits ahead of upstream base `6c0ea7fb`).
All work ran isolated in worktree `nax-experiments` (2026-08-06 → 08-08), adversarially reviewed and A/B-validated on real checkpoints. **The worktree and all code were removed on 2026-08-08 by owner decision; this file is the surviving record.** Appendices A–C preserve the dispatch diffs and the A/B harness verbatim; the three test suites are preserved as designs (§7).

---

## TL;DR

| #   | finding                                                                  | verdict                                                                                                                                                                                                                                                                                                           |
| --- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Opt A — quantized split-K→NAX reroute** (`qmm_splitk` at `split_k<=2`) | VALIDATED: +2.68% ttft@32 gemma4-26B, +2.31% ornith-35B; no regressions; decode untouched (never reaches `qmm_splitk`)                                                                                                                                                                                            |
| 2   | **Opt B — masked D=256 fused NAX SDPA** (bool-mask sliding layers)       | VALIDATED: kernel 1.22–1.28× vs unfused, +2.63% gemma4 6k-tok TTFT (needs `MLX_PAGED_PREFILL_CHUNK_SIZE=2048`), ~0.4–0.6 GB less transient/sliding layer                                                                                                                                                          |
| 3   | **`mlx agent` default-path audit + measurements**                        | warm turns are append-prefill-bound (sliding ladder healthy, replay_delta=0); **T=0 sampling = +3.05% decode REAL** on the 26B default; thinking-clamp lever REFUTED (`reasoningTokens` always 0); warm-TTFT floor 230–360 ms with a small-append prefill inversion (190 tok slower than 543 tok — needs a sweep) |
| 4   | **flashqla closure v2** (chunked WY-UT GDN on NAX)                       | RE-CLOSED with corrected data: chunked loses to **serialization/launch latency, not GEMM speed**. The June closure's premises were wrong (geometry K=192 vs real 128; 5.5 TF floor vs 7–16 TF today) but the verdict stands                                                                                       |

The June "K∈[128,256) NAX GEMM garbage band" is **empirically FIXED** on this pin (K=64/128/192 all bf16-rounding-clean), despite `gemm_nax.h`/`nax.h` being byte-identical since the garbage era — the fix came from the toolchain/metallib side, not kernel source. In-tree `test_support.rs` canary docs are stale and should be re-run before being believed.

---

## 1. Opt A — quantized split-K → NAX reroute

**Problem**: `qmm_splitk` (chosen when `transpose && B==1 && M >= vector_limit` and the split-K heuristic fires) had NO NAX branch — short incremental-turn prefill (tens of tokens) ran all projections on classic simdgroup.

**Design refuted by data**: the first patch gated on `M >= MIN_M`. A 45-point sweep showed the crossover is the **split factor, not M**: NAX wins at `split_k==2` (+5–36% micro, uniform across M 12–128), classic split-K wins at `split_k>=4` (0.79–0.90×), at ALL M.

**Final gate** (in `qmm_splitk`, after the split-K math — full diff in Appendix A):
reroute to plain `qmm` iff `split_k <= MLX_QMM_SPLITK_NAX_MAX_SK` (default **2**) && `MLX_ENABLE_QMM_SPLITK_NAX` (default **1**) && `!is_kquant_mode(mode)` (K-quants held out, unmeasured) && `qmm_t_prefers_nax(x,K,mode)` (= `is_nax_available && nax_supports_mode(QmmT) && K%64==0 && (tf32 || !f32)`). Debug trace: `MLX_QMM_SPLITK_NAX_DEBUG=1` prints `[splitk-dbg]` per dispatch.

**split_k formula** (M5 's'-class): `sk = max(1, 512/(ceil(N/32)*ceil(M/32)))`, clamp `min(sk, K/k_align)` with `k_align=max(gs,32)`, decrement while `K%(sk*k_align)!=0`. `vector_limit` = 18 (K,N≤2048), 12 (≤4096), 10 — 's' falls to the `default:` arm in `get_qmv_batch_limit`; an Ultra ('d') is 32/18/12 and shifts every band.

**E2E (paired, prewarmed, hardened verdict gate)**: gemma4-26B mxfp4 ttft@32 **+2.68% REAL**, ornith-35B **+2.31% REAL**; qwen3.6-27B / lfm2.5-8B flat (their shapes exit the sk≤2 band early); long-prefill and decode: no regression (a −8.9% two-pair decode scare re-adjudicated to +0.12% inconsistent over 4 pairs; `MLX_QMM_SPLITK_NAX_DEBUG` proved decode never reaches `qmm_splitk` — only M≈112 prefill lines appear).

**Coverage on the 26B agent default** (FLOP-weighted share of 2-D quantized MACs rerouted, by append size M): peak **74.6% at M∈[33,64]**; 39% at 65–96; ~0 at M≥257 (already-NAX). Routed experts (`gather_qmm`, 46.5% of quantized MACs) and the bf16 tied lm_head are outside `qmm_splitk` entirely.

## 2. Opt B — masked D=256 fused NAX SDPA

**Problem**: gemma4 sliding layers (D=256, 25/30 layers on the 26B) always pass a bool mask → the fused NAX SDPA (fork `7e61cdee`, causal-only) declined them → unfused graph with materialized scores (hundreds of MB per layer at long qL).

**Discovery**: the `bd256 wn2 maskbool` kernel variants were ALREADY in the shipped metallib — this was a pure dispatcher change. Gate `d256_masked_sdpa_would_use`: `has_bool_mask && qD==vD==256 && qL>8 && kL>=qL && qL%64==0 && kL%32==0 && qL>=1024`; kill switch `MLX_ENABLE_D256_MASKED_SDPA` (default 1). The alignment gate exists because the padding path can't host a mask (AttnParams qL/kL are the mask bound). wn=2 mask addressing has no d_half term (verified against the kernel source). A probe FFI `mlx_metal_d256_masked_sdpa_would_use` (Appendix B) lets Rust tests assert routing without dispatch tracing.

**Numbers**: kernel 1.22–1.28× vs unfused (3.3–3.5 ms vs 4.2 ms at bench shapes) + ~400–580 MB less transient per sliding layer; e2e gemma4-26B 6k-token TTFT **+2.63% REAL** — but ONLY with `MLX_PAGED_PREFILL_CHUNK_SIZE=2048` (default paged chunks are ≤512 → qL never reaches 1024; flat gemma4 hardcodes `GEMMA4_PREFILL_STEP_SIZE=512` — an untouched follow-up lever). `mlx agent` already sets chunk 2048, so the route fires on session-start cold prefill and on the first chunk of ≥2048-token appends.

**Liveness oracle** (mutation-tested): with the route disabled the output is bit-identical to a manual unfused reference; enabled, it differs (reduction order) while staying within ~80× tolerance headroom — non-bit-identity IS the "kernel ran" signal.

## 3. `mlx agent` default path — audit + measurements

Real default on this machine = `Gemma-4-26B-A4B-Unsloth-MXFP4-mlx` via `~/.mlx-node/agent/settings.json` (catalog `isDefault` only orders the first-run wizard). Paged cache required; gemma4 `draft/` hidden (speculative executor is flat-cache-only ⇒ structurally exclusive with paged prefix reuse); sampling preset T=0.7/topP.95/topK64, `maxNewTokens` 16384; full-history jinja re-render + retokenize every turn (the token-splice fast path is Continue-only; agent always sends Start).

Measured (traced live session, 4 warm turns):

- **Sliding-rung ladder healthy**: warm turns all `prefix_checkpoint` with `replay_delta_tokens=0` — pure append-prefill. Per-turn `cold_restore_declined reason=no_backed_boundary` lines are benign. Cross-process SSD restore: 4096/7466 tokens → TTFT 1270 ms vs 2019 cold.
- **Warm TTFT floor 230–360 ms regardless of append size**; single-sample inversion: 190-tok append prefill 353 ms (538 tok/s) vs 543-tok 228 ms (2381 tok/s) — fixed per-turn cost (O(context) KV gather + underfilled GEMMs + full re-render). Needs an append-size sweep before acting.
- **T=0 vs agent preset decode A/B: +3.05% REAL** (89.1→91.9 tok/s, 4/4 pairs +2.9–3.5%, clears strict control hurdle 2.16%). The cost is compiled top-k/top-p over the 262,144-entry vocab every step; T=0 collapses to argmax. Product tradeoff open (T=0 + repetitionPenalty 1.0 + cutoffs-off = loop risk).
- **Thinking lever REFUTED**: `reasoningTokens=0` on every turn including a pure-reasoning no-tools prompt — gemma4 emits no separate thinking stream in the agent, so the medium→high clamp costs nothing.
- `mtp_cycles=0` (draft hidden, expected). Decode 67–74 tok/s at 7–9k ctx vs 89–92 at tiny ctx.
- Open decode-side structural levers: bf16 tied lm_head `[262144,2816]` ≈ 1.475 GB read per decode step (quantize-to-q8 prior art exists: PR #77); NAX for `gather_qmm` experts.

## 4. flashqla closure v2 (chunked WY-UT GDN)

June-2026 closure premises re-litigated and found wrong — verdict re-derived and **upheld** on corrected data:

- **Geometry**: closure said K=Dk=192 "pinned by head geometry". Every shipped qwen3.5 checkpoint (0.8B/4B/27B/35B-A3B) has **Dk=Dv=128, Hk=16** (75/75 configs). `config.rs` _defaults_ to 192 — the likely source of the June harnesses' error.
- **NAX correctness**: the K∈[128,256) garbage band is gone (measured, both arms, full-mantissa inputs).
- **GEMM floor**: 7–16 TF at the true batched WY shapes (vs June's 5.5 TF) — but **NAX vs simdgroup is a wash at the big batch-2048 rows (0.93–1.11×, bandwidth-bound)**; real NAX wins only on small-batch bf16 state rows (1.38–1.69×). The ops path's internal f32 cast alone halves throughput.
- **The actual killer, measured**: modeled floor **332 ms faithful / 170 ms optimistic** vs today's per-step bar **272–386 ms** (2.77 µs/tok warm; June: 293–483). Split: **63-step serial tri-solve = 71%** (9.88 ms/layer faithful; 70.6 µs/serial step; the full-M `where_` scatter doubles the 4.45 ms gemv-only chain), chunk-serial state carry 16% (2.24 ms), ALL GEMM terms **12%** (41 ms/stack). Scalar chunked Metal kernel re-measured 15–16× slower than per-step (PR #68 confirmed).
- **The one open door (not pursued)**: batched binary-doubling solve (6 batched [2048,64,64] matmuls ≈ 0.1 ms each) ⇒ modeled floor ~110–118 ms ⇒ recurrence ceiling 2.3–3.5× ⇒ **≤ +10–15% TTFT** at GDN's 28% Amdahl share — IF the June 5× floor-to-reality overhead gap doesn't reappear. Requires implementing the solve before any A/B can break even.

## 5. Methodology bank (hard-won, reusable)

- **Kernel-identity oracle**: FULL-MANTISSA f32 activations; tf32 gap >5e-5 ⇔ tensor op ran, ~0 ⇔ simdgroup. 8-bit dyadic test data is tf32-EXACT and blinds the oracle.
- **Stale-build trap**: cargo `rerun-if-changed=mlx` MISSES submodule `.cpp` edits → `touch crates/mlx-sys/build.rs`, then verify dispatch changes BEHAVIORALLY (env-gated fprintf, throughput delta), never by compile success. Cargo GPU tests load metallibs from the exe dir: `cp packages/core/{mlx,paged_attn}.metallib target/aarch64-apple-darwin/release/deps/` — stale/missing metallib = garbage outputs, not errors.
- **26–35B paired TTFT benches REQUIRE page-cache prewarm** (`cat *.safetensors >/dev/null` per model): without it control pairs on identical code swing ±50% (cold P4510 mmap); with it ±1.5%.
- **Paired A/B protocol** (Appendix C): same-binary env-toggle two-process arms back-to-back, median of per-pair ratios, alternating order, control pairs on identical code; verdict requires signal > max(MAD band floored at 1.5%, worst control deviation) AND ≥75% sign consistency AND ≥2 pairs.
- **DVFS/thermal**: a 3.3× cold-start clock ramp poisons the first ~1.5 s — bake in a burn-in; repeat the first shape at the end as a drift anchor (23–31% anchor drift = a heat-soaked arm, rerun cooled); marginal-cost timing `(t_long−t_short)/(ops_long−ops_short)` cancels per-op encode overhead.
- **MLX encode floor**: single small ops cap at ~1.3–1.9 µs each (~7–8 TF for a 12.6-MFLOP op) — batch small GEMMs or the kernel speed is unobservable.
- Build env for cargo GPU tests on this box: `PATH=/usr/bin:$PATH`, `SDKROOT=$(xcrun --show-sdk-path)`, `MACOSX_DEPLOYMENT_TARGET=26.0`, `RUSTFLAGS="-C link-arg=/Library/Developer/CommandLineTools/usr/lib/clang/21/lib/darwin/libclang_rt.osx.a"` (compiler-rt for `___isPlatformVersionAtLeast`).

## 6. Env-var inventory introduced/used

| var                           | default | meaning                                                                              |
| ----------------------------- | ------- | ------------------------------------------------------------------------------------ |
| `MLX_ENABLE_QMM_SPLITK_NAX`   | 1       | Opt A master switch                                                                  |
| `MLX_QMM_SPLITK_NAX_MAX_SK`   | 2       | reroute only when split_k ≤ this                                                     |
| `MLX_QMM_SPLITK_NAX_DEBUG`    | off     | `[splitk-dbg]` per-dispatch trace                                                    |
| `MLX_ENABLE_D256_MASKED_SDPA` | 1       | Opt B master switch                                                                  |
| `MLX_DISABLE_GEMM_NAX`        | 0       | dense-GEMM NAX kill switch (read per dispatch; added for the flashqla re-litigation) |
| `MLX_GDN_KERNEL`              | auto    | `chunked` = scalar Metal chunked kernel; `chunked_ops` parses but is non-Metal-only  |
| `MLX_AB_SAMPLING`             | unset   | harness arm selector: `agent-gemma4` = T0.7/topP.95/topK64, unset = greedy           |

## 7. Test-suite designs (code removed; recreate from these)

- **`qmm_splitk_nax_band.rs`** (845 lines): parity + sweep for Opt A. Sweep: M_GRID [8,12,16,24,32,48,64,96,128] × 8 combos — C1 mxfp8 8192/2048, C2 mxfp8 2048/4096, C3 mxfp4 5120/17408, C4 mxfp4 2112/2816, C5 mxfp4 512/2048, C6 affine4/gs64 2048/4096, C7 nvfp4/gs16 2048/4096, C8 mxfp8+f16-act 2048/4096; cross-process anchors (A/B/AA) as drift tells; `reroute_active(m,n,k,gs)` mirrors the C++ split-K math to predict which cells reroute; parity at rerouting points per mode vs the toggle-off arm, full-mantissa activations (`(lcg as f32/u32::MAX)*2−1`). Two-process A/B required — the C++ gates are read-once statics.
- **`gemma4_d256_masked_sdpa.rs`** (418 lines): replicates `create_sliding_mask` exactly; parity vs a manual unfused SDPA reference; checkerboard masks; probe-decline cases (each gate condition violated singly); rollback child-process test (`MLX_ENABLE_D256_MASKED_SDPA=0` ⇒ bit-identical to reference = liveness oracle); wall-clock bench.
- **`gdn_nax_shapes.rs`** (1323 lines): dense-GEMM NAX correctness + marginal-cost bench at GDN shapes; `gdn_wy_batched_sweep` (batched WY rows incl. serial-chain measurements for the tri-solve and state carries); `gdn_perstep_bar` (drives pub `gated_delta_update` at real 4B call-site shapes, 2×2048-tok chunks with threaded state); 1.5 s burn-in + drift anchor + in-process `MLX_DISABLE_GEMM_NAX` flips with `mlx_synchronize()` before each flip.

## 8. Resurrection guide

Appendix A applies onto MLX fork pin `fef8890f` (branch `perf/qmm-splitk-nax` era, on `nax-macos-26-0-floor` base): `git -C crates/mlx-sys/mlx apply` the diff, `touch crates/mlx-sys/build.rs`, `yarn build:native`. Appendix B applies onto the superproject at `6eb2b040`-era main. Appendix C files go to `examples/`. Re-run order: qmm parity → sweep (cooled) → prewarmed paired e2e per model. Note `crates/mlx-core/index.d.cts` is hand-synced with `packages/core/index.d.cts` — the Appendix B index.d.cts hunk (removal of a stale block) must be mirrored if resurrected.

---

## Appendix A — MLX submodule diff (Opt A reroute + Opt B masked-D256 SDPA + MLX_DISABLE_GEMM_NAX)

Apply with `git -C crates/mlx-sys/mlx apply` on pin `fef8890f`.

```diff
diff --git a/mlx/backend/cuda/scaled_dot_product_attention.cpp b/mlx/backend/cuda/scaled_dot_product_attention.cpp
index ca411e91..7ba724e6 100644
--- a/mlx/backend/cuda/scaled_dot_product_attention.cpp
+++ b/mlx/backend/cuda/scaled_dot_product_attention.cpp
@@ -555,6 +555,7 @@ bool ScaledDotProductAttention::use_fallback(
     const array& v,
     bool has_mask,
     bool has_arr_mask,
+    bool has_bool_mask,
     bool do_causal,
     bool is_training,
     bool output_logsumexp,
diff --git a/mlx/backend/metal/matmul.cpp b/mlx/backend/metal/matmul.cpp
index ec3cb10c..a194729c 100644
--- a/mlx/backend/metal/matmul.cpp
+++ b/mlx/backend/metal/matmul.cpp
@@ -914,6 +914,7 @@ void steel_matmul_axpby(
   int _tk = K / 16;
   int64_t matrix_size = static_cast<int64_t>(M) * N;
   bool use_nax = metal::is_nax_available() &&
+      env::get_var("MLX_DISABLE_GEMM_NAX", 0) == 0 &&
       !issubdtype(a.dtype(), complexfloating) &&
       (env::enable_tf32() || a.dtype() != float32);
   char devc = d.get_architecture().back();
diff --git a/mlx/backend/metal/quantized.cpp b/mlx/backend/metal/quantized.cpp
index 17726635..062ff11b 100644
--- a/mlx/backend/metal/quantized.cpp
+++ b/mlx/backend/metal/quantized.cpp
@@ -70,6 +70,38 @@ bool nax_supports_mode(const std::string& mode, NaxPath path) {
   return path == NaxPath::QmmT;
 }

+// Whether a transposed qmm on this input would take the NAX tensor-op branch.
+// One predicate shared by `qmm` and the split-K reroute in `qmm_splitk`, so the
+// two call sites cannot drift apart: K % 64 == 0 is the NAX kernel's K-tile
+// requirement, and float32 rides along only under tf32 because the tensor op
+// carries a 10-bit mantissa.
+bool qmm_t_prefers_nax(const array& x, int K, const std::string& mode) {
+  return metal::is_nax_available() && nax_supports_mode(mode, NaxPath::QmmT) &&
+      (K % 64 == 0) && (env::enable_tf32() || x.dtype() != float32);
+}
+
+// `qmm_splitk` picks split_k > 1 whenever the output tile grid alone is short
+// of ~512 threadgroups, which for a transposed non-batched matmul is every M
+// between the qmv batch limit and roughly 128 at typical Ns. In that band the
+// NAX branch in `qmm` is never reached even when `qmm_t_prefers_nax` holds —
+// the simdgroup `qmm_t_splitk` runs instead. A 45-point sweep on M5 Max
+// (mxfp4/mxfp8, model shapes) put the crossover at the split factor, not M:
+// at split_k == 2 the tensor op wins 5-27% while at split_k >= 4 the deeper
+// K-parallelism beats it by 10-20%, uniformly across M 12..128. So only the
+// marginal split is rerouted. Both knobs are read once so a run's dispatch
+// cannot change mid-process and a two-process A/B
+// (MLX_ENABLE_QMM_SPLITK_NAX=0 against the default) compares stable arms;
+// with the switch off, dispatch is bit-identical to the pre-reroute behavior.
+bool qmm_splitk_nax_enabled() {
+  static bool enabled = env::get_var("MLX_ENABLE_QMM_SPLITK_NAX", 1);
+  return enabled;
+}
+
+int qmm_splitk_nax_max_sk() {
+  static int max_sk = env::get_var("MLX_QMM_SPLITK_NAX_MAX_SK", 2);
+  return max_sk;
+}
+
 const char* nax_quantized_kernel_family(
     std::string_view tag,
     const std::string& mode,
@@ -965,9 +997,7 @@ void qmm(
     metal::Device& d,
     const Stream& s,
     const std::string& mode) {
-  if (metal::is_nax_available() && nax_supports_mode(mode, NaxPath::QmmT) &&
-      transpose && (K % 64 == 0) &&
-      (env::enable_tf32() || x.dtype() != float32)) {
+  if (transpose && qmm_t_prefers_nax(x, K, mode)) {
     return qmm_nax(
         /* const array& x = */ x,
         /* const array& w = */ w,
@@ -1086,7 +1116,33 @@ void qmm_splitk(
   while (split_k > 1 && (K % (split_k * k_align) != 0)) {
     split_k--;
   }
-  if (split_k <= 1) {
+  if (env::get_var("MLX_QMM_SPLITK_NAX_DEBUG", 0)) {
+    fprintf(
+        stderr,
+        "[splitk-dbg] M=%d N=%d K=%d mode=%s sk=%d enabled=%d max_sk=%d "
+        "nax=%d supports=%d k64=%d tf32=%d xf32=%d\n",
+        M,
+        N,
+        K,
+        mode.c_str(),
+        split_k,
+        int(qmm_splitk_nax_enabled()),
+        qmm_splitk_nax_max_sk(),
+        int(metal::is_nax_available()),
+        int(nax_supports_mode(mode, NaxPath::QmmT)),
+        int(K % 64 == 0),
+        int(env::enable_tf32()),
+        int(x.dtype() == float32));
+  }
+  // A marginal split (split_k == 2 by default) goes to `qmm`, whose NAX branch
+  // beats the simdgroup split-K kernel there; deeper splits keep it, which
+  // beats the tensor op. See qmm_splitk_nax_max_sk above for the measurement.
+  // K-quants are held out: the sweep covered the affine and fp families only,
+  // and the ggml modes' small-M NAX behavior is unmeasured — they keep the
+  // pre-reroute dispatch until benchmarked.
+  if (split_k <= 1 ||
+      (qmm_splitk_nax_enabled() && split_k <= qmm_splitk_nax_max_sk() &&
+       !is_kquant_mode(mode) && qmm_t_prefers_nax(x, K, mode))) {
     return qmm(
         x, w, scales, biases, out, true, group_size, bits, M, N, K, d, s, mode);
   }
diff --git a/mlx/backend/metal/scaled_dot_product_attention.cpp b/mlx/backend/metal/scaled_dot_product_attention.cpp
index 697e9a61..a96d84f3 100644
--- a/mlx/backend/metal/scaled_dot_product_attention.cpp
+++ b/mlx/backend/metal/scaled_dot_product_attention.cpp
@@ -23,6 +23,14 @@ bool d256_full_sdpa_enabled() {
   return enabled;
 }

+bool d256_masked_sdpa_enabled() {
+  // Independent kill switch for the bool-array-masked D=256 route. It only
+  // narrows: the masked route also requires d256_full_sdpa_available(), so
+  // MLX_ENABLE_D256_FULL_SDPA=0 disables both routes.
+  static bool enabled = env::get_var("MLX_ENABLE_D256_MASKED_SDPA", 1);
+  return enabled;
+}
+
 void sdpa_full_self_attention_nax(
     const Stream& s,
     metal::Device& d,
@@ -695,12 +703,37 @@ bool d256_full_sdpa_would_use(
       !has_array_mask && d256_full_sdpa_available(effective_dtype_is_float32);
 }

+bool d256_masked_sdpa_would_use(
+    bool effective_dtype_is_float32,
+    int32_t query_head_dim,
+    int32_t value_head_dim,
+    int32_t query_length,
+    int32_t key_length,
+    bool has_bool_mask) {
+  // Bool-array-masked D=256 route (Gemma4 sliding-window prefill). Bool masks
+  // only: the kernel's out-of-bounds `load_safe` zero-fill reads as `false`
+  // (masked out) for bool but as additive `+0` (unmasked!) for float masks, so
+  // additive masks stay on the fallback. The alignment terms are required
+  // because the ragged-input recovery in `sdpa_full_self_attention_nax` pads
+  // qL/kL and cannot host an array mask (the padded AttnParams qL/kL would
+  // become the mask's read bounds, walking past the true mask storage); an
+  // unaligned masked dispatch would take the register-spilling unaligned
+  // pipelines instead. qL >= 1024 mirrors the causal route's measured
+  // break-even (512-row D=256 measured only 1.08x of unfused).
+  return query_head_dim == 256 && value_head_dim == 256 && has_bool_mask &&
+      query_length > 8 && key_length >= query_length &&
+      (query_length % 64 == 0) && (key_length % 32 == 0) &&
+      query_length >= 1024 && d256_masked_sdpa_enabled() &&
+      d256_full_sdpa_available(effective_dtype_is_float32);
+}
+
 bool ScaledDotProductAttention::use_fallback(
     const array& q,
     const array& k,
     const array& v,
     bool has_mask,
     bool has_arr_mask,
+    bool has_bool_mask,
     bool do_causal,
     bool is_training,
     bool output_logsumexp,
@@ -741,7 +774,14 @@ bool ScaledDotProductAttention::use_fallback(
       query_sequence_length,
       key_sequence_length,
       do_causal,
-      has_arr_mask);
+      has_arr_mask) ||
+      d256_masked_sdpa_would_use(
+          q.dtype() == float32,
+          query_head_dim,
+          value_head_dim,
+          query_sequence_length,
+          key_sequence_length,
+          has_arr_mask && has_bool_mask);
   const bool sdpa_full_supported_head_dim = query_head_dim == value_head_dim &&
       (query_head_dim == 64 || query_head_dim == 80 || query_head_dim == 128 ||
        (query_head_dim == 256 && sdpa_full_supported_256));
diff --git a/mlx/backend/no_gpu/primitives.cpp b/mlx/backend/no_gpu/primitives.cpp
index 4819ed27..23f78734 100644
--- a/mlx/backend/no_gpu/primitives.cpp
+++ b/mlx/backend/no_gpu/primitives.cpp
@@ -29,6 +29,7 @@ bool fast::ScaledDotProductAttention::use_fallback(
     const array& v,
     bool has_mask,
     bool has_arr_mask,
+    bool has_bool_mask,
     bool do_causal,
     bool is_training,
     bool output_logsumexp,
diff --git a/mlx/fast.cpp b/mlx/fast.cpp
index 8272be6f..79fa6325 100644
--- a/mlx/fast.cpp
+++ b/mlx/fast.cpp
@@ -831,6 +831,7 @@ array scaled_dot_product_attention(
           v,
           has_mask,
           has_arr_mask,
+          has_bool_mask,
           do_causal,
           is_training,
           output_logsumexp,
diff --git a/mlx/fast_primitives.h b/mlx/fast_primitives.h
index 0d2f8610..fda2fd35 100644
--- a/mlx/fast_primitives.h
+++ b/mlx/fast_primitives.h
@@ -225,6 +225,7 @@ class ScaledDotProductAttention : public Custom {
       const array& v,
       bool has_mask,
       bool has_arr_mask,
+      bool has_bool_mask,
       bool do_causal,
       bool is_training,
       bool output_logsumexp,
```

## Appendix B — superproject diff (D256 probe FFI + index.d.cts sync)

```diff
diff --git a/crates/mlx-sys/src/lib.rs b/crates/mlx-sys/src/lib.rs
index b4e35485..a79458bb 100644
--- a/crates/mlx-sys/src/lib.rs
+++ b/crates/mlx-sys/src/lib.rs
@@ -623,6 +623,18 @@ unsafe extern "C-unwind" {
         has_array_mask: bool,
         out_would_use: *mut bool,
     ) -> i32;
+    /// Evaluate the bool-array-masked D=256 eligibility predicate shared with
+    /// MLX's Metal dispatcher (Gemma4 sliding-window prefill route). Same 0/-1
+    /// fallible-output contract as `mlx_metal_d256_full_sdpa_would_use`.
+    pub fn mlx_metal_d256_masked_sdpa_would_use(
+        effective_dtype_is_float32: bool,
+        query_head_dim: i32,
+        value_head_dim: i32,
+        query_length: i32,
+        key_length: i32,
+        has_bool_mask: bool,
+        out_would_use: *mut bool,
+    ) -> i32;
     pub fn mlx_metal_device_info() -> *const std::os::raw::c_char;
     pub fn mlx_set_wired_limit(limit: u64, out_old_limit: *mut u64) -> i32;
     pub fn mlx_get_peak_memory(out_value: *mut u64) -> i32;
diff --git a/crates/mlx-sys/src/mlx_stream.cpp b/crates/mlx-sys/src/mlx_stream.cpp
index f70ac7c0..40286d2f 100644
--- a/crates/mlx-sys/src/mlx_stream.cpp
+++ b/crates/mlx-sys/src/mlx_stream.cpp
@@ -17,6 +17,13 @@ bool d256_full_sdpa_would_use(
     int32_t key_length,
     bool do_causal,
     bool has_array_mask);
+bool d256_masked_sdpa_would_use(
+    bool effective_dtype_is_float32,
+    int32_t query_head_dim,
+    int32_t value_head_dim,
+    int32_t query_length,
+    int32_t key_length,
+    bool has_bool_mask);
 }  // namespace mlx::core::fast
 #endif

@@ -221,6 +228,46 @@ int32_t mlx_metal_d256_full_sdpa_would_use(
 #endif
 }

+// Bool-array-masked D=256 eligibility probe (Gemma4 sliding-window prefill).
+// Calls the same helper used inside ScaledDotProductAttention::use_fallback;
+// callers still own the dispatcher's outer inference/stream gates. Same 0/-1
+// fallible-output contract as the causal D=256 probe above.
+int32_t mlx_metal_d256_masked_sdpa_would_use(
+    bool effective_dtype_is_float32,
+    int32_t query_head_dim,
+    int32_t value_head_dim,
+    int32_t query_length,
+    int32_t key_length,
+    bool has_bool_mask,
+    bool* out_would_use) {
+  if (out_would_use == nullptr) {
+    return -1;
+  }
+  *out_would_use = false;
+#ifdef MLX_NODE_METAL_ENABLED
+  try {
+    *out_would_use = mlx::core::fast::d256_masked_sdpa_would_use(
+        effective_dtype_is_float32,
+        query_head_dim,
+        value_head_dim,
+        query_length,
+        key_length,
+        has_bool_mask);
+    return 0;
+  } catch (...) {
+    return -1;
+  }
+#else
+  (void)effective_dtype_is_float32;
+  (void)query_head_dim;
+  (void)value_head_dim;
+  (void)query_length;
+  (void)key_length;
+  (void)has_bool_mask;
+  return 0;
+#endif
+}
+
 #ifndef MLX_NODE_METAL_ENABLED
 // CPU-only bridge definitions for the three entry points normally supplied by
 // mlx_paged_profile.cpp. That translation unit includes Metal/Metal.hpp and is
diff --git a/packages/core/index.d.cts b/packages/core/index.d.cts
index 50163cf2..8b83937e 100644
--- a/packages/core/index.d.cts
+++ b/packages/core/index.d.cts
@@ -2234,14 +2234,6 @@ export interface ChatConfig {
    * loop (pure-Rust eager; qwen3.5 dense and MoE). Requires the model
    * checkpoint to carry an MTP head (otherwise silently ignored). Default:
    * `false`.
-   *
-   * The MTP acceptance gate (`MLX_MTP_ACCEPT_GATE`, default ON) also
-   * applies to explicit requests at depth 1: once the aggregated
-   * first-draft acceptance sample is large enough for a 95% confidence
-   * bound to sit below the break-even, the model runs plain AR for
-   * subsequent depth-1 turns. The gate is depth-1-scoped and exempts
-   * adaptive-depth turns (depth > 1 turns are never gated). Set the env
-   * var to `0` to bypass the gate and always run MTP when requested.
    */
   enableMtp?: boolean | undefined;
   /**
```

## Appendix C — A/B harness (verbatim)

### examples/lfm2-perf-ab.ts

```typescript
#!/usr/bin/env node
/**
 * lfm2 perf A/B harness — single-arm measurement primitive.
 *
 * One invocation = one model load + warmup + N measured reps in ONE
 * thermal/process arm. The ARM (baseline vs optimized) is selected by the
 * caller via a `MLX_LFM2_DISABLE_<OPT>` env var, read once at process
 * start on the Rust side (the idiomatic toggle pattern in this repo).
 *
 * The thermally-fair A/B is done by the orchestrator
 * (`examples/lfm2-perf-pair.py`), which launches this script alternately
 * with/without the toggle env, pairs adjacent runs, and takes the median
 * of per-pair ratios (drift-canceling) plus a control band.
 *
 * Metrics come from the native `reportPerformance` path (measured AFTER
 * model load, so load variance does not pollute them).
 *
 * Usage:
 *   [MLX_LFM2_DISABLE_X=1] oxnode examples/lfm2-perf-ab.ts \
 *     --model lfm2.5-1.2b-thinking-mlx --mode ttft|decode \
 *     --prompt-tokens 1500 --max-new 4 --reps 4 --warmup 1 [--emit-text]
 *
 * Output: exactly one line beginning `RESULT_JSON:` followed by JSON.
 */

import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';

import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';

const { values } = parseArgs({
  args: process.argv.slice(2),
  options: {
    model: { type: 'string', default: 'lfm2.5-1.2b-thinking-mlx' },
    mode: { type: 'string', default: 'decode' }, // 'ttft' | 'decode'
    'prompt-tokens': { type: 'string', default: '64' },
    'max-new': { type: 'string', default: '256' },
    reps: { type: 'string', default: '4' },
    warmup: { type: 'string', default: '1' },
    'emit-text': { type: 'boolean', default: false },
  },
});

const modelName = values.model!;
const mode = values.mode!;
const promptTokens = Number.parseInt(values['prompt-tokens']!, 10);
const maxNew = Number.parseInt(values['max-new']!, 10);
const reps = Number.parseInt(values.reps!, 10);
const warmup = Number.parseInt(values.warmup!, 10);
const emitText = values['emit-text']!;

const MODEL_PATH = resolve(process.cwd(), '.cache', 'models', modelName);

const SENT = 'The quick brown fox jumps over the lazy dog beside the quiet river as the evening sun slowly sets. ';
function buildPrompt(nonce: string): string {
  const copies = Math.max(1, Math.ceil(promptTokens / 16));
  return `${nonce}Read the following text and then answer in detail.\n${SENT.repeat(copies)}\nNow write a long continuation.`;
}

function median(xs: number[]): number {
  const f = xs.filter((x) => Number.isFinite(x));
  if (f.length === 0) return Number.NaN;
  const s = [...f].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

const relevantToggles: Record<string, string> = {};
for (const [k, v] of Object.entries(process.env)) {
  if (k.startsWith('MLX_LFM2_') || k === 'MLX_NO_COMPILE' || k === 'MLX_DISABLE_COMPILE' || k === 'MLX_AB_SAMPLING') {
    relevantToggles[k] = v ?? '';
  }
}

// MLX_AB_SAMPLING selects the sampling config for this arm:
//   unset          -> greedy (temperature 0)
//   'agent-gemma4' -> the mlx agent's gemma4 preset (server presets.ts)
const samplingArm = process.env.MLX_AB_SAMPLING ?? '';
const samplingConfig = samplingArm === 'agent-gemma4' ? { temperature: 0.7, topP: 0.95, topK: 64 } : { temperature: 0 };
if (samplingArm && samplingArm !== 'agent-gemma4') {
  throw new Error(`unknown MLX_AB_SAMPLING arm: ${samplingArm}`);
}

const loaded = await loadModel(MODEL_PATH);

async function oneTurn(
  nonce: string,
): Promise<{ ttftMs: number; prefillTps: number; decodeTps: number; text: string }> {
  // Fresh session per turn → turn-1 cold prefill (no warm-continue confound).
  const session = new ChatSession(loaded as unknown as SessionCapableModel, {
    system: 'You are a helpful assistant.',
  });
  const res = await session.send(buildPrompt(nonce), {
    config: { maxNewTokens: maxNew, ...samplingConfig, reportPerformance: true },
  });
  const p = res.performance;
  return {
    ttftMs: p?.ttftMs ?? Number.NaN,
    prefillTps: p?.prefillTokensPerSecond ?? Number.NaN,
    decodeTps: p?.decodeTokensPerSecond ?? Number.NaN,
    text: res.text ?? '',
  };
}

for (let i = 0; i < warmup; i++) await oneTurn(`warmup-${i} `);

const ttftMs: number[] = [];
const prefillTps: number[] = [];
const decodeTps: number[] = [];
let firstText = '';
const hasher = createHash('sha256');

for (let r = 0; r < reps; r++) {
  // ttft: unique nonce per rep → cold prefill (miss the content-addressed
  // prefix cache) so we measure real prefill cost. decode: decodeTps is
  // cache-independent; keep the prompt FIXED so --emit-text is deterministic
  // across arms for byte-identical checks.
  const nonce = mode === 'ttft' ? `rep-${r} ` : '';
  const t = await oneTurn(nonce);
  ttftMs.push(t.ttftMs);
  prefillTps.push(t.prefillTps);
  decodeTps.push(t.decodeTps);
  if (r === 0) firstText = t.text;
  hasher.update(t.text);
}

const out = {
  model: modelName,
  mode,
  promptTokens,
  maxNew,
  reps,
  warmup,
  toggles: relevantToggles,
  ttftMs,
  prefillTps,
  decodeTps,
  medTtftMs: median(ttftMs),
  medPrefillTps: median(prefillTps),
  medDecodeTps: median(decodeTps),
  ...(emitText ? { textHash: hasher.digest('hex'), firstText: firstText.slice(0, 400) } : {}),
};

console.log(`RESULT_JSON:${JSON.stringify(out)}`);
```

### examples/lfm2-perf-pair.py

```python
#!/usr/bin/env python3
"""
lfm2 perf A/B orchestrator — thermally-fair paired measurement + verdict.

Launches the single-arm harness (examples/lfm2-perf-ab.ts) alternately with
and without a MLX_LFM2_DISABLE_<OPT> toggle, BACK TO BACK (adjacent processes
share a thermal window). Each pair yields one unit-free ratio; the median of
per-pair ratios cancels the ~15% cross-run thermal drift seen on M5 Max.

A CONTROL set runs BOTH arms with the toggle SET (identical code path) to
measure the ratio noise floor. An optimization counts as a REAL win only if
its median ratio improvement exceeds the worst control deviation AND the sign
is consistent across pairs.

Ratio convention (>1.0 = optimized faster):
  decode -> medDecodeTps_opt / medDecodeTps_base   (higher tok/s better)
  ttft   -> medTtftMs_base   / medTtftMs_opt        (lower ms better)
           plus prefill -> medPrefillTps_opt / medPrefillTps_base

Usage:
  python3 examples/lfm2-perf-pair.py \
    --model lfm2.5-1.2b-thinking-mlx --mode ttft \
    --toggle MLX_LFM2_DISABLE_LAST_TOKEN_SLICE \
    --prompt-tokens 1500 --max-new 4 --reps 4 --warmup 1 \
    --pairs 5 --control-pairs 3

Prints a human summary then one line `VERDICT_JSON:{...}`.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm2-perf-ab.ts")


def run_arm(args, toggle_set: bool) -> dict:
    env = dict(os.environ)
    if args.toggle:
        if toggle_set:
            env[args.toggle] = args.toggle_value
        else:
            env.pop(args.toggle, None)
    cmd = [
        "oxnode", HARNESS,
        "--model", args.model,
        "--mode", args.mode,
        "--prompt-tokens", str(args.prompt_tokens),
        "--max-new", str(args.max_new),
        "--reps", str(args.reps),
        "--warmup", str(args.warmup),
    ]
    p = subprocess.run(cmd, env=env, cwd=os.getcwd(), capture_output=True, text=True, timeout=args.timeout)
    if p.returncode != 0:
        sys.stderr.write(f"[arm toggle={toggle_set}] exit {p.returncode}. stderr tail:\n{p.stderr[-1500:]}\n")
        raise RuntimeError(f"harness exited nonzero ({p.returncode}); arm discarded")
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT_JSON:")), None)
    if line is None:
        sys.stderr.write(f"[arm toggle={toggle_set}] no RESULT_JSON. stderr tail:\n{p.stderr[-1500:]}\n")
        raise RuntimeError("harness produced no RESULT_JSON")
    return json.loads(line[len("RESULT_JSON:"):])


def metric(args, r: dict) -> float:
    return r["medTtftMs"] if args.mode == "ttft" else r["medDecodeTps"]


def ratio(args, base: dict, opt: dict) -> float:
    if args.mode == "ttft":
        return metric(args, base) / metric(args, opt)  # lower ms better -> invert
    return metric(args, opt) / metric(args, base)        # higher tok/s better


def prefill_ratio(base: dict, opt: dict):
    b, o = base.get("medPrefillTps"), opt.get("medPrefillTps")
    if b and o and b > 0:
        return o / b
    return None


def collect(args, pairs: int, control: bool, label: str):
    ratios, pref_ratios = [], []
    for i in range(pairs):
        # alternate order each pair to cancel order/thermal bias
        if i % 2 == 0:
            base = run_arm(args, True)
            opt = run_arm(args, True if control else False)
        else:
            opt = run_arm(args, True if control else False)
            base = run_arm(args, True)
        rr = ratio(args, base, opt)
        ratios.append(rr)
        pr = prefill_ratio(base, opt)
        if pr is not None:
            pref_ratios.append(pr)
        bm, om = metric(args, base), metric(args, opt)
        extra = f" prefillR={pr:.3f}" if pr is not None else ""
        print(f"  [{label} pair {i+1}/{pairs}] base={bm:.2f} opt={om:.2f} ratio={rr:.4f}{extra}", flush=True)
    return ratios, pref_ratios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lfm2.5-1.2b-thinking-mlx")
    ap.add_argument("--mode", default="decode", choices=["ttft", "decode"])
    ap.add_argument("--toggle", required=True, help="MLX_LFM2_DISABLE_* env var")
    ap.add_argument(
        "--toggle-value",
        default="1",
        help="value the baseline arm sets the toggle to (default '1'; use '0' "
        "for MLX_ENABLE_* vars whose baseline is the disabled state)",
    )
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--reps", type=int, default=4, help="inner reps per process arm")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--control-pairs", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    print(f"== MEASUREMENT ({args.mode}, toggle={args.toggle}, model={args.model}) ==", flush=True)
    m_ratios, m_pref = collect(args, args.pairs, control=False, label="measure")
    print(f"== CONTROL (both arms baseline; ratio noise floor) ==", flush=True)
    c_ratios, _ = collect(args, args.control_pairs, control=True, label="control")

    med = statistics.median(m_ratios)
    signal = med - 1.0
    # Robust noise floor: median absolute deviation of control ratios from 1.0
    # (the max-dev is too outlier-sensitive at small N). Floor at 1.5% so we
    # never claim a win smaller than the harness's own resolution.
    control_devs = sorted(abs(r - 1.0) for r in c_ratios)
    control_band = max(statistics.median(control_devs) if control_devs else 0.0, 0.015)
    control_strict = max((abs(r - 1.0) for r in c_ratios), default=0.0)
    control_med = statistics.median(c_ratios) if c_ratios else 1.0
    same_side = sum(1 for r in m_ratios if (r > 1.0) == (med > 1.0))
    consistent = same_side >= (len(m_ratios) * 3 + 3) // 4  # >=75%
    # A claim must clear BOTH the robust band and the worst observed control
    # deviation — a single wild control pair proves the harness can produce
    # that much noise on an identical code path, so any smaller signal is
    # unresolvable. Claims also need >= 2 measure pairs.
    hurdle = max(control_band, control_strict)
    enough = len(m_ratios) >= 2
    real_win = (signal > hurdle) and consistent and (med > 1.0) and enough
    regression = (med < 1.0) and (abs(signal) > hurdle) and consistent and enough

    pref_med = statistics.median(m_pref) if m_pref else None

    verdict = {
        "mode": args.mode,
        "toggle": args.toggle,
        "model": args.model,
        "median_ratio": round(med, 4),
        "pct_change": round(signal * 100, 2),
        "measure_ratios": [round(r, 4) for r in m_ratios],
        "control_ratios": [round(r, 4) for r in c_ratios],
        "control_band": round(control_band, 4),
        "control_strict": round(control_strict, 4),
        "control_median": round(control_med, 4),
        "prefill_median_ratio": round(pref_med, 4) if pref_med is not None else None,
        "consistent_sign": consistent,
        "real_win": bool(real_win),
        "regression": bool(regression),
    }
    print("\n== SUMMARY ==")
    print(f"  median ratio = {med:.4f}  ({signal*100:+.2f}%)   control band = ±{control_band*100:.2f}%")
    if pref_med is not None:
        print(f"  prefill median ratio = {pref_med:.4f}  ({(pref_med-1)*100:+.2f}%)")
    print(f"  REAL WIN = {real_win}   (signal {signal*100:+.2f}% vs noise ±{control_band*100:.2f}%, consistent={consistent})")
    print(f"VERDICT_JSON:{json.dumps(verdict)}")


if __name__ == "__main__":
    main()
```
