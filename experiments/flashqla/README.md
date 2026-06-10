# FlashQLA / M5-NA prefill research arc — worklog & banked lever

Host: Apple **M5 Max, gen 17** (apple10 / applegpu_g17s), macOS 26.x, MLX 0.31.x.
All work isolated in the `research-flashqla` worktree. All numbers below are **measured on this M5**, adversarially reproduced.

Started from "can FlashQLA (Qwen's chunked GDN linear-attention kernel) be implemented on M5 for a prefill speedup?" The answer turned into a full characterization of *where* M5 prefill time goes and *which* levers are real.

> **NOTE (2026-06-05):** the `experiments/flashqla/` `.py` harnesses listed in the artifact index below were **lost** (untracked scratch removed by an interrupted file-move). Every *finding* they produced is preserved here and in the project memory bank; the production int8 kernel is preserved verbatim in `crates/mlx-sys/src/metal/na_int8_gemm.metal.inc`. Harnesses can be re-derived from the descriptions below if needed.

---

## TL;DR

- **The one already-shipped prefill win:** PR #68 — per-step default GDN on M5 (chunked was a ~26× pessimization). ~2.5–3.5× prefill TTFT. Done.
- **The remaining real lever — now BUILT + e2e VERIFIED + ADVERSARIAL-REVIEWED (2026-06-05):** **native int8 W8A8** on the MLP + GDN-qkvz GEMMs. Measured **e2e prefill TTFT: +18.6–19.7% on bf16 27B dense, +9.4–13.4% on 4B** (paired-process, clears the control band 8/8–10/10 sign-consistent), accuracy coherent (greedy near-tie flips only), decode untouched. Opt-in behind `MLX_INT8_PREFILL` (MLP) + `MLX_INT8_PREFILL_QKVZ` (qkvz). Review found **no correctness bugs** (kernels bit-faithful, fail-soft to bf16); ship-ready **default-OFF opt-in for dense qwen3_5 greedy**; default-on still needs a checked-in real-distribution accuracy gate + long-context(>8k) + sampling/MTP validation.
- **Everything else is closed** (proofs below): chunked/FlashQLA/NA on the recurrence, the conv1d "win" (artifact), and weight-only quant for prefill (decode-only).
- **2026-06-10 status: the bf16 opt-in path above was REMOVED.** Lossy-on-bf16 by design, it never left default-OFF; its kernels live on as the **sym8** quant mode (per-channel symmetric int8 checkpoints: W8A8 prefill + W8A16 decode, default-ON for sym8 checkpoints). `MLX_INT8_PREFILL` / `MLX_INT8_PREFILL_QKVZ` no longer exist. This dir stays as the research record.

---

## Roofline ground truth (the key that explains everything)

Single-stream GDN prefill on M5 is **LATENCY-BOUND**, not bandwidth- or compute-bound.
- Per-step GDN, full 24-layer 4B stack @ T=4096 = **293–483 ms**, sitting **31–51× above the bandwidth floor** and **3.8–6.2× above the compute floor**. Cost tracks the *number of serial steps T* (flat ~82–99 µs/token), not bytes.
- Per-step state S is register-resident (read once / written once per layer) → state DRAM is <0.2% of wall.
- Prefill fraction @ 4B (T=4096): **GDN recurrence ~28%, MLP SwiGLU ~37%, GDN qkvz proj ~18%, out_proj ~7%, full-attn ~5%, conv1d ~4%.** (Corrects an earlier "GDN is 70–80%" note.)

Consequence: on the recurrence, "save bandwidth" and "engage the NA" are both the wrong objective. The savable mass is the dense projection GEMMs (MLP+qkvz ≈ 55%), which are bf16-NA-**compute**-bound at ~57 TF.

---

## What's CLOSED (with the deciding number)

### 1. FlashQLA / chunked / NA on the GDN recurrence — mathematically closed
A correct WY-UT `chunk_gated_delta_rule` was built + validated (f64 vs per-step to 1e-10; bf16 cos 0.99999) and measured: full stack T=4096 = **1473–1564 ms = 3–5× SLOWER** than per-step. Three independent NOs:
- Even an inverse-free, spill-free, launch-free **GEMM-only floor = 291 ms ≥ the 293 ms per-step bar**, at 5.5 TFLOP/s = 0.9% of NA peak.
- Tiles cap at 16–23 TF: N=Dv=128, K=Dk=192 are pinned by head geometry; only chunk-M grows, but the NA needs all of M,N,K large.
- The gated-delta DPLR transition composes to **dense rank-7** → no cheap associative scan (Blelloch combine costs Dk³/node).
- Reusable sub-result: truncated-Neumann inverse (P=3, ρ≈0.098) collapses the chunked triangular inverse 65%→15% — valid, doesn't rescue chunked.

### 2. conv1d → compiled stacked-slice — measurement artifact
The "~22× / 21–23% of prefill" claim was a strawman: it benchmarked `mx.conv1d(padding=3)` (symmetric-pad MLX depthwise pathology, ~14.7 ms). Production left-pads + `padding=0` valid conv = **1.06–1.32 ms/layer, ~11× cheaper, numerically identical**. Real win = **1.32× on a ~3% slice = ~0.4% of prefill** → below noise. Not shippable.

### 3. Weight-only quant for prefill — decode-only, regresses prefill
At M=4096 the MLP/qkvz GEMMs are compute-bound (AI = M = 4096 ≫ ridge ≈112) at ~59–61 TF (≥100% of the bf16 NA peak). `mx.quantized_matmul` (`qmm_nax`) **dequantizes weights to bf16 in threadgroup memory then runs the same `tile_matmad_nax`** — no native low-precision MAC — so it's **0.89–0.91× (9–13% slower)** at prefill M. Decode (M=1) win is real (1.7–3.2×) — that's the right use, and it's already wired.

---

## THE LEVER: native int8 W8A8 prefill GEMMs — BUILT + VERIFIED + REVIEWED (2026-06-05)

### Hardware door (M5, adversarially reproduced)
- Standalone `mpp::tensor_ops::matmul2d`: `int8×int8→int32` geomean **1.82× (1.65–2.14×)** faster than bf16 at the production GEMM shapes (M=4096). int8 ~100 TOPS vs bf16 ~57 TF. Bit-exact vs CPU int ref at full prod K; the int8 path is a DISTINCT native op (`_dv_i8_dv_i8_dv_i32`) MLX never emits (it dequantizes all quant to bf16).
- **W8A16 (int8 weight × bf16 act) is dead** (geomean 1.08×): the wide bf16 operand pins the unit at bf16 rate → the 1.8× requires BOTH operands int8 → activation quant is mandatory.
- Per-token (per-row) dynamic symmetric int8 activation quant holds accuracy (the MLP_DOWN kurtosis-~5000 outliers are token-localized → per-token beats per-channel; no SmoothQuant needed).

### In-engine reality (the standalone 1.8× does NOT fully transfer)
- Clean in-engine int8 GEMM (pre-transposed weight, no per-call transpose): **~74 TOPS gate_up / ~65 down at M=4096 = ~1.3–1.5× over bf16** (coolest baseline), NOT 1.8×. Gap to standalone (~103/85) root-caused: **~14% is MLX's per-call full-output zero-fill** (`init_value=0` for `multiply_accumulate`); a bit-exact **`mode::multiply` (overwrite, no fill) kernel removes it FREE (74→86 TOPS)**. Residual ~15–20% is JIT per-call encode overhead.
- An earlier "in-graph edge only 1.14×, dead-end" verdict was a **measurement artifact** (the profiler's "GEMM-only" arm re-ran a 302 MB weight transpose every iteration the hot path never pays).

### v1 fusion (the build that made it a real win)
Three kernels in `crates/mlx-sys/src/mlx_na_int8.cpp` + `metal/na_int8_{gemm,quant,rescale}.metal.inc`:
1. **nofill `mode::multiply` GEMM** as the production core (removes the per-call zero-fill).
2. **Fused per-row absmax→int8 act-quant** kernel: 2.7 ms → 0.33 ms, BIT-IDENTICAL to the old lazy chain (MLX `Round` *is* `metal::rint`; the kernel must DIVIDE `x/s_x`, not multiply by `1/s_x`, to stay bit-equal).
3. **Fused `acc_i32·s_x·s_w→bf16` rescale** kernel: 2 ms → 1.1 ms, bit-identical (left-assoc `(acc*s_x)*s_w`, bf16 narrow inside the kernel).

`mlx_w8a8_linear` = fused-quant → nofill-GEMM → fused-rescale, stays lazy (composes into the forward graph). Microbench MLP flipped from 0.82× regression → ~0.78× (4B) / ~0.68–0.82× (27B) FASTER.

### Wiring (opt-in, dense qwen3_5 only)
- MLP: `transformer/mlp.rs` — `finalize_gate_up()` quantizes gate/up/down at load behind `MLX_INT8_PREFILL`; `try_forward_int8()` gates on flag + fields-present + M≥`MLX_INT8_PREFILL_MIN_M` (256); Err→bf16 fallback; M=1 decode falls through.
- GDN qkvz: `qwen3_5/gated_delta_net.rs` — `try_forward_qkvz_int8()` behind independent `MLX_INT8_PREFILL_QKVZ`; splits the stacked qkvz+ba so only the big **qkvz** is int8 and **`in_proj_ba` (β/decay recurrence gates) stays bf16**.
- **int8 is a silent bf16 no-op on qwen3_5-MoE** (its persistence path never calls these finalizes). Dense-only by design.

### Measured e2e win (paired-process, control band)
| model | MLP-only | combined MLP+qkvz | qkvz marginal |
|---|---|---|---|
| 4B (1k–5k tok) | +7.3–9.6% | **+9.4–13.4%** | qkvz-only +2–2.6% |
| 27B dense | +11.1% | **+18.6–19.7%** | +7.5–8.6 pp |

All clear the control band with ≥75% sign-consistency. Accuracy coherent (27B byte-identical first ~110 greedy tokens; near-tie flips only). Decode untouched (M=1 → bf16, ~parity). Win grows with model width.

### Adversarial review (3 lenses, 2026-06-05) — no correctness bugs
Kernel numerics: overflow 7.6× headroom (K-ceiling ≈133k), rint==Round, divide-parity, rescale order/broadcast/bf16, weight [K,N] layout — all bit-faithful. Control-flow: invalidation airtight, fallback airtight, dtype/residual clean, quantized-ckpt skip clean, thread-safe. Open hardening: explicit partial-M-tile (M%128≠0) bit-exact test (empirically OK — e2e ran at M=1329 coherent), rescale grid int64→int downcast, a forward-path integration test, scope docs. Accuracy/design: opt-in greedy bf16 ≤5k = CONDITIONAL-YES; default-on = NO until a checked-in real-distribution accuracy gate + long-context(8–32k) + sampling/MTP validation land.

### Trigger / next steps
- **Ship now:** default-OFF opt-in for dense qwen3_5 greedy prefill — biggest value on 27–35B and batched/server prefill.
- **For default-on:** check in a real-distribution ppl/top-1 accuracy harness (the old `w8a8_e2e_accuracy.py` was untracked scratch and was lost — recreate as a tracked test), validate v/z at 8–32k context, and A/B MTP acceptance + T>0 KL.
- **v2 (deferred):** fold the rescale into the matmul2d epilogue (cooperative_tensor scaled store) to drop the int32 intermediate — widens the GEMM edge further. Wire MoE finalize if 35B-A3B prefill becomes a priority.

---

## Artifact index (`experiments/flashqla/`) — HARNESSES LOST 2026-06-05
The `.py`/`.cpp`/`.metal` harnesses below were untracked scratch and were removed by an interrupted file-move. Findings preserved above + in the memory bank. Recreate from the descriptions if reproduction is needed.
- Roofline: `roofline.py`, `latency_probe.py`, `prefill_fraction.py`, `nongdn_measure.py`, `prefill_breakdown_measured.py`
- Chunked-GDN (closed): `gdn_chunked_f64.py`, `gdn_chunked_batched.py`, `final_table.py`, `gdn_breakdown.py`, `chunk_economics_model.py`, `lens2_*.py`, `lens3_*.py`
- conv1d (artifact): `conv_remeasure.py`, `conv_pad_probe.py`, `conv_win_verdict.py`
- Quant prefill (decode-only): `prefill_quant_vs_bf16.py`, `prefill_gemm_roofline.py`, `quant_na_largeM_probe.py`
- int8 W8A8: `na_int8/gemm.metal` (preserved in-tree as `crates/mlx-sys/src/metal/na_int8_gemm.metal.inc`), `na_int8/harness.cpp`, `na_int8/harness_w8a16.cpp`, `na_int8/verify_prodK.cpp`, `na_int8/act_outlier_analysis.py`, `na_int8/act_quant_error.py`, `na_int8/w8a8_e2e_accuracy.py` (recreate as a tracked accuracy test)
