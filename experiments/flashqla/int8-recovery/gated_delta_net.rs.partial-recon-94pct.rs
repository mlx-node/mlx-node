//! NA (Neural Accelerator) int8 W8A8 prefill GEMM — isolated, proven primitive.
 "summary": "Confirm (or refute) the conv1d→compiled-stacked-slice prefill win on the PRODUCTION path: real prod conv op + dims, full additive prefill breakdown, parity incl cached-state seam, ship plan",
description: "MEASURED 2026-06-04 (M5 Max, isolated worktree): the chunked GDN prefill kernel is ~24-31x SLOWER than per-step on M5 at EVERY production T — CHUNK_MIN_GPU_GEN=17 is a backwards stale-M3 gate. GDN is 69-80% of prefill wall time (NOT the 3-8% cost-model guess). Flipping the gate to per-step → projected ~3-4x prefill TTFT. All FlashQLA levers are moot once chunked is off the hot path."
//! three C++ FFI ops (see `crates/mlx-sys/src/mlx_na_int8.cpp`):
//!   * [`matmul_int8`]      — int8 `x @ w^T -> int32` (bit-exact integer GEMM)
   "audit": "Confirmed the full chain: eager prefill → `decoder_layer.rs:168` `gdn.forward(...)` → `gated_delta_net.rs:226` `self.conv1d.forward(&conv_input)` → `mlx_conv1d` → `mlx::core::conv1d(..., groups=conv_dim)`. The slow grouped-depthwise op is the production conv on the real (eager) prefill path, and the compiled C++ prefill scaffolding (if ever wired) uses the identical op. \n\nI have everything needed. Let me verify one detail in the harness reference path correctness (padding=3 vs production padding=0+manual-prepend) to confirm the harness's conv math matches production, then report.\n\nThe harness `conv_verify.py` uses `mx.conv1d(xc, w, padding=3, groups=conv_dim)[:,:T,:]` — equivalent causal output to production's `padding=0` + manual `(K-1)`-zero left-prepend (both produce K-1 left context, both slice to T). The math matches; only conv_dim differs.\n\n## AUDIT REPORT: Production conv-path identity & dim reconciliation\n\n**pwd:** `/Users/brooklyn/workspace/github/mlx-node/.claude/worktrees/research-flashqla`\n\n### 1. Production conv op identity — SLOW-OP PREMISE HOLDS\nThe GDN short conv is the generic grouped/depthwise MLX conv, NOT an optimized/compiled kernel.\n\n- Eager path: `gated_delta_net.rs:226` `self.conv1d.forward(&conv_input)` → `nn/conv1d.rs:95-105` calls `sys::mlx_conv1d(...)` → `mlx_advanced_ops.cpp:1158` `mlx::core::conv1d(*inp, *wt, stride=1, padding=0, dilation=1, groups=conv_dim)`. No depthwise specialization anywhere; it dispatches straight into MLX's generic conv.\n- Constructed depthwise at `gated_delta_net.rs:69-78`: `groups = conv_dim`, K=4, `bias=false`, stride 1, padding 0. SiLU applied after (`gated_delta_net.rs:237`).\n- This is byte-for-byte the mlx-lm reference (`./mlx-lm/mlx_lm/models/qwen3_5.py:104-112,167` `nn.Conv1d(groups=conv_dim, kernel=4, bias=False)` + `nn.silu`).\n\n### 2. Real 4B conv_dim is 8192, NOT the harness's 14336 — DIM DISCREPANCY (the one failure)\n`conv_dim = key_dim*2 + value_dim` where `key_dim = num_k_heads*key_head_dim`, `value_dim = num_v_heads*value_head_dim` (`gated_delta_net.rs:52-55`; `qwen3_5.py:104`).\n\nActual 4B config (`/Volumes/P4510/.cache/models/qwen3.5-4b/config.json:54-57`): `num_key_heads=16, key_head_dim=128, num_value_heads=32, value_head_dim=128` →\n- key_dim = 16×128 = **2048**, value_dim = 32×128 = **4096**, **conv_dim = 2048×2 + 4096 = 8192**, K=4, T=4096, 24 GDN layers (32 total, full_attention_interval=4).\n\nThe harness `experiments/flashqla/conv_verify.py:4` hardcodes `conv_dim=14336` — that comes from the Rust **default** config values (`config.rs:103-110`: nk=16, dk=192, nv=64, dv=128 → 3072×2+8192=14336), which are a LARGER (80B-class) model, not the 4B. So conv_verify.py over-states the conv by **14336/8192 ≈ 1.75×**. The 14.88→0.668 ms/layer numbers and the ~341 ms / 21-23% prefill share are inflated by ~1.75× on the channel dimension. The conv is still BW/dispatch-bound and the speedup direction is real, but the magnitude needs re-measurement at conv_dim=8192. (T=4096 and K=4 are correct.)\n\n### 3. Conv is EAGER MLX on the real prefill path — standalone bench IS representative (op-wise)\n- The compiled C++ Qwen3.5 prefill (`mlx_qwen35_text_prefill.cpp`) is explicitly \"first-cut scaffolding only… no Rust dispatch wiring\" (file header :13-16, E53). It is FFI-declared (`lib.rs:2178`) but has **zero live Rust callers** — the only Rust hits are test-function *names* in chat_common.rs. Dead scaffolding.\n- Actual prefill runs eager: `decoder_layer.rs:168` `gdn.forward(...)` → `gated_delta_net.rs:226` conv. So the standalone `mx.conv1d` bench faithfully models the production op (same `mlx::core::conv1d`, same args). Even the compiled scaffolding, if ever wired, uses the identical slow op (`mlx_qwen35_common.h:1270` `gdn_prefill_fn` → `mlx::core::conv1d(conv_input, conv_w, 1,0,1, conv_dim)`; same at :788, :1825). No path uses anything but the generic grouped conv.\n\n### 4. Cache-state seam spec for a byte-correct stacked-slice replacement\nProduction left-context handling (`gated_delta_net.rs:192-234`, matches `qwen3_5.py:148-167`):\n- Conv input = `concatenate([conv_state, qkv], axis=1)` where `conv_state` is `[B, K-1, conv_dim]` (= `[B,3,8192]`). On first prefill with no cache, `conv_state` is **zeros** (`gated_delta_net.rs:204-211`).\n- New cache = last `K-1=3` timesteps of `conv_input` (`gated_delta_net.rs:217-221`).\n- `mlx::core::conv1d` is called with `padding=0`, so it consumes the prepended K-1 context and emits `T_in-(K-1)` outputs; production slices the last `seq_len` (`gated_delta_net.rs:228-234`).\n\n**Replacement spec:** the 4-tap MAC must operate on `xpad = concatenate([conv_state, qkv], axis=1)` (the cached/zero K-1 context), NOT a fresh zero-pad — i.e. `y[t] = Σ_{j=0..3} xpad[t+j]·w[:,j]` over t∈[0,T), reading the 3 carried-in rows for t<3. It must still write the last-3-rows cache identically. For the **first-prefill redesign target** (T=4096, no cache → conv_state=zeros), this reduces exactly to the harness's zero-left-pad form, so that path is clean and the harness's `mx.pad(x[:-lag],...)` zero-prepend is correct there. The harness uses `mx.conv1d(padding=3)` as its reference (equivalent causal output to production's padding=0+manual-prepend) — conv math matches; only conv_dim is wrong.\n\n### Verdict\n- Slow-op premise (a): **HOLDS** — production conv is the generic grouped/depthwise `mlx::core::conv1d`, eager, on the real prefill path; no optimized/compiled conv exists.\n- Standalone-bench representativeness (b): **op-correct but dim-wrong** — right op, right T/K, but conv_dim=14336 instead of the real **8192**. The ~22× ratio and eps-exactness are about the algorithm and likely survive; the absolute ms (14.88→0.668) and the ~341 ms / 21-23% prefill-share claim are **inflated ~1.75×** and must be re-measured at conv_dim=8192. Expect the real saving to be meaningfully smaller (rough scaling: ~195 ms range, ~13-15% rather than 21-23%, pending re-measure — and only if the additive breakdown in prefill_fraction.py is re-run with the corrected conv_dim).\n- Seam (c): clean and specifiable; first-prefill zero-context path matches the harness.\n\n**Action item before trusting the win:** re-run `conv_verify.py` and `prefill_fraction.py` with `conv_dim=8192` (real 4B), then re-confirm the conv share inside a breakdown that sums to the measured ~1457 ms prefill.",
   "remeasure": "All measurements are complete and consistent across repeats. Let me verify the conv share is right by also checking whether production uses bf16 (it does — the GDN cast_predicate keeps A_log f32 but conv runs in model dtype, typically bf16) and confirm the production conv1d binding isn't already using a faster path. The production Rust `Conv1d::forward` — let me quickly confirm it maps to `mx.conv1d` (the slow grouped op) and uses the left-pad form (it does: lines 199-226 prepend pad then call conv1d with the layer's own padding). The production conv1d is configured with `padding=0`-equivalent (it pre-pads), matching my faithful benchmark. \n\nThe investigation is complete. Here is my report.\n\n---\n\n## RE-MEASURE VERDICT: the conv1d \"22× / 21-23% prefill\" claim DOES NOT survive at production dims. The real win is ~1.3× and **<1% of prefill.**\n\n**Working dir:** `/Users/brooklyn/workspace/github/mlx-node/.claude/worktrees/research-flashqla` (M5 Max gen17, mlx 0.31.1, bf16, GPU).\n\n### 1. Conv vs compiled-stacked-slice MAC at EXACT production dims\nProduction GDN conv width is **fixed** across the Qwen3.5/Qwen3-Next family (mlx-lm `qwen3_5.py:35-39` and `config.rs:100-114` both hardcode `linear_num_key_heads=16, key_head_dim=192, num_value_heads=64, value_head_dim=128`, independent of hidden_size) → **conv_dim = 3072·2 + 8192 = 14336, K=4** for the 4B too. So 14336 is the right channel count — concern (c) on dims resolves in the claim's favor. Everything else does not.\n\n`conv_remeasure.py` (B=1, conv_dim=14336, K=4, bf16, production left-pad + silu, best-of-40):\n\n| T | conv1d ms | compiled-MAC ms | speedup | saved/layer | ×24 saved |\n|---|-----------|-----------------|---------|-------------|-----------|\n| 1024 | 0.501 | 0.416 | **1.20×** | 0.085 | 2.0 ms |\n| 2048 | 0.924 | 0.714 | **1.29×** | 0.210 | 5.0 ms |\n| 4096 | **1.746** | **1.324** | **1.32×** | 0.422 | **10.1 ms** |\n\n**Parity is fine** (f32 abs/rel = 0, cos = 1.0; bf16 cos = 0.99999) — but the **22× / 14.88→0.668 ms is wrong by ~17×**. Real numbers: **1.32×, 1.75→1.32 ms/layer.**\n\n**Root cause of the bogus 14.88 ms** (`conv_pad_probe.py`, parity max-abs-diff = 0.0): the original `conv_verify.py` measured `mx.conv1d(..., padding=3)[:,:T]` — **symmetric** padding (an MLX depthwise-conv pathology) = 14.73 ms. Production left-pads by K-1 and runs `padding=0` valid conv (exactly mlx-lm `qwen3_5.py:159-167` / `gated_delta_net.rs:199-226`) = **1.32 ms, 11.2× cheaper**, numerically identical. The claim benchmarked a strawman the production op never executes. Worse, the original's \"manual MAC\" used `mx.pad`-slice and clocks **5.0 ms — 2.85× SLOWER than the real conv**.\n\n### 2. Full additive measured prefill breakdown, T=4096, 4B class (H=2560, I=9728)\n`prefill_breakdown_measured.py`, all components measured, GDN recurrence = the supplied measured 24-layer band:\n\n| component | total ms | % prefill (GDN=293) |\n|---|---|---|\n| GDN recurrence (24-layer stack, band) | 293–483 | 27.9% |\n| MLP SwiGLU ×32 | 386.1 | 36.8% |\n| GDN in_proj_qkvz ×24 | 184.6 | 17.6% |\n| GDN out_proj ×24 | 70.0 | 6.7% |\n| full-attn block ×8 | 56.7 | 5.4% |\n| **GDN conv1d+silu ×24** | **42.2** | **4.0%** |\n| GDN in_proj_ba ×24 | 4.3 | 0.4% |\n| norms + lm_head | ~12 | ~1% |\n| **SUMMED TOTAL** | **1049–1239** | 100% |\n\nSum brackets the claim's stated 1457 ms (slightly under — unmeasured glue only shrinks conv's share further). **conv1d's entire ×24 cost is 42.2 ms = 3.4–4.0%** — and that is the cost to run it *at all*, not the savable delta.\n\n### 3. Corrected, defensible numbers (`conv_win_verdict.py`, 3 stability samples)\n- Real conv multiplier: **1.32×** (not 22×)\n- Real ms/layer saved: **0.42 ms**, ×24 = **+10.1 ms** (not 341 ms)\n- Real prefill-share of the win: **+0.70% to +0.97%** (vs 1457 ms / vs measured 1049 ms) — **not 21–23%.**\n\n**The lever is dead.** The 21-23% number was a measurement artifact (symmetric-padding conv strawman); at the production op the conv is already fast, the compiled-MAC merely ties-to-slightly-beats it, and even making the conv *free* would cut 4B prefill by ≤4%. The dominant savable prefill mass is MLP (37%) and the GDN qkvz projection (18%), not the conv.\n\nNew harness files: `experiments/flashqla/conv_remeasure.py`, `conv_pad_probe.py`, `prefill_breakdown_measured.py`, `conv_win_verdict.py`.",
   "characterize": "I have all the evidence needed. The verdict is unambiguous and independently confirmed by my own runs. Here is the characterization.\n\n---\n\n# CASE B — Quantizing MLP + GDN qkvz is a DECODE-ONLY lever. It does NOT speed up prefill; it makes those GEMMs ~10-13% SLOWER at M=4096.\n\nI re-ran all three probes on this M5 Max (MLX 0.31.1, best-of-40 warmed) and confirmed the verdict independently, plus measured the decode win directly and audited the convert/load infra and the kernel source.\n\n## The deciding number\n**quant_ms / bf16_ms at M=4096 (prefill) = 1.10–1.13× across every production GEMM** (i.e. quant is ~10-13% slower). No bit-width/scheme reaches parity. The crux is settled in the negative: **there is no large-M quant path that exceeds bf16 on the M5 NA.** My run:\n\n| GEMM | M=1 (decode) | M=512 | M=4096 (prefill) |\n|---|---|---|---|\n| MLP gate/up [2560→9216] | **0.55** (1.8× faster) | 1.04 | **1.10** (slower) |\n| MLP down [9216→2560] | **0.32** (3.1× faster) | 0.85 | **1.11** (slower) |\n| GDN qkvz [2560→12288] | **0.61** (1.6× faster) | 1.09 | **1.12** (slower) |\n\nRoofline confirms why: at M=4096 every GEMM runs at **57–61 TF = ≥100% of the bf16 NA peak**, arithmetic intensity AI = M = 4096 FLOP/byte ≫ ridge ≈ 112 → flat-out **compute-bound**. Weight bytes (all quant reduces) are irrelevant when compute-bound. 4-bit affine does not cut MACs.\n\n**Mechanism (source-verified, `quantized_nax.h:1011-1050`):** `qmm_nax` calls `dequantize<T,...>` to expand weights into threadgroup memory as the activation dtype `T` (= bf16), then `tile_matmad_nax` runs both tiles as `NAXTile<bf16>` with `AccumType=float` — **identical MAC precision to the bf16 dense path, just dequantized first**, plus per-block dequant overhead. There is NO INT8→INT32 path for `quantized_matmul`. Repo notes that \"M5 NA does INT8→INT32\" describe the *hardware*, but `mx.quantized_matmul` does not route MACs through it. Hence quant ≤ bf16 at large M, exactly as measured. (At M=4096 all three shapes get `split_k=1`, falling to `qmm`, which takes the same NA gate as bf16 since K=2560/9216 are both %64==0.)\n\n## Predicted prefill delta\n**NEGATIVE/none.** MLP+qkvz are ~55% of prefill mass but are bf16-NA-compute-bound. Quantizing them would *regress* TTFT by ~10-13% on that 55% → roughly **+5-7% e2e prefill ms** (worse). Amdahl gives no upside because the quant path is strictly slower here. Recurrence/conv/attn are untouched either way. **Do not quantize for TTFT.**\n\n## Measured decode delta (the real win)\nAt M=1, these same GEMMs are bandwidth-bound (AI=1 ≪ ridge), so quant wins big. Measured per-op M=1: gate/up **1.80×**, down **3.88×** (bf16 dispatches a poor gemv there; quant's qmv path wins most), qkvz **2.09×**. Aggregated over the model (32 MLP blocks + 24 GDN qkvz):\n- **MLP+qkvz projection wall @ M=1: 29.4 ms → 12.1 ms per token (2.42× on this mass, 17.2 ms saved/token).**\n- This is the projection portion only; full per-token decode also includes recurrence/attn/norms (untouched), so the e2e decode tok/s gain is the bandwidth-bound fraction — consistent with the existing measured decode wins in MEMORY.md.\n- **Memory: 6.04 GB → 1.60 GB on these tensors (3.76× smaller, 4.44 GB saved)** — a real footprint/working-set win that also helps decode bandwidth.\n\n## (1) Accuracy\n4-bit affine on MLP (gate/up/down) and GDN `in_proj_qkvz` is **safe and already the production default**. Evidence: `convert.rs` `should_quantize` already quantizes these at 4-bit affine (default recipe); router gates go 8-bit; AWQ-style imatrix pre-scaling is supported for low-bit quality (`docs/perf.md:431`). The one GDN projection deliberately kept at higher precision is **`in_proj_ba`** — but that exclusion is a **T=0 MTP bit-exactness / gemv-vs-steel argmax-tie fix, NOT an accuracy-of-weights concern** (`convert.rs:1282-1294`); it's tiny so size is moot. `A_log`/`dt_bias`/`conv1d`/norms stay bf16/f32 (non-matmul shapes, already excluded). No qkvz/MLP layer needs to stay bf16 for quality.\n\n## (2) Smallest ship-delta — it already exists, fully wired, for qwen3_5\nNothing to build. End-to-end today:\n- **Convert** (`convert.rs:1232` `should_quantize`): MLP gate/up/down and GDN `in_proj_qkv`+`in_proj_z` (merged → `in_proj_qkvz`) all pass (no exclusion) → 4-bit affine. Excluded: `lm_head`, `embed_tokens`, norms, `conv1d`, `A_log`/`dt_bias`, `in_proj_ba`.\n- **Persistence** (`persistence.rs`): merges the split GDN tensors (`merge_in_proj`, lines 47-75) and builds `QuantizedLinear` for MLP gate/up/down + qkvz when `.scales` present.\n- **Forward** (`quantized_linear.rs:264+`): calls `mlx_quantized_matmul(x, w, scales, biases, transpose=true, group_size=64, bits=4, \"affine\")` → the `qmm_nax` path probed above.\n\nThe minimal user action is just `mlx convert` with the default 4-bit recipe; the load path auto-detects via `.scales`. Ship it as the decode/memory feature it already is.\n\n## (3) Reconcile with the lfm2 memory note\nThe note *\"lfm2 bf16 decode bandwidth-bound; real lever = quantization (fewer bytes/token)\"* is a **DECODE finding (M=1)** and it **transfers directly to qwen3_5 decode** — same physics: at M=1 these projections are bandwidth-bound (AI=1 ≪ ridge), fewer bytes/token = faster, measured here at 2.4× on the projection mass. **It does NOT transfer to prefill.** Prefill is M=4096, AI=4096 ≫ ridge → compute-bound, where \"fewer bytes/token\" buys nothing and the dequant overhead makes it ~10% slower. Same weights, opposite regime — that's the whole prefill-vs-decode story, and it closes quantization as a single-stream *prefill* lever in the same direction as the rest of the arc (the prefill projection mass is bf16-NA-compute-bound; quantization is the wrong tool for TTFT, the right tool for decode/memory).\n\nProbe scripts (all reusable, on this M5):\n- `/Users/brooklyn/workspace/github/mlx-node/.claude/worktrees/research-flashqla/experiments/flashqla/prefill_quant_vs_bf16.py` — the decisive M-sweep\n- `/Users/brooklyn/workspace/github/mlx-node/.claude/worktrees/research-flashqla/experiments/flashqla/prefill_gemm_roofline.py` — compute-bound roofline\n- `/Users/brooklyn/workspace/github/mlx-node/.claude/worktrees/research-flashqla/experiments/flashqla/quant_na_largeM_probe.py` — multi-scheme (4b/8b affine, mxfp4/mxfp8) vs bf16\n- `/tmp/decode_block_probe.py` — M=1 whole-model projection-mass + memory delta (the decode-win quantification)\n\nProduction dims confirmed from the on-disk checkpoint headers (not just config): H=2560, MLP I=**9216** (task header's 9728 is stale), GDN input = `in_proj_qkv` [8192,2560] + `in_proj_z` [4096,2560] → merged qkvz width **12288** (the probe's value is exactly right; my Rust-default-derived 14336 was wrong — this checkpoint uses key_head_dim=128). 32 layers, 24 GDN, full_attention_interval=4.",
   "verdict": "Confirmed. `should_quantize` quantizes MLP gate/up/down and GDN `in_proj_qkv`/`in_proj_z` (which merge to qkvz) at the default 4-bit affine; only `in_proj_ba` is excluded (a T=0 MTP/AR argmax-tie bit-exactness fix, explicitly \"not a size optimization\"). The infra is end-to-end wired — convert quantizes these, persistence builds `QuantizedLinear`, forward calls `mx.quantized_matmul`. Nothing net-new to build.\n\nI have independently confirmed every load-bearing claim. The verdict is decisive and reproducible.\n\n---\n\n# ADVERSARIAL VERDICT — Quantizing MLP + GDN qkvz is a DECODE-ONLY lever. It does NOT speed up prefill on M5; it makes those GEMMs ~9-13% SLOWER at M=4096.\n\nI independently re-ran all three probes on this M5 Max (MLX 0.31.1, best-of-40 warmed) and audited the convert/load infra and dims from source. The four prior agents converged and I confirm them — but the user asked for a *prefill* win, and the brutally honest answer is **no, quantization is the wrong tool for TTFT**.\n\n## 1. Prefill / decode / neither — with the deciding ratio\n\n**Neither for prefill; a strong DECODE lever.** The crux (\"does the M5 NA run genuine INT8 above the 57 TF bf16 ceiling at large-M?\") is settled in the **negative**.\n\n**Deciding number — best-quant TFLOP/s vs bf16 at M=4096 (my run, dense-effective TF):**\n\n| Production GEMM | bf16 | best quant | ratio (quant/bf16) |\n|---|---|---|---|\n| MLP gate/up [2560→9216] | 59.8 TF | 54.4 (4b affine) | **0.91× (9% slower)** |\n| MLP down [9216→2560] | 59.1 TF | 52.3 (4b affine) | **0.89× (11% slower)** |\n| GDN qkvz [2560→12288] | 59.3 TF | 41.7 (4b affine) | **0.70× (30% slower)** |\n\nNo scheme (4b/8b affine, mxfp4, mxfp8) reached parity at M=4096 on any shape. The M=1 vs M=4096 separation is the whole story and I keep them explicit:\n\n- **M=1 (DECODE, bandwidth-bound):** quant **crushes** bf16 — 1.7–3.2× (down-proj 3.2× because bf16 dispatches a poor M=1 gemv there). Real, large win.\n- **M=512:** crossover — already at/below parity (0.89–1.13×).\n- **M=4096 (PREFILL, compute-bound):** quant **loses**, 0.70–0.91×.\n\nThis is exactly an M=1 decode win that must **not** be mis-sold as prefill. Roofline confirms why: at M=4096 every GEMM runs at 59–61 TF ≈100% of the bf16 NA peak; arithmetic intensity AI = M = 4096 FLOP/byte ≫ ridge ≈112 → flat-out compute-bound. Quant only cuts weight bytes, which is irrelevant when compute-bound; 4-bit affine does not cut MACs.\n\n**Mechanism (source-verified, not assumed):** `qmm_nax` (`crates/mlx-sys/mlx/mlx/backend/metal/kernels/quantized_nax.h`) calls `dequantize<T>` to expand weights to the activation dtype (bf16) in threadgroup memory, then runs `tile_matmad_nax` with both tiles as `NAXTile<bf16>`, `AccumType=float` — **identical MAC precision to the bf16 dense path, plus per-block dequant overhead**. There is no INT8→INT32 path for `mx.quantized_matmul`. The repo note \"M5 NA does INT8→INT32\" describes the hardware, not this op. Hence quant ≤ bf16 at large M, exactly as measured.\n\n## 2. The Amdahl ceiling — there is no positive ceiling\n\nThis is the adversarial kill-shot. Quant touches ~55% of prefill mass (MLP 37% + qkvz 18%), but on that mass it is **0.70–0.91× = slower**, not faster. Amdahl with a per-GEMM ratio <1 gives a **negative** delta:\n\n- Take the favorable per-GEMM number (~0.90× on MLP, but qkvz is 0.70×). On the 55% slice the realized slowdown is ~10–15% → e2e prefill **TTFT regresses ~+5–8%**, before counting the worse qkvz factor.\n- **There is no upside to clear the ~10–15% M5 thermal-noise floor — the sign is wrong.** Quantizing these projections would push TTFT *out the bottom* of the noise band in the wrong direction, or at best sit inside it as a wash-to-regression.\n\nRecurrence/conv/attention are untouched either way. **Prefill objective is not served — it is harmed.**\n\n## 3. Accuracy / risk / infra\n\n- **4-bit affine on MLP gate/up/down and GDN qkvz is safe and is already the production default.** `should_quantize` (`convert.rs:1232`) quantizes exactly these at 4-bit affine. The only GDN projection held back is `in_proj_ba`, and the source comment (`convert.rs:1282-1294`) is explicit that this is a **T=0 MTP/AR argmax-tie bit-exactness fix, \"not a size optimization\"** — not a weight-accuracy concern. Recipes apply v_proj/down_proj/SSM high-bit overrides for quality (lines 2026, 2105–2129).\n- **Infra is net-zero — fully wired today, nothing to build.** Convert quantizes these tensors → persistence merges split GDN tensors and builds `QuantizedLinear` when `.scales` present → forward calls `mlx_quantized_matmul(..., transpose=true, group_size=64, bits=4, \"affine\")` (the probed `qmm_nax` path). The minimal user action is `mlx convert` with the default recipe; the load path auto-detects via `.scales`.\n\n## 4. RECOMMENDATION\n\n- **For the user's PREFILL objective: NO-GO.** Deciding number: **best quant = 0.91×/0.89×/0.70× bf16 at M=4096 → ~+5–8% slower TTFT.** Quantization does not raise the compute-bound NA ceiling; it adds dequant overhead below it. This closes quantization as a single-stream *prefill* lever, consistent with the rest of the arc (the prefill projection mass is bf16-NA-compute-bound; the only prefill lever left on these GEMMs would be a genuinely-faster low-precision MAC path the M5 NA does not expose for this op).\n- **SHIP-FOR-DECODE-INSTEAD (already wired):** at M=1 these same projections are 1.7–3.2× faster and ~3.8× smaller (≈4.4 GB saved on these tensors). Aggregated over 32 MLP blocks + 24 GDN qkvz, the per-token projection wall drops ~2.4× — a real decode tok/s + working-set win that *also* helps decode bandwidth. This matches the existing decode findings in MEMORY.md and the lfm2 \"real lever is quantization (fewer bytes/token)\" note — which is a DECODE (M=1) finding and transfers to qwen3_5 decode, **not** to prefill.\n- **Next concrete experiment** (only if a real-weights e2e check is wanted, not for prefill hope): native-build same-binary A/B of a default-4-bit-recipe checkpoint vs bf16 on `/Volumes/P4510/.cache/models/qwen3.5-4b`, measuring **TTFT (prefill)** and **decode tok/s** separately with a T=0 accuracy/byte-equivalence gate. Predicted result: prefill TTFT ~+5–8% (worse, near/inside noise), decode tok/s materially up, output bit-stable under the standard 4-bit recipe. That A/B would confirm the deployment call: **ship the existing 4-bit quant for decode + memory, accept the small prefill cost, and do not pursue quantization as a TTFT optimization.**\n\nProbe scripts (all reusable on this M5, reproduced this session):\n- `/Users/brooklyn/workspace/github/mlx-node/.claude/worktrees/research-flashqla/experiments/flashqla/prefill_quant_vs_bf16.py` — decisive M-sweep\n- `/Users/brooklyn/workspace/github/mlx-node/.claude/worktrees/research-flashqla/experiments/flashqla/quant_na_largeM_probe.py` — multi-scheme (4b/8b affine, mxfp4/mxfp8) vs bf16\n- `/Users/brooklyn/workspace/github/mlx-node/.claude/worktrees/research-flashqla/experiments/flashqla/prefill_gemm_roofline.py` — compute-bound roofline\n\nDims confirmed from on-disk `/Volumes/P4510/.cache/models/qwen3.5-4b/config.json` + `crates/mlx-core/src/models/qwen3_5/config.rs:153-164`: H=2560, MLP I=**9216** (task header's 9728 is stale), GDN qkvz width **12288** (key_head_dim=128 in this checkpoint overrides the Rust default 192, so 14336 was wrong). 32 layers, 24 GDN, full_attention_interval=4."
Came out of the FlashQLA research arc (started by reading qwen.ai/blog?id=flashqla). The research never needed a FlashQLA kernel — profiling the chunked GDN path to evaluate FlashQLA levers exposed that **the chunked kernel should not be running on M5 at all.**
//! [`MxArray`]s holding **integer values in `[-127, 127]`** and casts them to
**Two measured facts (M5 Max gen 17, isolated worktree `research-flashqla`, two dynamic workflows, adversarially verified):**
* Set `MLX_INT8_PREFILL_QKVZ_DEBUG=1` to confirm the int8 qkvz branch fired on
1. **chunked GDN is ~24-31× SLOWER than per-step on M5**, same synthetic inputs (Hv=32, B=1, Dk=Dv=128), at EVERY production T. Per single GDN call: T=4096 → chunked 109ms vs per-step 3.7ms; T=32768 → 886ms vs 28.9ms. Flat ~26-30× across T=1024..32768. Output parity 1-2 bf16 ULP (maxabsdiff 0.002-0.004). Method: inline `#[cfg(test)]` calling private `gated_delta_chunked` vs `gated_delta_kernel` directly; chunked absolutes reproduce the standalone microbench within ~1%.
2. **GDN recurrence is 69-80% of prefill wall time** (paired same-session `MLX_GDN_ABLATE` ablation on real Qwen3.5-4B): 79.3% @1102 tok, 76.0% @4376, 69.5% @8739 (<3% per-pair spread). This **REPLACES the 3-8% cost-model estimate, which was wrong by ~10-25×.** Mechanism: GDN is a serial, tensor-core-free delta-rule recurrence running at ~0.1% of peak FLOPs, so a low-FLOP op legitimately dominates wall time.
//! Every op gates internally on **GPU gen >= 17 (M5+)** and **`K % 16 == 0`**,
**Root cause:** `CHUNK_MIN_GPU_GEN=17` (`crates/mlx-core/src/models/qwen3_5/gated_delta.rs:15/21`) gates chunked ON for M5+. It was set as the **INVERSE of an M3 result** (experiment E44: chunked 2.1-2.3× slower than per-step on M3 → so gated chunked to "M5+ where it wins") but the "M5+ chunked wins via bandwidth" premise was **NEVER A/B'd on M5** and is empirically FALSE. The in-source comments at `gated_delta.rs:7-14,17-20,439-441` claiming M5 chunked wins are wrong and must be corrected when the gate is flipped.
//! to a bf16 `matmul`. Stage 3 must check eligibility before routing a linear
**The lever (highest-value, ~zero cost):** flip the gate so M5 long-prefill (seq≥64) takes the **per-step** kernel — raise `CHUNK_MIN_GPU_GEN` above 17 or delete the chunked routing. Projected **~3-4× prefill TTFT** (per-step at ~1/26 of chunked, landing on the ~75% GDN share). Per-step is already the trusted production path on M1-M4, all seq<64, and all masked GDN calls.
//! `M` large enough to amortize quant; the threshold is a Stage-3 policy knob).
**RESOLVED + IMPLEMENTED (2026-06-04, worktree `research-flashqla`, branch `worktree-research-flashqla`, NOT pushed):**
- **Perf CONFIRMED (realized, not projected):** end-to-end T=0 greedy A/B (same binary, `MLX_GDN_KERNEL` env toggle, native `ttftMs`). bf16-4B: **2.80-3.53×** (580-5384 tok). Quant-27B-Q4 dense: **2.49×** (3102 tok). MoE 35B-A3B-mxfp8: **2.82×** sub-seam / **3.38×** cross-seam. Ratio grows with prompt length (GDN share rises).
- **Byte-identity FAILS (accepted):** per-step ≠ chunked at T=0 on SOME prompts — 1-2 bf16 ULP flips a greedy argmax → different-but-COHERENT continuation (bf16-4B 3/4 prompts fork; quant-27B prose IDENTICAL; MoE counting/list prompt forks @char ~83). Both kernels are valid orderings; per-step is the canonical reference on M1-M4 + all compiled-C++ paths + masked/short. User DECIDED: flip is acceptable (not a transparent byte-identical opt — a policy call that per-step is the M5 reference).
- **Ship mode = DEFAULT FLIP on M5** (per-step on every gen; chunked opt-in via `MLX_GDN_KERNEL=perstep|chunked`, legacy `MLX_GDN_FORCE_PERSTEP=1`). Removed the gen-17 gate. Impl: pure `should_use_chunked(seq_len,mask_none,gpu_gen,choice)` + `parse_gdn_kernel(..)` + 3 always-on regression tests locking "chunked never default on any arch (incl gen 17)". 4 false comments fixed (gated_delta.rs:6-21,439-441 + docs/perf.md). clippy/fmt/build green.
- **Quant+MoE characterization PASSED (the user's gate before flip):** flip reaches the SINGLE shared eager `gated_delta_update` for dense AND MoE × {flat default, paged} uniformly (quant only touches projections; GDN inputs are bf16 quant-independent; compiled-C++ is decode-only, imports eager-prefill caches, does NOT bypass the gate — C++ text-prefill is dead scaffolding). **Seam-compounding REFUTED** (source + empirical): state is f32 intra-kernel / bf16 carried symmetric across arms → cancels; slicing is memory-only/math-identical; decay gate α∈(0,1) geometrically DAMPS injected error. Empirical: cross-seam div@83 ≈ sub-seam div@86, both coherent (no early/gibberish fork).
- **Remaining:** adversarial code review (per CLAUDE.md) + commit + PR. Optional pre-default-broadening (deferred, flip already chosen): larger prompt set / downstream task-accuracy delta to characterize argmax-flip frequency.
       Ok(v) => {
**FlashQLA levers — all DEAD/moot** (see the arc): A (gate-out-of-inverse) — exp already hoisted out of our P3 solve; B (CP+warmup) — no occupancy headroom (V tiled into grid → 32·B·Hv TGs saturate the 40-core M5). Once per-step is the M5 path, any chunked-kernel rewrite is off the hot path. The only post-flip lever is the per-step kernel itself (now ~75% of prefill) — a simdgroup_matrix/NAX formulation or head-grid batching — to be re-scoped AFTER the flip ships and re-measures.
/// bf16/f32 [`MxArray`]s holding **exact integer values in `[-127, 127]`**. The
**M5 Neural-Accelerator (tensor-core) lever — DOA, CLOSED 2026-06-04** (7-agent deep-research workflow + adversarial). The M5 (gen 17, `apple10`) DOES have a tensor-core analog: the per-GPU-core **Neural Accelerator**, driven from a custom Metal kernel ONLY via Metal 4 `mpp::tensor_ops::matmul2d` (MetalPerformancePrimitives, over `metal_tensor`/`cooperative_tensor`). **`simdgroup_matrix<8,8>` is NOT the NA** (regular FP32 ALUs) → the old "simdgroup_matrix ~4×" comment in `gated_delta.rs:19`/`docs/perf.md` was always bogus. NA dtypes: INT8→INT32, FP16→FP16/F32, bf16 (macOS 26.1+); **no fp8/MXFP4**; **FP32 inputs fall off the NA**. Tile floor ~32×32, K mult of 32 (Dk=192=6×32 OK); win regime large-M≥1024, **M=1 ≈0.96× (slower)**. TOOLCHAIN IS READY (no blocker): vendored MLX 0.31.2 already has a NAX path (`crates/mlx-sys/mlx/.../steel/gemm/nax.h:401-448`), our metallib already ships `affine_qmm_*_nax_*`, Metal 4.0/SDK 26.4 compiles matmul2d — the NA already accelerates our quantized projection qmv/qmm. **But applying it to chunked-GDN prefill is DOA for an AMDAHL/structural reason** (stronger than the small-M floor): the chunked kernel has NO materialized GEMMs (P2 kk_dot/P4 qk_dot are scalar triangular `simd_sum` loops, never tiles); its real cost is the SERIAL spine — P3 triangular solve (forward-substitution data dependency), P5 cross-chunk state carry, per-chunk `exp()` decay matrix — all NA-immune; and per-step (the default) has ZERO GEMM, so there's nothing for the NA to beat. f32-state also trips MLX's `enable_tf32()||dtype!=f32` NAX gate (precision regression). Real GDN-prefill lever if ever revisited = the serial recurrence latency (P3 solve, P5 carry, exp traffic, barriers), NOT matrix throughput. Cheapest decider (~1hr, no code): profile `MLX_GDN_KERNEL=chunked` on M5, confirm P2+P4 <50% of wall. This CLOSES the "only NAX lever left = chunked-GDN prefill" thread from [[nax-smallm-qmm-slower-and-unsafe]].
/// reference `x @ w^T`.
**Experimental from-scratch FlashQLA — BUILT + VALIDATED + MEASURED fresh → NO-GO confirmed by DIRECT measurement (2026-06-04, worktree `research-flashqla`, 3-agent workflow, adversarially reproduced).** User rejected ALL prior metrics as possibly-stale/wrong-hardware and asked to implement a PROPER FlashQLA (FLA WY-UT chunked, matmul-dominated, NOT the bad scalar in-tree kernel) from scratch on M5 and let fresh numbers decide. Done. Result:
- **Correctness PASSED + a validated reference now exists** (`experiments/flashqla/*.py`): proper WY-UT chunk_gated_delta_rule (C=64) matches our per-step kernel to machine-eps in f64 (O rel-max ≤1.7e-10, S ≤1.5e-10). Two real bugs in the recon op-spec caught by the f64 gate + fixed against FLA (vLLM `fla/ops`): op5 binds U=T(βV) not raw V; op2 sign is `(I+A_strict)⁻¹` (FLA `solve_tril` negates first). bf16-operand/f32-accum precision is fine (cosine 0.99999, rel-L2 0.43%) — NOT the limiter. Derived gold: state S=[Dv,Dk], `S*=g_t; delta=(v−S@k)·β; S+=outer(delta,k); o=S@q` (POST-update); g=exp(−exp(a_log)·softplus(a+dt_bias))∈(0,1), β=sigmoid(b), q/k RMS-norm(no weight) then q·Dk⁻¹/k·Dk⁻⁰·⁵.
- **Perf LOST 3-5× (fresh, full 24-layer 4B GDN stack, T=4096): chunked-NA op-graph 1473-1564 ms vs per-step-kernel bar 293-483 ms = 0.19-0.33×.** Beats only a naive MLX per-step loop (~5×, irrelevant — engine ships a fast kernel).
- **THE CRUX NEW FACT (refutes the GO-PROTOTYPE microbench): NA is INERT at real GDN tile shapes.** bf16 ≡ f32 dead heat at every level (per-op NA gain 0.94-1.09×; tiles run 18-30 TFLOP/s, dispatch-bound, ~10% of the 600 TFLOP/s square-GEMM peak). The prior workflow's "37-54 vs 8 TFLOP/s bf16 headroom" only existed at LARGE SQUARE GEMMs that never occur in GDN; the actual per-head [64,192]@[192,64]/[64,64] tiles are too small to leave NA-immune dispatch overhead → the projected 2.6-5.2× evaporated. Heads are block-diagonal-independent so you CAN'T merge them into one large-M GEMM without wasting FLOPs on zeros → tile size is structurally capped by GDN dims (Dk192/Dv128/C64). This is why FlashQLA's Hopper-WGMMA win does NOT transfer to M5 NA.
- **Breakdown (T=4096): blocked triangular inverse op2 = ~64% (many tiny dispatch-bound matmuls), parallel matmuls ~28%, inter-chunk serial carry op6 only ~8% (linear in NT, no cliff).** A hand-fused matmul2d kernel could recover op2's launch overhead, BUT per-op NA-gain≈1.0× means it can't beat the f32-ALU throughput, and chunked does strictly MORE ALU work than per-step → still loses. CAVEAT for full honesty: a TRUE matmul2d-fused FlashQLA kernel was NOT built (the in-tree fused chunked kernel is the SCALAR simd_sum one, not matmul2d); but the per-op NA measurement makes a true fused kernel very unlikely to help — chunked's only advantage is high arithmetic intensity, worthless when the NA gives ~1.0× at these shapes.
- **VERDICT: chunked/FlashQLA/NA direction is DEAD on M5 for GDN prefill, now by DIRECT fresh measurement of a correct implementation (not just the earlier Amdahl/scalar-kernel argument).** Per-step remains the bar. The shipped gate flip (PR #68) stands. Redirect any further GDN-prefill effort to the per-step kernel itself (occupancy/bandwidth), NOT a chunked rewrite. Validated reference impl + measurement harness kept under `experiments/flashqla/` (untracked scratch — keep or delete per user).
fn int8_prefill_min_m() -> i64 {
**Mathematician redesign panel — ROOFLINE PROOF + a NEW off-recurrence lever (2026-06-04, 10-agent workflow, 4 lenses each adversarially verified).** User pushed for a first-principles MATH redesign (not a port) using M5's API. Verdict:
- **ROOFLINE: single-stream GDN prefill on M5 is LATENCY-BOUND** (serial T-dependency), decisively. Per-step full 24-layer 4B stack T=4096 = 293-483 ms sits **31-51× ABOVE the bandwidth floor (9.38 ms)** and **3.8-6.2× above the f32-ALU compute floor (77.5 ms)** — neither binds. Cost tracks the NUMBER of serial steps T (flat 82-99 µs/token across T), not bytes. GPU fully occupied (2048 TGs, 51/core — not occupancy-starved). Per-step S is REGISTER-resident (`float state[6]`=Dv·Dk/thread, read once at layer entry / written once at exit) → state DRAM = 0.60 ms/stack = <0.2% of wall. **CONSEQUENCE: "save bandwidth" and "engage NA" are BOTH the wrong objective for the recurrence** — its real shape is an M=1 GEMV (S[Dv,Dk]@k[Dk,1]) at bf16 1.13-1.42× ≈ 1% of NA peak; NA needs M≥32, saturates only at M=N=K≥1024. The ONLY axis that can beat per-step is breaking the serial T-dependency (= chunking), but every chunked reformulation pays a FLOP/DRAM/inverse tax exceeding the latency saved.
- **THE RECURRENCE IS MATHEMATICALLY CLOSED ON M5 (3 confirmed NOs, deciding numbers):** (1) chunk-economics: best C=128 = 895 ms; its BANDWIDTH FLOOR ALONE (~248 ms, from manufacturing [nblk,C,C] KKt/decay/A/Tinv/A_qk intermediates per-step never materializes) is already inside the bar. (2) tile-growth: best C=512 = 484 ms; intra-chunk tiles cap at 16-23 TF (NOT the 57 TF square ceiling) because N=Dv=128 & K=Dk=192 are STRUCTURALLY PINNED by head geometry — only M grows with C, NA needs all 3 of M,N,K large. (3) alt-scan: the inverse-free/spill-free/launch-free GEMM-ONLY floor = **291 ms ≥ BAR_lo 293** at 5.5 TFLOP/s = 0.9% of NA peak → a PERFECT fused chunk kernel can't beat its own GEMMs. Associative-scan escape is DEAD: the gated-delta DPLR transition (I−β k kᵀ, Householder rank-1) is NOT preserved under composition — composes to DENSE rank-7 after 7 steps → Blelloch combine costs Dk³=7.08 MFLOP/node, strictly worse than WY. Reusable sub-result: **truncated-Neumann inverse** (ρ(A)≈0.098, P=3 → O rel-err 1.8e-6, ~3 orders below bf16 floor) collapses the chunked triangular inverse 65%→~15% — VALID + validated, just doesn't rescue chunked.
- **PREFILL-FRACTION CORRECTION (4B harness dims H=2560/I=9728, T=4096):** GDN-recurrence = **40-52%** (293-483 ms), MLP SwiGLU = **41-52%** (382 ms, ×32), full-attn = 6-8% (58 ms, ×8). This **CORRECTS the prior "GDN is 69-80%"** (that came from a config where MLP was cheaper or attribution folded in projections/conv). MLP is CO-DOMINANT — and already at the ~57 TF NAX ceiling (not a lever). NOTE: the real production-checkpoint denominator is unconfirmed (assumption flagged).
- **The conv1d "off-recurrence lever" — REFUTED as a MEASUREMENT ARTIFACT (2026-06-04 production-path confirm workflow).** The panel's "~22× / 21-23% of prefill" conv1d→compiled-stacked-slice win was a STRAWMAN bench: `conv_verify.py` measured `mx.conv1d(padding=3)[:,:T]` (SYMMETRIC padding, an MLX depthwise pathology, ~14.7 ms) — but PRODUCTION (`gated_delta_net.rs:199-226`, matches mlx-lm `qwen3_5.py:159-167`) left-pads by K-1 then runs `padding=0` VALID conv = **1.06-1.32 ms/layer, ~11× cheaper, numerically identical** (`conv_pad_probe.py` max-abs-diff 0.0). At the real op the compiled-MAC only TIES-to-1.32× it. Full ADDITIVE measured prefill breakdown (T=4096, 4B): MLP SwiGLU ×32 **~37%**, GDN in_proj_qkvz ×24 **~18%**, GDN recurrence **~28%**, out_proj ×24 ~7%, full-attn ×8 ~5%, **conv1d+silu ×24 = only ~2-4%**, in_proj_ba ~0.4% (sums ~1049-1239 ms). So even making conv FREE caps the win <4%; real win = **~0.4-0.6% of prefill = +6-10 ms across 24 layers** — below the ~10-15% M5 thermal noise floor → UNMEASURABLE, NOT SHIPPABLE. (Aside: real 4B conv_dim=**8192** per on-disk config.json nk16/dk128/nv32/dv128; config.rs default 14336 is a larger-class model — doesn't change the verdict.) Parity IS eps-exact (f32 abs 0, bf16 cos 0.99999) — it's just not a win. The savable prefill mass is MLP (37%) + qkvz proj (18%), both already at the ~57 TF NAX ceiling (quantization, not a kernel/NA lever).
- **The ONLY NA path left for the recurrence = MULTI-SEQUENCE BATCHED prefill** (stack B sequences → tile-M grows past the block-diagonal head cap into the NA large-GEMM regime). Out of single-stream scope, unmeasured — the one genuinely-open, mathematically-grounded place a real M5 NA win on the recurrence could live (a server-batch-throughput objective, NOT single-stream TTFT). Validated reference impls + roofline/lens/conv harnesses kept under `experiments/flashqla/`.
- **Quantizing MLP+qkvz = DECODE lever, REGRESSES prefill — source-proven (2026-06-04 probe workflow).** Tested whether 4-bit/8-bit affine / mxfp4 / mxfp8 matmul beats bf16 at large-M on the M5 NA. DECIDING NUMBER: at M=4096 (prefill) best-quant/bf16 = **0.89-0.91× (9-13% SLOWER)**, qkvz worst run 0.70×. At M=4096 the MLP/qkvz GEMMs run at **59-61 TF ≈100% of the bf16 NA peak**, deeply COMPUTE-bound (AI = M = 4096 FLOP/byte ≫ ridge ≈112) → reducing weight bytes (all quant does) is irrelevant; 4-bit doesn't cut MACs. MECHANISM (source `quantized_nax.h:1011-1050`): `qmm_nax` calls `dequantize<T>` to expand weights to bf16 in threadgroup mem, then `tile_matmad_nax` runs both tiles as `NAXTile<bf16>`/`AccumType=float` — IDENTICAL MAC precision to bf16 dense + per-block dequant overhead → structurally ≤ bf16 at large-M. **There is NO INT8→INT32 path for `mx.quantized_matmul`** (the "M5 NA does INT8" repo note is hardware-true but this op doesn't route MACs through it). So quantizing MLP+qkvz would push prefill TTFT ~+5-8% (wrong sign). DECODE (M=1, bandwidth-bound, AI=1): quant CRUSHES bf16 **1.7-3.2×** (down-proj 3.2×), per-token projection-mass wall ~2.4× faster, **4.4 GB saved (~3.76× smaller)** — the real win, already FULLY WIRED (`convert.rs should_quantize` 4-bit-affines MLP gate/up/down + GDN qkvz by default; only `in_proj_ba` excluded as a T=0 MTP argmax-tie fix, not accuracy; nothing to build, just `mlx convert`). Transfers the [[lfm2-bf16-decode-bandwidth-bound-parity]] decode finding to qwen3_5 — DECODE only, NOT prefill.
- **★ NATIVE int8 W8A8 IS A REAL M5-NA PREFILL COMPUTE LEVER — hardware-verified (2026-06-04 custom-kernel microbench, adversarially reproduced 6 runs).** This VINDICATES the user's original "M5 NA / tensor-core gives a prefill boost" intuition — just on the dense projection GEMMs (via int8), NOT on the GDN recurrence. Wrote a custom `mpp::tensor_ops::matmul2d` Metal kernel (full Xcode SDK, `xcrun -sdk macosx26.4 metal -std=metal4.0`; metal-cpp host harness) at `experiments/flashqla/na_int8/` (gemm.metal/harness.cpp). MEASURED: **int8×int8→int32 runs ~1.8× FASTER than bf16×bf16→f32** at the production prefill GEMM shapes (MLP gate/up 2560→9216, down 9216→2560, GDN qkvz 2560→12288, M=4096): geomean **1.82× (range 1.65-2.14×)**; int8 saturates ~**100 TOPS** vs bf16 ~**57 TF**. VERIFIED to the hilt: int8 bit-exact vs CPU int ref at full prod K=2560 (no K-clamp); bf16 baseline calibrates 57-62 TF = MLX's real NA peak (disasm: bf16 uses MLX's exact `__tensorops_impl_matmul2d_op_run_dv_b16_dv_b16_dv_f32` relaxed=true; int8 the DISTINCT native `_dv_i8_dv_i8_dv_i32` relaxed=false — genuinely different hardware ops). The resident-tile "peak probe" (0.92×) is a serial-RMW latency-chain artifact, correctly DISREGARDED; the multi-output-tile production GEMM is the right saturated measurement. **This is the classic tensor-core int8=2×fp16 advantage, confirmed present on the M5 Max NA** — and the thing MLX leaves on the table (MLX dequantizes ALL quant to bf16, never emits the int8→int32 op).
- **ARC BOTTOM LINE (CORRECTED): there IS one real single-stream prefill compute lever on M5 — native int8 W8A8 — but it's a substantial UNBUILT lift, Amdahl-capped ~1.3× e2e.** What's CLOSED: FlashQLA/chunked/NA on the recurrence (mathematically closed, latency-bound); conv1d (artifact); WEIGHT-ONLY quant (dequants to bf16 → decode-only, regresses prefill). What's OPEN: **int8 W8A8** (above) — ~1.8× on the ~55% MLP+qkvz GEMM mass → **~1/(0.45+0.55/1.8) ≈ 1.3× e2e prefill TTFT** (recurrence/conv/attn unchanged). The BLOCKER is no longer "does the HW do it" (it does, proven) — it's the build: (1) MLX wires none of it → need a NEW custom int8 GEMM kernel in the forward pass (steel/NAX is bf16-only); (2) the HARD part = ACTIVATION quantization (per-token/channel int8 + scales every layer + int32→f dequant/rescale + accuracy work, SmoothQuant-class outlier handling — W8A8 is NOT byte-identical, needs perplexity/task-accuracy validation). Cheapest next decider BEFORE building: a pure-numerical W8A8 accuracy sim (quantize acts+weights of MLP+qkvz on real 4B, measure divergence vs bf16) — the kernel is proven; accuracy is the open risk. The one ALREADY-SHIPPED prefill win of the arc: PR #68 (per-step default, ~2.5-3.5×). Other open lever (different objective): multi-seq batched prefill (NA via M-growth, server throughput).
 'Read the following passage carefully and then write a clear, coherent multi-paragraph continuation that stays strictly on topic.\n' +
- **int8 W8A8 DE-RISK (2026-06-04, real 4B): realized e2e prefill is only ~1.18-1.20× (NOT 1.3×), marginal, and the EASY shortcut is DEAD.** (1) **W8A16 (int8 weight × bf16 act) is a NO — geomean 1.08×, peak probe 0.94×** (added bf16×int8→f32 + fp16×int8→f32 mixed matmul2d kernels to na_int8/, both compile+correct maxRel 0): the WIDE bf16 left operand pins the matrix unit at the bf16 rate (W8A16 ceiling 62.5 TF ≡ bf16 62.6 TF); the narrow int8 weight buys nothing for COMPUTE (only decode weight-bandwidth). So the ~1.8× requires FULL W8A8 (both operands int8) — no weight-only shortcut, activation quant is mandatory. (2) **Activation-quant difficulty is LOW** — direct per-token int8 reconstruction error on the real 4B (act_outlier_analysis.py/act_quant_error.py, ~630-tok prose): per-TOKEN(row) dynamic symmetric int8 holds for all 3 target classes (MLP_IN mean cos-err 7.3e-4, GDN_IN 1.5e-3, MLP_DOWN 2.4e-3). The pathological tensor is MLP_DOWN (SwiGLU intermediate, kurtosis ~5000) but its outliers are TOKEN-localized not channel-localized (cross-layer channel persistence 0.01) → per-row BEATS per-col → **NO SmoothQuant / per-channel migration needed**; GDN qkvz-input is NOT worse than MLP despite feeding the recurrence (q/k get RMS-normed downstream). (3) **Realized e2e prefill ~1.18-1.20×** (1.17 taxed → 1.22 no-tax → 1.28 ceiling): per-class SATURATED kernel speedups are 1.44/1.78/1.44 (gate-up/down/gdn-in, NOT uniform 1.8×), Amdahl-compressed by the ~50% non-target remainder (MLP 37% + gdn-in 12-18%), minus 8-12% activation-quant tax (per-token absmax reduction + int32→f rescale). This BRUSHES the ~1.10-1.15× M5 noise floor — clears it untaxed, brushes it taxed. (4) **Build = HIGH** (custom int8×int8→i32 GEMM in the C++ forward + Rust caller; per-token dynamic act-quant; int32→f rescale; weight int8 convert/load partly exists). (5) **e2e ACCURACY GATE PASSED (conditionally) — proven on real 4B** (`na_int8/w8a8_e2e_accuracy.py`, 2250-tok fake-quant teacher-forced, fp32-accum faithful to int32-exact, identity-path validated 100%/0-KL): W8A8 on MLP+qkvz gives ppl delta **+1.3%** worst (config C both; <2% ✓), top-1 agreement 94-96.5% LITERAL but that's a near-tie artifact — on COMMITTED predictions (ref top1p>0.5) **99.77%**, high-conf (>0.9) **100%**; all sub-1% misses confined to <0.1-margin coin-flip steps (mean KL 1.4e-2). So W8A8 is end-to-end accuracy-SAFE for greedy (residual = near-tie flips only; recommend post-build greedy-byte spot-check). BOTH gates (hardware ~1.8× + accuracy) now PASS. VERDICT: the lever is PROVEN + accuracy-safe + ready-to-build, but realized ~1.18-1.20× prefill-only on 4B BRUSHES the noise floor for a HIGH build → marginal; genuinely the user's cost/benefit call. SCALING NOTE: the win grows with model size (bigger MLP/proj share → higher Amdahl ceiling, ~1.25-1.35× on 27-35B) and with BATCHED/server prefill (GEMMs dominate more) — most valuable exactly where prefill hurts most; it's a prefill/large-M lever (decode already covered by weight-only quant). Artifacts: experiments/flashqla/na_int8/ (gemm.metal mixed kernels, harness_w8a16, act_*.py).
 '\nNow continue the passage in your own words, keeping it focused and grammatical.';
- **★ int8 W8A8 IN-ENGINE INTEGRATION BUILT + MEASURED — standalone 1.8× does NOT transfer; in-engine GEMM edge is ~1.3-1.5×, e2e is a WASH on 4B (2026-06-05, worktree, uncommitted).** Wired the proven kernel into the MLP prefill via `fast::metal_kernel` JIT (`crates/mlx-sys/src/mlx_na_int8.cpp` + `metal/na_int8_gemm.metal.inc` + `crates/mlx-core/src/models/qwen3_5/int8_gemm.rs`; load-time per-output-channel weight int8 in `transformer/mlp.rs` behind `MLX_INT8_PREFILL`, M≥`MLX_INT8_PREFILL_MIN_M`=256). Parity SOLID (S1 int32 bit-exact all shapes; S2 cosine 0.99998; e2e byte-identical). **But the real-world A/B is a NET REGRESSION (~0.82× at T=4096), and the root cause is now fully understood:** (1) the earlier "in-graph edge only 1.14×, bf16 near peak, dead-end" verdict was a **MEASUREMENT ARTIFACT** — the profiler's "int8 GEMM only" arm called `matmul_int8` which re-runs the [N,K]→[K,N] weight transpose (302 MB int8 copy) EVERY iteration; the hot path uses the pre-transposed weight and pays none of it. (2) **CLEAN in-engine int8 GEMM (pre-transposed weight, isolated FFI): ~74 TOPS gate_up / ~65 down at M=4096** = **~1.3-1.5× over bf16** (coolest baseline), NOT the standalone 1.8×. The gap to standalone (~103/85 TOPS) is root-caused: **~14% is MLX's per-call full-output zero-fill** — `init_value=0.0f` (required by `mode::multiply_accumulate`) makes MLX `fill_gpu` the entire C[M,N] int32 output every call (`backend/metal/custom_kernel.cpp:338`), worst at large N (gate_up); a **bit-exact `mode::multiply` (overwrite, no init_value) kernel removes it FREE → +14% (74→86 TOPS gate_up)**. The residual ~15-20% is JIT per-call scalar-rebind/encode overhead (MLX-internal; dispatch grid verified equivalent to standalone). (3) **e2e WASH** because the GEMM win (~1.5×) is eaten by UNFUSED overhead: per-token act-quant (~2.7 ms, a 5-op lazy f32 chain) + int32→f32→bf16 rescale (~2 ms multi-pass on the big [M,18432] tensor) + the fill. **Corrected Amdahl: ceiling ~1.12-1.22× e2e on 4B** (vs the banked 1.18-1.20 est) ONLY after FULL fusion (fused quant kernel + rescale-into-GEMM-epilogue + nofill) — brushes/at the noise floor on 4B single-stream; ~1.25-1.35× still holds for 27-35B / batched (bigger GEMM share). VERDICT unchanged from the bank but now IN-ENGINE-CONFIRMED: **not a 4B single-stream win; the trigger is big-model / batched prefill.** Free do-regardless lever banked: switch the production int8 GEMM to `mode::multiply` (nofill, bit-exact, +14% gate_up). Cheap measurement FFIs (`mlx_int8_gemm_pretransposed[_nofill]`) added test-side. Diagnostic agent a738175c70d352ce6 has full numbers.
   in_proj_qkvz: LinearProj, // hidden → key_dim*2 + value_dim*2 (q,k,v,z combined)
- **★ v1 FUSION BUILT + e2e VERIFIED — int8 W8A8 MLP prefill is a REAL ~1.07-1.11× e2e win (2026-06-05, worktree, uncommitted).** Per the user's pick ("build full fusion, verify on 27-35B"). Built 3 kernels in `mlx_na_int8.cpp` (+ `metal/na_int8_quant.metal.inc`, `na_int8_rescale.metal.inc`): (1) nofill `mode::multiply` GEMM as production core (removes the per-call output zero-fill); (2) fused per-row int8 act-quant kernel (1 tg/row, 2 strided global passes, simd+tg absmax reduce) → **2.7ms→0.33ms**, BIT-IDENTICAL to the old lazy chain (MLX `Round` IS `metal::rint` per `unary_ops.h:301`; kernel must DIVIDE x/s_x not multiply by 1/s_x to stay bit-equal); (3) fused `acc_i32·s_x·s_w→bf16` rescale kernel → **2ms→1.1ms**, bit-identical. `mlx_w8a8_linear` rewired to fused-quant→nofill-GEMM→fused-rescale, stays LAZY. Parity ALL green: S1 GEMM bit-exact, S2 cosine 0.99998, new fused-quant/rescale bit-parity tests, clippy/fmt clean. **Microbench MLP flipped from 0.82× regression → ~0.78× (4B) / ~0.68-0.82× (27B) FASTER** (int8 thermally stable; ratio vs coolest bf16). **e2e prefill TTFT (paired-process, control band, real bf16 models): 4B +7.3-9.6% (clears ±1.5%, 10/10 sign), 27B +11.1% (clears ±3.3%, 6/6 sign)** — modest, grows with model width, Amdahl-diluted from the MLP ~1.3× because MLP is ~37% of prefill. Accuracy coherent (near-tie argmax flips only; W8A8 lossy signature, not a regression). Decode (M=1) correctly falls through to bf16 (clean A/B ~parity). **NEXT: extend int8 to GDN in_proj_qkvz** (~18% of prefill → ~doubles the win to ~1.18-1.22×); split the stacked qkvz+ba, int8 the big qkvz, keep `in_proj_ba` bf16 (feeds recurrence decay/beta gates, excluded from quant per `convert.rs:1292`). **FLAGGED pre-existing (NOT int8):** the qwen3_5 **compiled** forward on 27B hits a Metal command-buffer watchdog timeout on cold-graph compile (bf16 baseline times out identically) → verify used `MLX_NO_COMPILE=1` eager on both arms (fair); compiled 27B cold-start needs warmup/watchdog mitigation independently. Impl agent a26bfbf8130fabcf8, verify agent ad01d774eddf6fea9.
///   * `s_w` is `f32 [N]`, the per-output-channel scale `max_k|w[n,k]| / 127`.
- **★ qkvz EXTENDED + COMBINED VERIFIED + ADVERSARIAL-REVIEWED — full int8 W8A8 prefill = +18-20% on 27B, ~+13% on 4B (2026-06-05, worktree, uncommitted).** Extended int8 to GDN `in_proj_qkvz` in `gated_delta_net.rs` (new `try_forward_qkvz_int8` + `qkvz_w_i8/qkvz_s_w` fields + `finalize_in_proj` quant), independent flag `MLX_INT8_PREFILL_QKVZ`, split the stacked qkvz+ba so only the big qkvz is int8 and `in_proj_ba` (β/decay recurrence gates) stays bf16. qkvz parity cosine 0.99998; 4B greedy COHERENT (near-tie flip only, zero blowup — the riskiest target held); qkvz microbench int8/bf16 0.79 (4B)/0.74 (27B). **COMBINED (MLP+qkvz) e2e prefill TTFT, paired-process + control band: 27B +18.6-19.7% (8/8 sign, clears band ~9×; qkvz adds ~+7.5-8.6pp on top of MLP's +11.1%), 4B +9.4-13.4% (qkvz-only +2-2.6%).** Accuracy coherent both (27B byte-identical first ~110 tok). Decode untouched. Verify needed `--arm-retries` for the pre-existing 27B compiled cold-start Metal watchdog (bf16 baseline times out identically; ran `MLX_NO_COMPILE=1` eager both arms). qkvz agent a554a942f77831030, combined-verify agent a07dcf5899a4ac7bb.
- **ADVERSARIAL REVIEW (3 parallel lenses, 2026-06-05) — NO correctness bugs; safe as the opt-in flag it is; gaps are validation-coverage not defects.** Codex review hit its 1MB input cap (the 3.6MB `experiments/` scratch + vendored metal_cpp swamp the working-tree sweep; same code hung Codex 2× before) → used 3 Claude adversarial subagents (kernel-numerics a4ba5c2ae80623e31, control-flow a52e238ba94abd788, accuracy/design aca8508527e09dcf5). **Kernel:** overflow 7.6× headroom (ceiling K≈133k), rint==MLX Round, divide-parity, rescale order/broadcast/bf16-narrow, weight [K,N] layout — ALL bit-faithful/SAFE; one open: `mode::multiply` nofill GEMM never bit-exact-tested at M%128≠0 (partial M-tile) — but e2e RAN at M=1329 coherent → empirically OK, adding explicit test; + rescale grid `static_cast<int>(M*N)` overflows >61k-tok prefill (latent, fixing). **Control-flow:** no BLOCKER/HIGH — invalidation airtight (every weight setter clears int8 fields), fallback airtight (Err→Ok(None)→bf16, no half-int8 state), dtype/residual clean, quantized-ckpt skip clean, thread-safe; MEDIUM: int8 is a silent no-op on qwen3_5-MoE (finalize never called in MoE persistence → documenting dense-only), flag-on holds bf16+int8 both resident (memory), no Rust forward-path integration test. **Accuracy/design:** opt-in greedy bf16 ≤5k-ctx = CONDITIONAL-YES; default-on = NO (no in-tree real-distribution accuracy gate — the w8a8 ppl evidence is in untracked `experiments/` scratch + synthetic-uniform unit tests; 8-32k v/z recurrence-accumulation untested; sink/CJK/code outlier class unreached by uniform tests); sampling(T>0)/MTP = NO (unvalidated — int8 prefill perturbs the hidden that seeds MTP draft). Hardening agent a2e8cfa6e999ccab7 closing: partial-M-tile bit-exact test, grid-overflow fix, forward integration test, scope-doc comments. **NET: ship-ready as default-OFF opt-in for dense qwen3_5 greedy prefill; default-on needs a checked-in real-distribution accuracy harness + long-context + sampling/MTP validation.**
const loaded = await loadModel(modelPath);
Corrects [[qwen35-prefill-arc]] (M3-era, chunked gated off there) and the "M5 runs M5-gated chunked GDN" framing in [[env-dev-machine-m5-max]] (it runs it, but it's a ~26× pessimization). All work was in the isolated worktree per [[never-build-in-shared-main-checkout]].
/// Stage 3 holds both handles alongside each quantized linear and passes them to
/// [`int8_w8a8_matmul`] on every forward.
pub fn quantize_weight_int8(w: &MxArray) -> Result<(MxArray, MxArray)> {
   let mut out_w_i8: *mut sys::mlx_array = std::ptr::null_mut();
   let mut out_s_w: *mut sys::mlx_array = std::ptr::null_mut();
   let ok = unsafe { sys::mlx_quantize_weight_int8(w.as_raw_ptr(), &mut out_w_i8, &mut out_s_w) };
   maxNewTokens: maxNew,
       return Err(Error::from_reason(
           "mlx_quantize_weight_int8 failed (see stderr)",
   reuseCache: false,
   // Surface token ids if the API exposes them; text is the load-bearing signal.
   let w_i8 = MxArray::from_handle(out_w_i8, "quantize_weight_int8:w_i8")?;
   let s_w = MxArray::from_handle(out_s_w, "quantize_weight_int8:s_w")?;
   Ok((w_i8, s_w))
// qwen3.5-4b is a reasoning model — most of the greedy output lands in
// `rawText` (full decoded stream incl. any <think>…</think>) and `thinking`;
/// W8A8 linear: per-token int8 activation quant + int8 GEMM + rescale -> bf16.
// token-by-token signal for an A/B coherence diff.
/// `x` is `[M, K]` bf16 activations; `w_i8` / `s_w` come from
/// [`quantize_weight_int8`] (the `w_i8` handle is opaque and pre-transposed to
/// the `[K, N]` kernel layout). Returns bf16 `[M, N] = x @ w^T`, lossy only by
/// int8 quantization noise (per-row cosine vs the bf16 reference is ≥ 0.999 on
/// real projection shapes — see the parity test below).
 numTokens: res.numTokens ?? null,
/// Stage 4b: the returned array is **lazy** — the C++ op no longer force-evals,
/// so the result composes into the surrounding forward graph (downstream swiglu +
/// down-matmul) and MLX keeps async pipelining/fusion across layers. The caller
/// must `eval` at the end of forward (the normal model loop already does).
 rawText: res.rawText ?? '',
/// The result is narrowed to bf16 **inside C++** before return, so a downstream
/// bf16 residual add is not promoted to f32 by an f32 scale.
pub fn int8_w8a8_matmul(x: &MxArray, w_i8: &MxArray, s_w: &MxArray) -> Result<MxArray> {
   let mut out: *mut sys::mlx_array = std::ptr::null_mut();
console.log(`RESULT_JSON:${JSON.stringify(out)}`);
       sys::mlx_w8a8_linear(
           x.as_raw_ptr(),
           w_i8.as_raw_ptr(),
           s_w.as_raw_ptr(),
           &mut out,
       )
   };
   if !ok {
       return Err(Error::from_reason(
           "mlx_w8a8_linear failed (unsupported gen/K or kernel error; see stderr)",
       ));
   }
   MxArray::from_handle(out, "int8_w8a8_matmul")
}

/// MEASUREMENT ONLY (profiler/test scope — NOT a production path).
///
/// Pure int8 `x @ w^T -> int32 [M,N]` with a PRE-TRANSPOSED `[K,N]` weight,
/// isolating the GEMM kernel from the per-call `int8_weight_to_kn` transpose
/// that [`matmul_int8`] pays every iteration. `x` is `[M,K]` bf16/f32 holding
/// integers in `[-127,127]`; `w_kn` is the opaque int8 `[K,N]` operand from
/// [`quantize_weight_int8`] (used directly, no transpose/contiguous/quant).
/// This is the apples-to-apples in-engine analogue of the standalone harness.
#[cfg(test)]
pub fn matmul_int8_kn(x: &MxArray, w_kn: &MxArray) -> Result<MxArray> {
   let mut out: *mut sys::mlx_array = std::ptr::null_mut();
   let ok =
       unsafe { sys::mlx_int8_gemm_pretransposed(x.as_raw_ptr(), w_kn.as_raw_ptr(), &mut out) };
   if !ok {
       return Err(Error::from_reason(
           "mlx_int8_gemm_pretransposed failed (unsupported gen/K or kernel error; see stderr)",
       ));
   }
   MxArray::from_handle(out, "matmul_int8_kn")
}

/// MEASUREMENT ONLY. Same as [`matmul_int8_kn`] but the kernel uses
       let norm = RMSNormGated::new(value_head_dim as u32, Some(config.rms_norm_eps))?;
       let out_proj = Linear::new(value_dim as u32, hidden_size as u32, Some(false))?;
#[cfg(test)]
pub fn matmul_int8_kn_nofill(x: &MxArray, w_kn: &MxArray) -> Result<MxArray> {
       let dt_bias = MxArray::ones(&[num_v_heads as i64], None)?;
       let a_log = MxArray::zeros(&[num_v_heads as i64], None)?; // Will be loaded from weights
       sys::mlx_int8_gemm_pretransposed_nofill(x.as_raw_ptr(), w_kn.as_raw_ptr(), &mut out)
       Ok(Self {
           in_proj_qkvz: LinearProj::Standard(in_proj_qkvz),
           in_proj_ba: LinearProj::Standard(in_proj_ba),
           "mlx_int8_gemm_pretransposed_nofill failed (see stderr)",
           norm,
           out_proj: LinearProj::Standard(out_proj),
   MxArray::from_handle(out, "matmul_int8_kn_nofill")
           a_log,
           num_k_heads,
/// MEASUREMENT ONLY (parity test scope). Runs the FUSED v1 activation-quant
/// kernel. `x` is `[M,K]` bf16; returns `(x_i8_as_i32, s_x)` where the int8
/// quant is widened to int32 `[M,K]` (Rust has no Int8 dtype) and `s_x` is f32
           key_dim,
           value_dim,
pub fn act_quant_fused(x: &MxArray) -> Result<(MxArray, MxArray)> {
   let mut out_i8: *mut sys::mlx_array = std::ptr::null_mut();
   let mut out_sx: *mut sys::mlx_array = std::ptr::null_mut();
   let ok = unsafe { sys::mlx_int8_act_quant_fused(x.as_raw_ptr(), &mut out_i8, &mut out_sx) };
           qkvz_s_w: None,
       return Err(Error::from_reason("mlx_int8_act_quant_fused failed"));
   }
   Ok((
   /// E51: precompute the stacked `[qkvz; ba]^T` weight once after both
   /// in_proj weights have been loaded. Forward will then use one matmul
   /// (x @ wqb_t) plus two axis-2 slices instead of two matmuls (x @ w_qkvz.T)
   /// + (x @ w_ba.T). Safe to call repeatedly (idempotent).
   ///
/// MEASUREMENT ONLY. The LAZY activation-quant chain (parity reference). Same
   /// Standard linears. Quantized models continue on the legacy 2-matmul
   /// path (no-op here).
pub fn act_quant_lazy(x: &MxArray) -> Result<(MxArray, MxArray)> {
   let mut out_i8: *mut sys::mlx_array = std::ptr::null_mut();
           (LinearProj::Standard(_), LinearProj::Standard(_)) => {}
   let ok = unsafe { sys::mlx_int8_act_quant_lazy(x.as_raw_ptr(), &mut out_i8, &mut out_sx) };
   if !ok {
       let w_qkvz = self.in_proj_qkvz.get_weight(); // [qkvz_dim, hidden]
       let w_ba = self.in_proj_ba.get_weight(); // [ba_dim, hidden]
       let stacked = MxArray::concatenate(&w_qkvz, &w_ba, 0)?; // [qkvz_dim+ba_dim, hidden]
       let stacked_t = stacked.transpose(Some(&[1, 0]))?; // [hidden, qkvz_dim+ba_dim]
       MxArray::from_handle(out_sx, "act_quant_lazy:s_x")?,
       self.in_proj_qkvz_ba_t = Some(stacked_t);
}
       // NA int8 W8A8 prefill (opt-in via MLX_INT8_PREFILL_QKVZ): quantize ONLY
       // the qkvz weight to int8 ONCE at load. `in_proj_ba` stays bf16 (its b/a
       // outputs gate the recurrence and are deliberately excluded from quant;
       // see convert.rs `in_proj_ba.` exclusion), so the int8 path forgoes the
       // E51 single-matmul fusion and does TWO matmuls (int8 qkvz + bf16 ba).
pub fn rescale_fused(acc: &MxArray, s_x: &MxArray, s_w: &MxArray) -> Result<MxArray> {
       // LAYOUT: `quantize_weight_int8` expects `[N, K]` with rows = output
       // channels so its `s_w[N]` scale broadcasts onto the GEMM accumulator
       // `acc[M, N]`. The UN-transposed `w_qkvz` is `[qkvz_dim, hidden]` =
       // `[N=qkvz_dim, K=hidden]`, already in `[N, K]`, so we pass it straight in
       // (the stacked `*_t` transposed form is for the bf16 matmul ONLY and must
       // NOT be quantized here). `quantize_weight_int8` ALSO hoists the
       // per-forward transpose, returning the opaque int8 weight already in the
       // `[K, N]` kernel layout (opaque to Rust).
       //
       // SCOPE: int8 prefill is DENSE qwen3_5 ONLY. qwen3_5_moe's decoder layer
       // never constructs/finalizes this GDN path with int8 wiring through the
       // MoE persistence route, so `MLX_INT8_PREFILL_QKVZ` is a silent bf16 no-op
       // on MoE (intended — not wired for MoE; documented, not fixed).
       //
       // MEMORY: with the flag ON, BOTH the bf16 stacked `in_proj_qkvz_ba_t`
       // (above) AND the int8 qkvz weight stay resident — the bf16 stacked form
       // is the per-call `Err`-fallback target in `try_forward_qkvz_int8`. Opt-in
       // trades extra weight memory for prefill speed.
pub fn rescale_lazy(acc: &MxArray, s_x: &MxArray, s_w: &MxArray) -> Result<MxArray> {
       // VALIDATED REGIME: opt-in, greedy (T=0), bf16 models, prefill M>=256.
       // Sampling / MTP / long-context (>8k) are NOT yet validation-gated, which
       // is why this path is deliberately default-OFF.
       self.qkvz_w_i8 = None;
       self.qkvz_s_w = None;
       if int8_prefill_qkvz_enabled() {
           // Fail-soft: if quant fails (e.g. unsupported shape), leave the
           // fields None so forward() stays on the unchanged bf16 path.
           if let Ok((qkvz_i8, qkvz_s)) = int8_gemm::quantize_weight_int8(&w_qkvz) {
               qkvz_i8.eval();
       return Err(Error::from_reason("mlx_int8_rescale_lazy failed"));
               self.qkvz_w_i8 = Some(qkvz_i8);
               self.qkvz_s_w = Some(qkvz_s);
           }
       }
       Ok(())
mod tests {
   use super::*;
   /// NA int8 W8A8 prefill path for the GDN `in_proj_qkvz` projection.
   use crate::nn::Activations;
   /// Returns:
   ///   * `Ok(Some(qkvz))` — the int8 path ran and produced the qkvz output
   ///     `[B, T, qkvz_dim]` (bf16). The caller computes `ba` separately as
   ///     bf16 (`in_proj_ba` stays full precision).
   ///   * `Ok(None)`       — not eligible (flag off / no int8 weights / `M`
   /// Deterministic pseudo-random integer in `[lo, hi]` from a linear-congruential
   ///   * `Err`            — only a genuine non-int8 error (reshape/shape).
   fn next_int(state: &mut u64, lo: i32, hi: i32) -> i32 {
       *state = state
           .wrapping_mul(6364136223846793005)
   ///   → `qkvz = int8_w8a8(x, qkvz_w_i8, qkvz_s_w)` `[M, qkvz_dim]`
   ///   → reshape back to `[B, T, qkvz_dim]`.
       lo + ((*state >> 33) % span) as i32
   /// The int8 op narrows its result to bf16 internally, so the downstream conv
   /// / recurrence (which expect bf16) are not promoted to f32.
   /// Build an `[rows, cols]` bf16 MxArray holding the given integer values.
   fn int_array_bf16(vals: &[i32], rows: i64, cols: i64) -> MxArray {
       let (Some(qkvz_i8), Some(qkvz_s)) = (&self.qkvz_w_i8, &self.qkvz_s_w) else {
       MxArray::from_float32(&f, &[rows, cols])
           .unwrap()
       if !int8_prefill_qkvz_enabled() {
           return Ok(None);
       }

       // Gate 2: M = product of leading dims (everything but the last).
   // GATE S1: int32 output BIT-EXACT (integer matmul is deterministic).
   // M ∈ {128,256,512}, K ∈ {2560,9216}, N ∈ {a tile multiple, a non-multiple}.
   // Tile is 128x64, so N=2560 is a multiple of 64; N=2570 is a non-multiple
   // (exercises the edge/tail tile + the contiguous w^T transpose path).
   #[test]
       let hidden = dims[dims.len() - 1];
       let m: i64 = dims[..dims.len() - 1].iter().product();
       // M == 1 (decode) and small prefill regress vs bf16 → fall through.
               "[s1] SKIP: gpu gen {} < 17 (NA matmul2d needs M5+)",
           return Ok(None);
           );
           return;
       // Reshape to 2D [M, hidden] for the int8 GEMM.
       let x2d = x.reshape(&[m, hidden])?;
       let ks = [2560usize, 9216];
       let ns = [2560usize, 2570]; // tile-multiple + non-multiple (edge tile)
       let qkvz2d = match int8_gemm::int8_w8a8_matmul(&x2d, qkvz_i8, qkvz_s) {
           Ok(v) => v,
           Err(_) => return Ok(None),
           for &k in &ks {
       let qkvz_dim = qkvz2d.shape_at(1)?;
                   // x[m,k], w[n,k] in [-127,127].
       if std::env::var("MLX_INT8_PREFILL_QKVZ_DEBUG").is_ok() {
           eprintln!("[int8-prefill-qkvz] fired: M={m} hidden={hidden} qkvz_dim={qkvz_dim}");
                       *v = next_int(&mut state, -127, 127);
                   }
       // Reshape [M, qkvz_dim] back to the original leading dims + qkvz_dim.
       let mut out_shape: Vec<i64> = dims[..dims.len() - 1].to_vec();
                       *v = next_int(&mut state, -127, 127);
       let qkvz = qkvz2d.reshape(&out_shape)?;
       Ok(Some(qkvz))
                   let x = int_array_bf16(&xv, m as i64, k as i64);
                   let w = int_array_bf16(&wv, n as i64, k as i64);
                   let out = matmul_int8(&x, &w).unwrap();
                   out.eval();
   /// # Arguments
                   assert_eq!(out.dtype().unwrap(), DType::Int32, "output must be int32");
                   let got = out.to_int32().unwrap();
   /// * `cache` - Optional ArraysCache with 2 slots: [conv_state, recurrent_state]
                   assert_eq!(got.len(), m * n, "size m={m} k={k} n={n}");
   /// # Returns
                   // i32 reference: ref[m,n] = sum_k x[m,k]*w[n,k]. Values in
                   // [-127,127] over k<=9216 fit comfortably in i32
                   // (127*127*9216 ~ 1.49e8 << 2.1e9).
                   let mut bad = 0usize;
                   let mut first: Option<(usize, i32, i32)> = None;
       mut cache: Option<&mut ArraysCache>,
                       for ni in 0..n {
                           let mut acc: i32 = 0;
                           for ki in 0..k {
                               acc += xv[mi * k + ki] * wv[ni * k + ki];
                           }
       // E51: when the stacked weight is available, do one matmul + two
       // slices. Env-toggle MLX_DISABLE_E51_STACKED_GDN_IN_PROJ=1 reverts to
                               bad += 1;
       let qkvz_dim = (self.key_dim * 2 + self.value_dim * 2) as i64;
                                   first = Some((mi * n + ni, g, acc));
       let (qkvz, ba) = if let Some(qkvz_i8) = self.try_forward_qkvz_int8(x)? {
           // NA int8 W8A8 prefill (opt-in): qkvz via int8 GEMM (bf16 out), ba
           // stays bf16 (recurrence gates excluded from quant). This forgoes
           // the E51 single-matmul fusion ONLY on the int8 path.
           let ba = self.in_proj_ba.forward(x)?;
                       bad, 0,
                       "NOT bit-exact at M={m} K={k} N={n}: {bad} mismatches, first {first:?}"
           && std::env::var("MLX_DISABLE_E51_STACKED_GDN_IN_PROJ").is_err()
                   eprintln!("[s1] BIT-EXACT M={m} K={k} N={n}");
           let combined = x.matmul(wqb_t)?; // [B, T, qkvz_dim + ba_dim]
           let qkvz = combined.slice_axis(2, 0, qkvz_dim)?;
           let ba = combined.slice_axis(2, qkvz_dim, qkvz_dim + ba_dim)?;
           (qkvz, ba)
       } else {
   // ====================== STAGE 1b (DECISIVE) ======================
   // GATE S1b: int32 output BIT-EXACT on PARTIAL tiles — the one open
   // correctness question for the production `mode::multiply` (overwrite, no
   // output zero-fill) GEMM (`int8_gemm_core_nofill`).
///   mask: [B, T] or None
   // `mode::multiply` overwrites C with NO MLX init_value fill, so it is only
   // safe if EVERY in-bounds output element is written exactly once — including
   // when the 128x64 tile overhangs M (M%128!=0) AND N (N%64!=0). The S1 test
       let a = ba.slice_axis(2, self.num_v_heads as i64, (self.num_v_heads * 2) as i64)?;
   // DOUBLE-PARTIAL corner tile (M%128!=0 AND N%64!=0 simultaneously) are
   // untested there. A garbage tail would surface here as a non-bit-exact
       // qkv: [B, T, key_dim*2 + value_dim] = [B, T, conv_dim]
       // z: [B, T, value_dim]
   // M in {300, 1025} (both %128!=0) x N in {2560 (%64==0), 2570 (%64!=0)}
   // x K in {2560, 9216}. M=1025 ^ N=2570 is the double-partial corner.
   // Same deterministic integer reference as S1.
           self.conv_dim as i64,
           (self.key_dim * 2 + self.value_dim * 2) as i64,
) -> Result<(MxArray, MxArray)> {
   let batch = q.shape_at(0)?;
               "[s1b] SKIP: gpu gen {} < 17 (NA matmul2d needs M5+)",
   let num_v_heads = v.shape_at(2)?;
   let v_dim = v.shape_at(3)?;
   let k_dim = q.shape_at(3)?;
       }
   // When use_kernel=false, use only ops-based paths for full differentiability (autograd).
   // compute_g builds a standard MLX expression graph via C++ (differentiable),
   // but the fused_gdn_gating Metal kernel and recurrence kernels are NOT differentiable.
       let mut state: u64 = 0xdead_1025_0300_2570;
       let beta = Activations::sigmoid(b)?;
       // compute_g returns exp(g_log) directly — use it as the decay gate without log/exp round-trip
       let g = compute_g(a_log, a, dt_bias)?;
               for &n in &ns {
                   // x[m,k], w[n,k] in [-127,127].
       let (q, k) = if num_v_heads != num_k_heads {
                   for v in xv.iter_mut() {
                       *v = next_int(&mut state, -127, 127);
                   "GatedDelta: num_k_heads is 0, cannot compute GQA repeat factor",
                   let mut wv = vec![0i32; n * k];
                   for v in wv.iter_mut() {
                       *v = next_int(&mut state, -127, 127);
               return Err(Error::from_reason(format!(
                   "GatedDelta: num_v_heads ({}) must be divisible by num_k_heads ({})",
                   let x = int_array_bf16(&xv, m as i64, k as i64);
                   let w = int_array_bf16(&wv, n as i64, k as i64);
                   // PRODUCTION path: matmul_int8 -> int8_gemm_core_nofill
           let repeat_factor = num_v_heads / num_k_heads;
           let q_expanded = q.repeat(repeat_factor as i32, 2)?;
           let k_expanded = k.repeat(repeat_factor as i32, 2)?;
           (q_expanded, k_expanded)
                   assert_eq!(out.dtype().unwrap(), DType::Int32, "output must be int32");
                   let got = out.to_int32().unwrap();
                   let got: &[i32] = &got;
                   assert_eq!(got.len(), m * n, "size m={m} k={k} n={n}");
       let initial_state = match state {
                   // SAME integer reference as S1: ref[m,n] = sum_k x[m,k]*w[n,k].
           None => MxArray::zeros(&[batch, num_v_heads, v_dim, k_dim], Some(v.dtype()?))?,
                   // element integer math is byte-for-byte identical to the serial
                   // S1 loop; only the iteration is split) so the O(M*N*K) debug
                   // reference for the M=1025/K=9216 corner stays in the seconds,
                   // not minutes. Each thread returns its first-mismatch + count.
                   let nthreads = std::thread::available_parallelism()
   // Compute beta = sigmoid(b) and g_log = -exp(A_log) * softplus(a + dt_bias)
   // Try fused Metal kernel first (single dispatch), fall back to separate ops.
   // g_log is the log-space gate; per-step kernel needs exp(g_log), chunked needs g_log directly.
   let (beta, g_log) = match fused_gdn_gating(b, a, a_log, dt_bias, num_v_heads as i32) {
                   let (xv_r, wv_r) = (&xv, &wv);
                   let results: Vec<(usize, Option<(usize, i32, i32)>)> =
           let beta = beta_flat.reshape(&[batch, seq_len_tmp, num_v_heads])?;
           let g_log = g_flat.reshape(&[batch, seq_len_tmp, num_v_heads])?;
                           for t in 0..nthreads {
                               let m_lo = t * chunk;
                               let m_hi = ((t + 1) * chunk).min(m);
                               handles.push(scope.spawn(move || {
           // compute_g returns exp(g_log), so take log to get g_log
                                   let mut first: Option<(usize, i32, i32)> = None;
                                   for mi in m_lo..m_hi {
                                       for ni in 0..n {
                                           let mut acc: i32 = 0;
                                           for ki in 0..k {
                                               acc += xv_r[mi * k + ki] * wv_r[ni * k + ki];
   // GQA head expansion: repeat q,k from Hk to Hv heads
                                           let g = got[mi * n + ni];
                                           if g != acc {
                                               bad += 1;
               "GatedDelta: num_k_heads is 0, cannot compute GQA repeat factor",
                                                   first = Some((mi * n + ni, g, acc));
                                               }
                                           }
           return Err(Error::from_reason(format!(
               "GatedDelta: num_v_heads ({}) must be divisible by num_k_heads ({})",
               "GatedDelta: num_k_heads is 0, cannot compute GQA repeat factor",
                               }));
                           }
                           handles.into_iter().map(|h| h.join().unwrap()).collect()
       let q_expanded = q.repeat(repeat_factor as i32, 2)?; // [B, T, Hv, Dk]
               "GatedDelta: num_v_heads ({}) must be divisible by num_k_heads ({})",
                   // First mismatch by lowest flat index across all row chunks.
                   let first: Option<(usize, i32, i32)> = results
                       .iter()
       let repeat_factor = num_v_heads / num_k_heads;
       let q_expanded = q.repeat(repeat_factor as i32, 2)?; // [B, T, Hv, Dk]
       let k_expanded = k.repeat(repeat_factor as i32, 2)?; // [B, T, Hv, Dk]
   // Use v's dtype to avoid f32 promotion for bf16/f16 models
                           "[s1b] MISMATCH M={m} K={k} N={n} at flat={idx} \
                            (mi={},ni={}) got={g} want={acc}",
       None => MxArray::zeros(&[batch, num_v_heads, v_dim, k_dim], Some(v.dtype()?))?,
                           idx % n
   // Initialize state if not provided: [B, Hv, Dv, Dk]
   // Use v's dtype to avoid f32 promotion for bf16/f16 models
                   let corner = if m % 128 != 0 && n % 64 != 0 {
   // Use Metal kernel for recurrence (requires Dk divisible by 32 for SIMD register blocking)
       None => MxArray::zeros(&[batch, num_v_heads, v_dim, k_dim], Some(v.dtype()?))?,
       // Chunked kernel for long sequences on M5+ (Neural Accelerator-accelerated prefill).
       // On M1–M4, per-step kernel is faster (no tensor cores for O(BT^2) matmuls).
       // Chunked kernel needs g in log-space directly (no exp/log roundtrip).
       if seq_len >= CHUNK_THRESHOLD
                       "NOT bit-exact at M={m} K={k} N={n}{corner}: {bad} mismatches, first {first:?}"
           && gpu_architecture_gen() >= CHUNK_MIN_GPU_GEN
       // Chunked kernel for long sequences on M5+ — wins via M5's memory bandwidth +
       // amortized Metal launch overhead, NOT tensor cores (the kernel is all scalar-FMA
       // + simd_sum, no simdgroup_matrix). On M1–M4 the per-step kernel is faster.
       // Chunked kernel needs g in log-space directly (no exp/log roundtrip).
                   // Fall through to per-step kernel
           && mask.is_none()
   // ====================== v1 FUSED-QUANT PARITY ======================
   // GATE: the fused activation-quant kernel (v1 kernel 2) must be BIT-IDENTICAL
           match gated_delta_chunked(&q, &k, v, &g_log, &beta, &initial_state) {
   // Exercised over the S2 shapes + a couple of MLP-real shapes, with realistic
   // bf16 magnitudes (so the per-row absmax / round / clip paths are all hit).
       if let Ok(result) = gated_delta_kernel(&q, &k, v, &g, &beta, &initial_state, mask) {
   fn v1_fused_quant_bit_parity() {
       if gpu_gen() < 17 {
           eprintln!("[v1q] SKIP gpu gen {} < 17", gpu_gen());
           return;
   // Ops-based sequential loop fallback (also needs exp(g_log))
       // (M, K). K must be %16==0. Mix of S2 shapes + MLP-real + a tail M.
       if let Ok(result) = gated_delta_kernel(&q, &k, v, &g, &beta, &initial_state, mask) {
           (512usize, 2560usize),
           (256, 2560),
           (4096, 2560),
           (4096, 9216),
   // Ops-based sequential loop fallback (also needs exp(g_log))
           (4096, 17408),
   gated_delta_ops(&q, &k, v, &g, &beta, &initial_state, mask)
       let mut state: u64 = 0x7e57_0f00_d15e_a5e0;
       for &(m, k) in &shapes {
           // Realistic-ish bf16 activations in ~[-0.2,0.2], plus deliberate
           // outliers so the per-row absmax differs from the bulk.
           let mut xf = vec![0f32; m * k];
           for v in xf.iter_mut() {
               *v = next_int(&mut state, -200, 200) as f32 / 1000.0;
           }
           // Inject one large outlier per row to stress absmax + clip.
           for mi in 0..m {
               let col = (next_int(&mut state, 0, (k - 1) as i32)) as usize;
               xf[mi * k + col] = if mi % 2 == 0 { 1.7 } else { -1.3 };
           }
           let x = MxArray::from_float32(&xf, &[m as i64, k as i64])
               .unwrap()
               .astype(DType::BFloat16)
               .unwrap();
           x.eval();

           let (qf, sxf) = act_quant_fused(&x).unwrap();
           let (ql, sxl) = act_quant_lazy(&x).unwrap();
           qf.eval();
           ql.eval();
           sxf.eval();
           sxl.eval();

           // int8 (widened to int32) must be BIT-IDENTICAL.
           let a = qf.to_int32().unwrap();
           let a: &[i32] = &a;
           let b = ql.to_int32().unwrap();
           let b: &[i32] = &b;
           assert_eq!(a.len(), m * k);
           assert_eq!(b.len(), m * k);
           let mut bad = 0usize;
           let mut first: Option<(usize, i32, i32)> = None;
           for i in 0..a.len() {
               if a[i] != b[i] {
                   bad += 1;
                   if first.is_none() {
                       first = Some((i, a[i], b[i]));
                   }
               }
           }
           assert_eq!(
               bad, 0,
               "fused-quant int8 NOT bit-identical at M={m} K={k}: {bad} diffs, first {first:?}"
           );

           // s_x must match exactly (same f32 arithmetic).
           let sa = sxf.to_float32().unwrap();
           let sa: &[f32] = &sa;
           let sb = sxl.to_float32().unwrap();
           let sb: &[f32] = &sb;
           assert_eq!(sa.len(), m);
           let mut bad_sx = 0usize;
           for i in 0..m {
               // Exact: both compute max(absmax,1e-12)/127 in f32 over the same
               // bf16-upcast values. Allow a 0-eps but report any drift.
               if (sa[i] - sb[i]).abs() > 0.0 {
                   bad_sx += 1;
               }
           }
           assert_eq!(
               bad_sx, 0,
               "fused-quant s_x NOT exact at M={m} K={k}: {bad_sx} diffs"
           );
           eprintln!("[v1q] BIT-IDENTICAL M={m} K={k} (int8 + s_x exact)");
       }
   }

   // ====================== v1 FUSED-RESCALE PARITY ======================
   // GATE: the fused int32->bf16 rescale kernel (v1 kernel 3) must match the
   // lazy multi-pass rescale to bf16 EPS (ideally bit-identical — both do
   // (acc*s_x)*s_w in f32 then narrow to bf16). Realistic acc/scale magnitudes.
   #[test]
   fn v1_fused_rescale_parity() {
       if gpu_gen() < 17 {
           eprintln!("[v1r] SKIP gpu gen {} < 17", gpu_gen());
           return;
       }
       let shapes = [
           (512usize, 9216usize),
           (256, 2560),
           (4096, 18432),
           (300, 34816),
           // N % 256 != 0 cases: exercise the PARTIAL threadgroup-in-x of the 2D
           // rescale dispatch (grid.x = N, threadgroup.x = 256). The other N here
           // are all multiples of 256, so without these the partial-x tail (where
           // dispatch_threads launches a sub-256 final group) is untested. 2570
           // and 9300 are both %256 != 0; 2570 also crosses M into a non-round M.
           (300, 2570),
           (1025, 9300),
       ];
       let mut state: u64 = 0x0badf00d_deadbeef;
       for &(m, n) in &shapes {
           // acc int32 in a realistic GEMM-accumulator range.
           let mut accv = vec![0i32; m * n];
           for v in accv.iter_mut() {
               *v = next_int(&mut state, -2_000_000, 2_000_000);
           }
           // s_x [M,1] and s_w [N] f32 in the load-time scale range (~absmax/127).
           let mut sxv = vec![0f32; m];
           for v in sxv.iter_mut() {
               *v = (next_int(&mut state, 1, 4000) as f32) / 1e6;
           }
           let mut swv = vec![0f32; n];
           for v in swv.iter_mut() {
               *v = (next_int(&mut state, 1, 4000) as f32) / 1e6;
           }
           let acc = MxArray::from_int32(&accv, &[m as i64, n as i64]).unwrap();
           let s_x = MxArray::from_float32(&sxv, &[m as i64, 1]).unwrap();
           let s_w = MxArray::from_float32(&swv, &[n as i64]).unwrap();
           acc.eval();
           s_x.eval();
           s_w.eval();

           let yf = rescale_fused(&acc, &s_x, &s_w).unwrap();
           let yl = rescale_lazy(&acc, &s_x, &s_w).unwrap();
           yf.eval();
           yl.eval();
           assert_eq!(yf.dtype().unwrap(), DType::BFloat16);
           assert_eq!(yl.dtype().unwrap(), DType::BFloat16);

           // Compare as raw bf16 bits via f32 readback (both narrowed identically).
           let a = yf.astype(DType::Float32).unwrap().to_float32().unwrap();
           let a: &[f32] = &a;
           let b = yl.astype(DType::Float32).unwrap().to_float32().unwrap();
           let b: &[f32] = &b;
           assert_eq!(a.len(), m * n);
           let mut bad = 0usize;
           let mut max_rel = 0.0f64;
           let mut first: Option<(usize, f32, f32)> = None;
           for i in 0..a.len() {
               let da = a[i] as f64;
               let db = b[i] as f64;
               let denom = db.abs().max(1e-6);
               let rel = (da - db).abs() / denom;
               max_rel = max_rel.max(rel);
               // bf16 has ~8 bits mantissa (~1/256 rel); equal narrowing should
               // be bit-identical, so any 1-ULP slip is at most ~1/256.
               if (da - db).abs() > 0.0 {
                   bad += 1;
                   if first.is_none() {
                       first = Some((i, a[i], b[i]));
                   }
               }
           }
           // Gate to bf16 eps (well under 1 ULP). Report exact-mismatch count too.
           assert!(
               max_rel <= 1.0 / 256.0,
               "fused-rescale beyond bf16 eps at M={m} N={n}: max_rel={max_rel:.6} \
                ({bad} non-identical, first {first:?})"
           );
           eprintln!(
               "[v1r] M={m} N={n}: max_rel={max_rel:.8} non_identical={bad}/{} (<= bf16 eps)",
               m * n
           );
       }
   }

   // ===================== STAGE 4b RESIDUAL PROFILE =====================
   // Localizes the residual ~18-22% prefill regression after the lazy +
   // load-time-transpose fixes. Times, at the real Qwen3.5-4B MLP shapes and
   // M=4096, the pieces of ONE MLP forward so we can attribute the gap:
   //   * bf16 fused-equivalent (2 matmuls + swiglu) — the BASELINE bar
   //   * int8 full W8A8 MLP (quant+gemm+rescale, both projections)
   //   * activation-quant ONLY (absmax/round/clip/astype) for both projections
   //   * int8 GEMM ONLY (pre-quantized acts, no quant, no rescale)
   // Run explicitly:
   //   cargo test -p mlx-core --lib int8_gemm::tests::profile_residual \
   //     -- --ignored --nocapture
   #[test]
   #[ignore = "manual residual profiler; run with --ignored"]
   fn profile_residual() {
       use crate::array::memory::synchronize;
       use std::time::Instant;
       if gpu_gen() < 17 {
           eprintln!("[profile] SKIP gpu gen {} < 17", gpu_gen());
           return;
       }
       let m: i64 = 4096;
       let hidden: i64 = 2560;
       let inter: i64 = 9216;
       let two_inter = 2 * inter;

       // Random bf16 activation + weights at the real shapes.
       let x = MxArray::random_normal(&[m, hidden], 0.0, 0.05, Some(DType::BFloat16)).unwrap();
       // gate_up weight [2*inter, hidden] (N,K); down weight [hidden, inter].
       let w_gu =
           MxArray::random_normal(&[two_inter, hidden], 0.0, 0.02, Some(DType::BFloat16)).unwrap();
       let w_d =
           MxArray::random_normal(&[hidden, inter], 0.0, 0.02, Some(DType::BFloat16)).unwrap();
       // bf16 transposed weights for the matmul baseline.
       let w_gu_t = w_gu.transpose(Some(&[1, 0])).unwrap();
       let w_d_t = w_d.transpose(Some(&[1, 0])).unwrap();
       w_gu_t.eval();
       w_d_t.eval();
       // int8 pre-quantized weights (load-time form).
       let (gu_i8, gu_s) = quantize_weight_int8(&w_gu).unwrap();
       let (d_i8, d_s) = quantize_weight_int8(&w_d).unwrap();
       gu_i8.eval();
       gu_s.eval();
       d_i8.eval();
       d_s.eval();
       x.eval();
       synchronize();

       let iters = 50;
       let warm = 10;

       // ---- (A) bf16 fused-equivalent: 2 matmuls + swiglu (silu*up) ----
       let bf16_mlp = || {
           let gate_up = x.matmul(&w_gu_t).unwrap(); // [M, 2*inter]
           let gate = gate_up.slice(&[0, 0], &[m, inter]).unwrap();
           let up = gate_up.slice(&[0, inter], &[m, two_inter]).unwrap();
           let gated = Activations::silu(&gate).unwrap().mul(&up).unwrap();
           gated.matmul(&w_d_t).unwrap()
       };
       // ---- (B) int8 full W8A8 MLP ----
       let int8_mlp = || {
           let gate_up = int8_w8a8_matmul(&x, &gu_i8, &gu_s).unwrap();
           let gate = gate_up.slice(&[0, 0], &[m, inter]).unwrap();
           let up = gate_up.slice(&[0, inter], &[m, two_inter]).unwrap();
           let gated = Activations::silu(&gate).unwrap().mul(&up).unwrap();
           int8_w8a8_matmul(&gated, &d_i8, &d_s).unwrap()
       };

       let bench = |label: &str, f: &dyn Fn() -> MxArray| {
           for _ in 0..warm {
               let o = f();
               o.eval();
           }
           synchronize();
           let t = Instant::now();
           for _ in 0..iters {
               let o = f();
               o.eval();
           }
           synchronize();
           let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
           eprintln!("[profile] {label:<28} {ms:8.3} ms/iter");
           ms
       };

       eprintln!(
           "[profile] M={m} hidden={hidden} inter={inter} (gate_up N={two_inter}, down N={hidden})"
       );
       // ---- (C) per-token activation quant ONLY, for the gate_up input ----
       // Mirrors mlx_w8a8_linear's quant block: absmax/round/clip/astype.
       // We emulate it in Rust ops over x[M,hidden] so we measure the same
       // arithmetic the C++ op builds into the graph.
       let quant_only_gu = || {
           let xf = x.astype(DType::Float32).unwrap();
           let absmax = xf.abs().unwrap().max(Some(&[1]), Some(true)).unwrap(); // [M,1]
           let sx = absmax.div_scalar(127.0).unwrap();
           let xq = xf.div(&sx).unwrap().round().unwrap();
           let xq = xq.clip(Some(-127.0), Some(127.0)).unwrap();
           xq.astype(DType::BFloat16).unwrap() // proxy for int8 cast (no Int8 dtype)
       };
       // ---- (D) int8 GEMM core ONLY (raw matmul_int8 on pre-int8 acts) ----
       // x already in [-127,127] integer range so values cast cleanly.
       let x_int = x
           .div_scalar(0.05)
           .unwrap()
           .round()
           .unwrap()
           .clip(Some(-127.0), Some(127.0))
           .unwrap()
           .astype(DType::BFloat16)
           .unwrap();
       let w_gu_int = w_gu
           .div_scalar(0.02)
           .unwrap()
           .round()
           .unwrap()
           .clip(Some(-127.0), Some(127.0))
           .unwrap()
           .astype(DType::BFloat16)
           .unwrap();
       x_int.eval();
       w_gu_int.eval();
       synchronize();
       let int8_gemm_gu = || matmul_int8(&x_int, &w_gu_int).unwrap();
       // ---- (E) bf16 matmul ONLY at the gate_up shape (the bar the GEMM beats) ----
       let bf16_gemm_gu = || x.matmul(&w_gu_t).unwrap();

       let t_bf16 = bench("A bf16 fused MLP", &bf16_mlp);
       let t_int8 = bench("B int8 W8A8 MLP", &int8_mlp);
       let t_quant = bench("C act-quant only (gate_up)", &quant_only_gu);
       let t_i8gemm = bench("D int8 GEMM only (gate_up)", &int8_gemm_gu);
       let t_bf16gemm = bench("E bf16 GEMM only (gate_up)", &bf16_gemm_gu);
       eprintln!(
           "[profile] int8/bf16 MLP ratio = {:.3} (>1 = int8 slower)",
           t_int8 / t_bf16
       );
       eprintln!(
           "[profile] bf16/int8 prefill-tps-equiv = {:.3} (matches harness prefill ratio)",
           t_bf16 / t_int8
       );
       eprintln!(
           "[profile] gate_up GEMM: int8={t_i8gemm:.3}ms bf16={t_bf16gemm:.3}ms \
            int8/bf16={:.3} (kernel-only; <1 = int8 GEMM faster)",
           t_i8gemm / t_bf16gemm
       );
       eprintln!(
           "[profile] act-quant (gate_up) = {t_quant:.3} ms; as %% of one int8 GEMM = {:.1}%%",
           100.0 * t_quant / t_i8gemm
       );
   }

   // ========================= v1 FUSED MLP PROFILE =========================
   // Reports the v1 fused int8 W8A8 MLP vs bf16 fused MLP wall ratio at M=4096
   // for BOTH the 4B and 27B MLP shapes, with the per-piece breakdown
   // (fused-quant ms / GEMM ms / fused-rescale ms / swiglu ms). The bf16
   // baseline thermally throttles, so we run the int8-vs-bf16 comparison 3x and
   // ratio the (thermally stable) int8 MLP against the COOLEST bf16 sample.
   //
   // Run:
   //   cargo test -p mlx-core --lib int8_gemm::tests::profile_fused \
   //     -- --ignored --nocapture
   #[test]
   #[ignore = "manual v1 fused MLP profiler; run with --ignored"]
   fn profile_fused() {
       use super::{act_quant_fused, matmul_int8_kn_nofill, rescale_fused};
       use crate::array::memory::synchronize;
       use std::time::Instant;
       if gpu_gen() < 17 {
           eprintln!("[fused] SKIP gpu gen {} < 17", gpu_gen());
           return;
       }
       let iters = 50;
       let warm = 12;

       let bench = |f: &dyn Fn() -> MxArray| -> f64 {
           for _ in 0..warm {
               let o = f();
               o.eval();
           }
           synchronize();
           let t = Instant::now();
           for _ in 0..iters {
               let o = f();
               o.eval();
           }
           synchronize();
           t.elapsed().as_secs_f64() * 1e3 / iters as f64
       };

       let run = |label: &str, m: i64, hidden: i64, inter: i64| {
           let two_inter = 2 * inter;
           // bf16 activations + weights at realistic magnitudes.
           let x = MxArray::random_normal(&[m, hidden], 0.0, 0.05, Some(DType::BFloat16)).unwrap();
           let w_gu =
               MxArray::random_normal(&[two_inter, hidden], 0.0, 0.02, Some(DType::BFloat16))
                   .unwrap();
           let w_d =
               MxArray::random_normal(&[hidden, inter], 0.0, 0.02, Some(DType::BFloat16)).unwrap();
           let w_gu_t = w_gu.transpose(Some(&[1, 0])).unwrap();
           let w_d_t = w_d.transpose(Some(&[1, 0])).unwrap();
           w_gu_t.eval();
           w_d_t.eval();
           let (gu_i8, gu_s) = quantize_weight_int8(&w_gu).unwrap();
           let (d_i8, d_s) = quantize_weight_int8(&w_d).unwrap();
           gu_i8.eval();
           gu_s.eval();
           d_i8.eval();
           d_s.eval();
           x.eval();
           synchronize();

           // ---- Full MLP: bf16 fused vs int8 W8A8 (production path) ----
           let bf16_mlp = || {
               let gate_up = x.matmul(&w_gu_t).unwrap();
               let gate = gate_up.slice(&[0, 0], &[m, inter]).unwrap();
               let up = gate_up.slice(&[0, inter], &[m, two_inter]).unwrap();
               let gated = Activations::silu(&gate).unwrap().mul(&up).unwrap();
               gated.matmul(&w_d_t).unwrap()
           };
           let int8_mlp = || {
               let gate_up = int8_w8a8_matmul(&x, &gu_i8, &gu_s).unwrap();
               let gate = gate_up.slice(&[0, 0], &[m, inter]).unwrap();
               let up = gate_up.slice(&[0, inter], &[m, two_inter]).unwrap();
               let gated = Activations::silu(&gate).unwrap().mul(&up).unwrap();
               int8_w8a8_matmul(&gated, &d_i8, &d_s).unwrap()
           };

           // 3 runs; int8 is thermally stable, bf16 throttles -> use coolest bf16.
           let mut bf16_runs = [0.0f64; 3];
           let mut int8_runs = [0.0f64; 3];
           for r in 0..3 {
               bf16_runs[r] = bench(&bf16_mlp);
               int8_runs[r] = bench(&int8_mlp);
           }
           let bf16_cool = bf16_runs.iter().cloned().fold(f64::INFINITY, f64::min);
           let int8_med = {
               let mut v = int8_runs;
               v.sort_by(|a, b| a.partial_cmp(b).unwrap());
               v[1]
           };
           let int8_cool = int8_runs.iter().cloned().fold(f64::INFINITY, f64::min);

           // ---- Per-piece breakdown (gate_up shape: K=hidden, N=two_inter) ----
           // Fused activation-quant (the [M,hidden] input).
           let t_quant = bench(&|| {
               let (q, _s) = act_quant_fused(&x).unwrap();
               q
           });
           // Build a pre-int8 [M,hidden] operand + the [K,N] weight for the GEMM.
           let (x_i8_i32, sx) = act_quant_fused(&x).unwrap();
           let x_i8_bf16 = x_i8_i32.astype(DType::BFloat16).unwrap(); // int-valued
           x_i8_bf16.eval();
           sx.eval();
           // gu_i8 is the opaque [K,N] int8 kernel operand from load.
           let t_gemm = bench(&|| matmul_int8_kn_nofill(&x_i8_bf16, &gu_i8).unwrap());
           // Fused rescale on an int32 acc of the gate_up shape.
           let acc = matmul_int8_kn_nofill(&x_i8_bf16, &gu_i8).unwrap();
           acc.eval();
           let t_rescale = bench(&|| rescale_fused(&acc, &sx, &gu_s).unwrap());
           // swiglu (silu(gate)*up) over the gate_up output [M, two_inter].
           let gate_up_bf16 =
               MxArray::random_normal(&[m, two_inter], 0.0, 0.1, Some(DType::BFloat16)).unwrap();
           gate_up_bf16.eval();
           let t_swiglu = bench(&|| {
               let gate = gate_up_bf16.slice(&[0, 0], &[m, inter]).unwrap();
               let up = gate_up_bf16.slice(&[0, inter], &[m, two_inter]).unwrap();
               Activations::silu(&gate).unwrap().mul(&up).unwrap()
           });

           eprintln!(
               "[fused] === {label}: M={m} hidden={hidden} inter={inter} \
                (gate_up N={two_inter} K={hidden}; down N={hidden} K={inter}) ==="
fn apply_weights_inner(
   inner: &mut Qwen35Inner,
               "[fused] bf16 MLP runs (ms): {:.3} {:.3} {:.3}  -> coolest {bf16_cool:.3}",
               bf16_runs[0], bf16_runs[1], bf16_runs[2]
   quant_bits: i32,
   quant_group_size: i32,
               "[fused] int8 MLP runs (ms): {:.3} {:.3} {:.3}  -> median {int8_med:.3} coolest {int8_cool:.3}",
               int8_runs[0], int8_runs[1], int8_runs[2]
) -> Result<()> {
   let is_quantized = is_quantized_checkpoint(params);
               "[fused] RATIO int8/bf16 (vs coolest bf16): median={:.3} coolest={:.3}  (<1.0 = int8 FASTER)",
   let default_mode = resolve_default_mode(top_level_mode, is_mxfp8);
   let default_plq = default_per_layer_quant(quant_bits, quant_group_size, default_mode);
           );
   let try_build_ql = |params: &HashMap<String, MxArray>, prefix: &str| {
               "[fused] per-piece (gate_up shape): fused-quant={t_quant:.3}ms  \
                GEMM={t_gemm:.3}ms  fused-rescale={t_rescale:.3}ms  swiglu={t_swiglu:.3}ms"
       // the two sides disagree we pick the higher-precision combination:
       //   1. higher `bits` wins,
       //   2. on equal bits, prefer Affine > Mxfp8 > Mxfp4 (most precise mode).
       eprintln!("[fused] === v1 FUSED int8 W8A8 MLP vs bf16, M=4096 ===");
       // 4B: hidden=2560, inter=9216.
       run("4B", 4096, 2560, 9216);
       // 27B: hidden=5120, inter=17408.
               if prefix.ends_with(".in_proj_qkvz") {
                   let base = prefix.strip_suffix(".in_proj_qkvz").unwrap();
                   let qkv = per_layer_quant.get(&format!("{}.in_proj_qkv", base));
   // =================== CLEAN PURE-GEMM THROUGHPUT PROFILE ===================
   // DIAGNOSTIC (measurement only). Times the in-engine int8 GEMM with a
   // PRE-TRANSPOSED [K,N] weight (zero per-call transpose) vs bf16 matmul at the
   // real Qwen3.5-4B MLP shapes, M in {512,4096}. Also times the OLD transpose-
                   let b_val = per_layer_quant.get(&format!("{}.in_proj_b", base));
                   let a_val = per_layer_quant.get(&format!("{}.in_proj_a", base));
                   merge_per_layer(b_val, a_val, "in_proj_ba", "b", "a")
               } else {
   //   cargo test -p mlx-core --lib int8_gemm::tests::profile_clean_gemm \
   //     -- --ignored --nocapture
           })
   #[ignore = "manual clean pure-GEMM throughput profiler; run with --ignored"]
   fn profile_clean_gemm() {
           PerLayerMode::Mxfp4 => try_build_mxfp4_quantized_linear(params, prefix),
           PerLayerMode::Mxfp8 => try_build_mxfp8_quantized_linear(params, prefix),
           PerLayerMode::Nvfp4 => try_build_nvfp4_quantized_linear(params, prefix),
           PerLayerMode::Affine => {
               try_build_quantized_linear(params, prefix, plq.group_size, plq.bits)
           return;
       }
   };
       // ---- bit-exact cross-check: matmul_int8_kn == matmul_int8 (small case) ----
       // Confirms removing the per-call transpose did not break the math: the
       // pre-transposed [K,N] weight must equal int8_weight_to_kn(w).
       let weight = params.get("embedding.weight").ok_or_else(|| {
           Error::from_reason("Missing embedding.weight for quantized embedding")
           let mut state: u64 = 0xabcd_1234_5678_9999;
       let biases = params.get("embedding.biases");
           for v in xv.iter_mut() {
               *v = next_int(&mut state, -127, 127);
           .copied()
           let mut wv = vec![0i32; n * k];
           for v in wv.iter_mut() {
               *v = next_int(&mut state, -127, 127);
           .load_quantized(weight, scales, biases, plq.group_size, plq.bits)?;
           let x = int_array_bf16(&xv, m as i64, k as i64); // [M,K]
           "Loaded quantized embedding ({}-bit, quantized_matmul on forward)",
           // Old contaminated path (transposes w internally).
           let out_old = matmul_int8(&x, &w).unwrap();
   } else if let Some(w) = params.get("embedding.weight") {
           // Pre-transpose w -> [K,N] contiguous via quantize? No — that rescales.
           // Build the [K,N] int-valued operand directly: transpose then force
           // contiguity by a round-trip through from_float32 (guaranteed C-order),
           // so matmul_int8_kn casts a genuinely row-contiguous [K,N] buffer.
           let mut wkn = vec![0f32; k * n]; // [K,N]: wkn[k*N + n] = w[n,k]
       inner.final_norm.set_weight(w)?;
               for ki in 0..k {
                   wkn[ki * n + ni] = wv[ni * k + ki] as f32;
               }
   if let Some(ref mut head) = inner.lm_head {
           let w_kn = MxArray::from_float32(&wkn, &[k as i64, n as i64])
           let weight = params.get("lm_head.weight").ok_or_else(|| {
               Error::from_reason("Missing lm_head.weight for quantized lm_head")
               .unwrap();
           let biases = params.get("lm_head.biases");
           let out_new = matmul_int8_kn(&x, &w_kn).unwrap();
               .get("lm_head")
           let a = out_old.to_int32().unwrap();
               .unwrap_or(default_plq);
           head.load_quantized(weight, scales, biases, plq.group_size, plq.bits)?;
           let b: &[i32] = &b;
               "Loaded quantized lm_head ({}-bit, quantized_matmul on forward)",
           let mut bad = 0usize;
           for i in 0..a.len() {
       } else if let Some(w) = params.get("lm_head.weight") {
           head.set_weight(w)?;
               }
           }
           assert_eq!(
   // Per-layer weights
               "matmul_int8_kn NOT bit-exact vs matmul_int8: {bad} diffs"
           );
           eprintln!(
               "[clean] cross-check BIT-EXACT: matmul_int8_kn == matmul_int8 (M={m} K={k} N={n})"
           );
           // nofill (mode::multiply) must also be bit-exact (overwrite, not accum).
           let out_nf = matmul_int8_kn_nofill(&x, &w_kn).unwrap();
           out_nf.eval();
           let c = out_nf.to_int32().unwrap();
           let c: &[i32] = &c;
           let mut bad2 = 0usize;
           for i in 0..a.len() {
               if a[i] != c[i] {
                   bad2 += 1;
               }
           }
           assert_eq!(bad2, 0, "matmul_int8_kn_nofill NOT bit-exact: {bad2} diffs");
           eprintln!("[clean] cross-check BIT-EXACT: matmul_int8_kn_nofill == matmul_int8");
       }

       let iters = 50;
       let warm = 10;

       // Generic timed comparison for one (M,K,N) shape.
       // Builds: x[M,K] bf16-int, bf16 weight w[N,K] + w^T[K,N], pre-transposed
       // int8-valued [K,N] operand (materialized contiguous, evaled ONCE).
       let run_shape = |label: &str, m: usize, k: usize, n: usize| {
           let mut state: u64 = 0x5151_a7a7_3939_c0c0 ^ ((m as u64) << 40) ^ ((n as u64) << 8);
           // x[M,K] integers in [-127,127] as bf16.
           let mut xv = vec![0i32; m * k];
           for v in xv.iter_mut() {
               *v = next_int(&mut state, -127, 127);
           }
           let x = int_array_bf16(&xv, m as i64, k as i64); // bf16 [M,K]

           // w[N,K] integers as bf16 (for matmul_int8 contaminated path + bf16 ref).
           let mut wv = vec![0i32; n * k];
           for v in wv.iter_mut() {
               *v = next_int(&mut state, -127, 127);
           }
           let w_nk = int_array_bf16(&wv, n as i64, k as i64); // bf16 [N,K]
           let w_t = w_nk.transpose(Some(&[1, 0])).unwrap(); // [K,N] (strided view)
           // Force the bf16 baseline weight contiguous (matches a stored w^T).
           let w_t = w_t.astype(DType::BFloat16).unwrap();

           // Pre-transposed int8-valued [K,N] operand, materialized row-contiguous.
           let mut wkn = vec![0f32; k * n];
           for ni in 0..n {
               for ki in 0..k {
                   wkn[ki * n + ni] = wv[ni * k + ki] as f32;
               }
           }
           let w_kn = MxArray::from_float32(&wkn, &[k as i64, n as i64])
               .unwrap()
               .astype(DType::BFloat16)
               .unwrap();

           x.eval();
           w_nk.eval();
           w_t.eval();
           w_kn.eval();
           synchronize();

           let bench = |f: &dyn Fn() -> MxArray| -> f64 {
               for _ in 0..warm {
                   let o = f();
                   o.eval();
               }
               synchronize();
               let t = Instant::now();
               for _ in 0..iters {
                   let o = f();
                   o.eval();
               }
               synchronize();
               t.elapsed().as_secs_f64() * 1e3 / iters as f64
           };

           // Clean pure int8 GEMM (pre-transposed weight, no per-call transpose).
           let clean = || matmul_int8_kn(&x, &w_kn).unwrap();
           // Clean int8 GEMM WITHOUT the per-call MLX zero-fill (mode::multiply).
           let nofill = || matmul_int8_kn_nofill(&x, &w_kn).unwrap();
           // bf16 matmul at identical logical shape: [M,K] @ [K,N].
           let bf16 = || x.matmul(&w_t).unwrap();
           // Contaminated: matmul_int8 transposes w[N,K]->[K,N] every call.
           let dirty = || matmul_int8(&x, &w_nk).unwrap();

           let ms_clean = bench(&clean);
           let ms_nofill = bench(&nofill);
           let ms_bf16 = bench(&bf16);
           let ms_dirty = bench(&dirty);

           // TOPS / TFLOPs = 2*M*N*K / sec / 1e12.
           let flop = 2.0 * m as f64 * n as f64 * k as f64;
           let tops_clean = flop / (ms_clean / 1e3) / 1e12;
           let tops_nofill = flop / (ms_nofill / 1e3) / 1e12;
           let tflops_bf16 = flop / (ms_bf16 / 1e3) / 1e12;
           let tops_dirty = flop / (ms_dirty / 1e3) / 1e12;

           eprintln!(
               "[clean] {label:<20} M={m:<5} N={n:<6} K={k:<5} | \
                int8(clean)={tops_clean:6.1} TOPS ({ms_clean:.3}ms)  \
                bf16={tflops_bf16:6.1} TF ({ms_bf16:.3}ms)  \
                ratio(int8/bf16)={:.3}  wall(bf16/int8)={:.3} || \
                int8(nofill)={tops_nofill:6.1} TOPS ({ms_nofill:.3}ms) \
                fill_delta={:.3}ms (nofill/bf16={:.3}) || \
                int8(dirty)={tops_dirty:6.1} TOPS ({ms_dirty:.3}ms) \
                contam_delta={:.3}ms ({:.2}x)",
               tops_clean / tflops_bf16,
               ms_bf16 / ms_clean,
               ms_clean - ms_nofill,
               tops_nofill / tflops_bf16,
               ms_dirty - ms_clean,
               ms_dirty / ms_clean,
           );
       };

       eprintln!("[clean] === pure-GEMM throughput, M5 NA int8 vs bf16 ===");
       // M=512 first (compare to standalone M=512), then M=4096.
       for &m in &[512usize, 4096usize] {
           // gate_up: x[M,2560] @ w[18432,2560]^T -> N=18432, K=2560.
           run_shape("gate_up", m, 2560, 18432);
           // down: x[M,9216] @ w[2560,9216]^T -> N=2560, K=9216.
           run_shape("down", m, 9216, 2560);
       }
   }

   // ==================== GDN in_proj_qkvz PARITY ====================
   // GATE (qkvz int8 wiring): per-ROW cosine >= 0.999 of the int8 W8A8 qkvz
   // output vs the bf16 `x @ w_qkvz^T` reference, at realistic GDN shapes.
   // qkvz feeds the GDN conv + recurrence so accuracy is load-bearing.
   //
   // Shapes (K=hidden must be %16==0):
   //   * 4B : hidden=2560, qkvz_dim = key_dim*2 + value_dim*2
   //          = (16*128)*2 + (32*128)*2 = 4096 + 8192 = 12288
   //   * 27B: hidden=5120, same head config -> qkvz_dim=12288
   // M=512 (a realistic prefill tile).
   #[test]
   fn qkvz_w8a8_cosine_parity() {
       if gpu_gen() < 17 {
           eprintln!(
               "[qkvz] SKIP: gpu gen {} < 17 (NA matmul2d needs M5+)",
// <<<RECOVERY-GAP line 1135>>>
// <<<RECOVERY-GAP line 1136>>>
// <<<RECOVERY-GAP line 1137>>>
// <<<RECOVERY-GAP line 1138>>>
// <<<RECOVERY-GAP line 1139>>>
       let shapes = [(512usize, 2560usize, 12288usize), (512, 5120, 12288)];
       let mut state: u64 = 0xb16e_5e7e_4242_d00d;

       for &(m, k, n) in &shapes {
           // bf16 activations + weights with realistic small magnitudes.
           let mut xf = vec![0f32; m * k];
           for v in xf.iter_mut() {
               *v = next_int(&mut state, -200, 200) as f32 / 1000.0;
           }
           let mut wf = vec![0f32; n * k];
           for v in wf.iter_mut() {
               *v = next_int(&mut state, -200, 200) as f32 / 1000.0;
           }

           let x = MxArray::from_float32(&xf, &[m as i64, k as i64])
               .unwrap()
               .astype(DType::BFloat16)
               .unwrap();
           // w_qkvz is [N=qkvz_dim, K=hidden], exactly quantize_weight_int8's [N,K].
           let w = MxArray::from_float32(&wf, &[n as i64, k as i64])
               .unwrap()
               .astype(DType::BFloat16)
               .unwrap();

           let (w_i8, s_w) = quantize_weight_int8(&w).unwrap();
           let y = int8_w8a8_matmul(&x, &w_i8, &s_w).unwrap();
           y.eval();
           assert_eq!(
               y.dtype().unwrap(),
               DType::BFloat16,
               "qkvz W8A8 output must be bf16"
           );

           // bf16 reference: y_ref = x @ w_qkvz^T.
           let wt = w.transpose(Some(&[1, 0])).unwrap();
           let y_ref = x.matmul(&wt).unwrap();
           y_ref.eval();

           let got = y.astype(DType::Float32).unwrap().to_float32().unwrap();
           let got: &[f32] = &got;
           let refv = y_ref.astype(DType::Float32).unwrap().to_float32().unwrap();
           let refv: &[f32] = &refv;
           assert_eq!(got.len(), m * n);
           assert_eq!(refv.len(), m * n);

           let mut min_cos = f64::INFINITY;
           let mut sum_cos = 0.0f64;
           for mi in 0..m {
               let mut dot = 0.0f64;
               let mut na = 0.0f64;
               let mut nb = 0.0f64;
               for ni in 0..n {
                   let a = got[mi * n + ni] as f64;
                   let b = refv[mi * n + ni] as f64;
                   dot += a * b;
                   na += a * a;
                   nb += b * b;
               }
               let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
               let cos = dot / denom;
               min_cos = min_cos.min(cos);
               sum_cos += cos;
           }
           let mean_cos = sum_cos / m as f64;
           eprintln!(
               "[qkvz] hidden={k} qkvz_dim={n} M={m}: min_row_cos={min_cos:.6} mean_row_cos={mean_cos:.6}"
           );
           assert!(
               min_cos >= 0.999,
               "qkvz W8A8 per-row cosine below gate at hidden={k} qkvz_dim={n}: min={min_cos:.6}"
           );
       }
   }

   // ==================== GDN in_proj_qkvz MICROBENCH ====================
// <<<RECOVERY-GAP line 1215>>>
// <<<RECOVERY-GAP line 1216>>>
// <<<RECOVERY-GAP line 1217>>>
// <<<RECOVERY-GAP line 1218>>>
// <<<RECOVERY-GAP line 1219>>>
// <<<RECOVERY-GAP line 1220>>>
// <<<RECOVERY-GAP line 1221>>>
// <<<RECOVERY-GAP line 1222>>>
// <<<RECOVERY-GAP line 1223>>>
// <<<RECOVERY-GAP line 1224>>>
// <<<RECOVERY-GAP line 1225>>>
// <<<RECOVERY-GAP line 1226>>>
// <<<RECOVERY-GAP line 1227>>>
// <<<RECOVERY-GAP line 1228>>>
// <<<RECOVERY-GAP line 1229>>>
// <<<RECOVERY-GAP line 1230>>>
// <<<RECOVERY-GAP line 1231>>>
// <<<RECOVERY-GAP line 1232>>>
// <<<RECOVERY-GAP line 1233>>>
// <<<RECOVERY-GAP line 1234>>>
// <<<RECOVERY-GAP line 1235>>>
// <<<RECOVERY-GAP line 1236>>>
// <<<RECOVERY-GAP line 1237>>>
// <<<RECOVERY-GAP line 1238>>>
// <<<RECOVERY-GAP line 1239>>>
// <<<RECOVERY-GAP line 1240>>>
// <<<RECOVERY-GAP line 1241>>>
// <<<RECOVERY-GAP line 1242>>>
// <<<RECOVERY-GAP line 1243>>>
// <<<RECOVERY-GAP line 1244>>>
// <<<RECOVERY-GAP line 1245>>>
// <<<RECOVERY-GAP line 1246>>>
// <<<RECOVERY-GAP line 1247>>>
// <<<RECOVERY-GAP line 1248>>>
// <<<RECOVERY-GAP line 1249>>>
// <<<RECOVERY-GAP line 1250>>>
// <<<RECOVERY-GAP line 1251>>>
// <<<RECOVERY-GAP line 1252>>>
// <<<RECOVERY-GAP line 1253>>>
// <<<RECOVERY-GAP line 1254>>>
// <<<RECOVERY-GAP line 1255>>>
// <<<RECOVERY-GAP line 1256>>>
// <<<RECOVERY-GAP line 1257>>>
// <<<RECOVERY-GAP line 1258>>>
// <<<RECOVERY-GAP line 1259>>>
// <<<RECOVERY-GAP line 1260>>>
// <<<RECOVERY-GAP line 1261>>>
// <<<RECOVERY-GAP line 1262>>>
// <<<RECOVERY-GAP line 1263>>>
// <<<RECOVERY-GAP line 1264>>>
// <<<RECOVERY-GAP line 1265>>>
// <<<RECOVERY-GAP line 1266>>>
// <<<RECOVERY-GAP line 1267>>>
// <<<RECOVERY-GAP line 1268>>>
// <<<RECOVERY-GAP line 1269>>>
// <<<RECOVERY-GAP line 1270>>>
// <<<RECOVERY-GAP line 1271>>>
// <<<RECOVERY-GAP line 1272>>>
// <<<RECOVERY-GAP line 1273>>>
// <<<RECOVERY-GAP line 1274>>>
// <<<RECOVERY-GAP line 1275>>>
// <<<RECOVERY-GAP line 1276>>>
// <<<RECOVERY-GAP line 1277>>>
// <<<RECOVERY-GAP line 1278>>>
// <<<RECOVERY-GAP line 1279>>>
// <<<RECOVERY-GAP line 1280>>>
// <<<RECOVERY-GAP line 1281>>>
// <<<RECOVERY-GAP line 1282>>>
// <<<RECOVERY-GAP line 1283>>>
// <<<RECOVERY-GAP line 1284>>>
// <<<RECOVERY-GAP line 1285>>>
// <<<RECOVERY-GAP line 1286>>>
// <<<RECOVERY-GAP line 1287>>>
// <<<RECOVERY-GAP line 1288>>>
// <<<RECOVERY-GAP line 1289>>>
// <<<RECOVERY-GAP line 1290>>>
// <<<RECOVERY-GAP line 1291>>>
// <<<RECOVERY-GAP line 1292>>>
// <<<RECOVERY-GAP line 1293>>>
// <<<RECOVERY-GAP line 1294>>>
// <<<RECOVERY-GAP line 1295>>>
// <<<RECOVERY-GAP line 1296>>>
// <<<RECOVERY-GAP line 1297>>>
// <<<RECOVERY-GAP line 1298>>>
// <<<RECOVERY-GAP line 1299>>>
           );
       };

       eprintln!("[qkvz-bench] === GDN in_proj_qkvz int8 W8A8 vs bf16, M=4096 ===");
       // 4B: hidden=2560, qkvz_dim=12288.
       run("4B", 4096, 2560, 12288);
       // 27B: hidden=5120, qkvz_dim=12288.
       run("27B", 4096, 5120, 12288);
   }

   // ============================ STAGE 2 ============================
   // GATE S2: per-ROW cosine >= 0.999 on a real projection shape.
   // x[M=512, hidden=2560] @ w[N=intermediate, K=2560]^T, weight quantized once.
   #[test]
   fn s2_w8a8_cosine_parity() {
       if gpu_gen() < 17 {
           eprintln!(
               "[s2] SKIP: gpu gen {} < 17 (NA matmul2d needs M5+)",
               gpu_gen()
           );
           return;
       }
       // Real-ish projection shapes (K must be % 16 == 0).
       // (M, K=hidden, N=intermediate)
       let shapes = [(512usize, 2560usize, 9216usize), (256, 2560, 2560)];
       let mut state: u64 = 0x0fed_cba9_8765_4321;

       for &(m, k, n) in &shapes {
           // bf16 activations and weights with realistic magnitudes (~N(0, 0.05)
           // emulated via small integers / 1000 -> values in ~[-0.127,0.127]).
           let mut xf = vec![0f32; m * k];
           for v in xf.iter_mut() {
               *v = next_int(&mut state, -200, 200) as f32 / 1000.0;
           }
           let mut wf = vec![0f32; n * k];
           for v in wf.iter_mut() {
               *v = next_int(&mut state, -200, 200) as f32 / 1000.0;
           }

           let x = MxArray::from_float32(&xf, &[m as i64, k as i64])
               .unwrap()
               .astype(DType::BFloat16)
               .unwrap();
           let w = MxArray::from_float32(&wf, &[n as i64, k as i64])
               .unwrap()
               .astype(DType::BFloat16)
               .unwrap();

           // Quantize weight ONCE, then run the W8A8 path.
           let (w_i8, s_w) = quantize_weight_int8(&w).unwrap();
           let y = int8_w8a8_matmul(&x, &w_i8, &s_w).unwrap();
           y.eval();
           assert_eq!(
               y.dtype().unwrap(),
               DType::BFloat16,
               "W8A8 output must be bf16"
           );

           // bf16 reference: y_ref = x @ w^T (matmul does x[M,K] @ w^T[K,N]).
           let wt = w.transpose(Some(&[1, 0])).unwrap();
           let y_ref = x.matmul(&wt).unwrap();
           y_ref.eval();

           let got = y.astype(DType::Float32).unwrap().to_float32().unwrap();
           let got: &[f32] = &got;
           let refv = y_ref.astype(DType::Float32).unwrap().to_float32().unwrap();
           let refv: &[f32] = &refv;
           assert_eq!(got.len(), m * n);
           assert_eq!(refv.len(), m * n);

           // Per-row cosine similarity.
           let mut min_cos = f64::INFINITY;
           let mut sum_cos = 0.0f64;
           for mi in 0..m {
               let mut dot = 0.0f64;
               let mut na = 0.0f64;
               let mut nb = 0.0f64;
               for ni in 0..n {
                   let a = got[mi * n + ni] as f64;
                   let b = refv[mi * n + ni] as f64;
                   dot += a * b;
                   na += a * a;
                   nb += b * b;
               }
               let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
               let cos = dot / denom;
               min_cos = min_cos.min(cos);
               sum_cos += cos;
           }
           let mean_cos = sum_cos / m as f64;
           eprintln!(
               "[s2] M={m} K={k} N={n}: min_row_cos={min_cos:.6} mean_row_cos={mean_cos:.6}"
           );
           assert!(
               min_cos >= 0.999,
               "W8A8 per-row cosine below gate at M={m} K={k} N={n}: min={min_cos:.6}"
           );
       }
   }
}
