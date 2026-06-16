# mlx-node CUDA PoC + Benchmark — Implementation Plan (GB10 / sm_121)

## Status (Spark de-risk)

```
PROVISIONED ✅  Ubuntu 24.04 aarch64, GB10 (cc 12.1, driver 580.159.03), 116GB RAM, 3.5TB free
               CUDA 13.0.88 (in MLX window >=12.0 <13.1) · cuDNN 9.23.1 · cmake 3.28.3
               gcc 13.3 · ninja · BLAS/LAPACK/LAPACKE · rust 1.96 · node 22 · vp 0.1.24 · uv
REPO CLONED ✅  /home/user/mlx-node @ fba240b8 (== Mac HEAD); submodule crates/mlx-sys/mlx
               @ a8776b7b (pinned, fork mlx-node/mlx, public); backend/cuda present (198 files)
MLX-CUDA GATE ✅ DECISIVE PASS — MLX backend/cuda BUILDS on this exact host:
               "CUDA architectures: 121a" · Found CUDNN · CUTLASS/CCCL/cudnn_frontend fetched
               built 100% → mlx-0.31.2.dev20260616+a8776b7b installed, 0 real errors
GPU SMOKE    ✅ RUNTIME PASS on Device(gpu,0): bf16 matmul · softmax · fast.rms_norm · fast.rope
               · quantized_matmul(affine-4bit) · sdpa-causal — all OK
MODELS       ✅ pre-quantized checkpoints downloading to GB10 ~/models (HF Brooooooklyn/Qwen3.6-*):
               27B + 35B-A3B × {Q4_K_XL, NVFP4, MXFP4, MXFP8, Q8_K_XL}. NO bf16 in collection →
               first-light = Q4/Q8 affine; perf = NVFP4 (+MXFP4/MXFP8). NO mlx-convert needed.
```

Central feasibility assumption is **proven**: the same `backend/cuda` code mlx-sys links compiles for `sm_121a` here. Remaining work is **all mlx-node-side** (build.rs Metal coupling, 3 paged C++ TUs, build.ts/napi, runtime eager gates), plus model bring-up and benchmark.

**Required env (prepend to every build/run):**
```bash
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda CUDA_HOME=/usr/local/cuda
export MLX_PTX_CACHE_DIR=$HOME/.cache/mlx-ptx   # persistent, writable, NOT noexec
```

---

## Phase 0 — Isolation & host pre-flight (blocking)

| Task | Action | Gate |
|---|---|---|
| 0.1 Branch | Create `feat/cuda-poc-gb10` on GB10 checkout. **Do not** work on `main` (shared/PR-active). | `git rev-parse --abbrev-ref HEAD` == feat branch |
| 0.2 Record base for no-regression | On **macOS**: `git rev-parse HEAD` of the feat branch base; capture it for the later diff target. | base SHA noted |
| 0.3 FetchContent vendoring check | MLX-CUDA cmake git-clones **FOUR** deps at configure: CCCL, NVTX3, cudnn-frontend, CUTLASS. The Spark fetched them once (network OK). Confirm cache or set `FETCHCONTENT_SOURCE_DIR_*` if host later goes air-gapped. | configure reaches "Built target cudnn_frontend" |
| 0.4 PTX cache + nvrtc runtime | `mkdir -p $MLX_PTX_CACHE_DIR`; verify writable + not `noexec`; confirm `libnvrtc.so` resolvable at runtime (`ldconfig -p \| grep nvrtc`). | dir writable; nvrtc present |
| 0.5 BLAS/stubs | `libopenblas`/lapack already installed. Confirm `libcuda.so` stub at `$CUDA_PATH/lib64/stubs` and real driver at runtime. | both present |

---

## Phase 1 — `crates/mlx-sys/build.rs` Linux/CUDA port (auto-detect by `target_os`)

**Selection = `CARGO_CFG_TARGET_OS` auto-detect. No Cargo feature.** macOS stays byte-for-byte; any `linux` build takes the CUDA branch.

### 1.1 Hoist OS read + gate Metal
- Move `let target_os = env::var("CARGO_CFG_TARGET_OS")` **above** the xcrun-metal check (~L211). Add `let is_macos = target_os == "macos"; let build_metal = is_macos && !metal_disabled;`
- Wrap the `metal_toolchain_available()` panic, `compile_paged_attn_metallib()` call, and metallib copy logic in `if build_metal` (Linux → `paged_metallib_path = None`).
- `MLX_BUILD_METAL` define = `if build_metal {"ON"} else {"OFF"}`; gate `CMAKE_OSX_ARCHITECTURES` behind `is_macos`.

### 1.2 Add Linux/CUDA cmake branch (after the macOS block)
```rust
} else if target_os == "linux" {
    let cuda_archs = env::var("MLX_CUDA_ARCHITECTURES").unwrap_or_else(|_| "121a".into());
    cfg.define("MLX_BUILD_CUDA", "ON")
       .define("MLX_BUILD_METAL", "OFF")
       .define("MLX_BUILD_CPU", "ON")
       .define("MLX_CUDA_ARCHITECTURES", &cuda_archs)
       .define("CMAKE_BUILD_TYPE", "Release");   // ← REVIEW FIX #6: Release or perf is skewed
}
```
> `121a` is what MLX auto-detected on this host. Pass it explicitly only for determinism / to dodge the empty-query FATAL on a GPU-less host. Native-on-GB10 auto-detect also works.

### 1.3 Per-OS link section (REVIEW FIX #6: `c++` vs `stdc++`)
```rust
println!("cargo:rustc-link-lib=static=mlx");
if is_macos {
    if build_metal { /* framework=Metal, =QuartzCore */ }
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=c++");
} else if target_os == "linux" {
    println!("cargo:rustc-link-search=native={CUDA_PATH}/lib64");
    println!("cargo:rustc-link-search=native={CUDA_PATH}/lib64/stubs"); // libcuda stub
    // PRIVATE deps on static libmlx — NOT re-exported, must re-declare:
    for l in ["cudart","cublas","cublasLt","cufft","nvrtc","cuda"] {
        println!("cargo:rustc-link-lib=dylib={l}");
    }
    // cuDNN 9 split — REVIEW FIX #3 / CRITIC: derive REAL names from configure, do not guess
    // capture from build/.../link.txt or `cargo build -vv`; expect cudnn umbrella + graph/ops/engines
    println!("cargo:rustc-link-lib=dylib=cudnn");
    for l in ["stdc++","dl","pthread"] { println!("cargo:rustc-link-lib=dylib={l}"); }
}
```

### 1.4 FFI source selection — exclude the 3 Metal-header TUs on Linux
Replace the blind `read_dir` glob with a skip-list when `!is_macos`:
```rust
const METAL_ONLY_TUS: &[&str] = &[
    "mlx_paged_dispatch.cpp",  // MTL:: types
    "mlx_paged_ops.cpp",       // #include mlx/backend/metal/device.h (L27)
    "mlx_paged_profile.cpp",   // already __APPLE__-guarded, belt-and-suspenders
];
```
Also gate the `metal_cpp` include block behind `is_macos`.

**Gate 1:** `cargo build -p mlx-sys --release` links. `nm -C .../libmlx_ffi.a | grep -c MTL` → **0**. `ldd` of the artifact resolves `cudart|cublas|nvrtc|cudnn`.

---

## Phase 2 — C++ FFI Metal/CUDA decoupling

**Verified decoupling map:** only the 3 TUs above are Metal-coupled *by header*. `mlx_na_int8/affine_w8a8/gated_delta/advanced_ops` + all 8 qwen35 TUs use only `fast::metal_kernel(...)` JIT strings → declared in cross-platform `mlx/fast.h`, satisfied by `no_metal.cpp` (throws at runtime) → **compile fine on Linux**.

We **guard-in-place** rather than exclude, so the FFI symbols the Rust externs need stay defined. (Build.rs in Phase 1 already excludes `mlx_paged_dispatch.cpp` / `mlx_paged_ops.cpp` from the Linux compile; the eval_gpu consumers are themselves guarded so no undefined-ref.)

| File | Change |
|---|---|
| `src/mlx_paged_dispatch.h` | Wrap `#include metal/device.h` + the whole `paged` decl block in `#if defined(__APPLE__) … #endif`. |
| `src/mlx_paged_dispatch.cpp` | Wrap entire body in `#if defined(__APPLE__)` (empty object on Linux). |
| `src/mlx_paged_ops.cpp` | Guard `#include metal/device.h` (L27) + the 3 `eval_gpu` Metal dispatch tails; non-Apple → `throw std::runtime_error("…no Metal backend…")`. Keep host-side validation unguarded. Make `mlx_paged_attention_forward`/`_kv_write_forward` factories return **null on `!__APPLE__`** so Rust callers take the SDPA fallback. |
| `src/lib.rs` | No cfg-gating of the paged extern block needed (symbols stay defined as null/throw stubs). Verify `mlx_metal_is_available()` extern decl exists. |

> **REVIEW FIX #1 (FALSE rationale corrected):** int8 TUs are NOT cfg-guarded. `forward_sym8` (`quantized_linear.rs:492`) is **runtime-mode-selected** (`mode==SYM8_MODE`) and fail-loud. They are safe **only because the bf16→nvfp4 matrix never sets SYM8_MODE**. Optional hardening: early `mlx_metal_is_available()` error in `forward_sym8`.

**Gate 2:** mlx-sys + the full FFI bridge compile and link on Linux; `grep -rc 'MTL::' src/*.cpp` outside `__APPLE__` guards → 0.

---

## Phase 3 — Runtime eager gates (load-bearing for correctness)

The compiled C++ forward uses `fast::metal_kernel` for GDN → throws on CUDA. Must force the **eager Rust flat path**.

| File | Change | Why |
|---|---|---|
| `crates/mlx-core/src/models/qwen3_5/persistence.rs` | At top of `register_weights_with_cpp` add hard gate: `if !unsafe { mlx_sys::mlx_metal_is_available() } { info!(…); return; }` (beside existing `MLX_QWEN35_FORCE_EAGER`). | model_id stays unset → `use_compiled=false` → GDN routes to `gated_delta_ops`. |
| `crates/mlx-core/src/models/qwen3_5_moe/persistence.rs` | **REVIEW FIX #7 / CRITIC (confirmed gap):** add the SAME gate at top of `register_moe_weights_with_cpp` (L1491, sets model_id at L1594, currently **no gate**). Share one helper to avoid drift. | 35B-A3B MoE otherwise hits compiled C++ MoE TU → throws at first decode. |
| `crates/mlx-core/src/models/qwen3_5/attention.rs` | **REVIEW FIX #2 (inter-spec contradiction):** make `paged_prefill_paged_attention_enabled()` return false when `!mlx_metal_is_available()`. | Flat-path cache-hit prefill (L426, `batch==1 && enabled`) would dispatch the C++ paged kernel on reuse turns. Single-turn fresh prompts don't hit it, but this hard-closes it. |
| `crates/mlx-core/src/models/.../gated_delta.rs` | **REVIEW FIX #8 / CRITIC:** add cached `mlx_metal_is_available()` probe so CUDA goes **straight to `gated_delta_ops`**, skipping the per-layer-per-token `metal_kernel` throw/catch + stderr spam. | Else DECODE-GAP number is contaminated by exception overhead on a GDN-heavy model. Must land **before** any GB10 perf run. |

**Gate 3:** Rust unit tests pass on Linux (`cargo test -p mlx-core` where applicable); a debug log confirms "no Metal backend → eager forward" fires on load for both dense + MoE.

---

## Phase 4 — Packaging: `build.ts` + napi target

| File | Change |
|---|---|
| `packages/core/build.ts` | (a) Gate `copyMetallibs()` behind `process.platform === 'darwin'` (throws otherwise — no metallib on CUDA). (b) **REVIEW FIX #5:** `copyNativeAddon` hard-codes `mlx-core.darwin-arm64.node` and throws **before** the metallib gate — derive `expectedName` + `npm/<triple>/` from `platform/arch` (`linux-arm64-gnu`). Both fixes required. |
| `packages/core/package.json` | Add `"aarch64-unknown-linux-gnu"` to `napi.targets`; add optional dep `@mlx-node/core-linux-arm64-gnu`. |
| `packages/core/npm/linux-arm64-gnu/package.json` | NEW: mirror darwin pkg; `os:["linux"]`, `cpu:["arm64"]`, `libc:["glibc"]`, `main: mlx-core.linux-arm64-gnu.node`, **no metallibs in `files`**. |
| `packages/core/index.cjs` | No manual change — regenerated by napi; linux-arm64-gnu branch already exists. |
| `crates/mlx-paged-attn` (build.rs/lib.rs/Cargo.toml) | **No change** — already `cfg(target_os="macos")`-gated; Linux-clean. |

**Gate 4:** `napi build --platform --target aarch64-unknown-linux-gnu --release` (via build.ts) produces `packages/core/mlx-core.linux-arm64-gnu.node`; `node -e "require('@mlx-node/core')"` loads without throwing.

---

## Phase 5 — Bring-up (staged: Q4 27B → NVFP4 27B → 35B-A3B MoE)

> **RESOLVED — no `mlx convert` on the critical path:** checkpoints ship **pre-quantized** from HF (`Brooooooklyn/Qwen3.6-*`), so the convert chicken-and-egg is moot. The collection has **no bf16**, so first-light correctness uses **Q4_K_XL** (then Q8_K_XL near-lossless reference); perf headline = **NVFP4_K_XL**; breadth = MXFP4/MXFP8. Download the identical repos on M5 for the baseline half to guarantee byte-identical weights+tokenizer on both hosts.

### 5.1 Op-coverage pre-flight (CRITIC — before any large model)
MLX-CUDA **hard-crashes** (no CPU fallback) on missing ops. Write a tiny op-probe harness that exercises each primitive the eager bf16/nvfp4 forward emits and names the first gap instead of crashing at token N:
- matmul, sdpa-causal, rope, rms_norm, softmax, gather_mm, gather_qmm, quantized_matmul (affine + nvfp4), conv1d (cuDNN)
- `gated_delta_ops` fallback primitives: cumsum / segment-sum / where / sigmoid / exp / concatenate / take

**Gate 5.1:** every probed op runs without abort, or the gap is reported as `unsupported-op:<name>`.

### 5.2 Models + acquisition — RESOLVED
- **Slugs pinned & public, pre-quantized:** `Brooooooklyn/Qwen3.6-27B-UD-<Q>_K_XL-mlx` and `Brooooooklyn/Qwen3.6-35B-A3B-UD-<Q>_K_XL-mlx`, `<Q>` ∈ {Q4, NVFP4, MXFP4, MXFP8, Q8}. Downloading to GB10 `~/models/` via `snapshot_download` (10 dirs, 271 GB; Q4 + NVFP4 first).
- **No `mlx convert`** — checkpoints are already quantized. No HF token needed (public).
- **M5 side:** download the same 10 repos on the Mac for the baseline half (none cached locally yet).

### 5.3 Staged load + smoke
Order: **Q4_K_XL 27B → NVFP4 27B → Q4_K_XL 35B-A3B MoE → NVFP4 35B-A3B** (Q8/MXFP4/MXFP8 as breadth). Each: load via `loadModel` → `ChatSession` → `session.send(T=0, enableMtp:false, reuseCache:false)` → coherent output + `finishReason`.

**Runtime invocation (forces eager flat, JIT-warmed):**
```bash
MLX_QWEN35_FORCE_EAGER=1   # belt-and-suspenders on top of the is_available gate
# block-paged stays OFF by default (use_block_paged_cache=None); SDPA causal+no-mask only
```

**Gate 5:** all 4 dirs load + produce coherent T=0 output; **flag OOM cells** (record GB10 unified-memory budget vs M5 128GB — large dense+MoE bf16 may not fit; OOM ≠ 0 tok/s).

---

## Phase 6 — Correctness gate

- **Harness** `examples/cuda-poc-golden.ts` (adapt `coherence-probe.ts`): `loadModel → ChatSession → send(FIXED_PROMPT, {maxNewTokens:256, temperature:0, topK:1, topP:1, reasoningEffort:'none', includeReasoning:false, enableMtp:false, reuseCache:false})` → print `sha256(rawText)` + rawText + finishReason + numTokens.
- **MTP-OFF trap:** ChatSession auto-enables MTP on MTP-bearing checkpoints (`chat-session.ts:1099`). `enableMtp:false` is mandatory in **every** config; `--q-mtp off` at convert removes the head entirely.
- **REVIEW FIX #8 — pass bar (sha256 demoted to stretch):** cross-backend fp ordering differs → greedy argmax can flip → bytes diverge for correct code.
  - **PASS** = same `finishReason` + coherent output (VERDICT regex) + first-divergence-token located.
  - **Divergence locator** (no public raw-logits API): loop `maxNewTokens:1`, `reuseCache:false`, append each greedy token, report first index where M5 vs GB10 disagree + preceding context.
  - **Real bug** only if divergence is at **token 1 with large logit gap**, not late-decode drift.

**Gate 6:** all 4 cells coherent + same finishReason; divergence index recorded per cell.

---

## Phase 7 — Benchmark

- **JIT warm (CRITIC):** every arm does one throwaway forward + verifies `$MLX_PTX_CACHE_DIR` populated before measured reps — cold nvrtc compile must never land in a TTFT number.
- **Harness:** reuse `examples/lfm2-perf-ab.ts` (loads any dir via `loadModel`, prints `RESULT_JSON{medTtftMs, medPrefillTps, medDecodeTps}`). **Add** `enableMtp:false` + `reuseCache: mode==='decode'`; emit `result.promptTokens` (true tokenized length) not requested `--prompt-tokens`.
- **Matrix:** {bf16, nvfp4} × {27B dense, 35B-A3B MoE}, MTP OFF, **flat cache** (MLX-CUDA SDPA rejects bool masks; MoE prefill is gather_qmv-only → slow-but-correct). Re-measure M5 **fresh** (do not reuse MEMORY.md AR/MTP numbers).
- **Sweep:** prefill TTFT over `--prompt-tokens {128,512,1024,2048,4096,8192}` (cap MoE at 2048 first pass — gather_qmv prefill is slow); decode tok/s at prompt 256, 256 gen; throughput = single-turn 1024-prompt + 256-gen wall.
- **Cross-host = median-vs-median** (the `lfm2-perf-pair.py` drift-cancel trick is **same-host A/B only**, e.g. flat-vs-paged via `MLX_QWEN35_PAGED_OVERRIDE`).
- **Telemetry per run (CRITIC):** `nvidia-smi` clocks/power/temp/throttle + free mem on GB10; MLX memory-limit state on both; confirm neither host is memory-capped.
- **Failure-mode labels:** `build-fail / configure-fetchcontent-fail / unsupported-op:<name> / jit-cache-fail / OOM / fp-divergence / numeric-bug` — never report a non-perf failure as tok/s.

**Report:** `experiments/cuda-gb10-poc/report.md` + raw `report-data.jsonl`. Host/arch table (M5 Max gen17 vs GB10 sm_121, MLX a8776b7b, CUDA 13.0, build flags) + 4-cell M5-vs-GB10 matrix with per-prompt TTFT/prefill-tps/decode-tps/throughput + ratios + the three characterizations: **PREFILL-WIN**, **DECODE-GAP**, **MOE-PREFILL-GAP**.

**Gate 7:** all non-OOM cells produce labeled numbers on both hosts; report committed.

---

## Phase 8 — macOS no-regression proof (must pass)

Raw `.node` byte-diff is invalid (timestamps/paths). **Concrete gate (CRITIC):**
1. On macOS, build `main@base` and the feat branch with the same toolchain; capture `cargo build -vv` **link/search lines** for each → diff THOSE (must be identical; only delta allowed = `target_os` read a few lines earlier).
2. Run full suite on both: `vp test --run`, `cargo test -p mlx-core`, `cargo test -p mlx-paged-attn`. **Baseline the 3 known-flaky Metal f32 tests on base first** (`env_pre_existing_metal_f32_test_failures.md`).
3. T=0 `rawText` sha256 of an existing model on macOS before/after → identical (eager + compiled paths untouched).

**Gate 8:** link lines identical · test deltas == baseline · macOS T=0 sha256 unchanged.

---

## Incorporated adversarial-review fixes

| # | Fix | Where |
|---|---|---|
| 1 | int8 safety rationale corrected (runtime-mode-selected, not cfg-guarded; safe only because matrix never sets SYM8_MODE) | Phase 2 note |
| 2 | flat-path cache-hit prefill CAN dispatch C++ paged kernel → gate `paged_prefill_…enabled()` on `mlx_metal_is_available()` | Phase 3 |
| 3 | cuDNN 9 split libs + 4× FetchContent network dep → derive real link names from configure; vendor if air-gapped | Phase 0.3, 1.3 |
| 4 | LAPACK/BLAS host prereq + libcuda stubs search path | Phase 0.5, 1.3 |
| 5 | build.ts darwin name/dir hard-code throws before metallib gate → fix both | Phase 4 |
| 6 | `c++`→`stdc++` on Linux + explicit `CMAKE_BUILD_TYPE=Release` | Phase 1.2/1.3 |
| 7 | MoE `register_moe_weights_with_cpp` FORCE_EAGER/availability gate (confirmed missing) | Phase 3 |
| 8 | GDN per-step throw/catch contaminates DECODE-GAP → probe-gate; sha256 demoted to stretch | Phase 3, 6 |

## Incorporated completeness-critic items

- nvrtc runtime + writable/non-noexec PTX cache (`MLX_PTX_CACHE_DIR`) + JIT warm before measure → Phase 0.4, 7
- 4× FetchContent (CCCL/NVTX3/cudnn-frontend/CUTLASS) not just cuDNN → Phase 0.3
- convert depends on working addon + CUDA ops → **convert on M5, rsync** → Phase 5
- HF slugs/token + checkpoint transfer/disk/mmap → Phase 5.2
- op-coverage pre-flight harness → Phase 5.1
- explicit build-bring-up DAG (this doc's phase order) + per-step gates
- macOS no-regression via link-line diff + test suite + T=0 sha256 → Phase 8
- benchmark comparability (JIT warm, nvidia-smi telemetry, identical weights/tokenizer, OOM labeling) → Phase 7
- cuda_driver stub-vs-runtime + MLX's pip-wheel `$ORIGIN/../../nvidia` rpath won't apply → set `LD_LIBRARY_PATH`/rpath, `ldd` the final `.node` → Phase 0.4/1.3
- branch isolation + diff vs branch base, not moving main → Phase 0.1/0.2

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| macOS regression from build.rs edits | Med | High | Every macOS stmt reachable under `is_macos`; link-line diff + full suite (Phase 8) |
| MLX-CUDA missing-op hard-crash mid-forward | Med | High | Op-probe pre-flight (5.1); label `unsupported-op:<name>` |
| MoE compiled path not gated → crash at decode | Was high | High | **Fixed** Phase 3 (MoE availability gate) |
| cuDNN 9 link names guessed wrong | Med | Med | Derive from `cargo build -vv`/`link.txt`, don't guess |
| Air-gapped host fails configure (4× FetchContent) | Low (net OK on Spark) | High | Vendor / `FETCHCONTENT_SOURCE_DIR_*` (0.3) |
| PTX cache unwritable/noexec → recompile every run | Med | High | Pin `MLX_PTX_CACHE_DIR` + verify + JIT warm (0.4, 7) |
| GB10 OOM on 27B/35B bf16 | Med | Med | Record mem budget; label OOM ≠ 0 tok/s; nvfp4 fallback |
| GDN throw/catch inflates decode | Was high | Med (perf) | **Fixed** probe-gate before any number (Phase 3) |
| Cross-backend T=0 bytes diverge (legit fp) | High | Low | sha256 = stretch; coherence + divergence-locator = pass bar |
| ~~Qwen3.6 HF slugs unknown~~ | RESOLVED | — | Public pre-quantized repos `Brooooooklyn/Qwen3.6-*`, downloading (5.2) |
| CUDA toolkit upgrade to 13.1 bricks build | Low | High | Pin 13.0; MLX rejects 13.1 |

## Deferred (out of scope this milestone)

- Custom Metal-kernel ports to CUDA (paged attention, GDN chunked, na_int8/W8A8, MoE tiled GEMM) — eager agnostic fallbacks only.
- Block-paged KV cache on CUDA (default OFF; SDPA bool-mask + MoE-prefill limits).
- MTP speculative decoding (OFF for first benchmark; `--q-mtp off`).
- Multi-turn / `reuseCache:true` cache-hit prefill on CUDA (unvalidated; gated false).
- nvfp4 quality tuning via `--q-recipe unsloth` + imatrix (bf16 is trustworthy baseline; nvfp4 secondary).
- Publishing the `@mlx-node/core-linux-arm64-gnu` package (artifact only).
- Read-only NAPI top-k logits debug export (would enable true logit/KLD compare; not needed for greedy locator).
- musl target (`linux-arm64-musl`); GB10 is glibc.
