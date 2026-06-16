# MLX-Node CUDA Backend — PoC + Perf Benchmark (Design Spec)

**Date:** 2026-06-16
**Branch:** `feat/cuda-backend-poc`
**Status:** Approved design → implementation planning
**Target hardware:** NVIDIA GB10 (Grace-Blackwell, `aarch64` Linux, compute capability **sm_121**), rented via Enverge (`spark-03`).

## 1. Problem & Goal

mlx-node is currently **macOS/Metal/Apple-Silicon only**. Upstream MLX (the C++ core vendored as a git submodule at `crates/mlx-sys/mlx`) now ships a mature **CUDA backend** (`mlx/backend/cuda/`, 198 files). The goal of *this milestone* is a **proof-of-concept + performance benchmark**, not a production backend:

- Build mlx-node's native NAPI addon on `aarch64-unknown-linux-gnu` against MLX's CUDA backend.
- Run **real Qwen3.6 inference on the GB10 GPU** through MLX's stock ops and mlx-node's existing **device-agnostic fallbacks** — **no custom Metal-kernel ports** this milestone.
- Verify correctness against the M5 Max.
- Benchmark prefill / decode / throughput vs the M5 Max.
- Keep the macOS build **provably unchanged**.

### Non-goals (this milestone)
- Porting custom Metal kernels to CUDA (paged attention, GDN / gated-delta, int8/W8A8).
- Fixing MLX's MoE-prefill tiled-GEMM gap.
- x86_64 Linux, CI, prebuilt/published binaries, distribution.
- MTP (speculative decoding) on CUDA — base AR engine only for the first benchmark.

## 2. Success Criteria

1. `yarn build:native` produces `mlx-core.linux-arm64-gnu.node` on the GB10 with MLX CUDA enabled.
2. **Correctness:** Qwen3.6 27B (bf16) greedy (T=0) generation on GB10 matches the M5 Max output on a fixed prompt (byte-equivalence target; logits spot-check tolerance as fallback).
3. **Perf:** prefill TTFT (varied prompt lengths), single-stream decode tok/s, and throughput measured for {bf16, nvfp4} × {Qwen3.6 27B dense, Qwen3.6 35B-A3B MoE}, compared against a **freshly re-measured** M5 baseline (same harness).
4. macOS `yarn build:native` + existing test subset still pass (no regression).

## 3. Decisions (locked)

| Axis | Decision |
|---|---|
| Scope | PoC + perf benchmark; device-agnostic fallbacks; no kernel ports |
| Models | Qwen3.6 27B dense → then Qwen3.6 35B-A3B MoE (Qwen3.5/3.6 arch path) |
| Precision | Staged: bf16 (first light) → nvfp4 (perf comparison) |
| Build approach | A+B hybrid: cfg-gate the existing crates + a `cuda` selection so macOS can't regress; preceded by an MLX-core sm_121 spike |
| Baseline | vs M5 Max, re-measured fresh same-harness (don't trust stored numbers; cross-session variance ~10–15%) |
| MTP | OFF for the first benchmark (follow-up) |
| cuDNN | Install lazily — only if the model trips MLX's cuDNN-fused SDPA path |

## 4. Architecture — Five Layers

```
0. sm_121 GATE     Build MLX-core CUDA from the submodule, run a tiny MLX program on GB10.
   (de-risk)       Proves the arch works BEFORE porting mlx-node. Kill-switch.

1. Toolchain       Spark: cmake>=3.25, rust, node+yarn(+vp), nvcc on PATH, LD_LIBRARY_PATH;
   (Spark)         blas/lapack dev libs; cuDNN if needed. (sudo + open internet confirmed.)

2. Repo            Fresh clone of mlx-node + mlx submodule (pinned a8776b7b; verify the fork
   (Spark)         commit is reachable). Fresh clone = isolated build by construction.

3. Build port      A+B hybrid `cuda` selection (Cargo feature and/or target_os):
   (the diff)        - mlx-sys/build.rs: macOS keeps xcrun/metallib/Metal-frameworks; Linux passes
                       MLX_BUILD_CUDA=ON, arch 121a (or auto-detect), links cudart/cublas/
                       cublasLt/nvrtc/cuda_driver (+cuDNN if needed), native gcc (no -isysroot),
                       skips paged_attn.metallib compile.
                     - C++ FFI (mlx_paged_dispatch / mlx_gated_delta / mlx_na_int8 /
                       mlx_affine_w8a8 / mlx_advanced_ops): exclude Metal-only translation units
                       from the Linux cc build; route Rust call-sites to the agnostic fallbacks
                       (SDPA / gated_delta_ops / bf16).
                     - mlx-paged-attn: confirm Linux path no-ops cleanly → engine uses flat/SDPA.
                     - packages/core/build.ts: gate metallib copy + "both metallibs exist" assert
                       on process.platform === 'darwin'.
                     - packages/core/package.json: add aarch64-unknown-linux-gnu NAPI target.

4. Bring-up        Qwen3.6 27B bf16 → forward + short gen → T=0 parity vs M5. Then nvfp4.
   & correctness   Then 35B-A3B MoE (expect the MoE-prefill gap to show).

5. Benchmark       Reuse the controlled paired harness (warm + interleaved). Measure prefill TTFT
   & report        (varied prompt len), decode tok/s (single-stream), throughput; {bf16, nvfp4} ×
                   {27B, 35B-A3B}. Re-run the M5 side fresh, same harness. Report prefill-win /
                   decode-gap / MoE-prefill-gap characterization.
```

## 5. Build Data Flow (Linux/CUDA path)

```
install toolchain → clone repo+submodule → yarn build:native (linux/cuda branch)
  → mlx-sys/build.rs: cmake MLX (MLX_BUILD_CUDA=ON, arch 121a) + cc FFI (Metal TUs excluded)
  → mlx-core: cargo build → mlx-core.linux-arm64-gnu.node
  → build.ts: skip metallib copy/assert on linux
  → @mlx-node/lm loads addon → ChatSession → MLX-CUDA ops on GB10
```

## 6. Risks → Mitigations

| Risk | Mitigation |
|---|---|
| sm_121 unproven on MLX (no public MLX-on-GB10 report) | Phase 0 gate (MLX-core spike) before any porting |
| MLX op missing → **hard crash** (MLX has no silent CPU fallback) | Enumerate ops each model hits; CPU-route or fallback; install cuDNN if SDPA needs it |
| MoE prefill slow (`GatherQMM` uses `gather_qmv` only, no tiled GEMM) | Expected — measure & report, do NOT fix this milestone |
| Custom-kernel fallback correctness drift | T=0 parity vs M5 gates it |
| Fork submodule commit unreachable from Spark | Verify in Phase 2 (upstream `ls-remote` already works; fork may need creds) |
| CUDA 13.0 only on GB10; MLX CMake **rejects 13.1**; nvfp4 needs ≥12.8 | We have 13.0 — inside the window; pin arch `121a` and force `-DMLX_CUDA_ARCHITECTURES` if auto-detect misfires |
| macOS regression | feature/cfg gating + macOS build check after the diff |
| Billing ($0.65/hr, spans create→delete) | Keep instance alive; be efficient; log build times |

## 7. Testing

- **Phase 0:** MLX-core CUDA smoke — tiny matmul + RoPE on GPU returns finite, correct-shaped output.
- **Correctness:** Qwen3.6 27B bf16 GB10 vs M5 T=0 byte-equivalence on a fixed prompt; logits spot-check.
- **macOS no-regression:** `yarn build:native` on the M5 + existing TS/Rust test subset still pass after the build-port diff.
- **Benchmark:** controlled paired A/B harness (warm + interleaved), median of per-pair ratios.

## 8. Open Items Carried Into Planning

- Exact per-file diff for the `cuda` selection (feature vs `target_os` auto, and where each guard lands).
- Which MLX ops Qwen3.6 dense vs MoE actually exercise on CUDA (to pre-empt missing-op crashes), and whether cuDNN is required at configure time.
- Whether mlx-paged-attn's Linux no-op leaves the engine on a working flat/SDPA attention path for these models.
- Benchmark prompt set + metrics definitions reused from the existing controlled-bench examples.
