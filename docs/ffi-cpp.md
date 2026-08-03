# C++ FFI bridge

The bridge between MLX (C++) and the NAPI/Rust layer lives in `crates/mlx-sys/`. The Rust side declares the FFI surface in `lib.rs`; the C++ side implements each declaration across topical `.cpp` files compiled by the `cc` crate.

## File inventory

`crates/mlx-sys/src/`:

| File                     | Purpose                                                                                                                                               |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mlx_array_ops.cpp`      | Array construction, arithmetic, indexing, dtype-safe scalar ops                                                                                       |
| `mlx_advanced_ops.cpp`   | quantized_matmul, gather_qmm, conv2d, FP8 dequant, PaddleOCR forward                                                                                  |
| `mlx_nn_ops.cpp`         | NN ops, data extraction, random, math                                                                                                                 |
| `mlx_fused_ops.cpp`      | Fused SwiGLU MLP and supporting ops                                                                                                                   |
| `mlx_misc_ops.cpp`       | Synchronization, compiled sampling helpers                                                                                                            |
| `mlx_stream.cpp`         | Stream/device management, memory limits                                                                                                               |
| `mlx_autograd.cpp`       | `value_and_grad` integration                                                                                                                          |
| `mlx_gated_delta.cpp`    | Metal GDN kernel opaque handles and shader indexing                                                                                                   |
| ~~`mlx_qwen35.cpp`~~     | **DELETED** in the chat-engine refactor (`ee88b92b`). Was the compiled Qwen3.5 dense forward (`mlx::core::compile`); decode is now pure-Rust eager (`paged_forward::run_paged_decode_step` / `forward_inner`). |
| ~~`mlx_qwen35_moe.cpp`~~ | **DELETED** (same refactor). Was the compiled MoE forward with expert routing. |
| ~~`mlx_qwen35_vlm.cpp`~~ | **DELETED** (same refactor). VLM prefill now runs in Rust (`models/qwen3_5/vision.rs` + `chunked_prefill`). |
| `mlx_qwen35_common.h`    | Shared compiled-forward helpers — only the compiled SwiGLU helper survives the deletion |
| `mlx_common.h`           | FFI macros, error handling, array conversion                                                                                                          |
| ~~`mlx_common_weights.cpp`~~ | **DELETED** (same refactor) — was the common weight storage for the compiled forward passes. |
| `mlx_paged_dispatch.cpp` | C++ paged-attention kernel dispatch                                                                                                                   |
| `mlx_paged_ops.cpp`      | `PagedKVWrite` / `PagedAttention` custom MLX ops (largest file in the bridge)                                                                         |
| `mlx_paged_profile.cpp`  | Profile-run helpers for auto-sizing the block pool                                                                                                    |

`crates/mlx-sys/src/lib.rs` is the FFI declaration root (~300 `pub fn` wrappers around `unsafe extern "C-unwind"` blocks).

Because the wrappers are `extern "C-unwind"`, a C++ function that lets an exception escape **aborts the process** rather than returning an `Err` — every `.cpp` here catches and returns `nullptr`, and the Rust side turns a null handle into an error. A panic on a load path is therefore not recoverable; guards that want to name a bad tensor must run *before* the FFI call, not rely on MLX rejecting it.

### Two build-side inputs that are not `.cpp` files

| Path | Purpose |
| ---- | ------- |
| `crates/mlx-sys/cmake/switch-exhaustiveness.cmake` | Injected into the vendored MLX build via `CMAKE_PROJECT_TOP_LEVEL_INCLUDES` from `crates/mlx-sys/build.rs` — **macOS only** |
| `crates/mlx-core/vendor/ggml/ggml_kquant_ref.{c,h}` | ggml's own Q4_K/Q5_K/Q6_K decoders, vendored verbatim, compiled by `crates/mlx-core/build.rs` as the K-quant parity oracle |

**`switch-exhaustiveness.cmake` exists because `-w` is absolute in clang.** The `cmake` crate composes `CMAKE_CXX_FLAGS` through `cc` with warnings off, which appends `-w`; clang then drops every non-error diagnostic and keeps dropping it however late a `-W`/`-Werror=` flag appears. The file rewrites `-w` to `-Wno-everything` (same silence, but later flags can re-enable individual diagnostics) and then sets `-Werror=switch`. That makes a `switch` over `QuantizationMode` with no `default:` label a **compile error** when an enumerator is missing — which is the point: `QuantizationMode` is append-only and serialized by ordinal through `export.cpp`, so adding a mode must break the build rather than fall through silently. (`primitives.h` states the same contract: reordering or removing a mode reinterprets every previously exported graph as a different quantization format.)

Two limits of that guard, both deliberate and both worth knowing before trusting it:

- **macOS only.** The Linux/CUDA branch of `build.rs` does not apply the file, because `-Wno-everything` is clang-only and nvcc drives host compilation through GCC. A `switch` that misses an enumerator in CUDA-only code compiles clean; CUDA sources are built nowhere else in CI, so those have to be walked by hand.
- **Needs CMake ≥ 3.29.** `CMAKE_PROJECT_TOP_LEVEL_INCLUDES` was introduced there, while the vendored `mlx/CMakeLists.txt` only declares `cmake_minimum_required(VERSION 3.25)` and nothing pins a toolchain cmake. On 3.25–3.28 the define is silently ignored and the guard is inert rather than failing loudly.

**The vendored ggml reference is a test oracle, not a runtime dependency.** It is compiled with `-ffp-contract=off` — contracting `d * sc * q` into an fma would round differently from ggml's own build and quietly move the reference. Its three functions are referenced by no production path, so the linker drops the object from the cdylib and only the test binaries pull it in. `ggml_quants_upstream.inc` is a byte-verbatim upstream anchor guarded by `vendored_ggml_reference_is_verbatim`; **never reformat it** (a repo-wide `vp fmt` once cut it from 117 lines to 43).

## Compiled forward paths — deleted

Qwen3.5 dense + MoE **used to** use `mlx::core::compile` to cache the decode
forward graph (trace once, reuse via `compile_replace`). Those compiled C++
paths — `mlx_qwen35.cpp`, `mlx_qwen35_moe.cpp`, `mlx_qwen35_vlm.cpp`, and the
MTP compiled helpers — were **deleted in the chat-engine refactor**
(`ee88b92b`, 2026-06-22) and replaced by pure-Rust eager forwards. The only
remnant is the compiled SwiGLU helper in `mlx_qwen35_common.h`.

Current state:

- The Qwen3.5 dense/MoE, LFM2, and Gemma4 chat forwards build the MLX op
  graph eagerly per step — `models/qwen3_5/paged_forward.rs::run_paged_decode_step`,
  `models/qwen3_5/model.rs::forward_inner` / `forward_pre_norm_inner`, and the
  per-family equivalents in `lfm2` / `gemma4`.
- The remaining fused C++ forwards are Qwen3's `mlx_qwen3_forward_step`
  (`mlx_advanced_ops.cpp`; one call per token vs ~300 per-op FFI calls on the
  eager paths) and the PaddleOCR-VL one-shot OCR steps
  `mlx_paddleocr_vl_forward_step` / `mlx_paddleocr_vl_forward_step_batched`
  (called from `models/paddleocr_vl/language.rs`).
- Per-token graph re-trace is the dominant CPU-side cost on the eager paths
  (several hundred FFI calls + lazy-node allocations per token). Restoring a
  compiled/traced forward for Qwen3.5 dense (and the deleted
  `MLX_MTP_BUCKETED_VERIFY` per-kv-len compiled verify graphs) is an open perf
  follow-up, not implemented.

### Pitfalls

- `mlx::core::array` has **no default constructor** — initialize via `mlx_array_from_scalar(...)` or other helpers.
- `int32` is not in scope inside inner namespaces — use `mlx::core::int32`.
- Adding a **new** `.cpp` file requires `rm -rf target/release/build/mlx-sys-*` once; the `cc` crate caches its source-file list across builds and won't pick up new files otherwise.

### Env vars

| Var                     | Effect                                                                  |
| ----------------------- | ----------------------------------------------------------------------- |
| ~~`MLX_NO_COMPILE=1`~~  | **REMOVED (dead)** — gated the deleted compiled forward; unread today.  |
| ~~`MLX_EVAL_ALL_CACHES=1`~~ | **REMOVED (dead)** — token-only eval is the only strategy; unread today. |

## Process-wide globals

The compiled paths that needed process-wide globals are deleted. The old
`DENSE_COMPILED_MUTEX` / `COMPILED_WEIGHTS_RWLOCK` in
`crates/mlx-core/src/models/qwen3_5/model.rs` no longer exist; today the
paged and flat decode paths are pure-Rust eager and take no compile-path
locks per step. `crates/mlx-core/src/engine/compiled_lock.rs` is now only an
`AtomicU64` model-id counter.

## Metal shaders

`crates/mlx-paged-attn/metal/`:

| File                              | Purpose                               |
| --------------------------------- | ------------------------------------- |
| `attention/paged_attention.metal` | Paged-attention attention kernel      |
| `cache/reshape_and_cache.metal`   | KV cache reshape operations           |
| `cache/copy_blocks.metal`         | Block copy for paged cache management |
| `float8.metal`                    | FP8 type conversions and helpers      |
| `utils.metal`                     | Common Metal utilities                |

`crates/mlx-sys/build.rs` compiles `.metal` sources into `paged_attn.metallib` and copies both `paged_attn.metallib` and `mlx.metallib` into `target/<profile>/` and `target/<profile>/deps/` so integration tests discover them.
