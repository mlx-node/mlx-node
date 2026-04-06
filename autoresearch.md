# Autoresearch: WebGPU Production Readiness

## Objective
Make the WebGPU browser backend for mlx-node production-ready. This spans two repos:
- **mlx** (`crates/mlx-sys/mlx/`): C++ WebGPU backend kernels
- **mlx-node-browser** (this repo): Rust bindings, TypeScript workers, test infra, demo

Optimization targets: decode tok/s, code quality, correctness (including temp > 0), architecture, test infra.

## Metrics
- **Primary**: `decode_tok_s` (tokens/second, higher is better) — measured from Playwright inference run
- **Secondary**:
  - `browser_tests_passed` — count of passing WebGPU op tests (out of 164)
  - `output_coherent` — 1 if inference output is meaningful, 0 if garbage
  - `temp_gt0_works` — 1 if temperature > 0 sampling works without crash, 0 otherwise
  - `ttft_ms` — time to first token in milliseconds (lower is better)
  - `wasm_build_s` — WASM build time in seconds (lower is better)

## How to Run
`./autoresearch.sh` — outputs `METRIC name=number` lines.

Requires:
- Vite dev server NOT running (script manages its own)
- Chrome installed (Playwright uses it)
- WASM already built OR `REBUILD_WASM=1` env var to trigger rebuild

## Files in Scope

### TypeScript (mlx-node-browser)
| File | Purpose |
|------|---------|
| `packages/browser/src/mlx-worker.ts` | WASM worker — model loading, chat, diagnostics |
| `packages/browser/src/gpu-worker.ts` | GPU worker — WebGPU device, RPC command loop |
| `packages/browser/src/webgpu-bridge-stub.ts` | RPC bridge — Atomics.wait/notify for GPU calls |
| `packages/browser/src/webgpu-bridge.ts` | Direct WebGPU bridge (non-RPC path) |
| `packages/browser/src/rpc-protocol.ts` | SharedArrayBuffer layout, constants |
| `packages/browser/src/test-worker.ts` | 164 browser op tests |
| `packages/browser/src/safetensors.ts` | Zero-copy weight parser |
| `packages/browser/src/cxx-stubs.ts` | C++ runtime stubs |
| `packages/browser/demo/app.ts` | Demo UI — streaming, thinking rendering |
| `packages/browser/demo/test-ops.html` | Test page |
| `packages/browser/demo/test-ops-runner.ts` | Test runner |

### Rust (crates/mlx-core/src/)
| File | Purpose |
|------|---------|
| `sampling.rs` | Temperature, top-k/p, min-p, categorical — **temp>0 crash here** |
| `models/qwen3_5/model.rs` | Qwen3.5 model — chatSync, streaming, forward pass |
| `models/qwen3_5/attention.rs` | Manual SDPA with GQA tile workaround |
| `array/random.rs` | categorical() — calls C++ mlx_array_categorical |
| `array/creation.rs` | test_categorical_sampling_bf16 |

### C++ WebGPU Backend (crates/mlx-sys/mlx/mlx/backend/webgpu/)
| File | Purpose |
|------|---------|
| `matmul.cpp` | GEMM/GEMV kernels (tiled 16x16) |
| `softmax.cpp` | Online softmax (3-phase) |
| `reduce.cpp` | All/row/col reductions |
| `unary.cpp` | Unary element-wise ops |
| `binary.cpp` | Binary element-wise ops |
| `copy.cpp` | Copy with strided access |
| `quantized.cpp` | Quantized matmul (int2/4/8) |
| `device.h/cpp` | Device, pipeline cache, command encoder |
| `allocator.h/cpp` | Buffer allocation with caching |
| `utils.h` | Codegen helpers, dtype mapping, offsets |
| `eval.cpp` | GPU eval entry point |

### Build
| File | Purpose |
|------|---------|
| `packages/browser/build-wasm.sh` | WASM build script |
| `.cargo/config.toml` | WASM build flags (lto=false!) |
| `packages/browser/vite.config.ts` | Vite dev server config |

## Off Limits
- Native Metal backend code (must not regress)
- Common MLX C++ code outside `mlx/backend/webgpu/` (no hacks in common modules)
- Common mlx-node Rust code outside `#[cfg(target_family = "wasm")]` paths
- `Cargo.toml` workspace structure
- `.cargo/config.toml` LTO settings (compiler bug)

## Constraints
1. All 164 browser WebGPU tests must pass
2. Native tests (`cargo test -p mlx-core`, `yarn test`) must still pass
3. Inference output must be coherent (no garbage tokens)
4. No hacks in MLX submodule common modules — only WebGPU-specific code
5. No hacks in mlx-node common code paths — only `#[cfg(target_family = "wasm")]`
6. WASM build must still work with `lto=false` (compiler vtable bug)

## What's Been Tried
*Nothing yet — this is the initial session.*

## Known Issues (Starting State)
1. **temperature > 0 crashes**: `categorical()` calls `cumsum` + `random::uniform` which exhausts WASM heap during decode → `memory access out of bounds` in dlmalloc
2. **bf16 buffer undersize warning**: Some bf16 GPU buffers allocated without 2x expansion for f32 storage
3. **WASM memory limit ~1.9GB**: SharedArrayBuffer addresses > 2GB overflow u32 in WebGPU bridge
4. **Diagnostics code in mlx-worker.ts**: ~130 lines of debug diagnostics run on first chat (GDN checkpoints, weight verification, layer forward tests)
5. **No Vitest browser tests**: Tests use custom worker-based test harness, not Vitest

## Experiment Priority
1. Clean up diagnostics code (quick win for code quality)
2. Fix temperature > 0 sampling (implement Gumbel-max trick or WebGPU categorical)
3. bf16→f32 early conversion architecture improvement
4. Performance optimization (kernel tuning, dispatch optimization)
5. Test infrastructure migration to Vitest browser mode
