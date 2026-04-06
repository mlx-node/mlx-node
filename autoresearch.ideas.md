# Autoresearch Ideas Backlog

## Priority 1: Fix WASM C++ Build Pipeline
The cmake for WASM with `WEBGPU_BACKEND=WASI_IMPORT` fails when the cmake build cache is cleared (BLAS not found despite `MLX_NO_BLAS=ON`). This means we can't reliably modify C++ WebGPU backend code.
- Root cause: clearing cmake build dir loses cached BLAS detection
- Fix: add `WASI_IMPORT` handling to CMakeLists.txt AND ensure `find_package(BLAS)` doesn't run on WASI
- Once fixed, we can apply: batch size increase (64→512), pipeline dedup, fused C++ dispatch

## Priority 2: RPC Batching - More Functions
Current fused dispatch saves 2 RPCs per dispatch. But other patterns are still expensive:
- `createBuffer + getMappedRange + memcpy + unmap` = 3 RPCs per uniform buffer
- `createBindGroup + releaseBindGroup` = 2 RPCs per dispatch
- `beginComputePass + endComputePass + releaseComputePass` = 3 RPCs per dispatch (due to C++ max_ops=64)

### Ideas:
- Fuse `createBindGroup(layout, buffers...)` from JS side by intercepting the descriptor
- Cache `beginComputePass` handle across dispatches (skip begin/end when pass is already active)
- Implement a "batch command" RPC that accepts multiple operations in one roundtrip

## Priority 3: Architecture Improvements
- bf16→f32 conversion at weight upload (once) vs during each kernel
- Reuse uniform buffers instead of creating new ones each dispatch
- Pool bind groups by layout signature

## Priority 4: Test Infrastructure
- Migrate 162 browser tests to Vitest browser mode
- Add Playwright-based E2E test for temp>0 inference
- Add perf regression test (ensure tok/s doesn't drop below threshold)
