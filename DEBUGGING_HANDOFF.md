# WebGPU Backend — Blocking Issue

## Status
- 119/119 unit tests pass
- `Compiled::eval_gpu` tape replay works (compile enabled)
- Model crashes on 3rd forward pass (decode step 1) with dlmalloc corruption
- **Root cause**: cumulative heap corruption from 48+ eval cycles in the Rust forward_inner path
- **Cannot use compiled C++ forward**: model is quantized (has .scales keys), C++ path only supports dense matmul

## Why It Crashes
Each forward pass through 24 layers creates ~500 array nodes (even with per-step GDN eval). When eval runs, the eval scheduler processes all nodes, allocates/frees hundreds of buffers, and cascading ~ArrayDesc destruction hammers dlmalloc. After 2 full passes (96 layer evals), the heap is corrupted.

## Why Tests Don't Reproduce
Tests use `fromFloat32` arrays (allocator-managed). The model uses `mlx_array_from_gpu_buffer` arrays (externally-managed). The difference in buffer lifecycle causes different heap fragmentation patterns that only manifest under the full model workload.

## Options
1. **Wrap Rust forward_inner in compile()** via new FFI — traces graph once, reuses via compile_replace. Requires adding `mlx_compile_forward_inner()` C++ wrapper.
2. **Fix the heap corruption root cause** — something in the eval→dispatch→free cycle corrupts dlmalloc. Needs WASM-level memory debugging (AddressSanitizer for WASM).
3. **Use non-quantized model weights** — enables the existing compiled C++ path. Requires a bf16/f32 Qwen3.5 0.8B model (not quantized).
4. **Reduce eval frequency** — eval every 4 layers instead of every layer. Reduces heap pressure but increases graph size.
