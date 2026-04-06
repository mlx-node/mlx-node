# Autoresearch Worklog: WebGPU Production Readiness

## Session: 2026-04-06

### Data Summary
- **Model**: Qwen3.5 0.8B (bf16, 24 layers)
- **Architecture**: Two-worker pattern (wasm-worker + gpu-worker via SharedArrayBuffer/Atomics)
- **Current state**: temp=0 inference works, temp>0 crashes (categorical sampling exhausts WASM heap)
- **Browser tests**: 164 WebGPU op tests in custom worker harness
- **Known bugs**: bf16 buffer undersize, diagnostics code in production path

### Key Insights
- `categorical()` in C++ MLX uses `cumsum` + `random::uniform` which exhausts WASM heap during decode
- WASM memory limited to ~1.9GB (SharedArrayBuffer u32 overflow at 2GB+)
- bf16 stored as f32 on GPU (WGSL has no bfloat16 type), conversion happens at upload time
- Diagnostics code (~130 lines) runs on first chat — significant code quality issue
- LTO must stay disabled (compiler bug corrupts C++ vtables)

### Next Ideas
1. Remove diagnostics from mlx-worker.ts (code quality, may slightly improve first-chat perf)
2. Implement Gumbel-max trick for temp>0 sampling (avoids categorical/cumsum)
3. Early bf16→f32 conversion at weight load time
4. Optimize matmul kernel tile sizes for WebGPU
5. Migrate browser tests to Vitest browser mode
