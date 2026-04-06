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

### Run 2: Remove diagnostics from mlx-worker.ts — decode_tok_s=2.4 (KEEP)
- Timestamp: 2026-04-06 23:40
- What changed: Removed ~130 lines of debug diagnostics (GDN checkpoints, weight verification, layer forward tests, bf16 verify) from handleChat in mlx-worker.ts
- Result: 2.4 tok/s (same as baseline), code quality improvement
- Insight: Diagnostics only ran on first chat. Equal perf → keep (simpler is better)
- Next: Fix temperature > 0 sampling

### Run 3: Fix temperature > 0 sampling — decode_tok_s=2.5, temp_gt0_works=1 (KEEP)
- Timestamp: 2026-04-07 00:00
- What changed: Implemented cpu_categorical_sample() in sampling.rs. Reads logits to CPU via to_float32_vec(), applies temperature/softmax/cumsum in Rust, samples with xorshift64 RNG. Also added to_float32_vec() internal method and fixed test-worker.ts missing mlx_stream_write/reset stubs.
- Result: 2.5 tok/s, TTFT 2796ms, temp>0 NOW WORKS
- Insight: Previous temp>0 crash was from MLX's categorical() using cumsum+random::uniform which create 248K-element temp arrays, exhausting the ~1.9GB WASM heap. Pure Rust sampling avoids all MLX array ops for intermediates. Also discovered WASM incremental builds were STALE — need to force recompilation when changing Rust sampling code.
- Next: Performance optimization, architecture improvements

### Key Insights
- MLX's categorical() on WASM exhausts heap via cumsum+random::uniform (248K temp arrays per token)
- WASM incremental builds can be stale — `cargo clean -p mlx-core` forces proper recompilation
- `to_float32()` returns napi::Float32Array which is NOT iterable in Rust — need internal `to_float32_vec()` returning Vec<f32>
- GPU readback of full vocab (248K * 4 = ~1MB) works fine via SharedArrayBuffer readback buffer (4MB limit)
- `RandomBits`, `Sort`, `Scan` have NO WebGPU implementation (NO_GPU in primitives.cpp)

### Run 4: Batch size 64→512 + setPipeline dedup — decode_tok_s=2.8 (KEEP)
- Timestamp: 2026-04-07 00:09
- What changed: Increased max_ops_per_commit from 64 to 512 in C++ device.cpp. Added pipeline/bindgroup dedup to skip redundant RPC calls when same pipeline is used consecutively.
- Result: 2.8 tok/s (+12% from 2.5), TTFT 2739ms
- Insight: Batch size increase helps but isn't the main bottleneck. The RPC overhead per individual call dominates.

### Run 5: Fused dispatch RPC — decode_tok_s=4.1 (KEEP)
- Timestamp: 2026-04-07 00:20
- What changed: In webgpu-bridge-stub.ts, buffer setPipeline and setBindGroup(0) calls locally. When dispatch is called, send all three as a single FUSED_DISPATCH RPC via Atomics.wait. Reduces from 3 separate roundtrips to 1 per dispatch.
- Result: **4.1 tok/s (+46%)**, TTFT 1585ms (-42%)
- Insight: RPC roundtrip batching is the single highest-impact optimization. Each Atomics.wait roundtrip costs ~100µs. With ~120 dispatches per token, saving 2 RPCs each = ~24ms saved per token (from ~36ms to ~12ms dispatch overhead).

### Key Insights
- MLX's categorical() on WASM exhausts heap via cumsum+random::uniform (248K temp arrays per token)
- WASM incremental builds can be stale — `cargo clean -p mlx-core` forces proper recompilation
- `to_float32()` returns napi::Float32Array which is NOT iterable in Rust — need internal `to_float32_vec()` returning Vec<f32>
- GPU readback of full vocab (248K * 4 = ~1MB) works fine via SharedArrayBuffer readback buffer (4MB limit)
- `RandomBits`, `Sort`, `Scan` have NO WebGPU implementation (NO_GPU in primitives.cpp)
- **RPC batching is the #1 perf lever**: each Atomics.wait/notify roundtrip costs ~100µs
- cmake for WASM doesn't handle WASI_IMPORT backend properly — can't easily modify C++ code without fixing cmake

### Next Ideas
1. Fuse createBuffer+getMappedRange+unmap for uniform buffers (3→1 RPC, needs C++ or complex JS interception)
2. Cache bind group layouts per pipeline (avoid recreation)
3. Use queueWriteBuffer instead of mappedAtCreation for small uniforms (1 RPC instead of 3)
4. Architecture: bf16→f32 early conversion at weight load time  
5. Test infra: migrate browser tests to Vitest browser mode
6. Code quality: add proper error handling and types to worker communication
