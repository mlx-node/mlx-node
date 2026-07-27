# Architecture

For the model-extension boundary and composable media, paged-attention, and
speculative-decoding plan, see [inference-architecture.md](inference-architecture.md).

```
┌──────────────────────────────────────────────────────────┐
│  TypeScript layer — 6 packages                           │
│  @mlx-node/lm      Inference, ChatSession, streaming      │
│  @mlx-node/trl     GRPO/SFT training, datasets            │
│  @mlx-node/vlm     VLM, OCR, document pipelines           │
│  @mlx-node/server  HTTP server (/v1/responses, /v1/messages)│
│  @mlx-node/cli     mlx download, mlx convert, mlx launch  │
│  @mlx-node/core    Native addon (NAPI bindings)           │
├──────────────────────────────────────────────────────────┤
│  Rust compute layer — 5 workspace crates                 │
│  mlx-core        Models, training, ops, vision (all NAPI) │
│  mlx-paged-attn  PagedAttention + Metal kernels           │
│  mlx-sys         Low-level MLX FFI bridge (cpp + headers) │
│  mlx-db          SQLite training persistence              │
│  mlx-tui         mlx-train Ratatui binary (no library deps)│
├──────────────────────────────────────────────────────────┤
│  C++ bridge → Compiled forward paths                      │
│  ~300 FFI declarations, compiled decode via mlx::compile  │
├──────────────────────────────────────────────────────────┤
│  MLX → Metal / CUDA / Accelerate GPUs                     │
└──────────────────────────────────────────────────────────┘
```

## Memory model: unified memory decides the cache hierarchy

MLX is built for Apple Silicon's unified memory — the CPU and the GPU address **one
physical pool**. There is no separate VRAM to spill out of. That single fact shapes
every caching decision in this repo, and it is why designs borrowed from vLLM have to
be re-derived rather than ported.

```
vLLM (discrete GPU)                 mlx-node (unified memory)

  GPU VRAM      scarce                unified pool   weights
     │  offload frees real bytes         │           paged KV pool
  host RAM      abundant                 │           sidecars / checkpoints
     │                                   │           ALL COMPETE FOR THE SAME BYTES
   disk         capacity               disk          the only tier below the pool

  two hops                            one hop
```

**A RAM tier would be a no-op.** vLLM's `CPUOffloadingManager` earns its keep because
moving KV from VRAM to host RAM frees a genuinely separate, scarce resource. Here the
destination *is* the source. Disk is the only place a block can go that gives memory
back, so the cold tier
([paged-cache.md](paged-cache.md#ssd-cold-tier-hybrid-families-and-the-auxiliary-sidecar))
is a single hop, not the bottom of a ladder.

**Every in-memory retention limit is a tax on the model.** A checkpoint kept for reuse
takes bytes from the weights and the paged pool. This is why the sliding-window and
GDN checkpoint stores derive their limits from a byte budget
(`GEMMA4_SLIDING_CHECKPOINT_MEMORY_BUDGET_BYTES`,
`GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES`, `cold_tier::gdn_prefix_checkpoint_limit`)
rather than from a tuned count — and why raising one is never free. The same pressure
runs the other way: oversizing the paged pool tanks long-context decode through
residency thrash, and pool bytes sit **outside** the MLX cache-limit budget, so the
pool is sized to the context, not to the machine.

**Persistence, not capacity.** vLLM's filesystem tier
(`vllm/v1/kv_offload/tiering/fs/`) buys capacity inside one process lifetime: it has no
`fsync`, no crash-durability contract, and nothing rebuilds an index from files already
on disk, so a fresh process cannot see what the last one wrote. Reuse *by a later
process* is exactly our feature, which is why the cold tier carries fingerprints,
payload checksums, version-skew rejection and a real `fsync(2)` — machinery vLLM does
not need and does not have.

**Unified memory makes the disk hop cheap, and cheaper the bigger the model.** Staging
buffers are `StorageModeShared`, so a "readback" is a blit inside the same memory with
no PCIe crossing. Measured on M5 Max, `--release`, after the single-command-buffer
batching. The restore figure is the batched `restore_block` cost also cited in
[paged-cache.md](paged-cache.md); the prefill column is per-model TTFT divided by
blocks prefilled, taken from `mlx agent` turns on real weights, so treat it as the
right order of magnitude rather than a bench-grade constant:

| model         | prefill ms/block | restore ms/block | restore wins by |
| ------------- | ---------------- | ---------------- | --------------- |
| `qwen3` 0.6B  | 3.80             | 1.158            | 3.3×            |
| `qwen3_5_moe` | 11.10            | 1.158            | 9.6×            |
| `qwen3_5` 27B | 24.82            | 1.158            | 21.4×           |

Capture costs the inference thread 0.323 ms/block for the readback (the disk write
itself is off-thread, on the writer). A restored block saves `prefill - restore` =
2.64 / 9.94 / 23.66 ms, so **one restored block pays for capturing 8 / 31 / 73 blocks**.
Restoring from SSD already beats recomputing, and
the margin **grows** with model size — the opposite of the usual "disk is too slow"
intuition. So per-block cost is not the constraint and there is no transfer-vs-recompute
heuristic to tune. What limits payoff is reuse **depth**: how far back a chain can
anchor. That is the axis the checkpoint ladders exist to move.

The rule that falls out:

```
scarce  →  the one unified pool, and the per-turn tax on the inference thread
cheap   →  disk capacity, and per-block restore
```

Bound memory hard, bound disk by quota, and spend the per-turn write budget on depth.

## Package dependency chain

```
@mlx-node/core (Rust/NAPI native addon)
    ├── @mlx-node/lm        inference, models, streaming, tools, profiling
    │     ├── @mlx-node/trl    training (GRPO, SFT, datasets, rewards)
    │     ├── @mlx-node/vlm    vision (VLM, OCR, document pipeline)
    │     └── @mlx-node/server HTTP server (SessionRegistry, /v1/* endpoints)
    └── @mlx-node/cli       depends on core + lm + server
```

`mlx-tui` is the workspace binary crate (Ratatui-based `mlx-train` TUI) — it's a workspace member but no other crate depends on it, so it's built separately via `cargo build -p mlx-tui`. `@mlx-node/internal-tools` lives in root `devDependencies` and is not part of the runtime chain.

## Repository layout

```
mlx-node/
├── Cargo.toml                  workspace manifest (5 crates)
├── package.json                npm workspaces (6 packages + examples)
├── vite.config.ts              Vitest + Oxlint + Oxfmt config
├── tsconfig.json               TypeScript project references
│
├── crates/
│   ├── mlx-sys/                MLX C/C++ FFI bridge — see ffi-cpp.md
│   ├── mlx-core/               All NAPI exports: models, training, ops, vision
│   ├── mlx-paged-attn/         PagedAttention + Metal shaders — see paged-cache.md
│   ├── mlx-db/                 SQLite training persistence
│   └── mlx-tui/                mlx-train Ratatui binary (standalone)
│
├── packages/
│   ├── core/                   @mlx-node/core (native addon + .d.cts)
│   ├── lm/                     @mlx-node/lm
│   │   └── src/
│   │       ├── chat-session.ts   ChatSession<M> cross-model wrapper
│   │       ├── stream.ts         Session-aware models + callback→AsyncGenerator bridge
│   │       ├── profiling.ts      JS profiling API
│   │       ├── models/           loadModel, loadSession, configs
│   │       └── tools/            Tool definition types
│   ├── trl/                    @mlx-node/trl (trainers/, data/, utils/)
│   ├── vlm/                    @mlx-node/vlm (models/, pipeline/)
│   ├── server/                 @mlx-node/server
│   │   └── src/
│   │       ├── endpoints/        /v1/responses, /v1/messages
│   │       ├── session-registry.ts  SessionRegistry — owns ChatSession lifetimes
│   │       └── host/             @mlx-node/server/host — reusable inference-host
│   │                             bootstrap (discovery, single-resident swap,
│   │                             paged-config overrides, engine env policy).
│   │                             Subpath export so the plain HTTP handler does
│   │                             not pull in model loading. Used by `mlx serve`,
│   │                             `mlx launch claude`, and the desktop sidecar.
│   │                             `host/paths` is a further, dependency-free
│   │                             subpath (~/.mlx-node layout) for callers that
│   │                             must not dlopen the native addon.
│   └── cli/                    @mlx-node/cli — see cli.md
│
├── __test__/                   TypeScript tests
└── examples/                   lm.ts, vlm-inference.ts, paddle-ocr-pipeline.ts, tool-use-example.ts, grpo/, sft/
```

## Build flow

| Command                            | Output                                                                                                                                                            |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `yarn build`                       | `yarn build:native && yarn build:ts`                                                                                                                              |
| `yarn build:native`                | macOS: `packages/core/index.cjs`, `mlx-core.darwin-arm64.node`, `mlx.metallib`, `paged_attn.metallib`. Linux/CUDA: `mlx-core.linux-arm64-gnu.node` (no metallibs) |
| `yarn build:ts`                    | `packages/*/dist/` via `tsc -b` (project references)                                                                                                              |
| `yarn typecheck`                   | TypeScript type-check only                                                                                                                                        |
| `cargo build --release -p mlx-tui` | `mlx-train` TUI binary                                                                                                                                            |

`yarn build:native` is the **canonical native build** — runs the napi-rs pipeline through `packages/core/build.ts` (executed via `oxnode`). Running `cargo build` directly does **not** produce the `.node` addon.

## Adding a new native operation

1. Add FFI declaration in `crates/mlx-sys/src/lib.rs`.
2. Add C++ bridge function in the appropriate `crates/mlx-sys/src/mlx_*.cpp` file (see [ffi-cpp.md](ffi-cpp.md) for which file owns what).
3. Add a Rust wrapper in `crates/mlx-core/src/` with `#[napi]` exports.
4. Run `yarn build:native` to regenerate NAPI bindings and `packages/core/index.d.cts`.
5. Add tests using TypedArray helpers.

If you added a **new** `.cpp` file, run `rm -rf target/release/build/mlx-sys-*` once — the `cc` crate caches the source-file list across builds and won't pick up new files otherwise.

`mlx-core` also has a `build.rs` that drives `cc`, for vendored C compiled as a test oracle rather than as part of the bridge (`crates/mlx-core/vendor/ggml/`, see [ffi-cpp.md](ffi-cpp.md)). The same caching applies, so a new `.c` file there needs `rm -rf target/release/build/mlx-core-*`.

## Adding a TypeScript utility

1. Pick the package by responsibility: `lm` (inference), `trl` (training), `vlm` (vision), `server` (HTTP), `cli` (CLI).
2. Add to `packages/<pkg>/src/`, export from `packages/<pkg>/src/index.ts`.
3. Run `yarn build:ts && yarn typecheck`.
