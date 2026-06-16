# Architecture

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
│   │       └── session-registry.ts  SessionRegistry — owns ChatSession lifetimes
│   └── cli/                    @mlx-node/cli — see cli.md
│
├── __test__/                   TypeScript tests
└── examples/                   lm.ts, vlm-inference.ts, paddle-ocr-pipeline.ts, tool-use-example.ts, grpo/, sft/
```

## Build flow

| Command                            | Output                                                                                         |
| ---------------------------------- | ---------------------------------------------------------------------------------------------- |
| `yarn build`                       | `yarn build:native && yarn build:ts`                                                           |
| `yarn build:native`                | macOS: `packages/core/index.cjs`, `mlx-core.darwin-arm64.node`, `mlx.metallib`, `paged_attn.metallib`. Linux/CUDA: `mlx-core.linux-arm64-gnu.node` (no metallibs) |
| `yarn build:ts`                    | `packages/*/dist/` via `tsc -b` (project references)                                           |
| `yarn typecheck`                   | TypeScript type-check only                                                                     |
| `cargo build --release -p mlx-tui` | `mlx-train` TUI binary                                                                         |

`yarn build:native` is the **canonical native build** — runs the napi-rs pipeline through `packages/core/build.ts` (executed via `oxnode`). Running `cargo build` directly does **not** produce the `.node` addon.

## Adding a new native operation

1. Add FFI declaration in `crates/mlx-sys/src/lib.rs`.
2. Add C++ bridge function in the appropriate `crates/mlx-sys/src/mlx_*.cpp` file (see [ffi-cpp.md](ffi-cpp.md) for which file owns what).
3. Add a Rust wrapper in `crates/mlx-core/src/` with `#[napi]` exports.
4. Run `yarn build:native` to regenerate NAPI bindings and `packages/core/index.d.cts`.
5. Add tests using TypedArray helpers.

If you added a **new** `.cpp` file, run `rm -rf target/release/build/mlx-sys-*` once — the `cc` crate caches the source-file list across builds and won't pick up new files otherwise.

## Adding a TypeScript utility

1. Pick the package by responsibility: `lm` (inference), `trl` (training), `vlm` (vision), `server` (HTTP), `cli` (CLI).
2. Add to `packages/<pkg>/src/`, export from `packages/<pkg>/src/index.ts`.
3. Run `yarn build:ts && yarn typecheck`.
