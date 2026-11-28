# Contributing to MLX-Node

Thank you for your interest in contributing to MLX-Node!

## Prerequisites

- **Node.js** `>= 22.19` (with Yarn 4 via `packageManager` / Vite+ `vp`)
- **Rust** `>= 1.89` (edition 2024)
- **Platform SDKs:**
  - macOS on Apple Silicon with macOS 26.2+ SDK (Xcode / Command Line Tools) for Metal builds
  - Or Linux `aarch64` / Windows ARM64 with CUDA 13.x + cuDNN 9 for experimental CUDA inference builds

## Getting Started

1. Clone with submodules (MLX lives in `crates/mlx-sys/mlx`):

   ```bash
   git clone --recurse-submodules https://github.com/mlx-node/mlx-node.git
   cd mlx-node
   ```

2. Install dependencies and build:

   ```bash
   vp install
   yarn build
   ```

   > `yarn build:native` is the canonical native build — running `cargo build` directly does not produce the `.node` addon or copy metallibs/headers.

## Common Commands

```bash
# Build
yarn build                                       # native addon + TypeScript packages
yarn build:native                                # Rust/NAPI native addon
yarn build:ts                                    # tsc -b across workspace packages
yarn typecheck                                   # TypeScript type-check

# Test
vp test                                          # all TypeScript tests (pure-ts, darwin-ts, addon)
vp test __test__/path/to.test.ts                 # single TypeScript test file
cargo test -p mlx-core                           # Rust unit tests
cargo test -p mlx-paged-attn                     # paged-attention tests

# Lint & Format
vp fmt                                           # Oxfmt via Vite+
vp lint --type-aware --type-check                # Oxlint with type checking
cargo clippy --all --fix --allow-dirty --allow-staged
cargo fmt

# Run TypeScript scripts
oxnode <file.ts>
```

## Project Layout

```text
mlx-node/
├── crates/                 # 8 Rust crates (mlx-sys, mlx-core, mlx-paged-attn, mlx-db, mlx-tts, mlx-audio, mlx-clairic, mlx-metal)
├── packages/               # 14 workspace packages
│   ├── core/               # @mlx-node/core native bindings + npm/{darwin-arm64,linux-arm64-gnu,win32-arm64-msvc}
│   ├── lm/                 # @mlx-node/lm text/chat inference
│   ├── vlm/                # @mlx-node/vlm vision-language & document pipelines
│   ├── trl/                # @mlx-node/trl GRPO & SFT training
│   ├── server/             # @mlx-node/server OpenAI/Anthropic-compatible HTTP server
│   ├── cli/                # @mlx-node/cli (`mlx` CLI)
│   ├── agent/              # @mlx-node/agent agent session/runtime helpers
│   ├── asr/, tts/, rerank/, privacy/
│   ├── dashboard/          # Control Panel UI & backend
│   ├── desktop/            # Portato Electron desktop app
│   └── internal-tools/     # Release versioning tools
├── __test__/               # Cross-package integration & unit tests
├── examples/               # End-to-end TypeScript examples
└── docs/                   # Architecture & subsystem guides
```

## Architecture Guides

- [docs/architecture.md](docs/architecture.md) — Workspace layout, packages, build flow, and native ops
- [docs/adding-a-model.md](docs/adding-a-model.md) — Checklist for onboarding a new model family
- [docs/ffi-cpp.md](docs/ffi-cpp.md) — C++ FFI bridge (`crates/mlx-sys/src/{common,metal,cuda}/`) and shaders
- [docs/monorepo-paths.md](docs/monorepo-paths.md) — `@mlx-node/*` export conditions and subpath resolution

## Pull Request Checklist

- Ensure `yarn typecheck`, `vp lint --type-aware --type-check`, `cargo fmt --all -- --check`, and relevant `vp test` / `cargo test` suites pass.
- When adding a new C++ source file under `crates/mlx-sys/src/`, clear `target/release/build/mlx-sys-*` so `build.rs` picks it up cleanly, and run `yarn build:native` to regenerate `packages/core/index.d.cts`.
- Keep PRs focused and include tests covering new behavior or bug fixes.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
