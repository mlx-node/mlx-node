# MLX-Node: High-Performance ML Framework for Node.js

MLX-Node brings Apple's MLX library to Node.js with Metal GPU acceleration (plus experimental NVIDIA CUDA inference) through a Rust/NAPI/C++ bridge. It supports inference (Qwen3, Qwen3.5, Gemma4, LFM2.5), training (GRPO, SFT), vision-language models, document processing (PaddleOCR-VL, PP-\* pipelines), and embeddings (Harrier).

## References

There are already some wild used inference implementations for your reference:

- ./mlx-lm, the MLX inference Python library from the official MLX team
- ./mlx-vlm, the MLX vision-language model inference library, more active maintenance
- ~/workspace/github/vllm, the state of art inference library, mostly optimized for CUDA/ROCm but we borrow a lot of paged attention design from it. Widely used in server production environments.

## Topic guides

- [docs/architecture.md](docs/architecture.md) — Workspace layout, packages, dependency chain, build flow, adding native ops / TS utilities
- [docs/inference-architecture.md](docs/inference-architecture.md) — Model registry, per-turn execution planning, media/paged/speculative composition, adding a model
- [docs/models.md](docs/models.md) — Model implementations, ChatSession API, streaming, VLM, document pipelines
- [docs/privacy-filter.md](docs/privacy-filter.md) — PII detection & redaction (openai/privacy-filter port)
- [docs/training.md](docs/training.md) — GRPO, SFT, autograd, optimizers, `mlx-train` TUI, persistence
- [docs/paged-cache.md](docs/paged-cache.md) — Block-paged KV cache support matrix and parity gates
- [docs/ffi-cpp.md](docs/ffi-cpp.md) — C++ FFI bridge, compiled Qwen3.5 forward paths, Metal shaders
- [docs/perf.md](docs/perf.md) — Profiling, env-var inventory, GPU arch detection, quantization
- [docs/cli.md](docs/cli.md) — `mlx download`, `mlx convert`, `mlx launch claude`
- [docs/convert-quantize.md](docs/convert-quantize.md) — Convert/quantize internals: on-disk formats, recipe decision engine, GGUF import, provenance, gotchas
- [docs/dashboard.md](docs/dashboard.md) — Control Panel window (desktop app) for models/sessions/metrics/cache; `app://` + MessagePort, no HTTP server, no `mlx dashboard`

## Build, test, lint

```bash
# Build
yarn build                                       # native + TS
yarn build:native                                # Rust/NAPI native addon (~70s incremental)
yarn build:ts                                    # tsc -b across packages
yarn typecheck                                   # TS type-check only
cargo build --release -p mlx-tui                 # mlx-train TUI binary

# Test
yarn vite run test                               # all TS tests
yarn vitest __test__/path/to.test.ts             # single TS test
cargo test -p mlx-core                           # Rust unit tests
cargo test -p mlx-paged-attn                     # paged-attention tests

# Lint & format
yarn vite fmt                                    # Oxfmt via Vite+
yarn vite lint --type-aware --type-check         # Oxlint with type checking
cargo clippy --all --fix --allow-dirty --allow-staged
cargo fmt

# Scripts
oxnode <file.ts>                                 # run a TS file (NOT tsx)
```

`yarn build:native` is the canonical native build — running `cargo build` directly does **not** produce the `.node` addon.

## Known limitations

- Primary platform: macOS / Apple Silicon (Metal) — full inference + training + VLM
- Experimental: Linux aarch64 (glibc) + NVIDIA CUDA — **inference-only preview**. Qwen3.6 dense/MoE validated on GB10 / DGX Spark (`sm_121`, CUDA 13.0) via device-agnostic eager fallbacks (no custom CUDA kernels; paged-attn forced off; perf below Apple Silicon). Training, other model families, and x86_64 Linux are untested. See README "Platform Support" + `docs/cuda-poc-benchmark.md`.
- Compiled C++ forward paths use process-wide globals (serialized via `std::sync::Mutex` + `RwLock` in `crates/mlx-core/src/models/qwen3_5/model.rs`)

---

# Using Vite+, the Unified Toolchain for the Web

This project is using Vite+, a unified toolchain built on top of Vite, Rolldown, Vitest, tsdown, Oxlint, Oxfmt, and Vite Task. Vite+ wraps runtime management, package management, and frontend tooling in a single global CLI called `vp`. Vite+ is distinct from Vite, but it invokes Vite through `vp dev` and `vp build`.

## Vite+ Workflow

`vp` is a global binary that handles the full development lifecycle. Run `vp help` for the command list and `vp <command> --help` for a specific command. It wraps the underlying package manager (pnpm/npm/Yarn, detected from `packageManager` / lockfiles), so `vp add`/`remove`/`update`/`install` replace direct package-manager use, and `vp dev`/`build`/`test`/`lint`/`fmt`/`check` map to Vite/Vitest/Oxlint/Oxfmt.

## Common Pitfalls

- **Using the package manager directly:** Do not use pnpm, npm, or Yarn directly. Vite+ can handle all package manager operations.
- **Always use Vite commands to run tools:** Don't attempt to run `vp vitest` or `vp oxlint`. They do not exist. Use `vp test` and `vp lint` instead.
- **Running scripts:** Vite+ built-in commands (`vp dev`, `vp build`, `vp test`, etc.) always run the Vite+ built-in tool, not any `package.json` script of the same name. To run a custom script that shares a name with a built-in command, use `vp run <script>`. For example, if you have a custom `dev` script that runs multiple services concurrently, run it with `vp run dev`, not `vp dev` (which always starts Vite's dev server).
- **Do not install Vitest, Oxlint, Oxfmt, or tsdown directly:** Vite+ wraps these tools. They must not be installed directly. You cannot upgrade these tools by installing their latest versions. Always use Vite+ commands.
- **Use Vite+ wrappers for one-off binaries:** Use `vp dlx` instead of package-manager-specific `dlx`/`npx` commands.
- **Import JavaScript modules from `vite-plus`:** Instead of importing from `vite` or `vitest`, all modules should be imported from the project's `vite-plus` dependency. For example, `import { defineConfig } from 'vite-plus';` or `import { expect, test, vi } from 'vite-plus/test';`. You must not install `vitest` to import test utilities.
- **Type-Aware Linting:** There is no need to install `oxlint-tsgolint`, `vp lint --type-aware` works out of the box.

## CI Integration

For GitHub Actions, consider using [`voidzero-dev/setup-vp`](https://github.com/voidzero-dev/setup-vp) to replace separate `actions/setup-node`, package-manager setup, cache, and install steps with a single action.

```yaml
- uses: voidzero-dev/setup-vp@v1
  with:
    cache: true
- run: vp check
- run: vp test
```

## Review Checklist for Agents

- [ ] Run `vp install` after pulling remote changes and before getting started.
- [ ] Run `vp check` and `vp test` and `cargo test` to validate changes.
