# MLX-Node: High-Performance ML Framework for Node.js

MLX-Node brings Apple's MLX library to Node.js with Metal GPU acceleration (plus experimental NVIDIA CUDA inference) through a Rust/NAPI/C++ bridge. It supports inference (Qwen3, Qwen3.5, Gemma4, LFM2.5, Nemotron 3.5 Lightning, DeepSeek V4.1 Flash and Hy4-flash with SSD expert streaming), training (GRPO, SFT), vision-language models, document processing (PaddleOCR-VL, PP-\* pipelines), and embeddings (Harrier).

## References

There are already some wild used inference implementations for your reference:

- ./mlx-lm, the MLX inference Python library from the official MLX team
- ./mlx-vlm, the MLX vision-language model inference library, more active maintenance
- ~/workspace/github/vllm, the state of art inference library, mostly optimized for CUDA/ROCm but we borrow a lot of paged attention design from it. Widely used in server production environments.

## Topic guides

- [docs/architecture.md](docs/architecture.md) — Workspace layout, packages, dependency chain, build flow, adding native ops / TS utilities
- [docs/monorepo-paths.md](docs/monorepo-paths.md) — How `@mlx-node/*` specifiers resolve: the `@mlx-node/source` export condition, dev/test vs published surfaces, adding a subpath
- [docs/inference-architecture.md](docs/inference-architecture.md) — Model registry, per-turn execution planning, media/paged/speculative composition, adding a model
- [docs/adding-a-model.md](docs/adding-a-model.md) — Per-stage checklist for onboarding a model family: required registration sites, failure modes, conditional/trainable fan-out, CI conventions
- [docs/models.md](docs/models.md) — Model implementations, ChatSession API, streaming, VLM, document pipelines
- [docs/privacy-filter.md](docs/privacy-filter.md) — PII detection & redaction (openai/privacy-filter port)
- [docs/training.md](docs/training.md) — GRPO, SFT, autograd, optimizers, persistence
- [docs/vllm-speculative-alignment.md](docs/vllm-speculative-alignment.md) — Speculative × paged KV vs vLLM: adopted laws, permanent divergences (with reopen triggers), and the remaining alignment ladder
- [docs/paged-cache.md](docs/paged-cache.md) — Block-paged KV cache support matrix and parity gates
- [docs/ffi-cpp.md](docs/ffi-cpp.md) — C++ FFI bridge, compiled Qwen3.5 forward paths, Metal shaders
- [docs/mlx-fork.md](docs/mlx-fork.md) — The MLX submodule: the 12 fork patches and why each stays, what moved into mlx-node, how to bump the pin (golden gates)
- [docs/perf.md](docs/perf.md) — Profiling, env-var inventory, GPU arch detection, quantization
- [docs/cli.md](docs/cli.md) — `mlx download`, `mlx convert`, `mlx launch claude`
- [docs/convert-quantize.md](docs/convert-quantize.md) — Convert/quantize internals: on-disk formats, recipe decision engine, GGUF import, provenance, gotchas
- [docs/dashboard.md](docs/dashboard.md) — Control Panel window (Portato desktop app) for models/sessions/metrics/cache; `app://` + MessagePort, no HTTP server, no `mlx dashboard`

## Build, test, lint

```bash
# Build
yarn build                                       # native + TS
yarn build:native                                # Rust/NAPI native addon (~70s incremental)
yarn build:ts                                    # tsc -b across packages
yarn typecheck                                   # TS type-check only

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
- Concurrent chat inference is per-family, not global. Sessions on one eligible **paged** Qwen3, LFM2, Qwen3.5 dense/MoE, Gemma4, Muse-Glimmer, Nemotron 3.5 Lightning (`nemotron_h`), or Hy4-flash (`hy_v4_flash`) model overlap in a continuous-batching scheduler on that model's single `"mlx-model"` OS thread. All of them are on by default with no env; `MLX_SERVE_FORCE_SERIAL=1` rolls any of them back to whole-turn for A/B. Qwen3.5 dense/MoE run up to 4 lanes, computed from the machine's memory at load; up to 8 on machines with 256 GB or more (lanes give way before the context window); a turn past the granted lanes waits for one. `hy_v4_flash` runs `min(4, MLX_SCHED_MAX_NUM_SEQS, wave row cap)` lanes at the context and prefill chunk its admission picks for one lane (fewer when memory admits fewer); its AR and MTP turns share waves (MTP always on by default, depth 1, at most 3, never adaptive), every Q8 call padded to at least 2 rows so a row's bits do not depend on its wave (fixture-tested only, not yet on real weights); `MLX_SERVE_FORCE_SERIAL=1` at load gives its flat mode (whole turns, no pool). It streams routed experts and Engram rows from the SSD, serves up to 256K tokens with chunked prefill (8,192 / 4,096 / 2,048 rows), and keeps prefix states on its own SSD tier, which shares the `prefix-cache/` root limit (32 GiB `auto`) with Qwen3.5 and Flash Next, so a 256K Hy4 conversation (about 3.6 GiB of blocks) pushes their idle namespaces out sooner. `nemotron_h` shares the scheduler but **not** the fused decode: any quantized checkpoint (the only published one included) sets `row_exact_decode_projections`, so its decode step is N serial single-row forwards, bit-identical to serial but paying N weight streams. Qwen3.5 MoE keeps its fused AR decode, but its scheduled MTP verify runs once per owner (N non-expert weight streams per verify step), so a lane's tokens are the tokens it has when it runs alone. Qwen3.5 dense DFlash2 text turns are scheduled lanes too: the lanes that reach verify in one step run one verify wave of 2-4 lanes (every K-quant projection once over the stacked `[N,8]` rows on the 16/32-row lane tier; attention and the GDN core per lane), so a lane's tokens are its tokens alone, sampled lanes included, draw for draw (an unpenalized sampled lane with `top_k` 1-256 draws its target rows on the device; other sampled lanes keep the host filter chain). A peer's prefill beside dense DFlash2 lanes runs in layer passes of about 100 ms between decode steps. An AR turn beside draft lanes runs its own one-row decode and is slower than alone. Devices before M5 have no lane tier, so their lanes verify per owner. Each lane past the first costs 370 MB at 27B (second GDN cell, draft ring, window, GDN tape). Measured speeds are in `docs/concurrent-inference.md`. Plain AR rows of a Qwen3.5 dense load that decode in one step share one fused `[N,1]` forward whose projection and GDN kernels are not the one-row kernels, so such a turn is not guaranteed bit-identical to the same turn alone (at temperature 0 its tokens can differ). Everything else still runs one whole turn at a time in the exclusive/barrier lane: flat-cache models (incl. DeepSeek V4.1 Flash, one sequence), Gemma4 media and MTP/DSpark owners, Qwen3.5 media (and speculative text turns after an image turn), adaptive-MTP and MoE DFlash2 turns, nemotron_h MTP turns, training, save, and reset. One `ChatSession` always allows only one turn in flight. Different loaded models run in parallel, one model thread each (MLX state is thread-local; residual coupling is the process-wide Metal allocator pool and wired limit). See `docs/concurrent-inference.md`.

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
