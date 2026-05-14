# MLX-Node: High-Performance ML Framework for Node.js

MLX-Node brings Apple's MLX library to Node.js with Metal GPU acceleration through a Rust/NAPI/C++ bridge. It supports inference (Qwen3, Qwen3.5, Gemma4, LFM2.5), training (GRPO, SFT), vision-language models, document processing (PaddleOCR-VL, PP-\* pipelines), and embeddings (Harrier).

## References

There are already some wild used inference implementations for your reference:

- ./mlx-lm, the MLX inference Python library from the official MLX team
- ./mlx-vlm, the MLX vision-language model inference library, more active maintenance
- ~/workspace/github/vllm, the state of art inference library, mostly optimized for CUDA/ROCm but we borrow a lot of paged attention design from it. Widely used in server production environments.

## Topic guides

- [docs/architecture.md](docs/architecture.md) — Workspace layout, packages, dependency chain, build flow, adding native ops / TS utilities
- [docs/models.md](docs/models.md) — Model implementations, ChatSession API, streaming, VLM, document pipelines
- [docs/privacy-filter.md](docs/privacy-filter.md) — PII detection & redaction (openai/privacy-filter port)
- [docs/training.md](docs/training.md) — GRPO, SFT, autograd, optimizers, `mlx-train` TUI, persistence
- [docs/paged-cache.md](docs/paged-cache.md) — Block-paged KV cache support matrix and parity gates
- [docs/ffi-cpp.md](docs/ffi-cpp.md) — C++ FFI bridge, compiled Qwen3.5 forward paths, Metal shaders
- [docs/perf.md](docs/perf.md) — Profiling, env-var inventory, GPU arch detection, quantization
- [docs/cli.md](docs/cli.md) — `mlx download`, `mlx convert`, `mlx launch claude`

## Top-level structure

```
crates/                           Rust workspace (5 crates)
├── mlx-sys/                      C++ FFI bridge → MLX
├── mlx-core/                     All NAPI exports
├── mlx-paged-attn/               Paged attention + Metal kernels
├── mlx-db/                       SQLite training persistence
└── mlx-tui/                      mlx-train Ratatui binary

packages/                         npm workspaces (@mlx-node/*)
├── core/                         Native addon + .d.cts
├── lm/                           Inference, ChatSession, streaming
├── trl/                          GRPO / SFT trainers
├── vlm/                          Vision + document pipelines
├── server/                       /v1/responses, /v1/messages, SessionRegistry
└── cli/                          mlx binary
```

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

## Imports

```typescript
// Inference + chat sessions
import {
  Qwen3Model,
  Qwen35Model,
  Qwen35MoeModel,
  Gemma4Model,
  Lfm2Model,
  loadModel,
  loadSession,
  ChatSession,
  QWEN3_CONFIGS,
  QWEN35_CONFIGS,
  LFM2_CONFIGS,
  enableProfiling,
  disableProfiling,
} from '@mlx-node/lm';

// Training
import { GRPOTrainer, GRPOTrainerConfig, SFTTrainer, SFTTrainerConfig } from '@mlx-node/trl';

// Vision + document processing
import {
  VLModel,
  QianfanOCRModel,
  StructureV3Pipeline,
  DocLayoutModel,
  TextDetModel,
  TextRecModel,
  DocOrientationModel,
  DocUnwarpModel,
} from '@mlx-node/vlm';

// Streaming chat via ChatSession
const session = await loadSession('./model-path');
for await (const event of session.sendStream('Hello!')) {
  if (!event.done) process.stdout.write(event.text);
}
```

## Known limitations

- macOS only (Metal backend, Apple Silicon)
- No CUDA support
- Compiled C++ forward paths use process-wide globals (serialized via `std::sync::Mutex` + `RwLock` in `crates/mlx-core/src/models/qwen3_5/model.rs`)

---

# Using Vite+, the Unified Toolchain for the Web

This project is using Vite+, a unified toolchain built on top of Vite, Rolldown, Vitest, tsdown, Oxlint, Oxfmt, and Vite Task. Vite+ wraps runtime management, package management, and frontend tooling in a single global CLI called `vp`. Vite+ is distinct from Vite, but it invokes Vite through `vp dev` and `vp build`.

## Vite+ Workflow

`vp` is a global binary that handles the full development lifecycle. Run `vp help` to print a list of commands and `vp <command> --help` for information about a specific command.

### Start

- create - Create a new project from a template
- migrate - Migrate an existing project to Vite+
- config - Configure hooks and agent integration
- staged - Run linters on staged files
- install (`i`) - Install dependencies
- env - Manage Node.js versions

### Develop

- dev - Run the development server
- check - Run format, lint, and TypeScript type checks
- lint - Lint code
- fmt - Format code
- test - Run tests

### Execute

- run - Run monorepo tasks
- exec - Execute a command from local `node_modules/.bin`
- dlx - Execute a package binary without installing it as a dependency
- cache - Manage the task cache

### Build

- build - Build for production
- pack - Build libraries
- preview - Preview production build

### Manage Dependencies

Vite+ automatically detects and wraps the underlying package manager such as pnpm, npm, or Yarn through the `packageManager` field in `package.json` or package manager-specific lockfiles.

- add - Add packages to dependencies
- remove (`rm`, `un`, `uninstall`) - Remove packages from dependencies
- update (`up`) - Update packages to latest versions
- dedupe - Deduplicate dependencies
- outdated - Check for outdated packages
- list (`ls`) - List installed packages
- why (`explain`) - Show why a package is installed
- info (`view`, `show`) - View package information from the registry
- link (`ln`) / unlink - Manage local package links
- pm - Forward a command to the package manager

### Maintain

- upgrade - Update `vp` itself to the latest version

These commands map to their corresponding tools. For example, `vp dev --port 3000` runs Vite's dev server and works the same as Vite. `vp test` runs JavaScript tests through the bundled Vitest. The version of all tools can be checked using `vp --version`. This is useful when researching documentation, features, and bugs.

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
