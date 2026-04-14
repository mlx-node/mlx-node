# MLX-Node: High-Performance ML Framework for Node.js

## Project Overview

MLX-Node is a high-performance machine learning framework for Node.js that brings Apple's MLX library to JavaScript/TypeScript. It supports inference (Qwen3, Qwen3.5, PaddleOCR-VL), training (GRPO, SFT), vision-language models, and document processing pipelines. Uses Metal GPU acceleration through a Rust/NAPI/C++ bridge.

### Core Technology Stack

- **MLX**: Apple's ML framework with Metal GPU acceleration
- **Rust**: ~100,000 lines across 5 crates (mlx-core, mlx-sys, mlx-paged-attn, mlx-tui, mlx-db)
- **C++**: ~7,700 lines (11 .cpp files + 2 headers) for compiled forward passes and FFI bridge
- **Metal**: ~2,500 lines of custom shaders (paged attention, gated delta recurrence)
- **NAPI-RS**: 328 NAPI exports, 221 FFI bindings
- **TypeScript**: 7,671 source lines across 5 packages + 9,125 test lines
- **Tests**: 1,315 total (900 Rust + 415 TypeScript)

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│  TypeScript Layer (7,671 lines, 5 packages)      │
│  @mlx-node/lm   - Inference, streaming, configs  │
│  @mlx-node/trl  - GRPO/SFT training, datasets    │
│  @mlx-node/vlm  - VLM, document pipelines        │
│  @mlx-node/cli  - Model download, conversion      │
│  @mlx-node/core - Native addon (NAPI bindings)    │
├──────────────────────────────────────────────────┤
│  Rust Compute Layer (~100,000 lines, 5 crates)   │
│  Models: Qwen3, Qwen3.5 Dense/MoE, PaddleOCR-VL │
│  Document: DocLayout, TextDet, TextRec, Ori, Unwarp│
│  Training: GRPO engine, SFT engine, autograd      │
│  Infra: transformers, sampling, tokenizer, KVCache │
│  Paged attention with Metal kernels               │
├──────────────────────────────────────────────────┤
│  C++ Bridge (7,700 lines) → Compiled forward paths│
│  221 FFI functions, compiled decode (mlx::compile) │
├──────────────────────────────────────────────────┤
│  MLX → Metal/Accelerate GPU Backend               │
└──────────────────────────────────────────────────┘
```

### Package Dependency Chain

```
@mlx-node/core (Rust/NAPI native addon - 328 exports)
    ├── @mlx-node/lm   (inference: models, streaming, profiling, tools)
    │     ├── @mlx-node/trl (training: GRPO, SFT, datasets, rewards)
    │     └── @mlx-node/vlm (vision: VLM, OCR, document pipeline)
    └── @mlx-node/cli  (CLI: download, convert)
```

### Rust Crate Inventory

| Crate              | Lines                  | Purpose                                         |
| ------------------ | ---------------------- | ----------------------------------------------- |
| **mlx-core**       | 82,339                 | All NAPI exports: models, training, ops, vision |
| **mlx-paged-attn** | 8,473 + 2,043 Metal    | PagedAttention with Metal kernels               |
| **mlx-tui**        | 5,567                  | Ratatui training TUI (`mlx-train` binary)       |
| **mlx-db**         | 2,345                  | SQLite training output persistence              |
| **mlx-sys**        | 1,148 Rust + 7,717 C++ | Low-level MLX FFI bridge                        |

### mlx-core Key Modules

| Module               | Lines  | Purpose                                                                  |
| -------------------- | ------ | ------------------------------------------------------------------------ |
| `models/`            | 36,867 | 9 model implementations (see Models section)                             |
| `transformer/`       | 9,228  | Attention, KVCache, BatchKVCache, RotatingKVCache, QuantizedKVCache, MLP |
| `utils/`             | 6,981  | GGUF, foreign weights, SafeTensors, functional, pickle, imatrix          |
| `grpo/`              | 6,505  | GRPO/DAPO/Dr.GRPO/BNPO loss, advantages, entropy, engine, rewards        |
| `array/`             | 3,972  | 90+ core ops, padding, masking, thread-safe handles                      |
| `nn/`                | 4,010  | Activations, Linear, Conv1d, RMSNorm, LayerNorm, Embedding, Losses       |
| `vision/`            | 2,178  | Conv2d, interpolate, vision encoder, embeddings, projector, 2D RoPE      |
| `optimizers/`        | 2,203  | Adam, AdamW, SGD, RMSprop                                                |
| `convert.rs`         | 1,765  | Model conversion (dtype, quantization, FP8, recipes)                     |
| `sft/`               | 1,192  | SFT training engine with autograd                                        |
| `output_store/`      | 1,155  | Training output persistence (SQLite)                                     |
| `tools/`             | 1,004  | Tool call/thinking parsing (`<tool_call>`, `<think>` tags)               |
| `tokenizer.rs`       | 1,054  | HuggingFace tokenizers + Jinja2 chat templates                           |
| `sampling.rs`        | 882    | Temperature, top-k/p, min-p, repetition penalty                          |
| `decode_profiler.rs` | 706    | Per-generation profiling with TTFT, memory snapshots                     |
| `autograd.rs`        | 503    | MLX value_and_grad integration                                           |
| `profiling.rs`       | 417    | Global profiling store, NAPI exports                                     |

---

## Models (9 implementations)

### Language Models

All generative wrappers share a uniform `ChatSession<M>` surface (`send` / `sendStream` / `sendToolResult` / `reset`) driven by the native `chatSessionStart` / `chatSessionContinue` / `chatSessionContinueTool` NAPI entry points. The legacy `model.chat()` / `model.chatStream()` methods have been removed.

| Model             | Lines | generate() | session | Training | Special                               |
| ----------------- | ----- | :--------: | :-----: | :------: | ------------------------------------- |
| **Qwen3**         | 7,061 |    Yes     |   Yes   | GRPO/SFT | Speculative decoding, Paged attention |
| **Qwen3.5 Dense** | 6,768 |    Yes     |   Yes   | GRPO/SFT | Compiled C++ forward, VLM variant     |
| **Qwen3.5 MoE**   | 4,267 |    Yes     |   Yes   | GRPO/SFT | Compiled C++ forward, VLM variant     |
| **Gemma4**        | —     |    Yes     |   Yes   |    No    | Session-driven streaming              |
| **LFM2.5**        | —     |    Yes     |   Yes   |    No    | Hybrid conv + attention               |

### Vision-Language Models

| Model            | Lines      | Purpose                                                              |
| ---------------- | ---------- | -------------------------------------------------------------------- |
| **PaddleOCR-VL** | 6,770      | OCR + document understanding (ERNIE language model + vision encoder) |
| **Qwen3.5 VLM**  | integrated | Vision encoder (27 layers) on Qwen3.5 dense/MoE with LRU image cache |

### Document Processing Pipeline

| Model                 | Lines | Purpose                                                              |
| --------------------- | ----- | -------------------------------------------------------------------- |
| **PP-DocLayoutV3**    | 6,680 | Document layout analysis (RT-DETR + HGNetV2 backbone, 25 categories) |
| **PP-TextDet**        | 2,082 | Text line detection (DBNet with PPHGNetV2 backbone)                  |
| **PP-TextRec**        | 1,635 | Text recognition (SVTR neck + CTC head, character dictionary)        |
| **PP-DocOrientation** | 695   | 4-class orientation classifier (0/90/180/270 degrees)                |
| **PP-DocUnwarp**      | 895   | Document dewarping via 2D displacement field (UVDocNet)              |

### Compiled C++ Forward Paths

Qwen3.5 models use `mlx::core::compile` for graph caching — traces forward pass once, reuses via `compile_replace`. Files:

- `mlx_qwen35.cpp` (285 lines) — dense compiled decode
- `mlx_qwen35_moe.cpp` (772 lines) — MoE compiled decode with expert routing
- `mlx_qwen35_vlm.cpp` (199 lines) — VLM compiled prefill
- `mlx_qwen35_common.h` (712 lines) — shared helpers (linear_proj, attn, GDN, RoPE)

---

## Training

### GRPO (Group-based Relative Policy Optimization)

Production-ready with 4 loss variants (GRPO, DAPO, Dr.GRPO, BNPO), entropy filtering, importance sampling, batch generation, reward functions.

```typescript
import { loadModel } from '@mlx-node/lm';
import { GRPOTrainer, GRPOConfig } from '@mlx-node/trl';
```

### SFT (Supervised Fine-Tuning)

Full SFT training engine with autograd, gradient accumulation, NaN detection, gradient clipping, weight decay, checkpoint resume.

```typescript
import { SFTTrainer, SFTTrainerConfig } from '@mlx-node/trl';
```

### Autograd

Functional forward pass architecture — stateless transformer components enable MLX to trace the computation graph from parameters to loss. 311 gradients computed automatically for full Qwen3 model.

---

## Key Features

### Streaming API

AsyncGenerator-based streaming for every generative model flows through `ChatSession.sendStream()`:

```typescript
import { loadSession } from '@mlx-node/lm';

const session = await loadSession('./models/Qwen3.5-0.8B');
for await (const event of session.sendStream('Hello!')) {
  if (!event.done) process.stdout.write(event.text);
}
```

### ChatSession Architecture

`ChatSession<M>` is the cross-model chat wrapper in `packages/lm/src/chat-session.ts`. It holds a `SessionCapableModel` and exposes `send`, `sendStream`, `sendToolResult`, `sendToolResultStream`, and `reset`, plus `primeHistory` / `startFromHistory` / `startFromHistoryStream` for server-side cold-start replay. All generative wrappers (Qwen3, Qwen3.5 Dense, Qwen3.5 MoE, Lfm2, Gemma4, and the VLM `QianfanOCRModel` in `@mlx-node/vlm`) structurally satisfy `SessionCapableModel` — any of them can be passed to `new ChatSession(model)`.

Turn 0 (and any turn whose image set changed) dispatches through the native `chatSessionStart` with the full rebuilt history; later turns take the cheap `chatSessionContinue` delta path that reuses the live KV cache. Tool-result turns always use `chatSessionContinueTool`. The server-side `/v1/responses` and `/v1/messages` endpoints route through a per-model `SessionRegistry` that owns the `ChatSession` lifetimes — clients pass `previous_response_id` and the registry handles resume vs. cold-start replay internally.

The legacy `.chat()` / `.chatStream()` surface is fully removed from all generative models. The only exception is PaddleOCR-VL's `VLModel.chat()`, which remains as a deliberate single-turn OCR entry point and is intentionally kept out of the session API.

### CLI Tool (`@mlx-node/cli`)

```bash
mlx download model --model Qwen/Qwen3-0.6B   # Download from HuggingFace
mlx download dataset                           # Download + Parquet→JSONL
mlx convert --model ./model --dtype bf16       # Weight conversion
mlx convert --model ./model --quantize --q-recipe mixed_4_6  # Quantization
mlx convert --model ./model.gguf               # GGUF→SafeTensors
```

### Document Processing Pipeline

```typescript
import { StructureV3Pipeline } from '@mlx-node/vlm';
const pipeline = await StructureV3Pipeline.load(modelDir);
const result = await pipeline.analyze(imageBuffer);
```

### Profiling

Enable via `enableProfiling()` from `@mlx-node/lm` or `MLX_PROFILE_DECODE=1` env var. Reports per-generation timing, phase breakdown (forward, sample, eval, extract), memory snapshots, and TTFT.

**Note**: MLX lazy evaluation means `prefillMs` measures only graph construction (~1ms). Use `timeToFirstTokenMs` (TTFT) as the real prefill latency indicator.

---

## Project Structure

```
mlx-node/
├── Cargo.toml                        # Cargo workspace (5 crates)
├── package.json                      # npm workspaces (5 packages + examples)
├── vite.config.ts                    # Vitest + Oxlint + Oxfmt config
├── tsconfig.json                     # TypeScript project references
│
├── crates/
│   ├── mlx-sys/                      # Low-level MLX C/C++ bindings
│   │   ├── src/lib.rs                # 221 FFI declarations
│   │   ├── src/mlx_*.cpp             # 11 C++ bridge files (6,702 lines)
│   │   ├── src/mlx_*.h               # 2 headers (1,015 lines)
│   │   └── mlx/                      # MLX git submodule
│   │
│   ├── mlx-core/                     # @mlx-node/core - All NAPI exports
│   │   └── src/
│   │       ├── array/                # Core tensor ops (3,972 lines)
│   │       ├── nn/                   # Neural network layers (4,010 lines)
│   │       ├── transformer/          # Attention, caches, blocks (9,228 lines)
│   │       ├── models/               # 9 model implementations (36,867 lines)
│   │       │   ├── qwen3/            # Qwen3 (7,061 lines)
│   │       │   ├── qwen3_5/          # Qwen3.5 Dense + VLM (6,768 lines)
│   │       │   ├── qwen3_5_moe/      # Qwen3.5 MoE (4,267 lines)
│   │       │   ├── paddleocr_vl/     # PaddleOCR VLM (6,770 lines)
│   │       │   ├── pp_doclayout_v3/  # Document layout (6,680 lines)
│   │       │   ├── pp_text_det/      # Text detection (2,082 lines)
│   │       │   ├── pp_text_rec/      # Text recognition (1,635 lines)
│   │       │   ├── pp_doc_ori/       # Orientation (695 lines)
│   │       │   └── pp_doc_unwarp/    # Dewarping (895 lines)
│   │       ├── vision/               # Shared vision components (2,178 lines)
│   │       ├── grpo/                 # GRPO training (6,505 lines)
│   │       ├── sft/                  # SFT training (1,192 lines)
│   │       ├── optimizers/           # Adam, AdamW, SGD, RMSprop (2,203 lines)
│   │       ├── gradients/            # Manual backward passes (974 lines)
│   │       ├── utils/                # GGUF, foreign weights, functional (6,981 lines)
│   │       ├── output_store/         # Training persistence (1,155 lines)
│   │       ├── tools/                # Tool call/thinking parsing (1,004 lines)
│   │       ├── convert.rs            # Model conversion pipeline (1,765 lines)
│   │       ├── tokenizer.rs          # HuggingFace + Jinja2 (1,054 lines)
│   │       ├── sampling.rs           # All sampling strategies (882 lines)
│   │       ├── decode_profiler.rs    # Generation profiling (706 lines)
│   │       ├── autograd.rs           # Automatic differentiation (503 lines)
│   │       └── profiling.rs          # Global profiling store (417 lines)
│   │
│   ├── mlx-paged-attn/              # PagedAttention + Metal kernels
│   │   ├── src/                      # 8,473 Rust lines
│   │   └── metal/                    # 2,043 lines of Metal shaders
│   │
│   ├── mlx-tui/                      # Training TUI (Ratatui, 5,567 lines)
│   └── mlx-db/                       # SQLite persistence (2,345 lines)
│
├── packages/
│   ├── core/                         # @mlx-node/core (native addon)
│   ├── lm/                           # @mlx-node/lm (808 lines)
│   │   └── src/
│   │       ├── stream.ts             # Session-aware model wrappers + `_runChatStream` callback→AsyncGenerator bridge
│   │       ├── chat-session.ts       # `ChatSession<M>` cross-model chat wrapper (send/sendStream/sendToolResult/reset)
│   │       ├── profiling.ts          # JS profiling API
│   │       ├── models/               # loadModel, Qwen3/3.5 configs
│   │       └── tools/                # Tool definition types
│   ├── trl/                          # @mlx-node/trl (5,298 lines)
│   │   └── src/
│   │       ├── trainers/             # GRPO trainer, SFT trainer, logger, configs
│   │       ├── data/                 # Dataset, SFT dataset
│   │       └── utils/                # XML parser, path security
│   ├── vlm/                          # @mlx-node/vlm (633 lines)
│   │   └── src/
│   │       ├── models/               # PaddleOCR-VL configs
│   │       └── pipeline/             # StructureV3Pipeline
│   └── cli/                          # @mlx-node/cli (932 lines)
│       └── src/commands/             # download-model, download-dataset, convert
│
├── __test__/                         # 415 TypeScript tests (9,125 lines)
│   ├── core/                         # Autograd, functional, profiling
│   ├── models/                       # All model tests
│   ├── trainers/                     # GRPO, SFT tests
│   ├── tokenization/                 # Tokenizer tests
│   ├── tools/                        # Chat tests
│   └── utils/                        # Dataset, XML, path security
│
└── examples/                         # Example scripts
```

---

## Development Guide

### Building

```bash
yarn install                      # Install dependencies
yarn build                        # Build native + TypeScript
yarn build:native                 # Build native addon only (~70s incremental)
yarn build:ts                     # Build TypeScript packages only (tsc -b)
cargo build --release -p mlx-tui  # Build mlx-train TUI binary
```

### Testing

```bash
yarn vite run test                      # Run all TS tests (415 tests)
yarn vitest __test__/path/to.test.ts    # Run specific TS test
cargo test -p mlx-core                  # Run Rust tests (810 tests)
cargo test -p mlx-paged-attn            # Run paged attention tests (83 tests)
```

### Lint & Format

```bash
yarn vite fmt                                           # Format TS/JS
yarn vite lint --type-aware --type-check                # Lint TS/JS
cargo clippy --all --fix --allow-dirty --allow-staged   # Rust lint
cargo fmt                                               # Rust format
```

### Build Flow

```
yarn build:native → packages/core/index.cjs + mlx-core.darwin-arm64.node + mlx.metallib
yarn build:ts     → packages/*/dist/ (via tsc -b with project references)
```

### Adding New Native Operations

1. Add FFI binding in `crates/mlx-sys/src/lib.rs`
2. Add C++ bridge in appropriate `crates/mlx-sys/src/mlx_*.cpp` file
3. Add Rust wrapper in `crates/mlx-core/src/` with `#[napi]` exports
4. Run `yarn build:native` to generate NAPI binding + TypeScript definitions
5. Add tests using TypedArray helpers

### Adding TypeScript Utilities

1. Add to appropriate package (`lm` for inference, `trl` for training, `vlm` for vision, `cli` for CLI)
2. Export from `packages/{package}/src/index.ts`
3. Run `yarn build:ts && yarn typecheck`

---

## Import Patterns

```typescript
// Inference + chat sessions
import { Qwen3Model, Qwen35Model, loadModel, loadSession, ChatSession, QWEN3_CONFIGS } from '@mlx-node/lm';

// Training
import { GRPOTrainer, GRPOConfig, SFTTrainer } from '@mlx-node/trl';

// Vision & Document Processing
import { VLModel, QianfanOCRModel, StructureV3Pipeline, DocLayoutModel } from '@mlx-node/vlm';

// Streaming chat via ChatSession
const session = await loadSession('./model-path');
for await (const event of session.sendStream('Hello!')) { ... }
```

---

## Performance

- **Metal GPU acceleration** on Apple Silicon
- **Compiled forward passes** via `mlx::core::compile` for graph caching
- **Qwen3.5 Dense**: ~6.4 tok/s on M3 Max (matches Python mlx-lm)
- **Qwen3.5 MoE**: ~47-52 tok/s (compiled + MXFP8 quantization)
- **Paged attention** with Metal kernels for memory-efficient serving
- **Quantization**: 4-bit affine, MXFP8, FP8 dequant, mixed-precision recipes
- **Quantized KV cache**: FP8/INT8 for reduced memory during inference

### Key Performance Patterns

- `token.eval()` after sampling — prevents unbounded lazy graph
- `synchronize_and_clear_cache()` every 256 steps — prevents memory accumulation
- Dtype-aware scalar ops — ANY f32 scalar in binary op with bf16 promotes entire result to f32
- Token-only eval — caches materialize through dependency graph (no need to eval all caches)

---

## C++ FFI Files (crates/mlx-sys/src/)

| File                   | Lines | Purpose                                                         |
| ---------------------- | ----- | --------------------------------------------------------------- |
| `mlx_advanced_ops.cpp` | 1,932 | quantized_matmul, gather_qmm, conv2d, FP8, PaddleOCR forward    |
| `mlx_nn_ops.cpp`       | 875   | Neural network ops, data extraction, random, math               |
| `mlx_qwen35_moe.cpp`   | 772   | Compiled MoE forward with expert routing                        |
| `mlx_fused_ops.cpp`    | 628   | Fused attention, SwiGLU MLP, transformer block                  |
| `mlx_array_ops.cpp`    | 622   | Array construction, arithmetic, indexing, dtype-safe scalar ops |
| `mlx_misc_ops.cpp`     | 591   | Synchronization, compiled sampling pipeline                     |
| `mlx_gated_delta.cpp`  | 388   | Metal GDN kernels, GPU architecture detection                   |
| `mlx_qwen35.cpp`       | 285   | Compiled Qwen3.5 dense forward                                  |
| `mlx_autograd.cpp`     | 223   | value_and_grad integration                                      |
| `mlx_qwen35_vlm.cpp`   | 199   | VLM compiled prefill                                            |
| `mlx_stream.cpp`       | 187   | Stream/device management, memory limits                         |
| `mlx_qwen35_common.h`  | 712   | Shared helpers: linear_proj, attn, GDN, RoPE                    |
| `mlx_common.h`         | 303   | FFI macros, error handling, array conversion                    |

---

## Security Model

Model files are assumed from **trusted sources**. Jinja2 chat templates from `tokenizer_config.json` are executed with user message content (minijinja sandbox — no file/code access, but malicious templates could cause DoS).

---

## Known Limitations

- macOS only (Metal backend, Apple Silicon)
- No CUDA support
- Compiled C++ forward paths use process-wide globals (serialized via Tokio mutex)

---

_Last updated: March 2026_
_Total Code: ~120,000 lines (100K Rust + 7.7K C++ + 2.5K Metal + 7.7K TS source + 9.1K TS tests)_
_Tests: 1,315 (900 Rust + 415 TypeScript)_
_Models: 9 (3 language + 2 VLM + 5 document processing)_
_NAPI Exports: 328 | FFI Functions: 221_

<!--VITE PLUS START-->

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
- [ ] Run `vp check` and `vp test` to validate changes.
<!--VITE PLUS END-->
