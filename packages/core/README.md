# @mlx-node/core

Native NAPI addon providing low-level bindings to Apple's [MLX](https://github.com/ml-explore/mlx) framework for Node.js. Powers all `@mlx-node/*` packages with Metal GPU-accelerated tensor operations, model inference, training engines, and document processing.

> **Note:** This package is an internal dependency. For most use cases, import from [`@mlx-node/lm`](https://github.com/mlx-node/mlx-node/tree/main/packages/lm) (inference), [`@mlx-node/trl`](https://github.com/mlx-node/mlx-node/tree/main/packages/trl) (training), or [`@mlx-node/vlm`](https://github.com/mlx-node/mlx-node/tree/main/packages/vlm) (vision) instead.

## Requirements

- macOS with Apple Silicon (M1 or later)
- Node.js 18+

## What's Inside

328 NAPI exports bridging ~100,000 lines of Rust, 7,700 lines of C++, and 2,500 lines of Metal shaders:

| Category              | Exports                                       | Description                                                                               |
| --------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **Tensor Ops**        | `MxArray` class (90+ methods)                 | Construction, arithmetic, reductions, indexing, shape manipulation, random, data transfer |
| **Neural Network**    | `nn` module                                   | Linear, Conv1d/2d, RMSNorm, LayerNorm, Embedding, activations, losses                     |
| **Language Models**   | `Qwen3Model`, `Qwen35Model`, `Qwen35MoeModel` | Full inference with generate, chat, streaming, paged attention, speculative decoding      |
| **Vision-Language**   | `VLModel`                                     | PaddleOCR-VL architecture — OCR, chat, batch inference                                    |
| **Document Pipeline** | 5 model classes                               | DocLayout (25 categories), TextDet, TextRec, DocOrientation, DocUnwarp                    |
| **Training**          | `GrpoTrainingEngine`, `SftTrainingEngine`     | GRPO/DAPO/Dr.GRPO/BNPO and SFT with autograd, gradient accumulation, NaN recovery         |
| **Optimizers**        | Adam, AdamW, SGD, RMSprop                     | Full optimizer suite with weight decay and gradient clipping                              |
| **Tokenization**      | `Qwen3Tokenizer`                              | HuggingFace tokenizers + Jinja2 chat templates                                            |
| **Sampling**          | Configurable pipeline                         | Temperature, top-k/p, min-p, repetition penalty                                           |
| **Conversion**        | `convertModel`, `convertGgufToSafetensors`    | Dtype casting, 4/8-bit quantization, MXFP8, quantization recipes, GGUF import             |
| **Profiling**         | Profiling store                               | Per-generation timing, TTFT, memory snapshots, phase breakdown                            |
| **Persistence**       | `OutputStore`                                 | SQLite-backed training run recording and querying                                         |

## Architecture

```
Node.js (JavaScript/TypeScript)
    │
    ▼
@mlx-node/core  ←  NAPI-RS auto-generated bindings
    │
    ▼
mlx-core (Rust)  ←  models, training, ops, vision
    │
    ▼
mlx-sys (C++)    ←  compiled forward passes, FFI bridge
    │
    ▼
MLX (C++)        ←  Apple's ML framework
    │
    ▼
Metal / Accelerate  ←  GPU compute + BLAS
```

## Key Types

```typescript
import {
  // Tensors
  MxArray,
  DType,
  // Models
  Qwen3Model,
  Qwen35Model,
  Qwen35MoeModel,
  // Vision
  VLModel,
  DocLayoutModel,
  TextDetModel,
  TextRecModel,
  // Training
  GrpoTrainingEngine,
  SftTrainingEngine,
  NativeRewardRegistry,
  // Tokenizer
  Qwen3Tokenizer,
  // Config
  GenerationConfig,
  ChatConfig,
  GrpoEngineConfig,
  SftEngineConfig,
  // Conversion
  ConversionOptions,
  // Persistence
  OutputStore,
} from '@mlx-node/core';
```

## Platform Support

| Platform | Architecture          | Status      |
| -------- | --------------------- | ----------- |
| macOS    | Apple Silicon (arm64) | Supported   |
| macOS    | Intel (x86_64)        | Unsupported |
| Linux    | CUDA                  | Coming soon |
| Windows  | CUDA                  | Coming soon |

## Related Packages

| Package                                                                        | Purpose                                                         |
| ------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| [`@mlx-node/lm`](https://github.com/mlx-node/mlx-node/tree/main/packages/lm)   | High-level inference API with streaming and model loading       |
| [`@mlx-node/trl`](https://github.com/mlx-node/mlx-node/tree/main/packages/trl) | GRPO and SFT training with datasets and rewards                 |
| [`@mlx-node/vlm`](https://github.com/mlx-node/mlx-node/tree/main/packages/vlm) | Vision-language models and document processing pipelines        |
| [`@mlx-node/cli`](https://github.com/mlx-node/mlx-node/tree/main/packages/cli) | CLI for model download, dataset download, and weight conversion |

## License

[MIT](https://github.com/mlx-node/mlx-node/blob/main/LICENSE)
