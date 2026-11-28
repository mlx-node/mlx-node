# MLX-Node Examples

Build the workspace once before running examples:

```bash
yarn build
```

Run any TypeScript example with `oxnode` (not `tsx`).

## Downloading Models

Use the built-in CLI to fetch checkpoints into `.cache/models/` (no Python tooling required):

```bash
yarn mlx download model
```

## Examples

### Chat & Language Models (`@mlx-node/lm`)

- **`examples/lm.ts`** — 4-turn streaming `ChatSession` demo with TTFT / prefill / decode throughput metrics and KV cache reuse checks. Defaults to `.cache/models/qwen3.5-9B-unsloth` and optionally accepts `[model-name]` plus `--image <path>` / `--image2 <path>` for VLM turns:
  ```bash
  oxnode examples/lm.ts
  oxnode examples/lm.ts qwen3.5-9B-unsloth --image examples/ocr.png
  ```
- **`examples/tool-use-example.ts`** — Tool calling with `ChatSession` and `sendToolResult`.
- **`examples/gemma4-qat-smoke.ts`** — Gemma 4 E2B QAT greedy text (and optional image) smoke check.

### Vision & Document OCR (`@mlx-node/vlm`)

- **`examples/vlm-inference.ts`** — Multi-turn Qianfan-OCR VLM chat.
- **`examples/paddle-ocr-pipeline.ts`** — PP-DocLayoutV3 + PaddleOCR-VL-1.5 document OCR pipeline (uses `examples/ocr.png`).

### Text-to-Speech (`@mlx-node/tts`)

- **`examples/tts/streaming.ts`** — Streaming Qwen3-TTS synthesis.
- **`examples/tts/instruct.ts`** — Instruction-controlled / VoiceDesign synthesis.
- **`examples/tts/http-stream.ts`** — Streaming `/v1/audio/speech/stream` against `@mlx-node/server`.

### Training (`@mlx-node/trl`)

- **`examples/sft/`** — SFT fine-tuning (`sft-octokit.ts`) and dataset generation (`generate-dataset.ts`).
- **`examples/grpo/`** — Multi-turn GRPO GitHub tool training (`train-github-tool.ts`; see [examples/grpo/README.md](grpo/README.md)).
